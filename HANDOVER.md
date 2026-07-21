# Handover: OSS model PR + behavioural testing prototype (Hawesko voice agent)

## Goal
Two intertwined workstreams:
1. **Model PR (#340, branch `glm-5.2-model` → `staging`)** — bring cheaper/faster OSS models that are
   on-par-or-better than gpt-5.2 and beat gpt-oss-120b (which underperforms for the Hawesko voice agent)
   into the platform, to eventually A/B in prod and cut the ~$40k/mo OpenAI spend.
2. **Testing prototype** — a lightweight local tool to rigorously, defensibly decide which models to
   include/exclude, and to seed the future test-cases-rework / evals RFC. Recorded for FDE demo
   (Marc/Kevin) to gauge if ablation testing helps them too.

## The core mechanic (how routing/function-calling works — verified in code)
A `dialogue` stage sends ONE LLM request: `system = agent.system_message + "\n\n" + stage.stage_message`
(with `{{field}}` substitution), history in OpenAI format, **one tool per transition + one per function**,
`tool_choice="auto"`, temp 0.1, `reasoning_effort="none"` forced for gpt-oss-120b & gpt-5.2
(`core/leaping/llm.py:113`). Transition tool name = display name with spaces→underscores, zero params
(`transition.py:openai_schema`). Firing the transition = calling that tool; **speaking the transition
label as text instead is the gpt-oss-120b prod failure** (conv `06a5a0a2`, spoke "Bestellung – Beratung"
×3). Model alias→string in `config.py:llm_model_mapping`. Stage/tools built via
`ctx.init_stages(stages, stage_id)` then `ctx.get_system_message()` +
`stage.functions/transitions[*].openai_schema`.

## Artifacts produced (all committed/saved)
**In monorepo `/Users/PWR/Documents/Professional/Leaping/core_repos/leaping`:**
- `backend/scripts/hawesko_turn_compare.py` — quick CLI: hand-embedded (trimmed) Intenterkennung prompt,
  replays 2 intent turns (Bestellung, Lieferstatus) across all models, scores routing/leak +
  latency/tokens/cost. Includes Nemotron Ultra (slug 404s) + DeepSeek-reason-off. Writes
  `hawesko_turn_results.json`.
  Run: `cd <repo> && set -a && source voice/.env && set +a && .venv/bin/python backend/scripts/hawesko_turn_compare.py --reps 3`
- `backend/scripts/hawesko_snapshot_replay.py` — faithful CLI: builds request **through core** from a
  snapshot+transcript (no trimming). `init_stages` fix applied (uncommitted in monorepo). Needs snapshot
  JSON or staging DB creds.

**New separate git repo `/Users/PWR/Documents/Professional/Leaping/core_repos/snapshot-replay-lab`**
(git-inited locally, NOT yet pushed — push to your own `PaWeRe` account; named to avoid clashing with the
existing `voice-agent-evals`):
- `replay_engine.py` — core-driven request builder + streamed model runner + scoring. Imports only
  `leaping`+`openai`. Auto-infers active stage + expected tool from the real transcript. `PRESET_MODELS`.
- `app.py` — Streamlit UI: drag-drop snapshot+transcript → pick cutoff turn → auto stage/expected-tool
  (overridable) → model multiselect + custom slugs → prompt overrides → behavior scoreboard → optional
  A/B snapshot.
- `run.sh` — **the way to launch** (see gotcha below). `README.md`, `.gitignore`, this `HANDOVER.md`.
- Run: `cd snapshot-replay-lab && ./run.sh` (drag in Hawesko snapshot + Call-2 "Bestellung" transcript,
  pick `…bestellen` turn → auto-selects `Intenterkennung` + `Bestellung_-_Beratung`).

**Canvas** (latency/cost/caching report):
`/Users/PWR/.cursor/projects/Users-PWR-Documents-Professional-Leaping-core-repos-leaping/canvases/hawesko-model-vibe-check.canvas.tsx`

## Environment gotchas (important)
- **`leaping` cannot be resolved standalone** anymore: only safe `semantic-router` (post
  CVE-2026-42208) forces `litellm≥1.84 → openai≥2 / tiktoken 0.12 / tokenizers 0.22`, conflicting with
  leaping's pins. Monorepo only works via its committed `uv.lock`. → The Streamlit tool **runs on the
  monorepo venv** (`run.sh` adds streamlit+pandas additively — verified 12 new pkgs, no changes to
  existing). Don't `uv run streamlit run app.py`.
- **Streamlit doesn't hot-reload imported modules** — after editing `replay_engine.py`, restart `./run.sh`.
- Everything needs `voice/.env` sourced (Baseten/OpenAI keys + Google/Vertex creds validated at
  `leaping.config` import).

## Key findings so far
**Behaviour (faithful-ish single-turn intent replay, 3 reps):**
- **gpt-oss-120b**: 6/6 clean in isolation — the prod loop did NOT reproduce with the trimmed prompt
  (fidelity caveat; use the snapshot-driven replay with the verbatim/huge Intenterkennung prompt — likely
  trigger).
- **Kimi K2.6**: 6/6 clean, fast, reasoning-free → strongest OSS candidate to promote.
- **DeepSeek V4 Pro**: 6/6 clean; `reasoning_effort=none` works (out tok 106→29, total ~1.36→0.74s) → run
  it reasoning-off.
- **GLM 5.2**: 6/6 clean but slow + jittery TTFT (1.2→2.2s on shared API).
- **Nemotron Super**: FAIL — leaks `</think>` into the speech channel + unstable/empty tool calls. Reject
  for voice.
- **gpt-5.2**: 6/6 clean baseline (priciest output).

**Cost (June CSVs, reconciled):** $41,557 billed (all endpoints); ~$23,848 completions-only ≈ the $26k
quick-math (gap = realtime audio/STT/TTS/embeddings + priority premium). 20.0B input / 0.333B output =
**60:1** (input is O(n²) since we resend transcript each turn). **45.8% cached** but wildly uneven:
gpt-5.2 84%, gpt-4.1 73%, **gpt-4o only 7%, gpt-4o-mini 0%**. gpt-4o alone ≈ $13.3k/mo — biggest lever.
On cached input, OSS ($0.03–0.26) still beats OpenAI cached ($0.18–1.25) *if* Baseten's shared API caches
our prefixes (unknown; guaranteed only on a dedicated deployment). API vs dedicated: shared API = same
latency jitter as OpenAI; dedicated = fixed GPU-hour cost + owned latency (supervisor: "API now,
self-host winner later").

## Open items / next steps
**Model PR:**
- [ ] Get exact **Baseten slugs** for Nemotron Ultra (my `nvidia/Nemotron-Ultra-253B-A22B` 404'd), Kimi
      K3, Gemma "64b" — paste into the tool's custom-models box.
- [ ] Run the **snapshot-driven replay** with the verbatim Hawesko snapshot to (a) confirm whether the
      full Intenterkennung prompt reproduces the gpt-oss loop, (b) lock the include/exclude list.
- [ ] Spot-check 1–2 harder turns (auth birthday parse, order quantity) — pick a different cutoff turn.
- [ ] Keep PR #340 scoped to model integration + latency/cost/behavioural rationale; behaviour validated
      later in prod A/B.

**Testing prototype:**
- [ ] Add multi-turn / saved test-suite support + pluggable scoring (LLM judge, regex leak rules, tool-arg
      assertions) — this is the evals/test-cases-rework RFC.
- [ ] Optionally add 2–3 seed turns (auth/order) for the FDE demo to show it catching a
      function-call/arg-extraction failure, not just routing.
- [ ] Push repo to PaWeRe account; record usage; get FDE feedback.

**Separate future RFCs (not this PR):** Context-Engineering (O(n) rolling-state context + prefix-cache
hardening + track `prompt_tokens_details.cached_tokens` per model/agent — investigate why gpt-4o caches
at 7%: likely volatile content early in the system prompt / prompt <1024 tok / many distinct templates —
ask Marc); Analytics (cache/cost dashboards); Testing + Evals RFCs.

## To provide in the fresh session
Hawesko agent snapshot `809ee65d` (snapshot `019f662a`) as JSON + the two Call transcripts (Call 1
success Lieferauskunft, Call 2 failed Bestellung) — needed to run the faithful replay. Both were pasted
in the original handover; export them as `.json` for drag-drop, or run against staging DB creds.

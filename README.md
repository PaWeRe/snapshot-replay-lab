# Snapshot Replay Lab

A **super lightweight, local** harness for **behavioural** ablation of voice-agent
LLM turns. Drag-drop a real agent **snapshot** + a real call **transcript**, pick a
tricky turn, and compare how different **models / prompts / snapshot versions**
behave on the *exact* request production would send.

It exists to answer one question fast and defensibly: **"if I swap the model (or
edit the prompt) on this stage, does the agent still do the right thing?"** —
before committing anything to a prod A/B.

## Why it's faithful

The request (system prompt with `{{field}}` substitution + one tool per
transition/function) is built by importing **`core/leaping` itself** — nothing is
hand-reconstructed. Swapping the model changes only the model; the prompt and
tools are byte-for-byte what the platform sends.

## What you can do

- Load a snapshot + transcript (drag-drop, files stay local).
- Select a **cutoff turn** (a customer utterance). The tool **auto-infers**:
  - the **active stage** at that turn (last stage the real call transitioned into),
  - the **ground-truth continuation** (what the real call did next), split into the
    *immediate next response* (the fair bar for one request) and the *full
    trajectory* (context), and
  - the **decision turns** of the call (turns where the model actually chose an
    action), so you can score a whole call in one click.
- **Inspect the request byte-for-byte** — the exact thing the API receives:
  - both system messages (base+stage with `{{fields}}` substituted from the call,
    plus the trailing `extra_instructions`),
  - the full OpenAI message history (roles, content, tool_calls **with arguments**,
    tool results),
  - every exposed tool's full JSON schema, and
  - the exact per-provider request payload (params, `reasoning_effort`, etc.).
- Multi-select **models** (all 26 presets by default — OpenAI, Gemini-via-Vertex,
  Baseten OSS; add your own `name,provider,slug,in$,out$` lines).
- Optionally **override the system or stage prompt** to test prompt changes.
- **Run** → each model's output is scored by an **LLM judge (gpt-5.2)** that returns
  **PASS/FAIL** against the real call's crucial actions — allowing multiple valid
  solutions, argument-equivalence, and order-tolerance where the stage permits.
  Plus measured **TTFT / total latency / output tokens / cached tokens** and
  **dynamic $ cost** (LiteLLM's maintained pricing × measured tokens; falls back to
  a flagged config estimate for slugs LiteLLM doesn't know).
- **Whole-call comparison**: replay every decision turn independently (each with its
  real history + ground truth), judge each, and aggregate into a **per-model
  conversation pass rate**. Deterministic, no simulated user — the right instrument
  for *model selection* (simulation lives in `voice-agent-evals`).
- Load a **Snapshot B** to A/B two agent versions on the same turn.

### Why the reconstruction is faithful

- History is rebuilt through leaping's **voice** path (`merge_event_logs` +
  `to_openai_messages`) — the messages the prod voice LLM actually saw. Hybrid
  exports that carry *both* voice `message` and core `chat_message` records no longer
  double-count turns (the earlier "duplicated turns" bug), and human turns (which
  live only as voice `message` events) are no longer dropped.
- `{{field}}` values are replayed from the call's `field_update` events, so the
  system prompt matches what prod sent at that point in time — not an empty template.
- Tools = stage functions + all transitions (+ KB tool), `tool_choice="auto"`,
  matching `generative_stage.stream_response`.

Seed example: the bath appointment cancellation call — at the address turn, the prod
model (gpt-4.1) silently fires `update_street`+`update_city`; gpt-4.1-mini
hallucinates a ZIP, gemini-2.5-flash drops the city, gpt-oss-120b speaks instead of
routing — all caught by the judge, missed by naive tool-name matching.

## Setup

The tool imports `leaping`, and `leaping`'s pinned deps can **no longer be
resolved standalone** (the only safe `semantic-router`, post CVE-2026-42208,
pulls `litellm>=1.84 → openai>=2 / tiktoken 0.12 / tokenizers 0.22`, which
conflict with leaping's pins — the monorepo only works via its committed
`uv.lock`). So this tool **runs on the monorepo's venv** instead of building its
own. `./run.sh` handles it:

```bash
cd snapshot-replay-lab
./run.sh                          # adds streamlit+pandas to ../leaping/.venv (once), sources env, launches
# monorepo elsewhere?  LEAPING_REPO=/path/to/leaping ./run.sh
```

`run.sh` sources `../leaping/voice/.env` for `BASETEN_API_KEY`, `OPENAI_API_KEY`
and the Google/Vertex creds that `leaping.config` validates at import. Then open
the local URL, drag in your two JSON files, and go.

> Do **not** `uv run streamlit run app.py` here — that would try to resolve
> `leaping` standalone and fail. Use `./run.sh`.

## Roadmap (candidate for the test-cases-rework RFC)

Done in this iteration: faithful voice-path history reconstruction, `{{field}}`
substitution, `extra_instructions`, byte-for-byte request inspector, LLM judge
(gpt-5.2) with immediate-vs-trajectory reference, dynamic cost + cached tokens,
whole-call per-model pass rate, parallel 26-model sweeps.

Next:
- Pull snapshot/transcript by ID from the platform (and by call-id) instead of
  drag-drop; batch a *selection* of prod calls per agent for a stable per-model score.
- Persisted runs + shareable behaviour reports; regression suites of decision turns.
- Judge calibration: small human-labelled set to confirm the gpt-5.2 verdicts, and a
  cheaper judge option. Reuse core's stage/conversation judges
  (`core/leaping/eval/criteria/llm_judge/`) where useful.
- Optional: additive tool-arg assertions / regex leak rules alongside the judge.
- Deliberately NOT here: user-simulation (that's `voice-agent-evals`); this tool
  stays deterministic replay so the model is the only variable.

Files never committed: uploaded snapshots/transcripts, `.venv`, results.

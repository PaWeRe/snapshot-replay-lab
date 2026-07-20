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

## What you can do (v0)

- Load a snapshot + transcript (drag-drop, files stay local).
- Select a **cutoff turn** (a customer utterance). The tool **auto-infers**:
  - the **active stage** at that turn (last stage the real call transitioned into), and
  - the **expected tool call** (what the real call did next) — the behavioural ground truth.
  Both are overridable.
- Multi-select **models** (presets + your own `name,provider,slug,in$,out$` lines,
  e.g. Gemma / Kimi K3 once you have the Baseten slug).
- Optionally **override the system or stage prompt** to test prompt changes.
- **Run** → behaviour scoreboard: did each model fire the right tool, or leak the
  label / `<think>` into speech? Plus TTFT / total / output tokens / $ per 1k calls.
- Load a **Snapshot B** to A/B two agent versions on the same turn.

Seed example: the Hawesko `Intenterkennung` turn where gpt-oss-120b looped in prod
(spoke "Bestellung – Beratung" instead of firing the transition).

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

- Scoring plugins (LLM judge, regex leak rules, tool-arg assertions) — the evals RFC.
- Multi-turn replay (run several tricky turns at once) + saved test suites.
- Pull snapshot/transcript by ID from the platform instead of drag-drop.
- Persisted runs + shareable behaviour reports.

Files never committed: uploaded snapshots/transcripts, `.venv`, results.

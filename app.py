"""
Snapshot Replay Lab — a lightweight local harness for behavioural ablation of
voice-agent LLM turns.

Workflow:
  1. Drag-drop an agent snapshot JSON + a call transcript JSON (both locally
     downloaded from the platform).
  2. Pick a "tricky" cutoff turn (a customer utterance). The tool auto-infers
     the active stage and the ground-truth expected tool (what the real call did
     next), both overridable.
  3. Swap models / edit the system or stage prompt.
  4. Run → behaviour scoreboard: did each model fire the right tool, or leak the
     label / <think> into speech? + latency / tokens / cost.

Run:
    set -a && source ../leaping/voice/.env && set +a && uv run streamlit run app.py
(voice/.env provides BASETEN_API_KEY, OPENAI_API_KEY and the Google/Vertex creds
 that leaping.config validates at import.)
"""

from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

import replay_engine as re

st.set_page_config(page_title="Snapshot Replay Lab", layout="wide")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _read_json(uploaded) -> dict | list:
    return json.loads(uploaded.getvalue().decode("utf-8"))


def _stage_message_of(agent_model, stage_id: str) -> str:
    for s in agent_model.stages:
        if str(s.id) == stage_id:
            return getattr(s, "stage_message", "") or ""
    return ""


def _outcome_emoji(o: str) -> str:
    return {"PASS": "✅", "pass_extra_tool": "✅"}.get(o, "•")


# --------------------------------------------------------------------------- #
# Sidebar — inputs
# --------------------------------------------------------------------------- #

st.sidebar.title("Snapshot Replay Lab")
st.sidebar.caption("Faithful (core-built) single-turn ablation for voice agents.")

have_baseten = bool(os.environ.get("BASETEN_API_KEY"))
have_openai = bool(os.environ.get("OPENAI_API_KEY"))
if not (have_baseten or have_openai):
    st.sidebar.error("No API keys in env. Source voice/.env before launching.")
else:
    st.sidebar.caption(
        f"keys: {'baseten ✓ ' if have_baseten else 'baseten ✗ '}"
        f"{'openai ✓' if have_openai else 'openai ✗'}"
    )

snap_file = st.sidebar.file_uploader("Agent snapshot JSON", type=["json"], key="snap")
tx_file = st.sidebar.file_uploader("Call transcript JSON", type=["json"], key="tx")
snap_b_file = st.sidebar.file_uploader("Snapshot B (optional, for A/B)", type=["json"], key="snapb")

reps = st.sidebar.number_input("Reps per model", 1, 10, 3)
model_names = st.sidebar.multiselect(
    "Models",
    [m.name for m in re.PRESET_MODELS],
    default=["gpt-oss-120b", "Kimi K2.6", "gpt-5.2"],
)
custom = st.sidebar.text_area(
    "Custom models (one per line: name,provider,slug,in$,out$)",
    placeholder="Gemma 64B,baseten,google/gemma-64b,0.20,0.60\nKimi K3,baseten,moonshotai/Kimi-K3,1.20,4.50",
    height=80,
)

if not snap_file or not tx_file:
    st.info("Load a snapshot + transcript to begin. Files stay local; only the LLM request leaves your machine.")
    st.stop()

# --------------------------------------------------------------------------- #
# Parse inputs
# --------------------------------------------------------------------------- #

try:
    agent_model = re.load_snapshot(_read_json(snap_file))
    events = re.load_transcript_events(_read_json(tx_file))
except Exception as e:  # noqa: BLE001
    st.error(f"Failed to parse inputs: {e}")
    st.stop()

user_turns = re.list_user_turns(events)
if not user_turns:
    st.error("No customer (human) turns found in transcript.")
    st.stop()

# --------------------------------------------------------------------------- #
# Turn / stage / expected-tool selection
# --------------------------------------------------------------------------- #

st.header(f"{agent_model.name}")
st.caption(f"{len(events)} events · {len(user_turns)} customer turns")

turn_labels = [f"#{t.index}  ·  {t.text[:80]}" for t in user_turns]
sel = st.selectbox("Cutoff turn (customer utterance to test)", range(len(user_turns)), format_func=lambda i: turn_labels[i], index=len(user_turns) - 1)
turn = user_turns[sel]

stages = re.list_stages(agent_model)
stage_ids = [sid for sid, _ in stages]
stage_labels = {sid: name for sid, name in stages}

inferred_stage = re.infer_stage_at(events, turn.index, str(agent_model.first_stage))
inferred_tool = re.infer_expected_tool_after(events, turn.index)

c1, c2 = st.columns(2)
with c1:
    stage_id = st.selectbox(
        "Active stage (auto-inferred)",
        stage_ids,
        index=stage_ids.index(inferred_stage) if inferred_stage in stage_ids else 0,
        format_func=lambda sid: f"{stage_labels.get(sid, sid)}",
    )
with c2:
    expected_tool = st.text_input(
        "Expected tool call (ground truth from the real call)",
        value=inferred_tool or "",
        help="What the real call did next at this turn. Leave blank to just observe.",
    )

with st.expander("Prompt overrides (optional — test prompt changes)"):
    system_override = st.text_area("System message override (blank = snapshot default)", height=100)
    stage_message_override = st.text_area(
        "Stage message override (prefilled with current stage prompt)",
        value=_stage_message_of(agent_model, stage_id),
        height=160,
    )

# --------------------------------------------------------------------------- #
# Build request (no network) — preview
# --------------------------------------------------------------------------- #

default_stage_msg = _stage_message_of(agent_model, stage_id)


def _build(model, snap_events):
    return re.build_request(
        model,
        snap_events,
        turn.index,
        stage_id,
        system_override=system_override or None,
        stage_message_override=stage_message_override if stage_message_override != default_stage_msg else None,
    )


try:
    req = _build(agent_model, events)
except Exception as e:  # noqa: BLE001
    st.error(f"Could not build request via core: {e}")
    st.stop()

st.caption(f"Stage **{req.stage_name}** exposes {len(req.exposed_tools)} tools: {', '.join(req.exposed_tools)}")
with st.expander(f"Request preview ({len(req.messages)} messages, system {len(req.system_message)} chars)"):
    st.code(req.system_message[:4000] + ("…" if len(req.system_message) > 4000 else ""), language="markdown")
    st.json([{"role": m.get("role"), "content": (m.get("content") or "")[:200], "tool_calls": bool(m.get("tool_calls"))} for m in req.messages])

# --------------------------------------------------------------------------- #
# Assemble model specs
# --------------------------------------------------------------------------- #

specs = [m for m in re.PRESET_MODELS if m.name in model_names]
for line in custom.splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) >= 3:
        name, provider, slug = parts[0], parts[1], parts[2]
        pin = float(parts[3]) if len(parts) > 3 and parts[3] else 0.0
        pout = float(parts[4]) if len(parts) > 4 and parts[4] else 0.0
        specs.append(re.ModelSpec(name, provider, slug, pin, pout))

# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #


def _report_df(reports: list[re.ModelReport]) -> pd.DataFrame:
    rows = []
    for r in reports:
        rows.append(
            {
                "Model": r.name,
                "Pass": f"{r.passes}/{r.reps}",
                "Outcomes": " ".join(f"{_outcome_emoji(o)}{o}" for o in r.outcomes),
                "TTFT s": r.ttft_med,
                "Total s": r.total_med,
                "Out tok": r.out_tok_med,
                "$/1k": round(r.cost_med * 1000, 3) if r.cost_med else None,
                "Error": r.error or "",
            }
        )
    return pd.DataFrame(rows)


def _show_reports(reports: list[re.ModelReport], key: str):
    st.dataframe(_report_df(reports), use_container_width=True, hide_index=True)
    for r in reports:
        if r.sample_content or r.sample_tools:
            with st.expander(f"{r.name} — sample output", expanded=False):
                st.write("tool calls:", r.sample_tools or "—")
                st.write("spoken content:", repr(r.sample_content) or "—")


if st.button("Run comparison", type="primary", disabled=not specs):
    with st.spinner(f"Running {len(specs)} models × {reps} reps…"):
        reports_a = re.run_comparison(req, specs, int(reps), expected_tool or None)
    st.subheader("Snapshot A" if snap_b_file else "Results")
    _show_reports(reports_a, "a")

    if snap_b_file:
        try:
            agent_b = re.load_snapshot(_read_json(snap_b_file))
            req_b = _build(agent_b, events)
            with st.spinner("Running Snapshot B…"):
                reports_b = re.run_comparison(req_b, specs, int(reps), expected_tool or None)
            st.subheader("Snapshot B")
            st.caption(f"Stage **{req_b.stage_name}** · {len(req_b.exposed_tools)} tools")
            _show_reports(reports_b, "b")
        except Exception as e:  # noqa: BLE001
            st.error(f"Snapshot B failed: {e}")

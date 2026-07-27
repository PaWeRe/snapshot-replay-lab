"""
Snapshot Replay Lab — a local harness for behavioural ablation of voice-agent
LLM turns.

Workflow:
  1. Drop an agent snapshot JSON + a call transcript JSON (both locally
     downloaded from the platform).
  2. Pick a cutoff turn (a customer utterance). The tool auto-infers the active
     stage and the ground-truth continuation (what the real call did next).
  3. Inspect the *exact* request production would send (byte-for-byte).
  4. Replay it across any set of models → an LLM judge (gpt-5.2) returns a
     per-turn PASS/FAIL against the real call's crucial actions, plus measured
     latency / tokens / dynamic cost.
  5. Score a model across every decision turn of the call → per-conversation
     pass rate.

Run:
    ./run.sh
(sources voice/.env for BASETEN/OPENAI keys + Vertex creds and runs on the
 monorepo venv — see README.)
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


def _verdict_emoji(v: str) -> str:
    return {"PASS": ":material/check_circle:", "FAIL": ":material/cancel:"}.get(v, "·")


# --------------------------------------------------------------------------- #
# Sidebar — inputs
# --------------------------------------------------------------------------- #

st.sidebar.title("Snapshot Replay Lab")
st.sidebar.caption("Faithful single-request replay + LLM judge for voice agents.")

have_baseten = bool(os.environ.get("BASETEN_API_KEY"))
have_openai = bool(os.environ.get("OPENAI_API_KEY"))
have_gemini = bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_BASE64"))
if not (have_baseten or have_openai or have_gemini):
    st.sidebar.error("No API keys in env. Source voice/.env before launching.")
else:
    st.sidebar.caption(
        f"keys: {'baseten ✓ ' if have_baseten else 'baseten ✗ '}"
        f"{'openai ✓ ' if have_openai else 'openai ✗ '}"
        f"{'gemini ✓' if have_gemini else 'gemini ✗'}"
    )
if not have_openai:
    st.sidebar.warning("OpenAI key missing — the gpt-5.2 judge needs it.")

snap_file = st.sidebar.file_uploader("Agent snapshot JSON", type=["json"], key="snap")
tx_file = st.sidebar.file_uploader("Call transcript JSON", type=["json"], key="tx")
snap_b_file = st.sidebar.file_uploader("Snapshot B (optional, for A/B)", type=["json"], key="snapb")

st.sidebar.divider()
reps = st.sidebar.number_input("Reps per model", 1, 10, 1, help="Independent samples per model per turn. Keep low for a full 26-model sweep.")
use_judge = st.sidebar.toggle("LLM judge (pass/fail)", value=True, help="Judge each output against the real call's crucial actions.")
judge_model = st.sidebar.selectbox(
    "Judge model",
    [m.name for m in re.PRESET_MODELS if m.provider == "openai"],
    index=[m.name for m in re.PRESET_MODELS if m.provider == "openai"].index("gpt-5.2"),
    disabled=not use_judge,
)

_all_model_names = [m.name for m in re.PRESET_MODELS]
model_names = st.sidebar.multiselect("Models to compare", _all_model_names, default=_all_model_names)
custom = st.sidebar.text_area(
    "Custom models (one per line: name,provider,slug,in$,out$)",
    placeholder="name,provider,slug,in$,out$  (provider = baseten | openai | gemini)\n"
    "My model,gemini,gemini-3.6-flash,0.30,2.50",
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

api_calls = re.list_api_calls(agent_model, events, str(agent_model.first_stage))
decision_indices = set(re.list_decision_turns(events, str(agent_model.first_stage)))

# --------------------------------------------------------------------------- #
# Assemble model specs (needed for the payload inspector too)
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
# Cutoff selection (per-API-call or per-human-turn)
# --------------------------------------------------------------------------- #

st.header(agent_model.name)
hc1, hc2, hc3 = st.columns(3)
hc1.metric("LLM API calls", len(api_calls), help="Generative-stage completions the real call made (parallel tool calls = 1).")
hc2.metric("Customer turns", len(user_turns))
hc3.metric("Events", len(events))

_kind_icon = {"tools": ":material/build:", "transition": ":material/alt_route:", "speech": ":material/chat:"}

granularity = st.segmented_control(
    "Cutoff granularity",
    ["Every API call", "Human turns"],
    default="Every API call",
    help="Every API call = one selectable cutoff per LLM decision (recommended). Human turns = only the first response after each customer utterance.",
)

if granularity == "Every API call":
    cutoffs = [
        {"index": c.cutoff_index, "stage_id": c.stage_id,
         "label": f"{_kind_icon.get(c.kind, '')} {c.stage_name} · {c.label}"}
        for c in api_calls
    ]
else:
    cutoffs = [
        {"index": t.index, "stage_id": None,
         "label": (":material/bolt: " if t.index in decision_indices else "") + f"#{t.index} · {t.text[:70]}"}
        for t in user_turns
    ]

if not cutoffs:
    st.warning("No LLM API calls found in this transcript.")
    st.stop()

default_sel = len(cutoffs) - 1
sel = st.selectbox("Cutoff to test", range(len(cutoffs)), format_func=lambda i: cutoffs[i]["label"], index=default_sel)
cut = cutoffs[sel]
cut_index = cut["index"]

stages = re.list_stages(agent_model)
stage_ids = [sid for sid, _ in stages]
stage_labels = {sid: name for sid, name in stages}

inferred_stage = cut["stage_id"] or re.infer_stage_at(events, cut_index, str(agent_model.first_stage))
inferred_tool = re.infer_expected_tool_after(events, cut_index)
reference = re.reference_continuation(events, cut_index)

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
        "Expected tool call (heuristic ground truth)",
        value=inferred_tool or "",
        help="First LLM action the real call took next. The judge uses the fuller reference below.",
    )

with st.expander("Prompt overrides (optional — test prompt changes)"):
    system_override = st.text_area("System message override (blank = snapshot default)", height=100)
    stage_message_override = st.text_area(
        "Stage message override (prefilled with current stage prompt)",
        value=_stage_message_of(agent_model, stage_id),
        height=160,
    )

# --------------------------------------------------------------------------- #
# Build request (no network)
# --------------------------------------------------------------------------- #

default_stage_msg = _stage_message_of(agent_model, stage_id)


def _build(model, snap_events):
    return re.build_request(
        model,
        snap_events,
        cut_index,
        stage_id,
        system_override=system_override or None,
        stage_message_override=stage_message_override if stage_message_override != default_stage_msg else None,
    )


try:
    req = _build(agent_model, events)
except Exception as e:  # noqa: BLE001
    st.error(f"Could not build request via core: {e}")
    st.stop()

_history_msgs = [m for m in req.messages if m.get("role") in ("user", "assistant") or m.get("tool_calls")]
mc1, mc2, mc3 = st.columns(3)
mc1.metric("Exposed tools", len(req.exposed_tools))
mc2.metric("History messages", len(_history_msgs))
mc3.metric("System prompt", f"{len(req.system_message):,} chars")
st.caption(
    f"Stage **{req.stage_name}** (prod model: {req.prod_model_alias or 'agent default'}) exposes: "
    + ", ".join(req.exposed_tools)
)
if not _history_msgs:
    st.warning("Reconstructed history is empty — transcript shape unrecognized. Check the export.")

# --------------------------------------------------------------------------- #
# Byte-for-byte request inspector
# --------------------------------------------------------------------------- #

st.subheader("Request inspector — exactly what the API receives")
tabs = st.tabs(["System prompt", "Messages (history)", "Tools (schemas)", "Exact API payload", "Real call did next"])

with tabs[0]:
    st.caption("system message[0] — base system message + stage message, with {{fields}} substituted from the call.")
    st.code(req.system_message or "(empty)", language="markdown")
    if req.extra_instructions:
        st.caption("Trailing system message — stage extra_instructions (appended after history, raw).")
        st.code(req.extra_instructions, language="markdown")

with tabs[1]:
    st.caption("Full OpenAI-format message array (roles, content, tool_calls with arguments, tool results) — byte-for-byte.")
    st.json(req.messages)

with tabs[2]:
    st.caption("Every tool exposed this turn, full JSON schema (one per function + one per transition).")
    st.json(req.tools)

with tabs[3]:
    insp_name = st.selectbox("Model", [s.name for s in specs] or ["(select models)"], key="inspect_model")
    insp_spec = next((s for s in specs if s.name == insp_name), None)
    if insp_spec:
        st.caption(f"The exact request body sent to **{insp_spec.provider}** for `{insp_spec.model}`.")
        st.json(re.render_api_payload(insp_spec, req))

with tabs[4]:
    st.caption("Ground truth the judge compares against — the crucial actions the real production call took at this cutoff.")
    st.json(reference)

# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #


def _report_df(reports: list[re.ModelReport]) -> pd.DataFrame:
    rows = []
    for r in reports:
        pass_pct = round(100 * r.passes / r.reps) if r.reps else 0
        rows.append(
            {
                "Model": r.name,
                "Verdict": f"{r.passes}/{r.reps}" + (" (judge)" if r.judged else " (heur)"),
                "Pass %": pass_pct,
                "TTFT s": r.ttft_med,
                "Total s": r.total_med,
                "Out tok": r.out_tok_med,
                "Cached tok": r.cached_med,
                "$/1k calls": round(r.cost_med * 1000, 3) if r.cost_med else None,
                "$ src": r.cost_source,
                "Error": (r.error or "")[:60],
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Pass %", "Total s"], ascending=[False, True], na_position="last")
    return df


def _show_reports(reports: list[re.ModelReport]):
    st.dataframe(_report_df(reports), width="stretch", hide_index=True)
    for r in reports:
        with st.expander(f"{_verdict_emoji(r.verdicts[0] if r.verdicts else '-')} {r.name} — detail"):
            for i in range(r.reps):
                v = r.verdicts[i] if i < len(r.verdicts) else "-"
                reason = r.reasons[i] if i < len(r.reasons) else ""
                st.markdown(f"**Rep {i + 1}: {_verdict_emoji(v)} {v}** — {reason}")
            st.write("Sample tool calls:", r.sample_tools or "—")
            st.write("Sample spoken content:", repr(r.sample_content) or "—")


# --------------------------------------------------------------------------- #
# Run — single turn
# --------------------------------------------------------------------------- #

st.divider()
st.subheader("Single-turn comparison")

if st.button("Run comparison", type="primary", disabled=not specs):
    with st.spinner(f"Running {len(specs)} models × {reps} reps" + (" + judge" if use_judge else "") + "…"):
        reports_a = re.run_comparison(
            req, specs, int(reps), expected_tool or None,
            reference=reference, use_judge=use_judge, judge_model=judge_model,
        )
    st.markdown("**Snapshot A**" if snap_b_file else "**Results**")
    _show_reports(reports_a)

    if snap_b_file:
        try:
            agent_b = re.load_snapshot(_read_json(snap_b_file))
            req_b = _build(agent_b, events)
            with st.spinner("Running Snapshot B…"):
                reports_b = re.run_comparison(
                    req_b, specs, int(reps), expected_tool or None,
                    reference=reference, use_judge=use_judge, judge_model=judge_model,
                )
            st.markdown("**Snapshot B**")
            st.caption(f"Stage **{req_b.stage_name}** · {len(req_b.exposed_tools)} tools")
            _show_reports(reports_b)
        except Exception as e:  # noqa: BLE001
            st.error(f"Snapshot B failed: {e}")

# --------------------------------------------------------------------------- #
# Whole-call comparison (every decision turn, aggregated per model)
# --------------------------------------------------------------------------- #

st.divider()
st.subheader("Whole-call comparison")
st.caption(
    "Replay every LLM API call of this conversation independently (each with its "
    "real input context + ground-truth output), judge each, and aggregate into a "
    "per-model conversation pass rate. Deterministic, no simulated user, no chaining."
)

if granularity == "Every API call":
    _pick_pool = [(c.cutoff_index, f"{_kind_icon.get(c.kind,'')} {c.stage_name} · {c.label[:34]}") for c in api_calls]
else:
    _pick_pool = [(t.index, f"#{t.index} · {t.text[:34]}") for t in user_turns if t.index in decision_indices]
_pool_labels = dict(_pick_pool)
picked = st.multiselect(
    "API calls to score",
    [idx for idx, _ in _pick_pool],
    default=[idx for idx, _ in _pick_pool],
    format_func=lambda idx: _pool_labels.get(idx, str(idx)),
)

if st.button("Score whole call", disabled=not (specs and picked)):
    matrix: dict[str, dict[str, str]] = {s.name: {} for s in specs}
    totals: dict[str, list[int]] = {s.name: [0, 0] for s in specs}  # [passes, reps]
    detail: list[tuple[str, dict, list]] = []
    prog = st.progress(0.0, text="Replaying API calls…")
    for n, idx in enumerate(picked):
        sid = re.infer_stage_at(events, idx, str(agent_model.first_stage))
        exp = re.infer_expected_tool_after(events, idx)
        ref = re.reference_continuation(events, idx)
        col_label = _pool_labels.get(idx, f"@{idx}")
        try:
            r = re.build_request(agent_model, events, idx, sid)
            reports = re.run_comparison(
                r, specs, int(reps), exp or None,
                reference=ref, use_judge=use_judge, judge_model=judge_model,
            )
        except Exception as e:  # noqa: BLE001
            st.error(f"Cutoff @{idx} failed: {e}")
            continue
        for rep in reports:
            matrix[rep.name][col_label] = f"{rep.passes}/{rep.reps}"
            totals[rep.name][0] += rep.passes
            totals[rep.name][1] += rep.reps
        detail.append((col_label, ref, reports))
        prog.progress((n + 1) / len(picked), text=f"Replaying API calls… ({n + 1}/{len(picked)})")

    cols = [c for c, _, _ in detail]
    if cols:
        summary = pd.DataFrame(
            [
                {
                    "Model": name,
                    "Conversation pass": f"{totals[name][0]}/{totals[name][1]}",
                    "Pass %": round(100 * totals[name][0] / totals[name][1]) if totals[name][1] else 0,
                    **{c: matrix[name].get(c, "-") for c in cols},
                }
                for name in matrix
            ]
        ).sort_values("Pass %", ascending=False)
        st.markdown("**Conversation scoreboard (models × API calls)**")
        st.dataframe(summary, width="stretch", hide_index=True)

        for col_label, ref, reports in detail:
            imm = ref.get("immediate_actions", {})
            fired = imm.get("tool_calls") or imm.get("transitions") or "(spoke)"
            with st.expander(f"{col_label}  ·  real call did next: {fired}"):
                _show_reports(reports)

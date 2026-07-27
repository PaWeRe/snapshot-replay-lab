"""
Core-driven replay engine for behavioural ablation.

Given a real agent snapshot + a real call transcript, this rebuilds the LLM
request that core/leaping would send at a chosen turn (system prompt with
{{field}} substitution + one tool per transition/function + stage
extra_instructions), then replays it against any set of models and scores
behaviour with an LLM judge.

Fidelity comes from importing `leaping` and letting CORE assemble the system
prompt + tools, and by reproducing the *voice* transcript rendering
(`merge_event_logs` + `to_openai_messages`) byte-for-byte — the history the
production voice LLM actually saw (human turns live only as voice `message`
events; the finalized `chat_message` events are bot-only persistence records).

Only `leaping` + `openai` + `litellm` are imported here (no Streamlit), so this
module is reusable from a UI, a notebook, or a CLI.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import litellm
from openai import AsyncOpenAI

from leaping import models
from leaping.agent import Agent
from leaping.models import LeapingTranscript

# litellm silently drops params a given model rejects (matches prod NATIVE
# routing) and lets us reuse its maintained pricing DB for real cost.
litellm.drop_params = True

# Cap per-call wall time so a single slow/cold shared-endpoint model can't stall a run.
CALL_TIMEOUT_S = 60
# Concurrency caps so a 26-model sweep doesn't hammer rate limits.
CALL_CONCURRENCY = 8
JUDGE_CONCURRENCY = 4

# The judge model (overridable). gpt-5.2 with reasoning is the platform's own
# eval backbone (settings.eval_reasoning_llm_alias).
JUDGE_MODEL = "gpt-5.2"
JUDGE_REASONING_EFFORT = "high"

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


@dataclass
class ModelSpec:
    name: str
    provider: str  # "baseten" | "openai" | "gemini"
    model: str  # slug sent to the API
    price_in: float = 0.0  # $ / 1M tokens (fallback only)
    price_out: float = 0.0
    price_cached_in: float | None = None
    reasoning_effort: str | None = None
    default_temp: bool = False  # gpt-5.x reasoning models: no custom temperature
    price_estimated: bool = False  # $ unconfirmed; trust measured tokens


# Slugs verified against the live provider /v1/models (2026-07). Prices flagged
# price_estimated=True are unconfirmed and only used when LiteLLM's pricing DB
# doesn't know the slug. reasoning_effort suppresses hidden thinking on
# reasoning-capable models; the runner retries without it if a model rejects it.
PRESET_MODELS: list[ModelSpec] = [
    # OpenAI baselines
    ModelSpec("gpt-4o", "openai", "gpt-4o", 2.50, 10.00, 1.25),
    ModelSpec("gpt-4.1", "openai", "gpt-4.1", 2.00, 8.00, 0.50, price_estimated=True),
    ModelSpec("gpt-4.1-mini", "openai", "gpt-4.1-mini", 0.40, 1.60, 0.10, price_estimated=True),
    ModelSpec("gpt-5-mini", "openai", "gpt-5-mini", 0.25, 2.00, 0.025, reasoning_effort="minimal", default_temp=True, price_estimated=True),
    ModelSpec("gpt-5.2", "openai", "gpt-5.2", 1.75, 14.00, 0.175, reasoning_effort="none", default_temp=True),
    # OpenAI new (prices estimated)
    ModelSpec("gpt-5.4", "openai", "gpt-5.4", 1.75, 14.00, reasoning_effort="none", default_temp=True, price_estimated=True),
    ModelSpec("gpt-5.4-mini", "openai", "gpt-5.4-mini", 0.25, 2.00, reasoning_effort="none", default_temp=True, price_estimated=True),
    ModelSpec("gpt-5.4-nano", "openai", "gpt-5.4-nano", 0.05, 0.40, reasoning_effort="none", default_temp=True, price_estimated=True),
    ModelSpec("gpt-5.6-terra", "openai", "gpt-5.6-terra", 1.75, 14.00, reasoning_effort="none", default_temp=True, price_estimated=True),
    ModelSpec("gpt-5.6-luna", "openai", "gpt-5.6-luna", 1.75, 14.00, reasoning_effort="none", default_temp=True, price_estimated=True),
    # Gemini via Vertex (prices estimated)
    ModelSpec("gemini-2.5-flash", "gemini", "gemini-2.5-flash", 0.30, 2.50, reasoning_effort="none", price_estimated=True),
    ModelSpec("gemini-3.6-flash", "gemini", "gemini-3.6-flash", 0.30, 2.50, reasoning_effort="none", price_estimated=True),
    ModelSpec("gemini-3.5-flash", "gemini", "gemini-3.5-flash", 0.30, 2.50, reasoning_effort="none", price_estimated=True),
    ModelSpec("gemini-3.5-flash-lite", "gemini", "gemini-3.5-flash-lite", 0.10, 0.40, reasoning_effort="none", price_estimated=True),
    ModelSpec("gemini-3.1-flash-lite", "gemini", "gemini-3.1-flash-lite", 0.10, 0.40, reasoning_effort="none", price_estimated=True),
    # Baseten OSS Model API
    ModelSpec("gpt-oss-120b", "baseten", "openai/gpt-oss-120b", 0.10, 0.50, 0.03, reasoning_effort="none"),
    ModelSpec("Kimi K2.5", "baseten", "moonshotai/Kimi-K2.5", 0.60, 2.50, price_estimated=True),
    ModelSpec("Kimi K2.6", "baseten", "moonshotai/Kimi-K2.6", 0.95, 4.00, 0.16),
    ModelSpec("Kimi K2.7-Code", "baseten", "moonshotai/Kimi-K2.7-Code", 0.95, 4.00, reasoning_effort="none", price_estimated=True),
    ModelSpec("Kimi K3", "baseten", "moonshotai/Kimi-K3", 0.95, 4.00, price_estimated=True),
    ModelSpec("GLM 4.7", "baseten", "zai-org/GLM-4.7", 0.60, 2.00, reasoning_effort="none", price_estimated=True),
    ModelSpec("GLM 5", "baseten", "zai-org/GLM-5", 1.00, 3.50, reasoning_effort="none", price_estimated=True),
    ModelSpec("GLM 5.1", "baseten", "zai-org/GLM-5.1", 1.20, 4.00, reasoning_effort="none", price_estimated=True),
    ModelSpec("GLM 5.2", "baseten", "zai-org/GLM-5.2", 1.40, 4.40, 0.26, reasoning_effort="none"),
    ModelSpec("GLM 5.2 Fast", "baseten", "zai-org/GLM-5.2-Fast", 1.40, 4.40, reasoning_effort="none", price_estimated=True),
    ModelSpec("DeepSeek V4 Pro", "baseten", "deepseek-ai/DeepSeek-V4-Pro", 1.74, 3.48, 0.15, reasoning_effort="none"),
    ModelSpec("Nemotron Super", "baseten", "nvidia/Nemotron-120B-A12B", 0.30, 0.75, 0.06, reasoning_effort="none"),
    ModelSpec("Nemotron Ultra", "baseten", "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B", 0.60, 2.40, 0.12, reasoning_effort="none", price_estimated=True),
]


# Vertex config (populated by make_clients) for the LiteLLM-driven Gemini path.
_VERTEX: dict[str, str] = {}


def make_clients() -> dict[str, Any]:
    clients: dict[str, Any] = {}
    if os.environ.get("BASETEN_API_KEY"):
        clients["baseten"] = AsyncOpenAI(
            api_key=os.environ["BASETEN_API_KEY"],
            base_url=os.environ.get("BASETEN_API_INFERENCE_URL", "https://inference.baseten.co/v1"),
        )
    if os.environ.get("OPENAI_API_KEY"):
        clients["openai"] = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    b64 = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_BASE64")
    if b64:
        _VERTEX["credentials"] = base64.b64decode(b64).decode("utf-8")
        _VERTEX["project"] = os.environ.get("GOOGLE_PROJECT_ID", "")
        _VERTEX["location"] = os.environ.get("VERTEX_LOCATION") or "global"
        clients["gemini"] = "litellm"  # sentinel; Gemini goes through litellm.acompletion
    return clients


def litellm_model_str(spec: ModelSpec) -> str:
    """The provider-qualified model string LiteLLM uses for pricing / calls."""
    if spec.provider == "openai":
        return f"openai/{spec.model}"
    if spec.provider == "gemini":
        return f"vertex_ai/{spec.model}"
    if spec.provider == "baseten":
        return f"baseten/{spec.model}"
    return spec.model


# --------------------------------------------------------------------------- #
# Transcript parsing / turn selection
# --------------------------------------------------------------------------- #

_MSG_TYPES = ("message", "chat_message")


def load_transcript_events(data: Any) -> list[dict]:
    """Return the raw event list. We deliberately do NOT pre-fold voice events
    into chat_messages any more — the folding + rendering happens in
    `render_history`, exactly mirroring the voice service, so we never emit the
    duplicate-turn artifact that came from keeping both representations."""
    if isinstance(data, dict) and "event_logs" in data:
        data = data["event_logs"]
    if isinstance(data, dict) and "transcript" in data and isinstance(data["transcript"], list):
        data = data["transcript"]
    if not isinstance(data, list):
        raise ValueError("transcript must be a JSON array of events (or {event_logs: [...]})")
    return data


@dataclass
class TurnRef:
    index: int  # position in the raw event list
    sender: str  # "human" | "bot"
    text: str


def list_user_turns(events: list[dict]) -> list[TurnRef]:
    """Human utterances are the candidate cutoff points (the 'tricky turns').

    Human turns live as voice `message` events (finalized `chat_message` records
    are bot-only), so we read both types and de-duplicate consecutive human
    fragments that carry identical text."""
    out: list[TurnRef] = []
    last_text = None
    for i, ev in enumerate(events):
        if ev.get("type") in _MSG_TYPES and ev.get("sender") == "human":
            txt = (ev.get("text") or "").strip()
            if txt and txt != last_text:
                out.append(TurnRef(index=i, sender="human", text=txt))
                last_text = txt
    return out


def list_decision_turns(events: list[dict], first_stage_id: str | None) -> list[int]:
    """Indices of human turns where the real call's *next* action was an
    LLM-originated decision (a tool/transition) — i.e. the turns where the model
    actually chose something. These are the meaningful turns to score a model
    on. Falls back to all human turns if none are found."""
    decision: list[int] = []
    for t in list_user_turns(events):
        if infer_expected_tool_after(events, t.index):
            decision.append(t.index)
    return decision


@dataclass
class ApiCall:
    """One LLM API call the real conversation made. `cutoff_index` is the event
    to pass to build_request/reference_continuation (history = events[:cutoff+1]),
    i.e. everything BEFORE this completion's own outputs."""
    cutoff_index: int
    stage_id: str | None
    stage_name: str
    kind: str  # "tools" | "transition" | "speech"
    label: str
    n_tools: int = 0


def list_api_calls(agent_model: "models.Agent", events: list[dict], first_stage_id: str | None) -> list[ApiCall]:
    """Discretize the conversation into the individual LLM API calls the
    orchestrator made. The control flow between calls is deterministic glue
    (`generative_stage`/`leaping_agent`), so each API call is a self-contained,
    ground-truthed decision point — no chaining needed.

    We count only *generative* (dialogue/response) completions:
      - a group of consecutive LLM-origin function_call_requests = ONE call
        (parallel tool calls), and any content spoken in the same call,
      - a single LLM-origin transition = one call,
      - a bot chat_message emitted while a generative stage is active = one
        call that chose to speak (scripted/function stages emit text or run
        functions with no LLM call, so they're skipped).
    """
    gen_ids = {str(s.id) for s in agent_model.stages if getattr(s, "type", None) in ("dialogue", "response")}
    stage_names = {str(s.id): s.name for s in agent_model.stages}
    calls: list[ApiCall] = []
    n = len(events)
    i = 0
    while i < n:
        ev = events[i]
        t = ev.get("type")
        o = ev.get("origin")

        if t == "function_call_request" and o == "llm":
            start = i
            names: list[str] = []
            while i < n and events[i].get("type") == "function_call_request" and events[i].get("origin") == "llm":
                if events[i].get("name"):
                    names.append(events[i]["name"])
                i += 1
            if start >= 1:
                sid = infer_stage_at(events, start, first_stage_id)
                calls.append(ApiCall(start - 1, sid, stage_names.get(sid or "", sid or "?"), "tools", "call " + ", ".join(names), len(names)))
            continue

        if t == "transition" and o == "llm":
            if i >= 1:
                sid = infer_stage_at(events, i, first_stage_id)
                nm = ev.get("name") or ev.get("to_name") or "?"
                calls.append(ApiCall(i - 1, sid, stage_names.get(sid or "", sid or "?"), "transition", f"→ {nm}"))
            i += 1
            continue

        if t == "chat_message" and ev.get("sender") == "bot":
            if i >= 1:
                sid = infer_stage_at(events, i, first_stage_id)
                if sid in gen_ids:
                    txt = (ev.get("text") or "").strip()
                    calls.append(ApiCall(i - 1, sid, stage_names.get(sid or "", sid or "?"), "speech", "speak: " + txt[:48]))
            i += 1
            continue

        i += 1
    return calls


def count_api_calls(agent_model: "models.Agent", events: list[dict], first_stage_id: str | None) -> int:
    return len(list_api_calls(agent_model, events, first_stage_id))


def infer_stage_at(events: list[dict], upto_index: int, default_stage_id: str | None) -> str | None:
    """Active stage when the customer speaks = last stage transitioned INTO."""
    stage_id = default_stage_id
    for ev in events[:upto_index]:
        if ev.get("type") == "transition" and ev.get("to"):
            stage_id = ev["to"]
    return stage_id


def infer_expected_tool_after(events: list[dict], upto_index: int) -> str | None:
    """Ground truth: the first LLM-originated transition or function call AFTER
    the chosen turn in the real call. Serialized (spaces -> underscores)."""
    for ev in events[upto_index + 1:]:
        if ev.get("origin") == "llm" and ev.get("type") in ("transition", "function_call_request"):
            name = ev.get("name") or ev.get("to_name")
            if name:
                return name.replace(" ", "_")
        if ev.get("type") in _MSG_TYPES and ev.get("sender") == "human":
            break
    return None


def reference_continuation(events: list[dict], upto_index: int) -> dict:
    """What the REAL production call did after the cutoff, split into:

    - `immediate_actions`: the FIRST LLM response at this cutoff (the fair bar
      for a single replayed request) — the tool calls / transition it emitted
      before any tool result came back, or the utterance it spoke.
    - `full_continuation`: everything up to the next human turn (the real
      trajectory), for the judge to understand where this turn was heading.

    A single replayed request produces only one assistant response, so the
    judge is told to score against `immediate_actions` and use
    `full_continuation` only as context."""
    window = events[upto_index + 1:]
    # For the ground-truth reference we prefer the finalized `chat_message` text
    # (clean, one per completion) over the fragmented live voice `message`
    # events. (History rendering still uses voice messages — what the LLM saw.)
    speech_type = "chat_message" if any(e.get("type") == "chat_message" and e.get("sender") == "bot" for e in window) else "message"

    imm_tools: list[dict] = []
    imm_trans: list[str] = []
    imm_speech: list[str] = []
    started = False  # have we seen the first LLM action group yet
    imm_done = False

    full_tools: list[dict] = []
    full_trans: list[str] = []
    full_speech: list[str] = []

    for ev in window:
        t = ev.get("type")
        sender = ev.get("sender")
        if t in _MSG_TYPES and sender == "human":
            break

        is_llm_tool = ev.get("origin") == "llm" and t == "function_call_request"
        is_llm_trans = ev.get("origin") == "llm" and t == "transition"
        is_bot_speech = t == speech_type and sender == "bot"

        # --- full trajectory ---
        if is_llm_tool:
            full_tools.append({"name": ev.get("name"), "args": ev.get("args") or {}})
        elif is_llm_trans:
            nm = ev.get("name") or ev.get("to_name")
            if nm:
                full_trans.append(nm)
        elif is_bot_speech:
            txt = (ev.get("text") or "").strip()
            if txt:
                full_speech.append(txt)

        # --- immediate first-response group ---
        if not imm_done:
            if is_llm_tool:
                imm_tools.append({"name": ev.get("name"), "args": ev.get("args") or {}})
                started = True
            elif is_llm_trans:
                nm = ev.get("name") or ev.get("to_name")
                if nm:
                    imm_trans.append(nm)
                started = True
                # A transition is the last thing an API call emits; anything
                # after it (e.g. a scripted-stage farewell) is a different
                # completion, not part of this LLM call.
                imm_done = True
            elif is_bot_speech:
                txt = (ev.get("text") or "").strip()
                if txt:
                    imm_speech.append(txt)
                # a spoken turn also closes the first response group
                imm_done = True
            elif t == "function":
                # a tool result returned → the next LLM call is a new turn
                if started:
                    imm_done = True

    return {
        "immediate_actions": {
            "tool_calls": imm_tools,
            "transitions": imm_trans,
            "spoke": " ".join(imm_speech).strip(),
        },
        "full_continuation": {
            "tool_calls": full_tools,
            "transitions": full_trans,
            "bot_speech": " ".join(full_speech).strip(),
        },
    }


# --------------------------------------------------------------------------- #
# Faithful history rendering (mirrors voice/vocode Transcript.to_openai_messages)
# --------------------------------------------------------------------------- #


def _merge_bot_messages(events: list[dict]) -> list[dict]:
    """Port of voice `merge_event_logs`: join consecutive BOT `message` events
    into one; everything else passes through untouched."""
    out: list[dict] = []
    i, n = 0, len(events)
    while i < n:
        if events[i].get("type") == "message" and events[i].get("sender") == "bot":
            buf: list[dict] = []
            while i < n and events[i].get("type") == "message" and events[i].get("sender") == "bot":
                buf.append(events[i])
                i += 1
            merged = dict(buf[-1])
            merged["text"] = " ".join((e.get("text") or "") for e in buf)
            out.append(merged)
        else:
            out.append(events[i])
            i += 1
    return out


def _tool_pair(ev: dict) -> list[dict]:
    """Render a completed `function` event as an assistant tool_call + tool
    result pair (port of voice `get_openai_messages`)."""
    _id = ev.get("id")
    name = ev.get("name")
    args = ev.get("args") or {}
    returned = ev.get("returned")
    error = ev.get("error")
    call = {
        "role": "assistant",
        "tool_calls": [
            {"id": _id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
        ],
    }
    if error:
        content = (
            f"{returned}\n\nAdditionally, the following error occured: {error}"
            if returned is not None
            else f"Function returned an error: {error}"
        )
    else:
        content = "No content returned" if returned is None else str(returned)
    return [call, {"tool_call_id": _id, "role": "tool", "name": name, "content": content}]


def render_history(events_slice: list[dict]) -> list[dict]:
    """Rebuild the OpenAI message history the production LLM saw.

    If the transcript contains voice `message` events (a real voice call), use
    the voice rendering (merge consecutive bot messages, drop the duplicate
    bot-only `chat_message` records). Otherwise fall back to `chat_message`
    rendering (chat-mode transcripts). Completed `function` events become
    tool_call/tool pairs in both paths."""
    use_voice = any(e.get("type") == "message" for e in events_slice)
    source = _merge_bot_messages(events_slice) if use_voice else events_slice
    msgs: list[dict] = []
    for ev in source:
        t = ev.get("type")
        if t == "message" and use_voice:
            text = ev.get("text") or ""
            if not text.strip():
                continue
            if ev.get("sender") == "bot":
                if not ev.get("is_final", False):
                    text = f"{text}-"
                msgs.append({"role": "assistant", "content": text})
            else:
                msgs.append({"role": "user", "content": text})
        elif t == "chat_message" and not use_voice:
            text = ev.get("text") or ""
            if not text.strip():
                continue
            role = "assistant" if ev.get("sender") == "bot" else "user"
            msgs.append({"role": role, "content": text})
        elif t == "function":
            msgs.extend(_tool_pair(ev))
    return msgs


# --------------------------------------------------------------------------- #
# Build the request through core
# --------------------------------------------------------------------------- #


@dataclass
class BuiltRequest:
    messages: list[dict]
    tools: list[dict]
    stage_name: str
    exposed_tools: list[str]
    system_message: str
    extra_instructions: str
    prod_model_alias: str


def load_snapshot(data: dict) -> models.Agent:
    if isinstance(data, dict):
        for key in ("agent", "snapshot", "enriched_snapshot", "agent_snapshot"):
            inner = data.get(key)
            if isinstance(inner, dict) and ("stages" in inner or "name" in inner):
                data = inner
                break
    return models.Agent.model_validate(data)


def list_stages(agent_model: models.Agent) -> list[tuple[str, str]]:
    return [(str(s.id), s.name) for s in agent_model.stages]


def _schema(obj: Any) -> dict:
    return getattr(obj, "openai_schema", None) or getattr(obj, "json_schema")


def build_request(
    agent_model: models.Agent,
    events: list[dict],
    upto_index: int,
    stage_id: str,
    system_override: str | None = None,
    stage_message_override: str | None = None,
) -> BuiltRequest:
    events_slice = events[: upto_index + 1]
    start_time = events_slice[0].get("timestamp", 0) if events_slice else 0
    transcript = LeapingTranscript(event_logs=events_slice, start_time=start_time)  # type: ignore

    agent = Agent(model=agent_model, transcript=transcript, execute_functions=False)

    if system_override:
        agent.ctx.raw_base_system_message = system_override

    # Faithful {{field}} substitution: replay the field values the real call had
    # set by this point (from field_update events) so the system prompt is
    # byte-for-byte what prod sent, not a template with empty placeholders.
    for ev in events_slice:
        if ev.get("type") == "field_update":
            fname = ev.get("field")
            if fname and fname in agent.ctx.fields:
                try:
                    agent.ctx.set_field(fname, ev.get("new_value"))
                except Exception:  # noqa: BLE001
                    pass

    # Registers ctx.stages and sets current_stage to our target (no side effects).
    agent.ctx.init_stages(agent.stages, UUID(stage_id))
    stage = next((s for s in agent.stages if str(s.id) == stage_id), None)
    if stage is None:
        raise ValueError(f"stage {stage_id} not found in snapshot")
    if stage_message_override is not None and hasattr(stage, "raw_stage_message"):
        stage.raw_stage_message = stage_message_override  # type: ignore
    # Mid-conversation replay: transitions are exposed (prod excludes them only
    # on a stage's very first reply).
    if hasattr(stage, "first_reply_sent"):
        stage.first_reply_sent = True  # type: ignore

    system = agent.ctx.get_system_message() or ""
    history = render_history(events_slice)
    messages: list[dict] = [{"role": "system", "content": system}] + history

    # extra_instructions is appended by prod as a trailing system message, AFTER
    # the history (generative_stage.py:501), raw (no field substitution).
    extra = getattr(stage, "extra_instructions", None) or ""
    if extra:
        messages = messages + [{"role": "system", "content": extra}]

    tools: list[dict] = []
    for fn in getattr(stage, "functions", {}).values():
        tools.append(_schema(fn))
    for tr in stage.transitions.values():
        tools.append(_schema(tr))
    kb_tool = getattr(stage, "_knowledge_base_tool", None)
    if kb_tool:
        tools.append(kb_tool)

    # Which model prod actually used at this stage (for display).
    alias = None
    for s in agent_model.stages:
        if str(s.id) == stage_id:
            alias = getattr(s, "llm_model_alias", None)
            break
    alias = alias or getattr(agent_model, "llm_model_alias", None)

    return BuiltRequest(
        messages=messages,
        tools=tools,
        stage_name=stage.name,
        exposed_tools=[t["function"]["name"] for t in tools],
        system_message=system,
        extra_instructions=extra,
        prod_model_alias=str(alias or ""),
    )


def render_api_payload(spec: ModelSpec, req: BuiltRequest) -> dict:
    """The exact request body that would be sent to this model's API — for the
    byte-for-byte inspector in the UI."""
    payload: dict[str, Any] = {
        "model": litellm_model_str(spec) if spec.provider == "gemini" else spec.model,
        "messages": req.messages,
        "tools": req.tools,
        "tool_choice": "auto",
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if not spec.default_temp:
        payload["temperature"] = 0.1
    if spec.reasoning_effort is not None:
        payload["reasoning_effort"] = spec.reasoning_effort
    if spec.provider == "openai":
        payload["service_tier"] = "priority"
    return payload


# --------------------------------------------------------------------------- #
# One streamed call + scoring
# --------------------------------------------------------------------------- #


@dataclass
class CallResult:
    ttft: float | None = None
    total: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    tool_calls: list[str] = field(default_factory=list)  # names, in call order
    tool_calls_detailed: list[dict] = field(default_factory=list)  # {name, args}
    content: str = ""
    error: str | None = None


def _finalize_tool_calls(names: dict[int, str], args_raw: dict[int, str]) -> tuple[list[str], list[dict]]:
    ordered = sorted(names)
    tool_calls = [names[i] for i in ordered]
    detailed = []
    for i in ordered:
        raw = args_raw.get(i, "") or ""
        try:
            parsed = json.loads(raw) if raw.strip() else {}
        except Exception:  # noqa: BLE001
            parsed = {"_raw": raw}
        detailed.append({"name": names[i], "args": parsed})
    return tool_calls, detailed


async def run_call(client: Any, spec: ModelSpec, messages: list[dict], tools: list[dict]) -> CallResult:
    if spec.provider == "gemini":
        return await _run_call_litellm(spec, messages, tools)

    kwargs: dict[str, Any] = {
        "model": spec.model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if not spec.default_temp:
        kwargs["temperature"] = 0.1
    if spec.reasoning_effort is not None:
        kwargs["reasoning_effort"] = spec.reasoning_effort

    res = CallResult()
    start = time.perf_counter()
    stream = None
    for attempt in range(2):
        try:
            stream = await client.chat.completions.create(**kwargs)
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 0 and "reasoning_effort" in kwargs:
                kwargs.pop("reasoning_effort", None)
                continue
            res.error = f"{type(e).__name__}: {e}"
            return res

    names: dict[int, str] = {}
    args_raw: dict[int, str] = {}
    try:
        async for chunk in stream:
            if res.ttft is None:
                res.ttft = time.perf_counter() - start
            if getattr(chunk, "usage", None):
                res.prompt_tokens = chunk.usage.prompt_tokens
                res.completion_tokens = chunk.usage.completion_tokens
                res.cached_tokens = _cached_tokens(chunk.usage)
            if not chunk.choices:
                continue
            d = chunk.choices[0].delta
            if d and d.content:
                res.content += d.content
            if d and d.tool_calls:
                for tc in d.tool_calls:
                    if tc.function and tc.function.name:
                        names[tc.index] = names.get(tc.index, "") + tc.function.name
                    if tc.function and tc.function.arguments:
                        args_raw[tc.index] = args_raw.get(tc.index, "") + tc.function.arguments
    except Exception as e:  # noqa: BLE001
        res.error = f"{type(e).__name__}: {e}"
        return res

    res.total = time.perf_counter() - start
    res.tool_calls, res.tool_calls_detailed = _finalize_tool_calls(names, args_raw)
    return res


async def _run_call_litellm(spec: ModelSpec, messages: list[dict], tools: list[dict]) -> CallResult:
    """Gemini path: same streamed contract as run_call, via LiteLLM + Vertex creds."""
    kwargs: dict[str, Any] = {
        "model": f"vertex_ai/{spec.model}",
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "stream": True,
        "stream_options": {"include_usage": True},
        "vertex_project": _VERTEX.get("project", ""),
        "vertex_location": _VERTEX.get("location", "global"),
        "vertex_credentials": _VERTEX.get("credentials", ""),
        "timeout": 45,
        "num_retries": 0,
    }
    if not spec.default_temp:
        kwargs["temperature"] = 0.1
    if spec.reasoning_effort is not None:
        kwargs["reasoning_effort"] = spec.reasoning_effort

    res = CallResult()
    start = time.perf_counter()
    stream = None
    for attempt in range(2):
        try:
            stream = await litellm.acompletion(**kwargs)
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 0 and "reasoning_effort" in kwargs:
                kwargs.pop("reasoning_effort", None)
                continue
            res.error = f"{type(e).__name__}: {e}"
            return res

    names: dict[int, str] = {}
    args_raw: dict[int, str] = {}
    try:
        async for chunk in stream:
            if res.ttft is None:
                res.ttft = time.perf_counter() - start
            usage = getattr(chunk, "usage", None)
            if usage:
                res.prompt_tokens = usage.prompt_tokens
                res.completion_tokens = usage.completion_tokens
                res.cached_tokens = _cached_tokens(usage)
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            d = choices[0].delta
            if d and getattr(d, "content", None):
                res.content += d.content
            if d and getattr(d, "tool_calls", None):
                for tc in d.tool_calls:
                    if tc.function and tc.function.name:
                        names[tc.index] = names.get(tc.index, "") + tc.function.name
                    if tc.function and getattr(tc.function, "arguments", None):
                        args_raw[tc.index] = args_raw.get(tc.index, "") + tc.function.arguments
    except Exception as e:  # noqa: BLE001
        res.error = f"{type(e).__name__}: {e}"
        return res

    res.total = time.perf_counter() - start
    res.tool_calls, res.tool_calls_detailed = _finalize_tool_calls(names, args_raw)
    return res


def _cached_tokens(usage: Any) -> int | None:
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None:
        return None
    if isinstance(details, dict):
        return details.get("cached_tokens")
    return getattr(details, "cached_tokens", None)


DEFAULT_LEAK_LABELS = ["</think>", "<think>"]


def classify(expected_tool: str | None, leak_labels: list[str], r: CallResult) -> str:
    """Cheap heuristic outcome (kept as a secondary signal next to the judge)."""
    if r.error:
        return "error"
    fired = r.tool_calls
    content = (r.content or "").strip()
    if expected_tool and expected_tool in fired and not content:
        return "PASS" if len(fired) == 1 else "pass_extra_tool"
    if expected_tool and expected_tool in fired and content:
        return "routed+spoke"
    if fired and expected_tool and expected_tool not in fired:
        return "wrong_tool"
    if fired and not expected_tool:
        return "tool_fired"
    if content and any(lbl.lower() in content.lower() for lbl in leak_labels):
        return "LEAK"
    if content:
        return "spoke_no_route"
    return "empty"


def dynamic_cost(spec: ModelSpec, r: CallResult) -> tuple[float | None, str]:
    """Real cost = measured tokens x price. Prices come from LiteLLM's
    maintained DB when it knows the slug (accounts for cached-token discount),
    otherwise the configured fallback. Returns (cost, source)."""
    if r.prompt_tokens is None or r.completion_tokens is None:
        return None, "no-usage"
    cached = r.cached_tokens or 0
    fresh = max(r.prompt_tokens - cached, 0)

    try:
        info = litellm.get_model_info(litellm_model_str(spec))
        c_in = info.get("input_cost_per_token")
        c_out = info.get("output_cost_per_token")
        if c_in and c_out:
            c_cache = info.get("cache_read_input_token_cost") or c_in
            cost = fresh * c_in + cached * c_cache + r.completion_tokens * c_out
            return cost, "litellm"
    except Exception:  # noqa: BLE001
        pass

    c_in = spec.price_in / 1e6
    c_out = spec.price_out / 1e6
    c_cache = (spec.price_cached_in / 1e6) if spec.price_cached_in is not None else c_in
    cost = fresh * c_in + cached * c_cache + r.completion_tokens * c_out
    return cost, ("config-est" if spec.price_estimated else "config")


# --------------------------------------------------------------------------- #
# LLM judge (gpt-5.2)
# --------------------------------------------------------------------------- #

JUDGE_SYSTEM = """You are a rigorous evaluator for a production voice-agent platform.

At a specific point in a REAL phone call (the "cutoff"), the production agent sent ONE request to an LLM. We re-sent the EXACT same request (identical system prompt, tools, and conversation history) to a CANDIDATE model to see whether it behaves correctly. You decide PASS or FAIL for this single turn.

You are given:
- STAGE_INSTRUCTIONS: the system prompt / stage instructions that define correct behaviour at this turn.
- AVAILABLE_TOOLS: the tools the model could call (name, description, parameters). Firing a transition means CALLING that tool, not saying its name.
- REFERENCE.immediate_actions (what the real production call did as its VERY NEXT single response at this cutoff): the tool call(s) with arguments, transition(s), or the utterance it spoke. This is the fair bar — the candidate, like the real call, produces only ONE assistant response here.
- REFERENCE.full_continuation (the real call's fuller trajectory until the next customer turn): CONTEXT ONLY, so you understand where this turn was heading. Do NOT require the candidate to reproduce later steps that in the real call only became possible AFTER a tool result returned (those are separate turns).
- CANDIDATE (the model under test): the tool calls it made (with arguments) and what it said in its single response.

How to judge:
1. From STAGE_INSTRUCTIONS + REFERENCE.immediate_actions, identify the CRUCIAL action(s) this SINGLE turn requires (e.g. fire a specific transition, call specific function(s) with correct arguments, ask a specific question, or deliberately just speak). Extract the arguments that actually matter (e.g. the street/city value), ignoring incidental formatting or transcription spelling ("four ninety five" vs "495").
2. There are often multiple valid ways to reach the goal. Order sometimes matters (e.g. must verify identity before disclosing data) and sometimes does not (e.g. update_city vs update_street can be called together in any order). Use the stage instructions to decide which ordering/selection constraints are real. Do not reward hallucinated actions the reference did not take (e.g. inventing a ZIP the caller never gave).
3. PASS iff the candidate accomplishes the crucial immediate action(s) correctly: right tool(s) called, arguments correct or semantically equivalent, no harmful/contradictory/hallucinated extra actions, and NO leaking of a tool/transition/label or <think> text into the spoken channel. Minor wording differences in speech are fine.
   FAIL if a crucial tool call is missing or wrong, arguments are wrong or invented, routing is wrong or absent when required, or the model speaks a routing label instead of firing the transition.

Return ONLY valid JSON with this exact shape:
{"verdict": "PASS" | "FAIL",
 "crucial_actions": [{"action": "<short description>", "required": true|false, "candidate_did_it": true|false}],
 "speech_leak": true|false,
 "reasoning": "<=120 words explaining the verdict"}"""


def _tools_for_judge(req: BuiltRequest) -> list[dict]:
    out = []
    for t in req.tools:
        fn = t.get("function", {})
        out.append(
            {
                "name": fn.get("name"),
                "description": fn.get("description", ""),
                "parameters": list((fn.get("parameters", {}) or {}).get("properties", {}).keys()),
            }
        )
    return out


@dataclass
class JudgeVerdict:
    verdict: str  # "PASS" | "FAIL" | "error"
    reasoning: str = ""
    crucial_actions: list[dict] = field(default_factory=list)
    speech_leak: bool = False
    error: str | None = None


async def judge_turn(
    judge_client: AsyncOpenAI,
    req: BuiltRequest,
    reference: dict,
    candidate: dict,
    judge_model: str = JUDGE_MODEL,
) -> JudgeVerdict:
    user_payload = {
        "STAGE": req.stage_name,
        "STAGE_INSTRUCTIONS": (req.system_message + ("\n\n[extra_instructions]\n" + req.extra_instructions if req.extra_instructions else "")),
        "AVAILABLE_TOOLS": _tools_for_judge(req),
        "REFERENCE": reference,
        "CANDIDATE": candidate,
    }
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
    kwargs: dict[str, Any] = {
        "model": judge_model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "reasoning_effort": JUDGE_REASONING_EFFORT,
    }
    for attempt in range(2):
        try:
            resp = await judge_client.chat.completions.create(**kwargs)
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 0 and "reasoning_effort" in kwargs:
                kwargs.pop("reasoning_effort", None)
                continue
            return JudgeVerdict(verdict="error", error=f"{type(e).__name__}: {e}")
    try:
        data = json.loads(resp.choices[0].message.content or "{}")
    except Exception as e:  # noqa: BLE001
        return JudgeVerdict(verdict="error", error=f"judge JSON parse: {e}")
    verdict = str(data.get("verdict", "")).upper()
    if verdict not in ("PASS", "FAIL"):
        verdict = "FAIL"
    return JudgeVerdict(
        verdict=verdict,
        reasoning=str(data.get("reasoning", "")),
        crucial_actions=data.get("crucial_actions", []) or [],
        speech_leak=bool(data.get("speech_leak", False)),
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


@dataclass
class RepResult:
    call: CallResult
    outcome: str
    verdict: JudgeVerdict | None
    cost: float | None
    cost_source: str


@dataclass
class ModelReport:
    name: str
    reps: int
    passes: int  # judge PASS count (falls back to heuristic if no judge)
    judged: bool
    outcomes: list[str]
    verdicts: list[str]
    reasons: list[str]
    ttft_med: float | None
    total_med: float | None
    out_tok_med: int | None
    cached_med: int | None
    cost_med: float | None
    cost_source: str
    sample_content: str
    sample_tools: list[dict]
    error: str | None
    reps_detail: list[RepResult] = field(default_factory=list)


async def _run_models_async(
    req: BuiltRequest,
    specs: list[ModelSpec],
    reps: int,
    expected_tool: str | None,
    reference: dict | None,
    use_judge: bool,
    extra_leak_labels: list[str],
    judge_model: str = JUDGE_MODEL,
) -> list[ModelReport]:
    clients = make_clients()
    judge_client = clients.get("openai") if use_judge else None
    leak_labels = DEFAULT_LEAK_LABELS + extra_leak_labels + [
        t["function"]["name"].replace("_", " ") for t in req.tools
    ]
    call_sem = asyncio.Semaphore(CALL_CONCURRENCY)
    judge_sem = asyncio.Semaphore(JUDGE_CONCURRENCY)

    async def call_and_judge(spec: ModelSpec) -> RepResult:
        client = clients.get(spec.provider)
        if client is None:
            return RepResult(CallResult(error=f"missing {spec.provider} API key"), "no_api_key", None, None, "no-usage")
        async with call_sem:
            try:
                r = await asyncio.wait_for(run_call(client, spec, req.messages, req.tools), timeout=CALL_TIMEOUT_S)
            except asyncio.TimeoutError:
                r = CallResult(error=f"timeout>{CALL_TIMEOUT_S}s")
        outcome = classify(expected_tool, leak_labels, r)
        cost, source = dynamic_cost(spec, r)
        verdict = None
        if judge_client is not None and reference is not None and not r.error:
            candidate = {"tool_calls": r.tool_calls_detailed, "spoken": r.content}
            async with judge_sem:
                verdict = await judge_turn(judge_client, req, reference, candidate, judge_model)
        return RepResult(r, outcome, verdict, cost, source)

    # Schedule every (model, rep) call concurrently.
    index: list[str] = []
    tasks: list[asyncio.Task] = []
    for spec in specs:
        for _ in range(reps):
            index.append(spec.name)
            tasks.append(asyncio.create_task(call_and_judge(spec)))
    done = await asyncio.gather(*tasks)

    grouped: dict[str, list[RepResult]] = defaultdict(list)
    for name, rr in zip(index, done):
        grouped[name].append(rr)

    reports: list[ModelReport] = []
    for spec in specs:
        rrs = grouped[spec.name]
        results = [rr.call for rr in rrs]
        outcomes = [rr.outcome for rr in rrs]
        verdicts = [rr.verdict.verdict if rr.verdict else "-" for rr in rrs]
        reasons = [rr.verdict.reasoning if rr.verdict else "" for rr in rrs]
        judged = any(rr.verdict is not None for rr in rrs)

        if judged:
            passes = sum(1 for rr in rrs if rr.verdict and rr.verdict.verdict == "PASS")
        else:
            passes = sum(1 for o in outcomes if o in ("PASS", "pass_extra_tool"))

        ttfts = [r.ttft for r in results if r.ttft is not None]
        totals = [r.total for r in results if r.total is not None]
        outs = [r.completion_tokens for r in results if r.completion_tokens is not None]
        cacheds = [r.cached_tokens for r in results if r.cached_tokens is not None]
        costs = [rr.cost for rr in rrs if rr.cost is not None]
        cost_source = next((rr.cost_source for rr in rrs if rr.cost is not None), "no-usage")

        reports.append(
            ModelReport(
                name=spec.name,
                reps=reps,
                passes=passes,
                judged=judged,
                outcomes=outcomes,
                verdicts=verdicts,
                reasons=reasons,
                ttft_med=round(statistics.median(ttfts), 3) if ttfts else None,
                total_med=round(statistics.median(totals), 3) if totals else None,
                out_tok_med=round(statistics.median(outs)) if outs else None,
                cached_med=round(statistics.median(cacheds)) if cacheds else None,
                cost_med=round(statistics.median(costs), 6) if costs else None,
                cost_source=cost_source,
                sample_content=next((r.content for r in results if r.content), ""),
                sample_tools=next((r.tool_calls_detailed for r in results if r.tool_calls_detailed), []),
                error=next((r.error for r in results if r.error), None),
                reps_detail=rrs,
            )
        )
    return reports


def run_comparison(
    req: BuiltRequest,
    specs: list[ModelSpec],
    reps: int,
    expected_tool: str | None,
    reference: dict | None = None,
    use_judge: bool = True,
    extra_leak_labels: list[str] | None = None,
    judge_model: str = JUDGE_MODEL,
) -> list[ModelReport]:
    """Sync entrypoint for UIs/notebooks."""
    return asyncio.run(
        _run_models_async(
            req, specs, reps, expected_tool, reference, use_judge, extra_leak_labels or [], judge_model
        )
    )

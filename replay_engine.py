"""
Core-driven replay engine for behavioural ablation.

Given a real agent snapshot + a real call transcript, this rebuilds the LLM
request that core/leaping would send at a chosen turn (system prompt with
{{field}} substitution + one tool per transition/function), then replays it
against any set of models / prompt overrides and scores behaviour.

Fidelity comes from importing `leaping` and letting CORE assemble the prompt
and tools — nothing is hand-trimmed. Only `leaping` + `openai` are imported here
(no Streamlit), so this module is reusable from a UI, a notebook, or a CLI.
"""

from __future__ import annotations

import asyncio
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from openai import AsyncOpenAI

from leaping import models
from leaping.agent import Agent
from leaping.models import LeapingTranscript

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


@dataclass
class ModelSpec:
    name: str
    provider: str  # "baseten" | "openai"
    model: str  # slug sent to the API
    price_in: float = 0.0
    price_out: float = 0.0
    price_cached_in: float | None = None
    reasoning_effort: str | None = None
    default_temp: bool = False  # gpt-5.x: no custom temperature


PRESET_MODELS: list[ModelSpec] = [
    ModelSpec("gpt-oss-120b", "baseten", "openai/gpt-oss-120b", 0.10, 0.50, 0.03, reasoning_effort="none"),
    ModelSpec("Kimi K2.6", "baseten", "moonshotai/Kimi-K2.6", 0.95, 4.00, 0.16),
    ModelSpec("Nemotron Super", "baseten", "nvidia/Nemotron-120B-A12B", 0.30, 0.75, 0.06),
    ModelSpec("DeepSeek V4 Pro", "baseten", "deepseek-ai/DeepSeek-V4-Pro", 1.74, 3.48, 0.15),
    ModelSpec("DeepSeek V4 (reason off)", "baseten", "deepseek-ai/DeepSeek-V4-Pro", 1.74, 3.48, 0.15, reasoning_effort="none"),
    ModelSpec("GLM 5.2", "baseten", "zai-org/GLM-5.2", 1.40, 4.40, 0.26),
    ModelSpec("gpt-5.2", "openai", "gpt-5.2", 1.75, 14.00, 0.175, reasoning_effort="none", default_temp=True),
    ModelSpec("gpt-4o", "openai", "gpt-4o", 2.50, 10.00, 1.25),
]


def make_clients() -> dict[str, AsyncOpenAI]:
    clients: dict[str, AsyncOpenAI] = {}
    if os.environ.get("BASETEN_API_KEY"):
        clients["baseten"] = AsyncOpenAI(
            api_key=os.environ["BASETEN_API_KEY"],
            base_url=os.environ.get("BASETEN_API_INFERENCE_URL", "https://inference.baseten.co/v1"),
        )
    if os.environ.get("OPENAI_API_KEY"):
        clients["openai"] = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return clients


# --------------------------------------------------------------------------- #
# Transcript parsing / turn selection
# --------------------------------------------------------------------------- #

_MSG_TYPES = ("message", "chat_message")


def load_transcript_events(data: Any) -> list[dict]:
    if isinstance(data, dict) and "event_logs" in data:
        data = data["event_logs"]
    if not isinstance(data, list):
        raise ValueError("transcript must be a JSON array of events (or {event_logs: [...]})")
    return data


@dataclass
class TurnRef:
    index: int  # position in the raw event list
    sender: str  # "human" | "bot"
    text: str


def list_user_turns(events: list[dict]) -> list[TurnRef]:
    """Human utterances are the candidate cutoff points (the 'tricky turns')."""
    out: list[TurnRef] = []
    for i, ev in enumerate(events):
        if ev.get("type") in _MSG_TYPES and ev.get("sender") == "human":
            txt = (ev.get("text") or "").strip()
            if txt:
                out.append(TurnRef(index=i, sender="human", text=txt))
    return out


def infer_stage_at(events: list[dict], upto_index: int, default_stage_id: str | None) -> str | None:
    """Active stage when the customer speaks = last stage transitioned INTO."""
    stage_id = default_stage_id
    for ev in events[:upto_index]:
        if ev.get("type") == "transition" and ev.get("to"):
            stage_id = ev["to"]
    return stage_id


def infer_expected_tool_after(events: list[dict], upto_index: int) -> str | None:
    """
    Ground truth: the first LLM-originated transition or function call AFTER the
    chosen turn in the real call. Serialized (spaces -> underscores).
    """
    for ev in events[upto_index + 1:]:
        if ev.get("origin") == "llm" and ev.get("type") in ("transition", "function_call_request"):
            name = ev.get("name") or ev.get("to_name")
            if name:
                return name.replace(" ", "_")
        # stop scanning once the next human turn begins
        if ev.get("type") in _MSG_TYPES and ev.get("sender") == "human":
            break
    return None


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


def load_snapshot(data: dict) -> models.Agent:
    return models.Agent.model_validate(data)


def list_stages(agent_model: models.Agent) -> list[tuple[str, str]]:
    return [(str(s.id), s.name) for s in agent_model.stages]


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

    agent.ctx.change_stage(UUID(stage_id))
    stage = next((s for s in agent.stages if str(s.id) == stage_id), None)
    if stage is None:
        raise ValueError(f"stage {stage_id} not found in snapshot")
    if stage_message_override is not None and hasattr(stage, "raw_stage_message"):
        stage.raw_stage_message = stage_message_override  # type: ignore
    if hasattr(stage, "first_reply_sent"):
        stage.first_reply_sent = True  # type: ignore

    system = agent.ctx.get_system_message() or ""
    messages = agent.transcript.to_openai_messages(system)  # type: ignore

    def schema(obj: Any) -> dict:
        return getattr(obj, "openai_schema", None) or getattr(obj, "json_schema")

    tools: list[dict] = []
    for fn in getattr(stage, "functions", {}).values():
        tools.append(schema(fn))
    for tr in stage.transitions.values():
        tools.append(schema(tr))

    return BuiltRequest(
        messages=messages,
        tools=tools,
        stage_name=stage.name,
        exposed_tools=[t["function"]["name"] for t in tools],
        system_message=system,
    )


# --------------------------------------------------------------------------- #
# One streamed call + scoring
# --------------------------------------------------------------------------- #


@dataclass
class CallResult:
    ttft: float | None = None
    total: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tool_calls: list[str] = field(default_factory=list)
    content: str = ""
    error: str | None = None


async def run_call(client: AsyncOpenAI, spec: ModelSpec, messages: list[dict], tools: list[dict]) -> CallResult:
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
    try:
        stream = await client.chat.completions.create(**kwargs)
    except TypeError:
        kwargs.pop("reasoning_effort", None)
        stream = await client.chat.completions.create(**kwargs)
    except Exception as e:  # noqa: BLE001
        res.error = f"{type(e).__name__}: {e}"
        return res

    names: dict[int, str] = {}
    try:
        async for chunk in stream:
            if res.ttft is None:
                res.ttft = time.perf_counter() - start
            if getattr(chunk, "usage", None):
                res.prompt_tokens = chunk.usage.prompt_tokens
                res.completion_tokens = chunk.usage.completion_tokens
            if not chunk.choices:
                continue
            d = chunk.choices[0].delta
            if d and d.content:
                res.content += d.content
            if d and d.tool_calls:
                for tc in d.tool_calls:
                    if tc.function and tc.function.name:
                        names[tc.index] = names.get(tc.index, "") + tc.function.name
    except Exception as e:  # noqa: BLE001
        res.error = f"{type(e).__name__}: {e}"
        return res

    res.total = time.perf_counter() - start
    res.tool_calls = [names[i] for i in sorted(names)]
    return res


DEFAULT_LEAK_LABELS = ["</think>", "<think>"]


def classify(expected_tool: str | None, leak_labels: list[str], r: CallResult) -> str:
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


def cost_per_call(spec: ModelSpec, r: CallResult) -> float | None:
    if r.prompt_tokens is None or r.completion_tokens is None:
        return None
    return r.prompt_tokens / 1e6 * spec.price_in + r.completion_tokens / 1e6 * spec.price_out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


@dataclass
class ModelReport:
    name: str
    passes: int
    reps: int
    outcomes: list[str]
    ttft_med: float | None
    total_med: float | None
    out_tok_med: int | None
    cost_med: float | None
    sample_content: str
    sample_tools: list[str]
    error: str | None


async def _run_models_async(
    req: BuiltRequest,
    specs: list[ModelSpec],
    reps: int,
    expected_tool: str | None,
    extra_leak_labels: list[str],
) -> list[ModelReport]:
    clients = make_clients()
    leak_labels = DEFAULT_LEAK_LABELS + extra_leak_labels + [
        t["function"]["name"].replace("_", " ") for t in req.tools
    ]
    reports: list[ModelReport] = []
    for spec in specs:
        client = clients.get(spec.provider)
        if client is None:
            reports.append(ModelReport(spec.name, 0, reps, ["no_api_key"] * reps, None, None, None, None, "", [], f"missing {spec.provider} API key"))
            continue
        results: list[CallResult] = []
        for _ in range(reps):
            results.append(await run_call(client, spec, req.messages, req.tools))
            await asyncio.sleep(0.1)
        outcomes = [classify(expected_tool, leak_labels, r) for r in results]
        ttfts = [r.ttft for r in results if r.ttft is not None]
        totals = [r.total for r in results if r.total is not None]
        outs = [r.completion_tokens for r in results if r.completion_tokens is not None]
        costs = [c for r in results if (c := cost_per_call(spec, r)) is not None]
        reports.append(
            ModelReport(
                name=spec.name,
                passes=sum(1 for o in outcomes if o in ("PASS", "pass_extra_tool")),
                reps=reps,
                outcomes=outcomes,
                ttft_med=round(statistics.median(ttfts), 3) if ttfts else None,
                total_med=round(statistics.median(totals), 3) if totals else None,
                out_tok_med=round(statistics.median(outs)) if outs else None,
                cost_med=round(statistics.median(costs), 6) if costs else None,
                sample_content=next((r.content for r in results if r.content), ""),
                sample_tools=next((r.tool_calls for r in results if r.tool_calls), []),
                error=next((r.error for r in results if r.error), None),
            )
        )
    return reports


def run_comparison(
    req: BuiltRequest,
    specs: list[ModelSpec],
    reps: int,
    expected_tool: str | None,
    extra_leak_labels: list[str] | None = None,
) -> list[ModelReport]:
    """Sync entrypoint for UIs/notebooks."""
    return asyncio.run(
        _run_models_async(req, specs, reps, expected_tool, extra_leak_labels or [])
    )

"""Tool-calling assistant loop.

Guardrails (all enforced in code, see README "GenAI assistant"):
  1. Tool-call budget: at most MAX_TOOL_CALLS tool executions and MAX_TURNS model calls per question.
  2. Argument validation: tool inputs are checked against allow-lists before any API call (tools.py).
  3. Grounded citations: every factual answer must cite the tools it used as [tool_name], and may
     only cite tools that actually ran in this question. Answers with numbers but no valid citation
     get one corrective retry, then are replaced by a safe fallback.
  4. Domain scope: the system prompt restricts the assistant to shipping/logistics data questions.

Observability: every LLM call, tool call and final answer is written as one JSON line to
the LLM log (prompt sent, tools called, tool inputs/outputs, model output, tokens, cost).
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Protocol

from sci.assistant.tools import TOOL_NAMES, TOOL_SCHEMAS, ToolExecutor

MAX_TOOL_CALLS = 6
MAX_TURNS = 8
CITATION = re.compile(r"\[([a-z_]+)\]")
# Identifiers and dates are not "figures": they come from the question or the system prompt.
_NON_FIGURES = re.compile(r"SHP-\d+|VSL-\d+|CUST-\d+|\d{4}-\d{2}-\d{2}|Q[1-4]\s?\d{4}")
FALLBACK = (
    "I couldn't produce an answer that is grounded in the platform's data, so I won't guess. "
    "Try rephrasing, or ask about a specific route, shipment or time window."
)


# --------------------------------------------------------------------------- LLM interface


@dataclass
class LLMResponse:
    stop_reason: str
    content: list[dict]  # [{"type": "text", "text": ...} | {"type": "tool_use", "id", "name", "input"}]
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient(Protocol):
    model: str

    def create(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


class AnthropicLLM:
    """Thin adapter so the agent (and its tests) never depend on the SDK's object types."""

    def __init__(self, model: str, max_tokens: int = 1024):
        import anthropic

        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model
        self.max_tokens = max_tokens

    def create(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        resp = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system, messages=messages, tools=tools
        )
        content = []
        for block in resp.content:
            if block.type == "text":
                content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                content.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
        return LLMResponse(resp.stop_reason, content, resp.usage.input_tokens, resp.usage.output_tokens)


# --------------------------------------------------------------------------- logging


class LLMCallLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict) -> None:
        record = {"ts": datetime.now(UTC).isoformat(), **record}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")


# --------------------------------------------------------------------------- agent


@dataclass
class AgentAnswer:
    text: str
    tools_called: list[str] = field(default_factory=list)
    guardrail_events: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


def validate_citations(text: str, tools_called: list[str]) -> str | None:
    """Return a reason string if the answer is not properly grounded, else None."""
    cited = set(CITATION.findall(text))
    invalid = {c for c in cited if c in TOOL_NAMES and c not in tools_called}
    if invalid:
        return f"cited tools that were not called: {sorted(invalid)}"
    has_numbers = bool(re.search(r"\d", _NON_FIGURES.sub("", text)))
    if has_numbers and not tools_called:
        return "answer contains figures but no tool was called"
    if has_numbers and not (cited & set(tools_called)):
        return "answer contains figures without a [tool_name] citation"
    return None


def quarter_bounds(day: date) -> tuple[date, date]:
    q = (day.month - 1) // 3
    start = date(day.year, 3 * q + 1, 1)
    end = date(day.year, 12, 31) if q == 3 else date(day.year, 3 * q + 4, 1) - timedelta(days=1)
    return start, end


def build_system_prompt(ports: list[dict], data_summary: dict) -> str:
    port_lines = ", ".join(f"{p['port_name']}={p['port_code']}" for p in ports)
    last = str(data_summary.get("last_planned_departure") or "")[:10]
    quarter_note = ""
    if last:
        ref = date.fromisoformat(last)
        q_start, q_end = quarter_bounds(ref)
        quarter_note = (
            f"The data is historical; the latest planned departure is {last}. Treat that as 'today'. "
            f"'This quarter' means {q_start} to {q_end} (Q{(ref.month - 1) // 3 + 1} {ref.year}); "
            "say so when you use it."
        )
    return f"""You are a supply-chain analytics assistant for a container shipping data platform.

Scope: answer only questions about shipments, routes, ports, delays, on-time performance and
delay-risk predictions in this platform's data. For anything else, politely decline in one
sentence and say what you can help with.

Rules:
- Get every number from a tool. Never estimate or invent figures, shipment ids or port codes.
- After each factual claim, cite the tool that produced it in square brackets, e.g. [get_route_stats].
- Always mention the sample size (shipment or completed count) behind a rate or average.
- For predictions, report the probability and risk band, and the model's held-out AUC from the
  tool result; if the AUC is near 0.5, say the score is only weakly informative.
- Use port codes in tool calls. Port names to codes: {port_lines}.
- {quarter_note}
- Be concise: a few sentences or a short list."""


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        executor: ToolExecutor,
        system_prompt: str,
        call_logger: LLMCallLogger,
        max_tool_calls: int = MAX_TOOL_CALLS,
        max_turns: int = MAX_TURNS,
    ):
        self.llm = llm
        self.executor = executor
        self.system_prompt = system_prompt
        self.logger = call_logger
        self.max_tool_calls = max_tool_calls
        self.max_turns = max_turns
        self.price_in = float(os.getenv("SCI_LLM_INPUT_USD_PER_MTOK", "0.10"))
        self.price_out = float(os.getenv("SCI_LLM_OUTPUT_USD_PER_MTOK", "0.50"))

    def _cost(self, tokens_in: int, tokens_out: int) -> float:
        return round(tokens_in / 1e6 * self.price_in + tokens_out / 1e6 * self.price_out, 6)

    def ask(self, question: str, session_id: str | None = None) -> AgentAnswer:
        session_id = session_id or uuid.uuid4().hex
        question_id = uuid.uuid4().hex
        answer = AgentAnswer(text="")
        messages: list[dict] = [{"role": "user", "content": question}]
        retried_grounding = False

        for turn in range(1, self.max_turns + 1):
            start = time.perf_counter()
            resp = self.llm.create(self.system_prompt, messages, TOOL_SCHEMAS)
            answer.input_tokens += resp.input_tokens
            answer.output_tokens += resp.output_tokens
            self.logger.log(
                {
                    "type": "llm_call",
                    "session_id": session_id,
                    "question_id": question_id,
                    "turn": turn,
                    "model": self.llm.model,
                    "system_prompt": self.system_prompt,
                    "messages_sent": messages,
                    "tools_offered": sorted(TOOL_NAMES),
                    "response": resp.content,
                    "stop_reason": resp.stop_reason,
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "latency_ms": round((time.perf_counter() - start) * 1000, 1),
                }
            )
            messages = messages + [{"role": "assistant", "content": resp.content}]

            tool_uses = [b for b in resp.content if b["type"] == "tool_use"]
            if resp.stop_reason == "tool_use" and tool_uses:
                results = []
                for block in tool_uses:
                    if len(answer.tools_called) >= self.max_tool_calls:
                        answer.guardrail_events.append("tool_budget_exhausted")
                        output, is_error, latency = (
                            {"error": "Tool budget for this question is used up. Answer with what you have."},
                            True,
                            0.0,
                        )
                    else:
                        output, is_error, latency = self.executor.execute(block["name"], block["input"])
                        answer.tools_called.append(block["name"])
                        if is_error and str(output.get("error", "")).startswith("invalid arguments"):
                            answer.guardrail_events.append(f"invalid_tool_args:{block['name']}")
                    self.logger.log(
                        {
                            "type": "tool_call",
                            "session_id": session_id,
                            "question_id": question_id,
                            "turn": turn,
                            "tool": block["name"],
                            "input": block["input"],
                            "output": output,
                            "is_error": is_error,
                            "latency_ms": latency,
                        }
                    )
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block["id"],
                            "content": json.dumps(output, default=str),
                            "is_error": is_error,
                        }
                    )
                messages = messages + [{"role": "user", "content": results}]
                continue

            text = "\n".join(b["text"] for b in resp.content if b["type"] == "text").strip()
            problem = validate_citations(text, answer.tools_called)
            if problem and not retried_grounding:
                retried_grounding = True
                answer.guardrail_events.append(f"grounding_retry:{problem}")
                messages = messages + [
                    {
                        "role": "user",
                        "content": f"Your answer failed a grounding check ({problem}). Rewrite it using only "
                        "tool results, citing each fact as [tool_name]. If no tool applies, say you cannot answer.",
                    }
                ]
                continue
            if problem:
                answer.guardrail_events.append(f"grounding_fallback:{problem}")
                text = FALLBACK
            answer.text = text
            break
        else:
            answer.guardrail_events.append("max_turns_reached")
            answer.text = FALLBACK

        answer.cost_usd = self._cost(answer.input_tokens, answer.output_tokens)
        self.logger.log(
            {
                "type": "final_answer",
                "session_id": session_id,
                "question_id": question_id,
                "question": question,
                "answer": answer.text,
                "tools_called": answer.tools_called,
                "guardrail_events": answer.guardrail_events,
                "input_tokens": answer.input_tokens,
                "output_tokens": answer.output_tokens,
                "estimated_cost_usd": answer.cost_usd,
            }
        )
        return answer

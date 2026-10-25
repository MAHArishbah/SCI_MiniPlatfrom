import json

import httpx
import pytest

from sci.assistant.agent import FALLBACK, Agent, LLMCallLogger, LLMResponse, quarter_bounds, validate_citations
from sci.assistant.tools import ToolArgumentError, ToolExecutor, validate_args

PORTS = {"CNSHA", "NLRTM"}


def tool_use(name, args, id_="t1"):
    return LLMResponse("tool_use", [{"type": "tool_use", "id": id_, "name": name, "input": args}], 100, 10)


def text(msg):
    return LLMResponse("end_turn", [{"type": "text", "text": msg}], 100, 10)


class ScriptedLLM:
    model = "scripted"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.seen = []

    def create(self, system, messages, tools):
        self.seen.append(messages)
        return self.responses.pop(0)


class FakeExecutor:
    def __init__(self):
        self.calls = []

    def execute(self, name, args):
        self.calls.append((name, args))
        return {"on_time_rate": 0.75, "completed_count": 12}, False, 1.0


@pytest.fixture
def log_path(tmp_path):
    return tmp_path / "llm.jsonl"


def make_agent(llm, log_path, executor=None, **kw):
    return Agent(llm, executor or FakeExecutor(), "system prompt", LLMCallLogger(log_path), **kw)


# --------------------------------------------------------------- argument guardrail


def test_validate_args_normalizes_ports_and_caps_limits():
    assert validate_args("get_route_stats", {"origin": "cnsha", "destination": " nlrtm "}, PORTS) == {
        "origin": "CNSHA",
        "destination": "NLRTM",
    }
    assert validate_args("query_shipments", {"limit": 500}, PORTS)["page_size"] == 20


@pytest.mark.parametrize(
    "name,args",
    [
        ("get_route_stats", {"origin": "XXTST", "destination": "NLRTM"}),
        ("predict_delay", {"shipment_id": "'; DROP TABLE shipments"}),
        ("query_shipments", {"date_from": "last tuesday"}),
        ("delete_everything", {}),
    ],
)
def test_validate_args_rejects_bad_input(name, args):
    with pytest.raises(ToolArgumentError):
        validate_args(name, args, PORTS)


def test_executor_turns_api_errors_into_tool_errors():
    def handler(request):
        return httpx.Response(404, json={"error": {"code": "not_found", "message": "Shipment 'SHP-12345' not found"}})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://api")
    result, is_error, _ = ToolExecutor("http://api", PORTS, client=client).execute(
        "predict_delay", {"shipment_id": "SHP-12345"}
    )
    assert is_error and result == {"error": "Shipment 'SHP-12345' not found", "status_code": 404}


def test_executor_does_not_call_api_for_invalid_args():
    calls = []
    client = httpx.Client(transport=httpx.MockTransport(lambda r: calls.append(r)), base_url="http://api")
    result, is_error, _ = ToolExecutor("http://api", PORTS, client=client).execute(
        "get_route_stats", {"origin": "XXTST", "destination": "NLRTM"}
    )
    assert is_error and "invalid arguments" in result["error"] and calls == []


def test_executor_compacts_shipment_lists():
    item = {"shipment_id": "SHP-00001", "route_key": "A → B", "customer_id": "secret-ish", "status": "DELIVERED"}
    client = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"total": 1, "items": [item]})),
        base_url="http://api",
    )
    result, is_error, _ = ToolExecutor("http://api", PORTS, client=client).execute("query_shipments", {})
    assert not is_error and result["total_matching"] == 1
    assert "customer_id" not in result["items"][0]


# --------------------------------------------------------------- agent loop + guardrails


def test_happy_path_calls_tool_and_returns_cited_answer(log_path):
    llm = ScriptedLLM(
        tool_use("get_route_stats", {"origin": "CNSHA", "destination": "NLRTM"}),
        text("On-time rate is 75% across 12 completed shipments [get_route_stats]."),
    )
    answer = make_agent(llm, log_path).ask("On-time rate Shanghai to Rotterdam?")
    assert answer.tools_called == ["get_route_stats"]
    assert "[get_route_stats]" in answer.text and answer.guardrail_events == []
    # The tool result is fed back to the model as a tool_result block.
    tool_result = llm.seen[1][-1]["content"][0]
    assert tool_result["type"] == "tool_result" and json.loads(tool_result["content"])["on_time_rate"] == 0.75


def test_tool_budget_is_enforced(log_path):
    executor = FakeExecutor()
    llm = ScriptedLLM(
        *[tool_use("get_route_stats", {"origin": "CNSHA", "destination": "NLRTM"}, f"t{i}") for i in range(3)],
        text("Done [get_route_stats]."),
    )
    answer = make_agent(llm, log_path, executor=executor, max_tool_calls=2).ask("loop forever")
    assert len(executor.calls) == 2
    assert "tool_budget_exhausted" in answer.guardrail_events


def test_max_turns_stops_runaway_loops(log_path):
    llm = ScriptedLLM(*[tool_use("get_route_stats", {}, f"t{i}") for i in range(3)])
    answer = make_agent(llm, log_path, max_turns=3).ask("?")
    assert answer.text == FALLBACK and "max_turns_reached" in answer.guardrail_events


def test_ungrounded_numbers_get_one_retry_then_fallback(log_path):
    llm = ScriptedLLM(text("About 80% are on time."), text("Roughly 80%."))
    answer = make_agent(llm, log_path).ask("on-time rate?")
    assert answer.text == FALLBACK
    assert [e.split(":")[0] for e in answer.guardrail_events] == ["grounding_retry", "grounding_fallback"]


def test_retry_can_recover(log_path):
    llm = ScriptedLLM(
        text("About 80% are on time."),
        tool_use("get_route_stats", {"origin": "CNSHA", "destination": "NLRTM"}),
        text("75% on time across 12 completed shipments [get_route_stats]."),
    )
    answer = make_agent(llm, log_path).ask("on-time rate?")
    assert answer.text.startswith("75%")


def test_off_domain_refusal_passes_grounding(log_path):
    llm = ScriptedLLM(text("I can only help with shipping and logistics questions about this platform."))
    answer = make_agent(llm, log_path).ask("Write me a poem")
    assert answer.tools_called == [] and answer.guardrail_events == []


@pytest.mark.parametrize(
    "answer,tools,ok",
    [
        ("75% on time [get_route_stats]", ["get_route_stats"], True),
        ("75% on time", ["get_route_stats"], False),
        ("75% on time [predict_delay]", ["get_route_stats"], False),
        ("Please give a shipment id like SHP-00421.", [], True),
        ("This quarter is 2025-10-01 to 2025-12-31.", [], True),
    ],
)
def test_validate_citations(answer, tools, ok):
    assert (validate_citations(answer, tools) is None) is ok


def test_every_llm_and_tool_call_is_logged(log_path):
    llm = ScriptedLLM(
        tool_use("get_route_stats", {"origin": "CNSHA", "destination": "NLRTM"}),
        text("75% [get_route_stats]."),
    )
    make_agent(llm, log_path).ask("q")
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [r["type"] for r in records] == ["llm_call", "tool_call", "llm_call", "final_answer"]
    first = records[0]
    assert first["system_prompt"] == "system prompt" and first["messages_sent"][0]["content"] == "q"
    assert records[1]["tool"] == "get_route_stats" and records[1]["output"]["on_time_rate"] == 0.75
    assert records[-1]["answer"] == "75% [get_route_stats]." and records[-1]["estimated_cost_usd"] > 0


def test_quarter_bounds():
    from datetime import date

    assert quarter_bounds(date(2025, 10, 27)) == (date(2025, 10, 1), date(2025, 12, 31))
    assert quarter_bounds(date(2024, 2, 29)) == (date(2024, 1, 1), date(2024, 3, 31))

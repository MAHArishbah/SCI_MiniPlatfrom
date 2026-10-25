"""CLI for the GenAI assistant.

python -m sci.assistant.cli                       # interactive
python -m sci.assistant.cli -q "On-time rate Shanghai to Rotterdam?"
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

import httpx

from sci.assistant.agent import Agent, AnthropicLLM, LLMCallLogger, build_system_prompt
from sci.assistant.tools import ToolExecutor

DEFAULT_MODEL = "claude-haiku-5-5"


def build_agent(api_url: str, model: str, log_path: Path) -> Agent:
    client = httpx.Client(base_url=api_url, timeout=10.0)
    ports = client.get("/ports").raise_for_status().json()
    health = client.get("/health").json()
    executor = ToolExecutor(api_url, {p["port_code"] for p in ports}, client=client)
    return Agent(
        llm=AnthropicLLM(model),
        executor=executor,
        system_prompt=build_system_prompt(ports, health.get("data") or {}),
        call_logger=LLMCallLogger(log_path),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask the supply-chain assistant a question.")
    parser.add_argument("-q", "--question", help="Ask one question and exit")
    parser.add_argument("--api-url", default=os.getenv("SCI_API_URL", "http://localhost:8000"))
    parser.add_argument("--model", default=os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL))
    parser.add_argument("--log", type=Path, default=Path(os.getenv("SCI_LLM_LOG", "logs/llm_calls.jsonl")))
    args = parser.parse_args(argv)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set (see .env.example).", file=sys.stderr)
        return 2
    try:
        agent = build_agent(args.api_url, args.model, args.log)
    except httpx.HTTPError as exc:
        print(f"Cannot reach the API at {args.api_url}: {exc}", file=sys.stderr)
        return 2

    session = uuid.uuid4().hex
    questions = [args.question] if args.question else None
    print(f"Supply-chain assistant ({args.model}). LLM calls are logged to {args.log}.")
    while True:
        if questions is not None:
            if not questions:
                break
            question = questions.pop()
        else:
            try:
                question = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if question.lower() in {"exit", "quit", ""}:
                break
        answer = agent.ask(question, session_id=session)
        print(f"\n{answer.text}")
        meta = f"tools: {', '.join(answer.tools_called) or 'none'} | ~${answer.cost_usd:.5f}"
        if answer.guardrail_events:
            meta += f" | guardrails: {', '.join(answer.guardrail_events)}"
        print(f"  ({meta})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

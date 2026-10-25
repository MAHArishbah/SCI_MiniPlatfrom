"""Standalone, re-runnable data quality step. Reads the raw CSVs directly and does not
depend on (or touch) the pipeline or the data store.

    python dq_check.py --input ./data/raw --output ./artifacts/dq_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from sci import __version__
from sci.dq.checks import CRITICAL, INFO, WARNING, RawFrames, all_checks

logger = logging.getLogger("sci.dq")

RAW_STEMS = ("ports", "shipments", "port_events")


def _resolve(input_dir: Path, stem: str) -> Path:
    exact = input_dir / f"{stem}.csv"
    if exact.exists():
        return exact
    # Tolerate suffixed exports like shipments_2.csv, but refuse ambiguity.
    matches = sorted(input_dir.glob(f"{stem}_*.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one {stem}*.csv in {input_dir}, found {[m.name for m in matches]}")
    return matches[0]


def load_raw(input_dir: Path) -> tuple[RawFrames, dict[str, Path]]:
    paths = {name: _resolve(input_dir, name) for name in RAW_STEMS}
    # Read everything as strings: DQ must see the data exactly as delivered.
    frames = {
        name: pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8") for name, path in paths.items()
    }
    return RawFrames(**frames), paths


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_checks(frames: RawFrames) -> list[dict]:
    return [check(frames).to_dict() for check in all_checks()]


def build_report(frames: RawFrames, paths: dict[str, Path]) -> dict:
    results = run_checks(frames)
    failed = [r for r in results if r["status"] == "fail"]
    by_sev = {sev: sum(1 for r in failed if r["severity"] == sev) for sev in (CRITICAL, WARNING, INFO)}
    return {
        "report_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "tool_version": __version__,
        "inputs": {
            name: {"file": path.name, "rows": int(len(getattr(frames, name))), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "summary": {
            "checks_run": len(results),
            "checks_failed": len(failed),
            "failed_by_severity": by_sev,
            "assumptions": [
                "All timestamps are timezone-naive and treated as UTC.",
                "Severity: critical = record unusable for its core purpose; warning = fixable/flaggable; info = observation.",
                "rows_affected counts rows in the raw file; one row can fail several checks.",
            ],
        },
        "checks": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run data quality checks on the raw CSV files.")
    parser.add_argument("--input", type=Path, default=Path("data/raw"), help="Directory with the raw CSVs")
    parser.add_argument("--output", type=Path, default=Path("artifacts/dq_report.json"), help="JSON report path")
    parser.add_argument(
        "--fail-on-critical", action="store_true", help="Exit 1 if any critical check fails (use as a CI gate)"
    )
    args = parser.parse_args(argv)

    from sci.logging_config import configure_logging  # lazy loading

    configure_logging()
    frames, paths = load_raw(args.input)
    report = build_report(frames, paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))

    for r in report["checks"]:
        if r["status"] == "fail":
            logger.info(
                "dq_check_failed",
                extra={k: r[k] for k in ("check", "severity", "rows_affected", "pct_of_total")},
            )
    logger.info("dq_run_complete", extra={"output": str(args.output), **report["summary"]["failed_by_severity"]})

    if args.fail_on_critical and report["summary"]["failed_by_severity"][CRITICAL]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

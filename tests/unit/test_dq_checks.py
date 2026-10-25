import json

import pandas as pd
import pytest

from sci.dq import checks, rules
from sci.dq.runner import build_report, load_raw, main

# Planted issues in tests/fixtures/raw and the exact row counts we expect.
EXPECTED_FAILURES = {
    "shipments.duplicate_rows_exact": 1,
    "shipments.duplicate_id_conflicting": 1,
    "shipments.unknown_port_code": 1,
    "shipments.cargo_type_non_canonical": 2,
    "shipments.cargo_type_missing": 1,
    "shipments.weight_tons_missing": 1,
    "shipments.weight_tons_negative": 1,
    "shipments.container_count_zero": 1,
    "shipments.container_count_outlier": 1,
    "shipments.status_non_canonical": 2,
    "shipments.missing_actuals_non_cancelled": 2,
    "shipments.actual_departure_before_booking": 1,
    "ports.country_missing": 1,
    "port_events.event_id_collision": 2,
    "port_events.event_type_missing": 1,
    "port_events.unknown_vessel": 1,
    "port_events.delayed_event_zero_minutes": 1,
    "port_events.notes_missing": 1,
}


@pytest.fixture(scope="module")
def results(raw_frames):
    return {r.name: r for r in (check(raw_frames) for check in checks.all_checks())}


@pytest.mark.parametrize("name,expected", EXPECTED_FAILURES.items())
def test_planted_issue_is_detected_with_exact_count(results, name, expected):
    assert results[name].rows_affected == expected


def test_clean_checks_pass(results):
    failing = {name for name, r in results.items() if r.rows_affected}
    assert failing == set(EXPECTED_FAILURES)


def test_sample_keys_point_at_the_offending_rows(results):
    assert results["shipments.unknown_port_code"].to_dict()["sample_keys"] == ["SHP-00005"]
    assert results["port_events.unknown_vessel"].to_dict()["sample_keys"] == ["EVT-000003"]


def test_check_dict_has_required_report_fields(results):
    record = results["shipments.weight_tons_negative"].to_dict()
    for key in ("check", "rows_affected", "pct_of_total", "severity", "recommended_action", "status"):
        assert key in record
    assert record["severity"] in {"critical", "warning", "info"}
    assert record["pct_of_total"] == pytest.approx(100 / 12, abs=0.01)


@pytest.mark.parametrize("raw,expected", [("", True), ("N/A", True), (" null ", True), ("0", False), ("x", False)])
def test_is_blank(raw, expected):
    assert bool(rules.is_blank(pd.Series([raw])).iloc[0]) is expected


@pytest.mark.parametrize(
    "raw,expected",
    [("DELIVERED", "DELIVERED"), ("delivered", "DELIVERED"), ("Complete", "DELIVERED"), ("COMPLETED", "DELIVERED"),
     ("N/A", "UNKNOWN"), ("", "UNKNOWN"), ("lost at sea", "UNKNOWN"), ("CANCELLED", "CANCELLED")],
)  # fmt: skip
def test_normalize_status(raw, expected):
    assert rules.normalize_status(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [("FURNITURE", "Furniture"), (" Chemicals", "Chemicals"), ("machinery ", "Machinery"),
     ("food & beverage", "Food & Beverage"), ("", None), ("Spaceships", "Spaceships")],
)  # fmt: skip
def test_normalize_cargo_type(raw, expected):
    assert rules.normalize_cargo_type(raw) == expected


def test_iqr_fence_flags_only_extreme_values():
    counts = pd.Series([10, 12, 15, 8, 20, 6, 9, 900], dtype=float)
    fence = rules.iqr_upper_fence(counts)
    assert 20 < fence < 900


def test_report_summary_counts(raw_dir):
    frames, paths = load_raw(raw_dir)
    report = build_report(frames, paths)
    assert report["summary"]["checks_failed"] == len(EXPECTED_FAILURES)
    assert set(report["inputs"]) == {"ports", "shipments", "port_events"}
    assert all(len(v["sha256"]) == 64 for v in report["inputs"].values())


def test_cli_writes_report_and_gates_on_critical(raw_dir, tmp_path):
    out = tmp_path / "report.json"
    assert main(["--input", str(raw_dir), "--output", str(out)]) == 0
    assert json.loads(out.read_text())["checks"]
    # The fixture has critical issues, so the CI gate flag must fail the run.
    assert main(["--input", str(raw_dir), "--output", str(out), "--fail-on-critical"]) == 1


def test_loader_accepts_suffixed_file_names(raw_dir, tmp_path):
    for stem in ("ports", "shipments", "port_events"):
        (tmp_path / f"{stem}_2.csv").write_bytes((raw_dir / f"{stem}.csv").read_bytes())
    frames, paths = load_raw(tmp_path)
    assert paths["port_events"].name == "port_events_2.csv"
    assert len(frames.shipments) == 12

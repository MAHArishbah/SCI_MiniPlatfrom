from collections import Counter

import pandas as pd
import pytest

from sci.pipeline import transform


def _shipment(planned_arrival, actual_arrival, actual_departure="2025-01-02 00:00:00"):
    return pd.DataFrame(
        {
            "origin_port": ["CNSHA"],
            "destination_port": ["NLRTM"],
            "booking_date": pd.to_datetime(["2024-12-25 00:00:00"]),
            "planned_departure": pd.to_datetime(["2025-01-01 00:00:00"]),
            "actual_departure": pd.to_datetime([actual_departure]),
            "planned_arrival": pd.to_datetime([planned_arrival]),
            "actual_arrival": pd.to_datetime([actual_arrival]),
        }
    )


@pytest.mark.parametrize("hours_late,on_time", [(-5, True), (0, True), (24, True), (24.5, False), (100, False)])
def test_on_time_flag_boundary_is_24_hours_inclusive(hours_late, on_time):
    actual = pd.Timestamp("2025-02-01") + pd.Timedelta(hours=hours_late)
    out = transform.derive_shipment_fields(_shipment("2025-02-01 00:00:00", str(actual)))
    assert out["actual_delay_hours"].iloc[0] == pytest.approx(hours_late)
    assert bool(out["on_time_flag"].iloc[0]) is on_time


def test_delay_is_null_safe():
    out = transform.derive_shipment_fields(_shipment("2025-02-01 00:00:00", None, actual_departure=None))
    assert pd.isna(out["actual_delay_hours"].iloc[0])
    assert pd.isna(out["on_time_flag"].iloc[0])  # unknown, not False
    assert pd.isna(out["transit_days_actual"].iloc[0])


def test_route_key_and_transit_days():
    out = transform.derive_shipment_fields(_shipment("2025-01-31 00:00:00", "2025-02-01 12:00:00"))
    assert out["route_key"].iloc[0] == "CNSHA → NLRTM"
    assert out["transit_days_planned"].iloc[0] == pytest.approx(30.0)
    assert out["transit_days_actual"].iloc[0] == pytest.approx(30.5)
    assert out["booking_lead_days"].iloc[0] == pytest.approx(7.0)


@pytest.mark.parametrize(
    "tz,expected", [("UTC+8", 8.0), ("UTC-8", -8.0), ("UTC+5:30", 5.5), ("UTC+0", 0.0), ("PST", None)]
)
def test_parse_utc_offset(tz, expected):
    assert transform.parse_utc_offset(tz) == expected


def test_clean_ports_fills_country_from_locode(raw_frames):
    ports = transform.clean_ports(raw_frames.ports).set_index("port_code")
    assert ports.loc["BEANR", "country"] == "Belgium"
    assert ports.loc["BEANR", "dq_flags"] == ["country_filled_from_locode"]
    assert ports.loc["INNSA", "utc_offset_hours"] == 5.5


@pytest.fixture(scope="module")
def cleaned(raw_frames):
    curated, quarantine = transform.clean_shipments(raw_frames)
    return curated.set_index("shipment_id"), quarantine


def test_shipments_quarantine_reasons(cleaned):
    curated, quarantine = cleaned
    assert Counter(quarantine["quarantine_reason"]) == {
        "duplicate_exact": 1,
        "duplicate_id_conflicting": 1,
        "unknown_port_code": 1,
    }
    assert len(curated) == 9 and curated.index.is_unique
    assert "SHP-00005" not in curated.index


def test_conflicting_duplicate_keeps_most_complete_row(cleaned):
    curated, quarantine = cleaned
    row = curated.loc["SHP-00004"]
    assert row["cargo_type"] == "Machinery"  # the copy with a blank cargo was dropped
    assert "duplicate_id_resolved" in row["dq_flags"]
    dropped = quarantine[quarantine["quarantine_reason"] == "duplicate_id_conflicting"]
    assert dropped["cargo_type"].tolist() == [""]


def test_resolve_duplicates_tie_keeps_first_seen():
    raw = pd.DataFrame({"shipment_id": ["A", "A"], "weight_tons": ["10", "11"]})
    kept, exact, conflict, ids = transform.resolve_duplicate_shipments(raw)
    assert list(kept) == [0] and list(conflict) == [1] and ids == {"A"} and len(exact) == 0


def test_invalid_numbers_are_nulled_and_flagged(cleaned):
    curated, _ = cleaned
    row = curated.loc["SHP-00006"]
    assert pd.isna(row["weight_tons"]) and pd.isna(row["container_count"])
    assert row["cargo_type"] == "Chemicals"
    assert {"weight_negative_nulled", "container_count_zero_nulled", "cargo_type_normalized"} <= set(row["dq_flags"])


def test_outlier_and_impossible_departure(cleaned):
    curated, _ = cleaned
    row = curated.loc["SHP-00010"]
    assert pd.isna(row["container_count"])
    assert pd.isna(row["actual_departure"])  # departed before booking -> nulled
    assert row["actual_delay_hours"] == pytest.approx(12.0)  # arrival-based delay kept
    assert "actual_departure_before_booking_nulled" in row["dq_flags"]


def test_status_normalization_and_labelling(cleaned):
    curated, _ = cleaned
    assert curated.loc["SHP-00007", "status"] == "DELIVERED"
    assert "missing_actuals" in curated.loc["SHP-00007", "dq_flags"]
    assert not curated.loc["SHP-00007", "is_labelled"]
    assert curated.loc["SHP-00008", "status"] == "UNKNOWN"
    assert not curated.loc["SHP-00009", "is_labelled"]  # cancelled
    assert curated["is_labelled"].sum() == 6


def test_port_events_cleaning(raw_frames):
    events, quarantine = transform.clean_port_events(raw_frames)
    assert quarantine["quarantine_reason"].tolist() == ["unknown_vessel"]
    assert len(events) == 5
    assert events["event_sk"].is_unique  # event_id is not, the surrogate key is
    collided = events[events["event_id"] == "EVT-000002"]
    assert len(collided) == 2 and all("event_id_collision" in f for f in collided["dq_flags"])
    assert events.set_index("event_id").loc["EVT-000004", "event_type"] == "UNKNOWN"

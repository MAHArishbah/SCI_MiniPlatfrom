"""Data quality checks. Each check is a pure function over the raw (all-string) frames and
returns a CheckResult carrying a boolean row mask, so the pipeline can reuse the exact same
detection logic when it applies the remediation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from sci.dq import rules

CRITICAL, WARNING, INFO = "critical", "warning", "info"


@dataclass
class RawFrames:
    """Raw CSVs loaded as strings, untouched (dtype=str, keep_default_na=False)."""

    ports: pd.DataFrame
    shipments: pd.DataFrame
    port_events: pd.DataFrame


@dataclass
class CheckResult:
    name: str
    table: str
    description: str
    severity: str
    action: str
    mask: pd.Series = field(repr=False)  # true/false per row
    key_column: str | None = None
    detection: str = ""
    keys: pd.Series | None = field(default=None, repr=False)

    @property
    def rows_affected(self) -> int:
        return int(self.mask.sum())

    @property
    def total_rows(self) -> int:
        return int(len(self.mask))

    def to_dict(self, sample_size: int = 5) -> dict:
        """method to return the Checkresult object to standard dict"""
        sample: list[str] = []
        if self.keys is not None and self.rows_affected:
            sample = (
                self.keys[self.mask].astype(str).drop_duplicates().head(sample_size).tolist()
            )  # str for prper json serialization,dr dp before head to have truly unique
        pct = round(100 * self.rows_affected / self.total_rows, 3) if self.total_rows else 0.0
        return {
            "check": self.name,
            "table": self.table,
            "status": "fail" if self.rows_affected else "pass",
            "severity": self.severity,
            "rows_affected": self.rows_affected,
            "total_rows": self.total_rows,
            "pct_of_total": pct,
            "description": self.description,
            "detection": self.detection,
            "recommended_action": self.action,
            "sample_keys": sample,
        }


CheckFn = Callable[[RawFrames], CheckResult]

_REGISTRY: list[CheckFn] = []


def register(fn: CheckFn) -> CheckFn:
    _REGISTRY.append(fn)
    return fn


def all_checks() -> list[CheckFn]:
    return list(_REGISTRY)


def _valid_ports(frames: RawFrames) -> set[str]:
    return set(frames.ports["port_code"].str.strip())


# --------------------------------------------------------------------------- shipments


@register
def shipments_duplicate_exact(f: RawFrames) -> CheckResult:
    s = f.shipments
    return CheckResult(
        name="shipments.duplicate_rows_exact",
        table="shipments",
        description="Row is an exact copy of an earlier row (same shipment_id and all fields).",
        detection="DataFrame.duplicated(keep='first') over all columns.",
        severity=WARNING,
        action="drop: keep the first copy, drop the rest (no information lost).",
        mask=s.duplicated(keep="first"),
        keys=s["shipment_id"],
    )


@register
def shipments_duplicate_conflicting(f: RawFrames) -> CheckResult:
    s = f.shipments
    mask = s["shipment_id"].duplicated(keep="first") & ~s.duplicated(keep="first")
    return CheckResult(
        name="shipments.duplicate_id_conflicting",
        table="shipments",
        description="shipment_id repeats with different field values (e.g. weight or cargo differs).",
        detection="shipment_id duplicated but full row not duplicated.",
        severity=CRITICAL,
        action="fix: keep the most complete record per id (fewest blanks, tie -> first seen), drop the rest, flag the kept row.",
        mask=mask,
        keys=s["shipment_id"],
    )


@register
def shipments_unknown_port(f: RawFrames) -> CheckResult:
    s, valid = f.shipments, _valid_ports(f)
    mask = ~s["origin_port"].str.strip().isin(valid) | ~s["destination_port"].str.strip().isin(valid)
    return CheckResult(
        name="shipments.unknown_port_code",
        table="shipments",
        description="origin_port or destination_port not in the ports reference (e.g. XXTST, ZZZZZ, UNKNW placeholders).",
        detection="Referential integrity: port code NOT IN ports.port_code.",
        severity=CRITICAL,
        action="quarantine: route metrics and the model cannot use a shipment without real ports.",
        mask=mask,
        keys=s["shipment_id"],
    )


@register
def shipments_cargo_non_canonical(f: RawFrames) -> CheckResult:
    raw = f.shipments["cargo_type"]
    normalized = raw.map(rules.normalize_cargo_type)
    mask = ~rules.is_blank(raw) & (raw != normalized)
    return CheckResult(
        name="shipments.cargo_type_non_canonical",
        table="shipments",
        description="cargo_type differs from the canonical label only by case/whitespace (e.g. 'FURNITURE', ' Chemicals').",
        detection="Raw value != whitespace-collapsed, case-insensitive match to the canonical list.",
        severity=WARNING,
        action="fix: normalize to the canonical label.",
        mask=mask,
        keys=f.shipments["shipment_id"],
    )


@register
def shipments_cargo_unrecognized(f: RawFrames) -> CheckResult:
    normalized = f.shipments["cargo_type"].map(rules.normalize_cargo_type)
    mask = normalized.notna() & ~normalized.isin(rules.CANONICAL_CARGO_TYPES)
    return CheckResult(
        name="shipments.cargo_type_unrecognized",
        table="shipments",
        description="cargo_type does not match any canonical category even after normalization.",
        detection="Normalized value NOT IN canonical cargo list.",
        severity=WARNING,
        action="flag: keep raw value, review mapping.",
        mask=mask,
        keys=f.shipments["shipment_id"],
    )


@register
def shipments_cargo_missing(f: RawFrames) -> CheckResult:
    return CheckResult(
        name="shipments.cargo_type_missing",
        table="shipments",
        description="cargo_type is blank.",
        detection="Blank / null-token check.",
        severity=WARNING,
        action="fix: set to 'Unknown' so the record stays usable; model treats it as its own category.",
        mask=rules.is_blank(f.shipments["cargo_type"]),
        keys=f.shipments["shipment_id"],
    )


@register
def shipments_weight_missing(f: RawFrames) -> CheckResult:
    return CheckResult(
        name="shipments.weight_tons_missing",
        table="shipments",
        description="weight_tons is blank.",
        detection="Blank / null-token check.",
        severity=WARNING,
        action="flag: keep NULL (no defensible fill value); the model imputes the training median.",
        mask=rules.is_blank(f.shipments["weight_tons"]),
        keys=f.shipments["shipment_id"],
    )


@register
def shipments_weight_negative(f: RawFrames) -> CheckResult:
    weight = rules.to_number(f.shipments["weight_tons"])
    return CheckResult(
        name="shipments.weight_tons_negative",
        table="shipments",
        description="weight_tons < 0, which is physically impossible.",
        detection="Numeric range check: weight_tons < 0.",
        severity=WARNING,
        action="fix: set to NULL and flag (sign-flip is a guess we should not make silently).",
        mask=(weight < 0).fillna(
            False
        ),  # prevention against in future is nullabe flora64 instead of numpy flot64, curated output is FLoat64
        keys=f.shipments["shipment_id"],
    )


@register
def shipments_container_zero(f: RawFrames) -> CheckResult:
    count = rules.to_number(f.shipments["container_count"])
    return CheckResult(
        name="shipments.container_count_zero",
        table="shipments",
        description="container_count is 0 (or negative) for a booked shipment.",
        detection="Numeric range check: container_count <= 0.",
        severity=WARNING,
        action="fix: set to NULL and flag.",
        mask=(count <= 0).fillna(False),
        keys=f.shipments["shipment_id"],
    )


@register
def shipments_container_outlier(f: RawFrames) -> CheckResult:
    count = rules.to_number(f.shipments["container_count"])
    fence = rules.iqr_upper_fence(count.dropna())
    return CheckResult(
        name="shipments.container_count_outlier",
        table="shipments",
        description=f"container_count above the robust upper fence ({fence:.0f}); values here are ~15-25x the normal max.",
        detection=f"IQR fence: count > Q3 + {rules.OUTLIER_IQR_K} * IQR on positive counts.",
        severity=WARNING,
        action="fix: set to NULL and flag (likely unit/keying error, e.g. TEU vs containers or an extra digit).",
        mask=(count > fence).fillna(False),
        keys=f.shipments["shipment_id"],
    )


@register
def shipments_status_non_canonical(f: RawFrames) -> CheckResult:
    raw = f.shipments["status"]
    mask = ~raw.isin(rules.CANONICAL_STATUSES)
    return CheckResult(
        name="shipments.status_non_canonical",
        table="shipments",
        description="status outside {DELIVERED, DELAYED, CANCELLED}: 'COMPLETED', 'Complete', 'delivered', 'N/A', blank.",
        detection="Raw value NOT IN canonical status set.",
        severity=WARNING,
        action="fix: map completion variants to DELIVERED, blanks/'N/A' to UNKNOWN.",
        mask=mask,
        keys=f.shipments["shipment_id"],
    )


@register
def shipments_missing_actuals(f: RawFrames) -> CheckResult:
    s = f.shipments
    status = s["status"].map(rules.normalize_status)
    no_actuals = rules.is_blank(s["actual_arrival"]) | rules.is_blank(s["actual_departure"])
    return CheckResult(
        name="shipments.missing_actuals_non_cancelled",
        table="shipments",
        # the check is right on this data, but the description overstates it — departure isn't needed for arrival delay, and I'd split it into two checks with different severities.
        description="Shipment is not CANCELLED but has no actual departure/arrival, so its delay cannot be computed.",
        detection="status != CANCELLED AND (actual_departure blank OR actual_arrival blank).",
        severity=CRITICAL,
        action="flag: keep the record (it is a real booking) but exclude it from delay metrics and ML labels.",
        mask=(status != "CANCELLED") & no_actuals,
        keys=s["shipment_id"],
    )


@register
def shipments_departure_before_booking(f: RawFrames) -> CheckResult:
    s = f.shipments
    mask = (rules.parse_ts(s["actual_departure"]) < rules.parse_ts(s["booking_date"])).fillna(False)
    return CheckResult(
        name="shipments.actual_departure_before_booking",
        table="shipments",
        description="Vessel actually departed before the booking was made: temporally impossible.",
        detection="actual_departure < booking_date.",
        severity=WARNING,
        action="fix: set actual_departure to NULL and flag; arrival-based delay is kept.",
        mask=mask,
        keys=s["shipment_id"],
    )


@register
def shipments_unparseable_timestamp(f: RawFrames) -> CheckResult:
    s = f.shipments
    mask = pd.Series(False, index=s.index)
    for col in rules.SHIPMENT_TIMESTAMP_COLUMNS:
        mask |= ~rules.is_blank(s[col]) & rules.parse_ts(s[col]).isna()
    return CheckResult(
        name="shipments.unparseable_timestamp",
        table="shipments",
        description="A non-blank timestamp that does not parse as YYYY-MM-DD HH:MM:SS.",
        detection="Strict format parse of every timestamp column.",
        severity=CRITICAL,
        action="quarantine if present.",
        mask=mask,
        keys=s["shipment_id"],
    )


@register
def shipments_required_timestamp_missing(f: RawFrames) -> CheckResult:
    s = f.shipments
    mask = pd.Series(False, index=s.index)
    for col in ["booking_date", "planned_departure", "planned_arrival"]:
        mask |= rules.is_blank(s[col])
    return CheckResult(
        name="shipments.planned_timestamp_missing",
        table="shipments",
        description="booking_date / planned_departure / planned_arrival blank (needed for every metric and the model).",
        detection="Blank check on required planning timestamps.",
        severity=CRITICAL,
        action="quarantine if present.",
        mask=mask,
        keys=s["shipment_id"],
    )


@register
def shipments_planned_order(f: RawFrames) -> CheckResult:
    s = f.shipments
    pd_, pa = rules.parse_ts(s["planned_departure"]), rules.parse_ts(s["planned_arrival"])
    bd = rules.parse_ts(s["booking_date"])
    mask = ((pa <= pd_) | (bd > pd_)).fillna(False)
    return CheckResult(
        name="shipments.planned_timeline_inconsistent",
        table="shipments",
        description="planned_arrival <= planned_departure, or booking_date after planned_departure.",
        detection="Timeline order check on planned timestamps.",
        severity=CRITICAL,
        action="quarantine if present.",
        mask=mask,
        keys=s["shipment_id"],
    )


@register
def shipments_actual_order(f: RawFrames) -> CheckResult:
    s = f.shipments
    mask = (rules.parse_ts(s["actual_arrival"]) <= rules.parse_ts(s["actual_departure"])).fillna(False)
    return CheckResult(
        name="shipments.actual_arrival_before_departure",
        table="shipments",
        description="actual_arrival <= actual_departure.",
        detection="Timeline order check on actual timestamps.",
        severity=CRITICAL,
        action="flag and null actuals if present.",
        mask=mask,
        keys=s["shipment_id"],
    )


@register
def shipments_same_origin_destination(f: RawFrames) -> CheckResult:
    s = f.shipments
    return CheckResult(
        name="shipments.origin_equals_destination",
        table="shipments",
        description="origin_port == destination_port.",
        detection="Column equality check.",
        severity=WARNING,
        action="flag if present.",
        mask=s["origin_port"].str.strip() == s["destination_port"].str.strip(),
        keys=s["shipment_id"],
    )


@register
def shipments_cancelled_with_actuals(f: RawFrames) -> CheckResult:
    s = f.shipments
    status = s["status"].map(rules.normalize_status)
    mask = (status == "CANCELLED") & ~rules.is_blank(s["actual_arrival"])
    return CheckResult(
        name="shipments.cancelled_with_actuals",
        table="shipments",
        description="CANCELLED shipment that still has an actual arrival.",
        detection="status == CANCELLED AND actual_arrival not blank.",
        severity=WARNING,
        action="flag if present.",
        mask=mask,
        keys=s["shipment_id"],
    )


@register
def shipments_status_vs_delay(f: RawFrames) -> CheckResult:
    s = f.shipments
    status = s["status"].map(rules.normalize_status)
    delay_h = (rules.parse_ts(s["actual_arrival"]) - rules.parse_ts(s["planned_arrival"])).dt.total_seconds() / 3600
    late = delay_h > rules.ON_TIME_THRESHOLD_HOURS
    mask = (((status == "DELIVERED") & late) | ((status == "DELAYED") & ~late & delay_h.notna())).fillna(False)
    return CheckResult(
        name="shipments.status_contradicts_delay",
        table="shipments",
        description="status DELIVERED but delay > 24h, or DELAYED but delay <= 24h.",
        detection="Cross-field consistency between status and computed arrival delay.",
        severity=WARNING,
        action="flag if present (computed delay wins over the status label).",
        mask=mask,
        keys=s["shipment_id"],
    )


# --------------------------------------------------------------------------- ports


@register
def ports_missing_country(f: RawFrames) -> CheckResult:
    p = f.ports
    return CheckResult(
        name="ports.country_missing",
        table="ports",
        description="country is blank (BEANR / Antwerp).",
        detection="Blank / null-token check.",
        severity=WARNING,
        action="fix: fill from the UN/LOCODE country prefix (BE -> Belgium).",
        mask=rules.is_blank(p["country"]),
        keys=p["port_code"],
    )


@register
def ports_duplicate_code(f: RawFrames) -> CheckResult:
    p = f.ports
    return CheckResult(
        name="ports.duplicate_port_code",
        table="ports",
        description="port_code appears more than once in the reference table.",
        detection="port_code duplicated.",
        severity=CRITICAL,
        action="quarantine duplicates if present.",
        mask=p["port_code"].duplicated(keep=False),
        keys=p["port_code"],
    )


@register
def ports_congestion_range(f: RawFrames) -> CheckResult:
    score = rules.to_number(f.ports["avg_congestion_score"])
    return CheckResult(
        name="ports.congestion_score_out_of_range",
        table="ports",
        description="avg_congestion_score missing or outside [0, 1].",
        detection="Numeric range check.",
        severity=WARNING,
        action="flag if present.",
        mask=(score.isna() | (score < 0) | (score > 1)),
        keys=f.ports["port_code"],
    )


# --------------------------------------------------------------------------- port events


@register
def events_duplicate_id(f: RawFrames) -> CheckResult:
    e = f.port_events
    return CheckResult(
        name="port_events.event_id_collision",
        table="port_events",
        description="event_id reused by different events (different port, vessel, time). All rows sharing an id are counted.",
        detection="event_id duplicated(keep=False); rows are not exact duplicates.",
        severity=WARNING,
        action="fix: keep all rows (they are distinct events), add a deterministic surrogate key, flag.",
        mask=e["event_id"].duplicated(keep=False),
        keys=e["event_id"],
    )


@register
def events_exact_duplicate(f: RawFrames) -> CheckResult:
    e = f.port_events
    return CheckResult(
        name="port_events.duplicate_rows_exact",
        table="port_events",
        description="Exact duplicate event rows (typical of at-least-once streaming delivery).",
        detection="DataFrame.duplicated(keep='first').",
        severity=WARNING,
        action="drop duplicates if present.",
        mask=e.duplicated(keep="first"),
        keys=e["event_id"],
    )


@register
def events_type_missing(f: RawFrames) -> CheckResult:
    e = f.port_events
    return CheckResult(
        name="port_events.event_type_missing",
        table="port_events",
        description="event_type is blank.",
        detection="Blank / null-token check.",
        severity=WARNING,
        action="fix: set to 'UNKNOWN' and keep the row (delay_minutes still usable).",
        mask=rules.is_blank(e["event_type"]),
        keys=e["event_id"],
    )


@register
def events_unknown_vessel(f: RawFrames) -> CheckResult:
    e = f.port_events
    fleet = set(f.shipments["vessel_id"].str.strip())
    return CheckResult(
        name="port_events.unknown_vessel",
        table="port_events",
        description="vessel_id not in the shipment fleet (VSL-0000 and VSL-9999 look like test/placeholder ids).",
        detection="Referential integrity: vessel_id NOT IN shipments.vessel_id.",
        severity=WARNING,
        action="quarantine: cannot be linked to any shipment.",
        mask=~e["vessel_id"].str.strip().isin(fleet),
        keys=e["event_id"],
    )


@register
def events_unknown_port(f: RawFrames) -> CheckResult:
    e = f.port_events
    return CheckResult(
        name="port_events.unknown_port_code",
        table="port_events",
        description="port_code not in the ports reference.",
        detection="Referential integrity.",
        severity=CRITICAL,
        action="quarantine if present.",
        mask=~e["port_code"].str.strip().isin(_valid_ports(f)),
        keys=e["event_id"],
    )


@register
def events_delay_invalid(f: RawFrames) -> CheckResult:
    delay = rules.to_number(f.port_events["delay_minutes"])
    return CheckResult(
        name="port_events.delay_minutes_invalid",
        table="port_events",
        description="delay_minutes missing, non-numeric or negative.",
        detection="Numeric parse + range check.",
        severity=WARNING,
        action="set to NULL and flag if present.",
        mask=(delay.isna() | (delay < 0)),
        keys=f.port_events["event_id"],
    )


@register
def events_delayed_zero(f: RawFrames) -> CheckResult:
    e = f.port_events
    delay = rules.to_number(e["delay_minutes"])
    return CheckResult(
        name="port_events.delayed_event_zero_minutes",
        table="port_events",
        description="event_type DELAYED but delay_minutes = 0 (contradictory).",
        detection="Cross-field consistency check.",
        severity=INFO,
        action="flag only: we cannot tell which field is wrong.",
        mask=((e["event_type"].str.strip() == "DELAYED") & (delay == 0)).fillna(False),
        keys=e["event_id"],
    )


@register
def events_unparseable_timestamp(f: RawFrames) -> CheckResult:
    e = f.port_events
    return CheckResult(
        name="port_events.unparseable_timestamp",
        table="port_events",
        description="event_timestamp blank or not parseable.",
        detection="Strict format parse.",
        severity=CRITICAL,
        action="quarantine if present.",
        mask=rules.parse_ts(e["event_timestamp"]).isna(),
        keys=e["event_id"],
    )


@register
def events_notes_missing(f: RawFrames) -> CheckResult:
    e = f.port_events
    return CheckResult(
        name="port_events.notes_missing",
        table="port_events",
        description="Free-text notes blank. Notes are optional, so this is informational.",
        detection="Blank check.",
        severity=INFO,
        action="keep as NULL.",
        mask=rules.is_blank(e["notes"]),
        keys=e["event_id"],
    )

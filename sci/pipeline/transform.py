"""Raw -> curated transformations. Pure functions (DataFrame in, DataFrame out) so every
decision is unit-testable without a database. Detection reuses the DQ check masks."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from sci.dq import checks, rules


def _flags_column(flag_masks: dict[str, pd.Series], index: pd.Index) -> pd.Series:
    """Combine named boolean masks into a per-row list of flag names. fine for 5k rows , vectorize for 5M"""
    flags = pd.Series([[] for _ in range(len(index))], index=index, dtype=object)
    for name, mask in flag_masks.items():
        aligned = mask.reindex(index, fill_value=False)
        for idx in aligned[aligned].index:  # keeps only the true values
            flags.at[idx] = flags.at[idx] + [name]
    return flags


def _quarantine(raw: pd.DataFrame, mask: pd.Series, reason: str) -> pd.DataFrame:
    out = raw.loc[mask[mask].index].copy()
    out["quarantine_reason"] = reason
    return out


# --------------------------------------------------------------------------- ports


def parse_utc_offset(tz: str) -> float | None:
    """'UTC+5:30' -> 5.5, 'UTC-8' -> -8.0."""
    if not isinstance(tz, str) or not tz.upper().startswith("UTC"):
        return None
    body = tz[3:].strip() or "+0"
    sign = -1.0 if body.startswith("-") else 1.0
    hours, _, minutes = body.lstrip("+-").partition(":")
    return sign * (int(hours) + int(minutes or 0) / 60)


def clean_ports(raw: pd.DataFrame) -> pd.DataFrame:
    p = raw.copy()
    for col in p.columns:
        p[col] = p[col].str.strip()
    missing_country = rules.is_blank(p["country"])
    p.loc[missing_country, "country"] = p.loc[missing_country, "port_code"].str[:2].map(rules.COUNTRY_BY_LOCODE_PREFIX)
    p["avg_congestion_score"] = rules.to_number(p["avg_congestion_score"])
    p["utc_offset_hours"] = p["timezone"].map(parse_utc_offset)
    p["dq_flags"] = _flags_column({"country_filled_from_locode": missing_country}, p.index)
    return p[
        [
            "port_code",
            "port_name",
            "country",
            "region",
            "timezone",
            "utc_offset_hours",
            "avg_congestion_score",
            "dq_flags",
        ]
    ].reset_index(drop=True)


# --------------------------------------------------------------------------- shipments


def resolve_duplicate_shipments(raw: pd.DataFrame) -> tuple[pd.Index, pd.Index, pd.Index, set[str]]:
    """Return (kept_index, exact_dup_index, conflict_dropped_index, resolved_ids).

    Exact copies are dropped. For ids that still repeat, keep the most complete row
    (fewest blank fields); ties go to the first row seen in the file."""
    exact = raw.duplicated(keep="first")
    remaining = raw[~exact]
    blanks = remaining.apply(rules.is_blank).sum(axis=1)
    ranked = remaining.assign(_blanks=blanks, _pos=np.arange(len(remaining))).sort_values(
        ["shipment_id", "_blanks", "_pos"]
    )  # to ensure the first row of each group is the complete one
    kept = ranked.drop_duplicates("shipment_id", keep="first").index
    conflict_dropped = remaining.index.difference(kept)  # gives the row indexes having conflicts
    resolved_ids = set(remaining.loc[conflict_dropped, "shipment_id"])
    return (
        raw.index.intersection(kept),
        exact[exact].index,
        conflict_dropped,
        resolved_ids,
    )  # intersection to keep the order determinisitic as kept was sorted


def derive_shipment_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add the curated derived fields. Null-safe: missing actuals give NULL, never 0."""
    out = df.copy()
    hours = pd.Timedelta(
        hours=1
    )  # unit constraints , timedelta/timedelta= timedelta . div by one hour gives plain hours in float
    days = pd.Timedelta(days=1)
    out["actual_delay_hours"] = (out["actual_arrival"] - out["planned_arrival"]) / hours
    out["on_time_flag"] = (out["actual_delay_hours"] <= rules.ON_TIME_THRESHOLD_HOURS).astype(
        "boolean"
    )  # nullable null not numpy nan
    out.loc[out["actual_delay_hours"].isna(), "on_time_flag"] = (
        pd.NA
    )  # NAN < 24 evaluates false in numpy , so pandas nullable
    out["route_key"] = out["origin_port"] + " → " + out["destination_port"]
    out["transit_days_planned"] = (out["planned_arrival"] - out["planned_departure"]) / days
    out["transit_days_actual"] = (out["actual_arrival"] - out["actual_departure"]) / days
    out["booking_lead_days"] = (out["planned_departure"] - out["booking_date"]) / days
    out["departure_delay_hours"] = (out["actual_departure"] - out["planned_departure"]) / hours
    return out


def clean_shipments(frames: checks.RawFrames) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (curated_shipments, quarantined_rows)."""
    raw = frames.shipments.copy()
    raw.index = pd.RangeIndex(len(raw), name="source_row")  # anchor

    kept_idx, exact_idx, conflict_idx, resolved_ids = resolve_duplicate_shipments(raw)

    def as_mask(values) -> pd.Series:
        return pd.Series(values, index=raw.index)

    quarantine = [
        _quarantine(raw, as_mask(raw.index.isin(exact_idx)), "duplicate_exact"),
        _quarantine(raw, as_mask(raw.index.isin(conflict_idx)), "duplicate_id_conflicting"),
    ]

    bad_port = checks.shipments_unknown_port(frames).mask.set_axis(raw.index)
    bad_planned = (
        checks.shipments_required_timestamp_missing(frames).mask.set_axis(raw.index)
        | checks.shipments_planned_order(frames).mask.set_axis(raw.index)
        | checks.shipments_unparseable_timestamp(frames).mask.set_axis(raw.index)
    )
    in_scope = as_mask(raw.index.isin(kept_idx))
    quarantine.append(_quarantine(raw, in_scope & bad_port, "unknown_port_code"))
    quarantine.append(_quarantine(raw, in_scope & ~bad_port & bad_planned, "invalid_planned_timeline"))

    keep = in_scope & ~bad_port & ~bad_planned
    s = raw[keep].copy()  # Deduplicated and valid ports and valid timeline

    # Detection masks, computed on raw and aligned by source row.
    def m(check) -> pd.Series:
        return check(frames).mask.set_axis(raw.index)[keep]

    flag_masks = {
        "duplicate_id_resolved": s["shipment_id"].isin(resolved_ids),
        "cargo_type_normalized": m(checks.shipments_cargo_non_canonical),
        "cargo_type_missing": m(checks.shipments_cargo_missing),
        "weight_missing": m(checks.shipments_weight_missing),
        "weight_negative_nulled": m(checks.shipments_weight_negative),
        "container_count_zero_nulled": m(checks.shipments_container_zero),
        "container_count_outlier_nulled": m(checks.shipments_container_outlier),
        "status_normalized": m(checks.shipments_status_non_canonical),
        "missing_actuals": m(checks.shipments_missing_actuals),
        "actual_departure_before_booking_nulled": m(checks.shipments_departure_before_booking),
    }

    for col in ["shipment_id", "customer_id", "origin_port", "destination_port", "vessel_id"]:
        s[col] = s[col].str.strip()
    for col in rules.SHIPMENT_TIMESTAMP_COLUMNS:
        s[col] = rules.parse_ts(s[col])

    s["cargo_type"] = s["cargo_type"].map(rules.normalize_cargo_type).fillna(rules.UNKNOWN_CARGO)
    s["status"] = s["status"].map(rules.normalize_status)

    s["weight_tons"] = rules.to_number(s["weight_tons"])
    s.loc[flag_masks["weight_negative_nulled"], "weight_tons"] = np.nan

    s["container_count"] = rules.to_number(s["container_count"])
    s.loc[
        flag_masks["container_count_zero_nulled"] | flag_masks["container_count_outlier_nulled"], "container_count"
    ] = np.nan
    s["container_count"] = s["container_count"].astype("Int64")  # castas contianer count is int not np float .

    s.loc[flag_masks["actual_departure_before_booking_nulled"], "actual_departure"] = pd.NaT

    s = derive_shipment_fields(s)
    s["dq_flags"] = _flags_column(flag_masks, s.index)
    s["is_labelled"] = s["actual_delay_hours"].notna() & s["status"].isin(["DELIVERED", "DELAYED"])  # ml gate
    s = s.reset_index()

    quarantined = pd.concat(quarantine).reset_index()
    return s, quarantined


# --------------------------------------------------------------------------- port events


def event_surrogate_key(row: pd.Series) -> str:
    """Deterministic key over the event's content (event_id alone is not unique)."""
    payload = "|".join(str(row[c]) for c in ["event_id", "event_timestamp", "port_code", "vessel_id", "event_type"])
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def clean_port_events(frames: checks.RawFrames) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = frames.port_events.copy()
    raw.index = pd.RangeIndex(len(raw), name="source_row")

    def m(check) -> pd.Series:
        return check(frames).mask.set_axis(raw.index)

    drop_reasons = {
        "duplicate_exact": m(checks.events_exact_duplicate),
        "unknown_vessel": m(checks.events_unknown_vessel),
        "unknown_port_code": m(checks.events_unknown_port),
        "unparseable_timestamp": m(checks.events_unparseable_timestamp),
    }
    quarantine, dropped = [], pd.Series(False, index=raw.index)
    for reason, mask in drop_reasons.items():
        quarantine.append(_quarantine(raw, mask & ~dropped, reason))
        dropped |= mask

    e = raw[~dropped].copy()
    flag_masks = {
        "event_id_collision": m(checks.events_duplicate_id)[~dropped],
        "event_type_missing": m(checks.events_type_missing)[~dropped],
        "delayed_event_zero_minutes": m(checks.events_delayed_zero)[~dropped],
        "delay_minutes_invalid": m(checks.events_delay_invalid)[~dropped],
    }
    for col in e.columns:
        e[col] = e[col].str.strip()
    e["event_sk"] = e.apply(event_surrogate_key, axis=1)
    e["event_timestamp"] = rules.parse_ts(e["event_timestamp"])
    e["event_type"] = e["event_type"].where(~rules.is_blank(e["event_type"]), rules.UNKNOWN_EVENT_TYPE)
    delay = rules.to_number(e["delay_minutes"])
    e["delay_minutes"] = delay.where(delay >= 0).astype("Int64")
    e["notes"] = e["notes"].where(~rules.is_blank(e["notes"]))
    e["dq_flags"] = _flags_column(flag_masks, e.index)
    e = e.reset_index()

    quarantined = pd.concat(quarantine).reset_index()
    return e, quarantined

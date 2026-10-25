"""Feature engineering for delay-risk prediction at booking time.

Leakage guard: build_features only ever reads BOOKING_TIME_INPUTS. Anything known only
after the booking (actual times, status, delay) is listed in FORBIDDEN_COLUMNS and the
function raises if a forbidden column would end up in the feature matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sci.dq.rules import ON_TIME_THRESHOLD_HOURS

BOOKING_TIME_INPUTS = [
    "origin_port",
    "destination_port",
    "cargo_type",
    "container_count",
    "weight_tons",
    "booking_date",
    "planned_departure",
    "planned_arrival",
]

FORBIDDEN_COLUMNS = {
    "actual_departure",
    "actual_arrival",
    "status",
    "actual_delay_hours",
    "on_time_flag",
    "transit_days_actual",
    "departure_delay_hours",
}

CATEGORICAL_FEATURES = ["origin_port", "destination_port", "origin_region", "destination_region", "cargo_type"]
NUMERIC_FEATURES = [
    "origin_congestion",
    "destination_congestion",
    "booking_lead_days",
    "planned_transit_days",
    "departure_month_sin",
    "departure_month_cos",
    "departure_dow",
    "container_count",
    "log_weight_tons",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def build_features(df: pd.DataFrame, ports: pd.DataFrame) -> pd.DataFrame:
    """Map booking-time inputs (+ static port reference data) to the model feature matrix."""
    missing = [c for c in BOOKING_TIME_INPUTS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing booking-time inputs: {missing}")

    x = df[BOOKING_TIME_INPUTS].copy()  # only booking-time columns from here on
    for col in ("booking_date", "planned_departure", "planned_arrival"):
        x[col] = pd.to_datetime(x[col])

    ref = ports.set_index("port_code")
    out = pd.DataFrame(index=x.index)
    out["origin_port"] = x["origin_port"]
    out["destination_port"] = x["destination_port"]
    out["origin_region"] = x["origin_port"].map(ref["region"])
    out["destination_region"] = x["destination_port"].map(ref["region"])
    out["cargo_type"] = x["cargo_type"]
    out["origin_congestion"] = x["origin_port"].map(ref["avg_congestion_score"])
    out["destination_congestion"] = x["destination_port"].map(ref["avg_congestion_score"])
    out["booking_lead_days"] = (x["planned_departure"] - x["booking_date"]) / pd.Timedelta(days=1)
    out["planned_transit_days"] = (x["planned_arrival"] - x["planned_departure"]) / pd.Timedelta(days=1)
    month = x["planned_departure"].dt.month
    out["departure_month_sin"] = np.sin(2 * np.pi * month / 12)
    out["departure_month_cos"] = np.cos(2 * np.pi * month / 12)
    out["departure_dow"] = x["planned_departure"].dt.dayofweek
    out["container_count"] = pd.to_numeric(x["container_count"], errors="coerce").astype(float)
    weight = pd.to_numeric(x["weight_tons"], errors="coerce")
    out["log_weight_tons"] = np.log1p(weight.where(weight >= 0))

    leaked = FORBIDDEN_COLUMNS & set(out.columns)
    if leaked:
        raise AssertionError(f"Post-booking columns leaked into features: {leaked}")
    return out[FEATURE_COLUMNS]


def build_label(df: pd.DataFrame) -> pd.Series:
    """1 if the shipment arrived more than 24h after the planned arrival."""
    return (df["actual_delay_hours"] > ON_TIME_THRESHOLD_HOURS).astype(int)

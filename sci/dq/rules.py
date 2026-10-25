"""Reference rules for data quality. Shared by the DQ checks and the pipeline so that
detection and remediation use exactly one definition of each rule."""

from __future__ import annotations

import pandas as pd

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

SHIPMENT_TIMESTAMP_COLUMNS = [
    "booking_date",
    "planned_departure",
    "actual_departure",
    "planned_arrival",
    "actual_arrival",
]

CANONICAL_CARGO_TYPES = {
    "Apparel",
    "Automotive Parts",
    "Chemicals",
    "Consumer Goods",
    "Electronics",
    "Food & Beverage",
    "Furniture",
    "Machinery",
    "Pharmaceuticals",
    "Raw Materials",
}
UNKNOWN_CARGO = "Unknown"

CANONICAL_STATUSES = {"DELIVERED", "DELAYED", "CANCELLED"}
# Raw variants seen in the feed, mapped to the canonical value.
STATUS_ALIASES = {"COMPLETED": "DELIVERED", "COMPLETE": "DELIVERED"}
UNKNOWN_STATUS = "UNKNOWN"
NULL_TOKENS = {"", "N/A", "NA", "NULL", "NONE"}

UNKNOWN_EVENT_TYPE = "UNKNOWN"

# Reference-data correction for a missing country. Sourced from the UN/LOCODE prefix
# (BE = Belgium); applied only when the country is blank.
COUNTRY_BY_LOCODE_PREFIX = {
    "SG": "Singapore", "CN": "China", "HK": "Hong Kong", "KR": "South Korea", "JP": "Japan",
    "IN": "India", "AE": "UAE", "NL": "Netherlands", "DE": "Germany", "BE": "Belgium",
    "GB": "UK", "ES": "Spain", "US": "USA", "BR": "Brazil", "ZA": "South Africa",
    "MY": "Malaysia", "TH": "Thailand", "PK": "Pakistan", "EG": "Egypt",
}  # fmt: skip

ON_TIME_THRESHOLD_HOURS = 24.0
# Container-count outlier fence: Q3 + k * IQR on positive counts (robust, data-driven).# extreme outlier
OUTLIER_IQR_K = 3.0


def is_blank(series: pd.Series) -> pd.Series:
    """True where a raw string cell is empty or a null token like 'N/A'."""
    return series.fillna("").astype(str).str.strip().str.upper().isin(NULL_TOKENS)


def parse_ts(series: pd.Series) -> pd.Series:
    """Parse raw timestamp strings; blanks become NaT, malformed values become NaT too."""
    cleaned = series.where(~is_blank(series))
    return pd.to_datetime(cleaned, format=TIMESTAMP_FORMAT, errors="coerce")


def to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.where(~is_blank(series)), errors="coerce")


def normalize_cargo_type(value: object) -> str | None:
    """Map raw cargo text to a canonical value. Returns None if blank, the raw stripped
    text if it matches nothing canonical (so it stays visible rather than silently guessed)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = " ".join(str(value).split())
    if text.upper() in NULL_TOKENS:
        return None
    by_lower = {c.lower(): c for c in CANONICAL_CARGO_TYPES}
    return by_lower.get(text.lower(), text)


def normalize_status(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return UNKNOWN_STATUS
    text = str(value).strip().upper()
    if text in NULL_TOKENS:
        return UNKNOWN_STATUS
    text = STATUS_ALIASES.get(text, text)
    return text if text in CANONICAL_STATUSES else UNKNOWN_STATUS


def iqr_upper_fence(values: pd.Series, k: float = OUTLIER_IQR_K) -> float:
    positive = values[values > 0]
    q1, q3 = positive.quantile([0.25, 0.75])
    return float(q3 + k * (q3 - q1))

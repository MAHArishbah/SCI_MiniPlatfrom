"""Read-only data access over the serving layer. All SQL lives here; parameters are always
bound (never string-formatted from user input)."""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import duckdb
import pandas as pd

SHIPMENT_COLUMNS = """
    shipment_id, customer_id, origin_port, origin_port_name, destination_port, destination_port_name,
    route_key, vessel_id, booking_date, planned_departure, actual_departure, planned_arrival,
    actual_arrival, container_count, cargo_type, weight_tons, status, actual_delay_hours,
    on_time_flag, transit_days_planned, transit_days_actual, booking_lead_days, dq_flags
"""


def _clean(value: Any) -> Any:
    """Convert pandas/numpy NA and timestamps into JSON-friendly Python values."""
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if hasattr(value, "tolist") and not isinstance(value, str):
        return value.tolist()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: _clean(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


class Repository:
    def __init__(self, con: duckdb.DuckDBPyConnection):
        self._con = con

    def _query(self, sql: str, params: list | None = None) -> pd.DataFrame:
        # A cursor per call: DuckDB connections are not safe to share across threads,
        # cursors are.
        cur = self._con.cursor()
        try:
            return cur.execute(sql, params or []).df()
        finally:
            cur.close()

    def ping(self) -> bool:
        return self._query("SELECT 1 AS ok")["ok"].iloc[0] == 1

    def data_summary(self) -> dict:
        df = self._query(
            """
            SELECT COUNT(*) AS shipments,
                   MIN(planned_departure) AS first_planned_departure,
                   MAX(planned_departure) AS last_planned_departure,
                   MAX(actual_arrival)    AS last_actual_arrival
            FROM serving.shipments
            """
        )
        return {k: (str(v) if isinstance(v, pd.Timestamp) else _clean(v)) for k, v in df.iloc[0].items()}

    def ports(self) -> pd.DataFrame:
        return self._query("SELECT * FROM curated.ports ORDER BY port_code")

    def get_shipment(self, shipment_id: str) -> dict | None:
        df = self._query(f"SELECT {SHIPMENT_COLUMNS} FROM serving.shipments WHERE shipment_id = ?", [shipment_id])
        return _records(df)[0] if len(df) else None

    def list_shipments(
        self,
        origin: str | None,
        destination: str | None,
        status: str | None,
        date_from: date | None,
        date_to: date | None,
        page: int,
        page_size: int,
    ) -> tuple[list[dict], int]:
        where, params = ["1=1"], []
        if origin:
            where.append("origin_port = ?")
            params.append(origin)
        if destination:
            where.append("destination_port = ?")
            params.append(destination)
        if status:
            where.append("status = ?")
            params.append(status)
        if date_from:
            where.append("CAST(planned_departure AS DATE) >= ?")
            params.append(date_from)
        if date_to:
            where.append("CAST(planned_departure AS DATE) <= ?")
            params.append(date_to)
        clause = " AND ".join(where)
        total = int(self._query(f"SELECT COUNT(*) AS n FROM serving.shipments WHERE {clause}", params)["n"].iloc[0])
        df = self._query(
            f"""SELECT {SHIPMENT_COLUMNS} FROM serving.shipments WHERE {clause}
                ORDER BY planned_departure DESC, shipment_id LIMIT ? OFFSET ?""",
            params + [page_size, (page - 1) * page_size],
        )
        return _records(df), total

    def route_stats(self, origin: str, destination: str, date_from: date | None, date_to: date | None) -> dict:
        where, params = ["origin_port = ?", "destination_port = ?"], [origin, destination]
        if date_from:
            where.append("departure_date >= ?")
            params.append(date_from)
        if date_to:
            where.append("departure_date <= ?")
            params.append(date_to)
        df = self._query(
            f"""
            SELECT COALESCE(SUM(shipment_count), 0)  AS shipment_count,
                   COALESCE(SUM(completed_count), 0) AS completed_count,
                   COALESCE(SUM(cancelled_count), 0) AS cancelled_count,
                   SUM(total_delay_hours) / NULLIF(SUM(completed_count), 0) AS avg_delay_hours,
                   SUM(on_time_count)::DOUBLE / NULLIF(SUM(completed_count), 0) AS on_time_rate
            FROM serving.route_daily_stats WHERE {' AND '.join(where)}
            """,
            params,
        )
        row = {k: _clean(v) for k, v in df.iloc[0].items()}
        for key in ("avg_delay_hours", "on_time_rate"):
            if row[key] is not None:
                row[key] = round(float(row[key]), 4)
        return {k: int(v) if k.endswith("_count") else v for k, v in row.items()}

    def top_routes(
        self, date_from: date | None, date_to: date | None, order_by: str, limit: int, min_completed: int
    ) -> list[dict]:
        if order_by not in {"avg_delay_hours", "on_time_rate"}:
            raise ValueError("unsupported order_by")
        direction = "DESC" if order_by == "avg_delay_hours" else "ASC"  # worst routes first
        where, params = ["1=1"], []
        if date_from:
            where.append("departure_date >= ?")
            params.append(date_from)
        if date_to:
            where.append("departure_date <= ?")
            params.append(date_to)
        df = self._query(
            f"""
            SELECT origin_port, destination_port, route_key,
                   SUM(shipment_count)::BIGINT AS shipment_count,
                   SUM(completed_count)::BIGINT AS completed_count,
                   ROUND(SUM(total_delay_hours) / NULLIF(SUM(completed_count), 0), 2) AS avg_delay_hours,
                   ROUND(SUM(on_time_count)::DOUBLE / NULLIF(SUM(completed_count), 0), 4) AS on_time_rate
            FROM serving.route_daily_stats WHERE {' AND '.join(where)}
            GROUP BY origin_port, destination_port, route_key
            HAVING SUM(completed_count) >= ?
            ORDER BY {order_by} {direction}, route_key
            LIMIT ?
            """,
            params + [min_completed, limit],
        )
        return _records(df)

    def booking_inputs(self, shipment_id: str) -> dict | None:
        df = self._query(
            """SELECT shipment_id, origin_port, destination_port, cargo_type, container_count, weight_tons,
                      booking_date, planned_departure, planned_arrival
               FROM curated.shipments WHERE shipment_id = ?""",
            [shipment_id],
        )
        return _records(df)[0] if len(df) else None

"""Build the DuckDB data product: raw -> curated (+ quarantine) -> serving.

Idempotent by construction: every run rebuilds all tables from the raw files into a
temporary database file, then atomically swaps it into place with os.replace. Running
it twice gives identical tables, and readers never see a half-built database.

    python -m sci.pipeline.build --input data/raw --db warehouse/sci.duckdb
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd

from sci.dq.runner import load_raw
from sci.pipeline import transform

logger = logging.getLogger("sci.pipeline")

SERVING_SQL = """
CREATE OR REPLACE VIEW serving.shipments AS
SELECT
    s.shipment_id, s.customer_id, s.origin_port, o.port_name AS origin_port_name,
    s.destination_port, d.port_name AS destination_port_name, s.route_key, s.vessel_id,
    s.booking_date, s.planned_departure, s.actual_departure, s.planned_arrival, s.actual_arrival,
    s.container_count, s.cargo_type, s.weight_tons, s.status,
    s.actual_delay_hours, s.on_time_flag, s.transit_days_planned, s.transit_days_actual,
    s.booking_lead_days, s.dq_flags
FROM curated.shipments s
JOIN curated.ports o ON o.port_code = s.origin_port
JOIN curated.ports d ON d.port_code = s.destination_port;

-- Pre-aggregated route x day grain. The API aggregates any date range from this table,
-- so query cost scales with routes x days rather than with raw shipment volume.
CREATE OR REPLACE TABLE serving.route_daily_stats AS
SELECT
    origin_port, destination_port, route_key,
    CAST(planned_departure AS DATE)                         AS departure_date,
    COUNT(*)                                                AS shipment_count,
    COUNT(*) FILTER (WHERE status = 'CANCELLED')            AS cancelled_count,
    COUNT(actual_delay_hours)                               AS completed_count,
    COALESCE(SUM(actual_delay_hours), 0)                    AS total_delay_hours,
    COUNT(*) FILTER (WHERE on_time_flag)                    AS on_time_count
FROM curated.shipments
GROUP BY ALL;
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame) -> None:
    con.register("_df", df)
    con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM _df")
    con.unregister("_df")


def build(input_dir: Path, db_path: Path) -> dict:
    started = datetime.now(UTC)
    frames, paths = load_raw(input_dir)

    ports = transform.clean_ports(frames.ports)
    shipments, shipments_q = transform.clean_shipments(frames)
    events, events_q = transform.clean_port_events(frames)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = db_path.with_name(f".{db_path.name}.{uuid.uuid4().hex[:8]}.tmp")
    con = duckdb.connect(str(tmp_path))
    try:
        for schema in ("raw", "curated", "quarantine", "serving", "meta"):
            con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")

        ingested_at = started.isoformat()
        for name in ("ports", "shipments", "port_events"):
            raw_df = getattr(frames, name).copy()
            raw_df.insert(0, "_source_row", range(len(raw_df)))
            raw_df["_source_file"] = paths[name].name
            raw_df["_ingested_at"] = ingested_at
            _write(con, f"raw.{name}", raw_df)

        _write(con, "curated.ports", ports)
        _write(con, "curated.shipments", shipments)
        _write(con, "curated.port_events", events)
        _write(con, "quarantine.shipments", shipments_q)
        _write(con, "quarantine.port_events", events_q)
        con.execute(SERVING_SQL)

        counts = {
            t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in (
                "raw.ports", "raw.shipments", "raw.port_events",
                "curated.ports", "curated.shipments", "curated.port_events",
                "quarantine.shipments", "quarantine.port_events", "serving.route_daily_stats",
            )
        }  # fmt: skip
        run = {
            "run_id": uuid.uuid4().hex,
            "started_at": ingested_at,
            "finished_at": datetime.now(UTC).isoformat(),
            "input_sha256": {n: _sha256(p) for n, p in paths.items()},
            "row_counts": counts,
        }
        _write(con, "meta.pipeline_runs", pd.DataFrame([{**run, "input_sha256": str(run["input_sha256"]),
                                                         "row_counts": str(counts)}]))  # fmt: skip
        con.execute("CHECKPOINT")
    finally:
        con.close()

    os.replace(tmp_path, db_path)  # atomic swap
    logger.info(
        "pipeline_complete", extra={"db_path": str(db_path), **{k.replace(".", "_"): v for k, v in counts.items()}}
    )
    return run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the DuckDB data product from raw CSVs.")
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--db", type=Path, default=Path("warehouse/sci.duckdb"))
    args = parser.parse_args(argv)

    from sci.logging_config import configure_logging

    configure_logging()
    build(args.input, args.db)
    return 0


if __name__ == "__main__":
    sys.exit(main())

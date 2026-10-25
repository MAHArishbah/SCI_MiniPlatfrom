import duckdb
import pandas as pd

from sci.pipeline.build import build

TABLES = [
    "curated.shipments",
    "curated.port_events",
    "curated.ports",
    "quarantine.shipments",
    "serving.route_daily_stats",
]


def _snapshot(db_path):
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return {t: con.execute(f"SELECT * FROM {t} ORDER BY ALL").df() for t in TABLES}
    finally:
        con.close()


def test_pipeline_is_idempotent(raw_dir, tmp_path):
    db = tmp_path / "sci.duckdb"
    build(raw_dir, db)
    first = _snapshot(db)
    build(raw_dir, db)
    second = _snapshot(db)
    for table in TABLES:
        pd.testing.assert_frame_equal(
            first[table].astype(str), second[table].astype(str), obj=table
        )  # same rows, no duplicates appended
    assert [p.name for p in tmp_path.iterdir()] == ["sci.duckdb"]  # temp build file swapped away


def test_layers_and_row_counts(settings):
    con = duckdb.connect(str(settings.db_path), read_only=True)
    try:
        count = lambda t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]  # noqa: E731
        assert count("raw.shipments") == 12  # raw keeps everything, as delivered
        assert count("curated.shipments") + count("quarantine.shipments") == 12
        assert count("serving.shipments") == count("curated.shipments")
        stats = con.execute(
            "SELECT SUM(shipment_count), SUM(completed_count) FROM serving.route_daily_stats"
        ).fetchone()
        assert stats == (9, 6)
    finally:
        con.close()

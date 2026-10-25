"""In-process API tests (FastAPI TestClient) against a DuckDB built from the fixture CSVs.
The model is stubbed, so these run in milliseconds with no infrastructure."""

import pytest

BOOKING = {
    "origin_port": "cnsha",
    "destination_port": "NLRTM",
    "cargo_type": "electronics",
    "container_count": 5,
    "booking_date": "2025-03-01T00:00:00",
    "planned_departure": "2025-03-08T00:00:00",
    "planned_arrival": "2025-04-08T00:00:00",
}


def assert_error(resp, status, code):
    assert resp.status_code == status
    body = resp.json()["error"]
    assert body["code"] == code and body["message"] and body["request_id"]


def test_health_reports_components(client):
    resp = client.get("/health")
    # The test env has no DQ-independent model file; the stub predictor counts as loaded.
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok" and body["version"]
    assert body["checks"] == {"database": "ok", "model": "ok", "dq_report": "ok"}
    assert body["data"]["shipments"] == 9


def test_get_shipment(client):
    resp = client.get("/shipments/SHP-00001")
    assert resp.status_code == 200
    body = resp.json()
    assert body["route_key"] == "CNSHA → NLRTM"
    assert body["actual_delay_hours"] == 10.0 and body["on_time_flag"] is True
    assert body["origin_port_name"] == "Shanghai"


def test_get_shipment_is_case_insensitive(client):
    assert client.get("/shipments/shp-00001").status_code == 200


def test_unknown_shipment_returns_structured_404(client):
    assert_error(client.get("/shipments/SHP-99999"), 404, "not_found")


def test_quarantined_shipment_is_not_served(client):
    assert client.get("/shipments/SHP-00005").status_code == 404


def test_list_filters_and_pagination(client):
    resp = client.get("/shipments", params={"origin": "CNSHA", "page_size": 1, "page": 2})
    body = resp.json()
    assert resp.status_code == 200
    assert body["total"] == 3 and body["total_pages"] == 3 and len(body["items"]) == 1
    assert all(i["origin_port"] == "CNSHA" for i in body["items"])


def test_list_status_and_date_filters(client):
    body = client.get("/shipments", params={"status": "DELAYED"}).json()
    assert {i["shipment_id"] for i in body["items"]} == {"SHP-00002", "SHP-00011"}
    body = client.get("/shipments", params={"date_from": "2025-01-15", "date_to": "2025-01-16"}).json()
    assert {i["shipment_id"] for i in body["items"]} == {"SHP-00009", "SHP-00010"}


@pytest.mark.parametrize(
    "params,status,code",
    [
        ({"date_from": "2025-02-01", "date_to": "2025-01-01"}, 400, "bad_request"),
        ({"status": "LOST"}, 422, "validation_error"),
        ({"page_size": 1000}, 422, "validation_error"),
        ({"origin": "XXTST"}, 404, "not_found"),
    ],
)
def test_list_rejects_bad_input(client, params, status, code):
    assert_error(client.get("/shipments", params=params), status, code)


def test_route_stats(client):
    # SHP-00001 (10h late) + SHP-00002 (48h late; its exact duplicate was dropped, not double-counted)
    body = client.get("/routes/CNSHA/NLRTM/stats").json()
    assert body["shipment_count"] == 2 and body["completed_count"] == 2
    assert body["avg_delay_hours"] == 29.0 and body["on_time_rate"] == 0.5


def test_route_stats_unknown_port(client):
    assert_error(client.get("/routes/CNSHA/ZZZZZ/stats"), 404, "not_found")


def test_route_stats_empty_window_is_200_with_zero(client):
    body = client.get("/routes/CNSHA/NLRTM/stats", params={"date_from": "2030-01-01"}).json()
    assert body["shipment_count"] == 0 and body["avg_delay_hours"] is None


def test_top_routes(client):
    body = client.get("/routes/top", params={"min_completed": 1, "limit": 2}).json()
    assert body["routes"][0]["route_key"] == "BEANR → CNSHA"  # 72h late, the worst
    assert len(body["routes"]) == 2


def test_predict_by_shipment_id_uses_stored_booking_fields(client, stub_predictor):
    resp = client.post("/predict-delay", json={"shipment_id": "SHP-00001"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["delay_probability"] == 0.42 and body["model_version"] == "stub-1"
    assert body["model_test_roc_auc"] == 0.51
    sent = stub_predictor.calls[-1]
    assert sent["origin_port"] == "CNSHA" and "actual_arrival" not in sent


def test_predict_with_booking_fields_normalizes_input(client, stub_predictor):
    assert client.post("/predict-delay", json=BOOKING).status_code == 200
    sent = stub_predictor.calls[-1]
    assert sent["origin_port"] == "CNSHA" and sent["cargo_type"] == "Electronics"


@pytest.mark.parametrize(
    "body,status",
    [
        ({"origin_port": "CNSHA"}, 422),  # incomplete booking
        ({**BOOKING, "planned_arrival": "2025-03-01T00:00:00"}, 422),  # arrival before departure
        ({**BOOKING, "cargo_type": "Spaceships"}, 422),
        ({**BOOKING, "origin_port": "XXTST"}, 422),
        ({"shipment_id": "SHP-99999"}, 404),
    ],
)
def test_predict_rejects_bad_input(client, body, status):
    assert client.post("/predict-delay", json=body).status_code == status


def test_dq_report_endpoint(client):
    body = client.get("/data-quality/report").json()
    assert body["summary"]["checks_failed"] == 18
    assert {"check", "rows_affected", "severity", "recommended_action"} <= set(body["checks"][0])


def test_request_id_is_echoed(client):
    resp = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert resp.headers["X-Request-ID"] == "abc123"


def test_request_is_logged_as_json(client, caplog):
    caplog.set_level("INFO", logger="sci.api")
    client.get("/shipments/SHP-00001")
    record = next(r for r in caplog.records if r.getMessage() == "http_request")
    assert record.method == "GET" and record.path == "/shipments/SHP-00001"
    assert record.status_code == 200 and record.latency_ms >= 0


def test_service_degrades_cleanly_without_database(settings, tmp_path):
    from dataclasses import replace

    from fastapi.testclient import TestClient

    from sci.api.main import create_app

    broken = replace(settings, db_path=tmp_path / "missing.duckdb", model_path=tmp_path / "missing.joblib")
    with TestClient(create_app(broken)) as c:
        health = c.get("/health")
        assert health.status_code == 503 and health.json()["status"] == "degraded"
        assert_error(c.get("/shipments/SHP-00001"), 503, "service_unavailable")

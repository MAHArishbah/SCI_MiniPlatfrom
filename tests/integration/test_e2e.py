"""End-to-end tests against a RUNNING service (real HTTP, real DuckDB, real model).

docker compose up -d --build --wait
SCI_BASE_URL=http://localhost:8000 pytest tests/integration -m integration
"""

import os
import time

import httpx
import pytest

pytestmark = pytest.mark.integration
BASE_URL = os.getenv("SCI_BASE_URL", "http://localhost:8000")


@pytest.fixture(scope="session")
def api():
    client = httpx.Client(base_url=BASE_URL, timeout=10.0)
    deadline = time.time() + 90  # the container builds the warehouse on start-up
    while True:
        try:
            if client.get("/health").status_code == 200:
                break
        except httpx.HTTPError:
            pass
        if time.time() > deadline:
            pytest.fail(f"Service at {BASE_URL} did not become healthy within 90s")
        time.sleep(2)
    yield client
    client.close()


def test_health(api):
    body = api.get("/health").json()
    assert body["status"] == "ok"
    assert body["checks"] == {"database": "ok", "model": "ok", "dq_report": "ok"}
    assert body["data"]["shipments"] > 4000


def test_shipment_lookup(api):
    resp = api.get("/shipments/SHP-00421")
    assert resp.status_code == 200
    body = resp.json()
    assert body["shipment_id"] == "SHP-00421"
    assert body["route_key"] == f"{body['origin_port']} → {body['destination_port']}"
    assert resp.headers["X-Request-ID"]


def test_unknown_shipment_is_404(api):
    resp = api.get("/shipments/SHP-99999")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


def test_list_with_filters_and_pagination(api):
    body = api.get(
        "/shipments", params={"origin": "CNSHA", "status": "DELAYED", "date_from": "2024-06-01", "page_size": 5}
    ).json()
    assert body["page_size"] == 5 and len(body["items"]) <= 5 and body["total"] >= len(body["items"])
    assert all(i["origin_port"] == "CNSHA" and i["status"] == "DELAYED" for i in body["items"])


def test_route_stats(api):
    body = api.get("/routes/CNSHA/NLRTM/stats").json()
    assert body["shipment_count"] >= body["completed_count"] > 0
    assert 0 <= body["on_time_rate"] <= 1


def test_dq_report(api):
    body = api.get("/data-quality/report").json()
    assert body["summary"]["checks_failed"] >= 12
    failed = {c["check"] for c in body["checks"] if c["status"] == "fail"}
    assert {"shipments.duplicate_id_conflicting", "shipments.unknown_port_code"} <= failed


def test_predict_delay_by_id(api):
    resp = api.post("/predict-delay", json={"shipment_id": "SHP-00421"})
    assert resp.status_code == 200
    body = resp.json()
    assert 0 <= body["delay_probability"] <= 1
    assert body["risk_band"] in {"low", "medium", "high"} and body["model_version"]


def test_predict_delay_new_booking(api):
    resp = api.post(
        "/predict-delay",
        json={
            "origin_port": "CNSHA",
            "destination_port": "NLRTM",
            "cargo_type": "Electronics",
            "container_count": 12,
            "weight_tons": 300,
            "booking_date": "2025-11-01T00:00:00",
            "planned_departure": "2025-11-08T00:00:00",
            "planned_arrival": "2025-12-10T00:00:00",
        },
    )
    assert resp.status_code == 200


def test_predict_delay_validation_error(api):
    resp = api.post("/predict-delay", json={"origin_port": "CNSHA"})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

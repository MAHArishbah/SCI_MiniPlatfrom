"""Tools the assistant can call. Every tool goes through the HTTP API (the same contract
external clients use), so the assistant never touches the database directly.

Guardrail: arguments are validated against allow-lists (known port codes, shipment-id
pattern, ISO dates, capped limits) BEFORE any request is made. Invalid arguments come back
to the model as a tool error it can recover from."""

from __future__ import annotations

import re
import time
from datetime import date

import httpx

SHIPMENT_ID = re.compile(r"^SHP-\d{5}$")
MAX_LIST_LIMIT = 20

TOOL_SCHEMAS = [
    {
        "name": "query_shipments",
        "description": (
            "List shipments from the curated data product, newest planned departure first. "
            "Use for questions about specific shipments or samples of shipments. Returns the total "
            "match count plus up to `limit` rows."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "Origin port code, e.g. CNSHA"},
                "destination": {"type": "string", "description": "Destination port code, e.g. NLRTM"},
                "status": {"type": "string", "enum": ["DELIVERED", "DELAYED", "CANCELLED", "UNKNOWN"]},
                "date_from": {"type": "string", "description": "planned_departure on/after, YYYY-MM-DD"},
                "date_to": {"type": "string", "description": "planned_departure on/before, YYYY-MM-DD"},
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LIST_LIMIT},
            },
        },
    },
    {
        "name": "get_route_stats",
        "description": (
            "Aggregated delay metrics for one origin->destination route: shipment count, completed "
            "count, average delay hours and on-time rate (delay <= 24h). Optional date window on planned departure."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "Origin port code"},
                "destination": {"type": "string", "description": "Destination port code"},
                "date_from": {"type": "string", "description": "YYYY-MM-DD"},
                "date_to": {"type": "string", "description": "YYYY-MM-DD"},
            },
            "required": ["origin", "destination"],
        },
    },
    {
        "name": "get_top_routes",
        "description": (
            "Rank routes by average delay (worst first) or by on-time rate (worst first) within an "
            "optional planned-departure window. Routes with fewer than `min_completed` completed "
            "shipments are excluded so tiny samples do not dominate."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_by": {"type": "string", "enum": ["avg_delay_hours", "on_time_rate"]},
                "date_from": {"type": "string", "description": "YYYY-MM-DD"},
                "date_to": {"type": "string", "description": "YYYY-MM-DD"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                "min_completed": {"type": "integer", "minimum": 1},
            },
        },
    },
    {
        "name": "predict_delay",
        "description": (
            "Delay-risk prediction from the ML model for a known shipment (probability of arriving "
            "more than 24h late, based only on booking-time information). The response includes the "
            "model's held-out AUC; report it so the user can judge reliability."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"shipment_id": {"type": "string", "description": "e.g. SHP-00421"}},
            "required": ["shipment_id"],
        },
    },
]

TOOL_NAMES = {t["name"] for t in TOOL_SCHEMAS}


class ToolArgumentError(ValueError):
    pass


def _port(value: object, valid_ports: set[str], field: str) -> str:
    code = str(value or "").strip().upper()
    if code not in valid_ports:
        raise ToolArgumentError(f"{field} '{value}' is not a known port code")
    return code


def _iso_date(value: object, field: str) -> str:
    try:
        return date.fromisoformat(str(value)).isoformat()
    except ValueError as exc:
        raise ToolArgumentError(f"{field} must be YYYY-MM-DD, got '{value}'") from exc


def _bounded_int(value: object, field: str, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ToolArgumentError(f"{field} must be an integer") from exc
    return max(low, min(high, number))


def validate_args(name: str, args: dict, valid_ports: set[str]) -> dict:
    """Return sanitized query/body params for the API, or raise ToolArgumentError."""
    if name not in TOOL_NAMES:
        raise ToolArgumentError(f"Unknown tool '{name}'")
    clean: dict = {}
    for key in ("date_from", "date_to"):
        if args.get(key):
            clean[key] = _iso_date(args[key], key)
    if name == "query_shipments":
        for key in ("origin", "destination"):
            if args.get(key):
                clean[key] = _port(args[key], valid_ports, key)
        if args.get("status"):
            clean["status"] = str(args["status"]).upper()
        clean["page_size"] = _bounded_int(args.get("limit", 10), "limit", 1, MAX_LIST_LIMIT)
    elif name == "get_route_stats":
        clean["origin"] = _port(args.get("origin"), valid_ports, "origin")
        clean["destination"] = _port(args.get("destination"), valid_ports, "destination")
    elif name == "get_top_routes":
        clean["order_by"] = args.get("order_by", "avg_delay_hours")
        if clean["order_by"] not in {"avg_delay_hours", "on_time_rate"}:
            raise ToolArgumentError("order_by must be avg_delay_hours or on_time_rate")
        clean["limit"] = _bounded_int(args.get("limit", 5), "limit", 1, 10)
        clean["min_completed"] = _bounded_int(args.get("min_completed", 10), "min_completed", 1, 1000)
    elif name == "predict_delay":
        shipment_id = str(args.get("shipment_id", "")).strip().upper()
        if not SHIPMENT_ID.match(shipment_id):
            raise ToolArgumentError(f"shipment_id must look like SHP-00421, got '{args.get('shipment_id')}'")
        clean["shipment_id"] = shipment_id
    return clean


_COMPACT_FIELDS = (
    "shipment_id", "route_key", "status", "cargo_type", "planned_departure",
    "planned_arrival", "actual_arrival", "actual_delay_hours", "on_time_flag",
)  # fmt: skip


class ToolExecutor:
    def __init__(self, api_base_url: str, valid_ports: set[str], client: httpx.Client | None = None):
        self.client = client or httpx.Client(base_url=api_base_url, timeout=10.0)
        self.valid_ports = valid_ports

    def execute(self, name: str, args: dict) -> tuple[dict, bool, float]:
        """Run a tool. Returns (result, is_error, latency_ms). Never raises."""
        start = time.perf_counter()
        try:
            params = validate_args(name, args, self.valid_ports)
            result = self._call(name, params)
            return result, False, round((time.perf_counter() - start) * 1000, 1)
        except ToolArgumentError as exc:
            return {"error": f"invalid arguments: {exc}"}, True, round((time.perf_counter() - start) * 1000, 1)
        except httpx.HTTPStatusError as exc:
            try:
                detail = exc.response.json().get("error", {}).get("message")
            except ValueError:
                detail = exc.response.text[:200]
            return (
                {"error": detail or "API error", "status_code": exc.response.status_code},
                True,
                round((time.perf_counter() - start) * 1000, 1),
            )
        except httpx.HTTPError as exc:
            return {"error": f"API unreachable: {exc}"}, True, round((time.perf_counter() - start) * 1000, 1)

    def _call(self, name: str, params: dict) -> dict:
        if name == "query_shipments":
            resp = self.client.get("/shipments", params=params)
            resp.raise_for_status()
            body = resp.json()
            return {
                "total_matching": body["total"],
                "returned": len(body["items"]),
                "items": [{k: item.get(k) for k in _COMPACT_FIELDS} for item in body["items"]],
            }
        if name == "get_route_stats":
            origin, destination = params.pop("origin"), params.pop("destination")
            resp = self.client.get(f"/routes/{origin}/{destination}/stats", params=params)
        elif name == "get_top_routes":
            resp = self.client.get("/routes/top", params=params)
        else:  # predict_delay
            resp = self.client.post("/predict-delay", json={"shipment_id": params["shipment_id"]})
        resp.raise_for_status()
        return resp.json()

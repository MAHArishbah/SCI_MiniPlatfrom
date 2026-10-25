"""HTTP API for the supply-chain intelligence platform.

uvicorn sci.api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from contextlib import asynccontextmanager
from datetime import date
from enum import Enum

import duckdb
from fastapi import Depends, FastAPI, Path, Query, Request
from fastapi.responses import JSONResponse

from sci.api.errors import (
    BadRequestError,
    NotFoundError,
    ServiceUnavailableError,
    UnprocessableError,
    install_error_handlers,
)
from sci.api.repository import Repository
from sci.api.schemas import (
    HealthResponse,
    Port,
    PredictRequest,
    PredictResponse,
    RouteStats,
    Shipment,
    ShipmentPage,
)
from sci.config import Settings, get_settings
from sci.logging_config import configure_logging
from sci.ml.predict import DelayPredictor

logger = logging.getLogger("sci.api")


class StatusFilter(str, Enum):
    DELIVERED = "DELIVERED"
    DELAYED = "DELAYED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


class RouteMetric(str, Enum):
    avg_delay_hours = "avg_delay_hours"
    on_time_rate = "on_time_rate"


def _load_resources(app: FastAPI, settings: Settings) -> None:
    app.state.repo, app.state.ports, app.state.predictor = None, None, None
    if settings.db_path.exists():
        con = duckdb.connect(str(settings.db_path), read_only=True)
        app.state.con = con
        app.state.repo = Repository(con)
        app.state.ports = app.state.repo.ports()
    else:
        logger.error("database_missing", extra={"db_path": str(settings.db_path)})
    if app.state.predictor_override is not None:
        app.state.predictor = app.state.predictor_override
    elif settings.model_path.exists() and settings.model_card_path.exists():
        app.state.predictor = DelayPredictor.load(settings.model_path, settings.model_card_path)
    else:
        logger.error("model_missing", extra={"model_path": str(settings.model_path)})


def create_app(settings: Settings | None = None, predictor: DelayPredictor | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _load_resources(app, settings)
        logger.info("startup", extra={"version": settings.version, "git_sha": settings.git_sha})
        yield
        if getattr(app.state, "con", None) is not None:
            app.state.con.close()

    app = FastAPI(title="Supply Chain Intelligence API", version=settings.version, lifespan=lifespan)
    app.state.settings = settings
    app.state.predictor_override = predictor
    install_error_handlers(app)

    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            logger.info(
                "http_request",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                },
            )

    def repo(request: Request) -> Repository:
        if request.app.state.repo is None:
            raise ServiceUnavailableError("Data store not available; run the pipeline first")
        return request.app.state.repo

    def known_port(request: Request, code: str) -> str:
        code = code.strip().upper()
        if code not in set(request.app.state.ports["port_code"]):
            raise NotFoundError(f"Unknown port code '{code}'")
        return code

    def check_range(date_from: date | None, date_to: date | None) -> None:
        if date_from and date_to and date_from > date_to:
            raise BadRequestError("date_from must be on or before date_to")

    # ------------------------------------------------------------------ endpoints

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health(request: Request):
        state = request.app.state
        checks, data = {}, None
        try:
            checks["database"] = "ok" if state.repo is not None and state.repo.ping() else "unavailable"
            if checks["database"] == "ok":
                data = state.repo.data_summary()
        except Exception:  # health must answer even when the DB is broken
            logger.exception("health_db_check_failed")
            checks["database"] = "error"
        checks["model"] = "ok" if state.predictor is not None else "unavailable"
        checks["dq_report"] = "ok" if settings.dq_report_path.exists() else "unavailable"
        healthy = all(v == "ok" for v in checks.values())
        body = HealthResponse(
            status="ok" if healthy else "degraded",
            version=settings.version,
            git_sha=settings.git_sha,
            checks=checks,
            data=data,
        )
        return JSONResponse(status_code=200 if healthy else 503, content=body.model_dump(mode="json"))

    @app.get("/ports", response_model=list[Port], tags=["reference"])
    def list_ports(request: Request, r: Repository = Depends(repo)):
        return r.ports()[["port_code", "port_name", "country", "region"]].to_dict(orient="records")

    @app.get("/shipments/{shipment_id}", response_model=Shipment, tags=["shipments"])
    def get_shipment(shipment_id: str = Path(..., max_length=32), r: Repository = Depends(repo)):
        record = r.get_shipment(shipment_id.strip().upper())
        if record is None:
            raise NotFoundError(f"Shipment '{shipment_id}' not found")
        return record

    @app.get("/shipments", response_model=ShipmentPage, tags=["shipments"])
    def list_shipments(
        request: Request,
        origin: str | None = Query(None, description="Origin port code, e.g. CNSHA"),
        destination: str | None = Query(None, description="Destination port code, e.g. NLRTM"),
        status: StatusFilter | None = Query(None),
        date_from: date | None = Query(None, description="planned_departure on/after (YYYY-MM-DD)"),
        date_to: date | None = Query(None, description="planned_departure on/before (YYYY-MM-DD)"),
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=100),
        r: Repository = Depends(repo),
    ):
        check_range(date_from, date_to)
        origin = known_port(request, origin) if origin else None
        destination = known_port(request, destination) if destination else None
        items, total = r.list_shipments(
            origin, destination, status.value if status else None, date_from, date_to, page, page_size
        )
        return {
            "items": items,
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": math.ceil(total / page_size) if total else 0,
        }

    @app.get("/routes/top", tags=["routes"])
    def top_routes(
        date_from: date | None = None,
        date_to: date | None = None,
        order_by: RouteMetric = RouteMetric.avg_delay_hours,
        limit: int = Query(5, ge=1, le=50),
        min_completed: int = Query(10, ge=1, description="Ignore routes with too few completed shipments"),
        r: Repository = Depends(repo),
    ):
        check_range(date_from, date_to)
        return {
            "order_by": order_by.value,
            "date_from": str(date_from) if date_from else None,
            "date_to": str(date_to) if date_to else None,
            "min_completed": min_completed,
            "routes": r.top_routes(date_from, date_to, order_by.value, limit, min_completed),
        }

    @app.get("/routes/{origin}/{destination}/stats", response_model=RouteStats, tags=["routes"])
    def route_stats(
        request: Request,
        origin: str,
        destination: str,
        date_from: date | None = None,
        date_to: date | None = None,
        r: Repository = Depends(repo),
    ):
        check_range(date_from, date_to)
        origin, destination = known_port(request, origin), known_port(request, destination)
        stats = r.route_stats(origin, destination, date_from, date_to)
        return {
            "origin_port": origin,
            "destination_port": destination,
            "route_key": f"{origin} → {destination}",
            "date_from": str(date_from) if date_from else None,
            "date_to": str(date_to) if date_to else None,
            **stats,
        }

    @app.post("/predict-delay", response_model=PredictResponse, tags=["ml"])
    def predict_delay(body: PredictRequest, request: Request, r: Repository = Depends(repo)):
        predictor: DelayPredictor | None = request.app.state.predictor
        if predictor is None:
            raise ServiceUnavailableError("Model artefact not loaded")
        if body.shipment_id:
            booking = r.booking_inputs(body.shipment_id.strip().upper())
            if booking is None:
                raise NotFoundError(f"Shipment '{body.shipment_id}' not found")
        else:
            booking = body.model_dump(exclude={"shipment_id"})
            ports = set(request.app.state.ports["port_code"])
            unknown = [p for p in (booking["origin_port"], booking["destination_port"]) if p not in ports]
            if unknown:
                raise UnprocessableError(f"Unknown port code(s): {unknown}")
        result = predictor.predict(booking, request.app.state.ports)
        lr = predictor.card.get("metrics_test", {}).get("logistic_regression", {})
        return {
            "shipment_id": booking.get("shipment_id"),
            **result.__dict__,
            "model_test_roc_auc": lr.get("roc_auc"),
        }

    @app.get("/data-quality/report", tags=["data-quality"])
    def dq_report():
        path = settings.dq_report_path
        if not path.exists():
            raise ServiceUnavailableError("DQ report not generated yet; run `python dq_check.py`")
        return json.loads(path.read_text())

    return app


app = create_app()

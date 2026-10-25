"""Request/response models. These double as the OpenAPI contract at /docs."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from sci.dq.rules import CANONICAL_CARGO_TYPES, UNKNOWN_CARGO, normalize_cargo_type


class Shipment(BaseModel):
    shipment_id: str
    customer_id: str | None
    origin_port: str
    origin_port_name: str | None
    destination_port: str
    destination_port_name: str | None
    route_key: str
    vessel_id: str | None
    booking_date: datetime
    planned_departure: datetime
    actual_departure: datetime | None
    planned_arrival: datetime
    actual_arrival: datetime | None
    container_count: int | None
    cargo_type: str
    weight_tons: float | None
    status: str
    actual_delay_hours: float | None
    on_time_flag: bool | None
    transit_days_planned: float | None
    transit_days_actual: float | None
    booking_lead_days: float | None
    dq_flags: list[str] = []


class ShipmentPage(BaseModel):
    items: list[Shipment]
    page: int
    page_size: int
    total: int
    total_pages: int


class RouteStats(BaseModel):
    origin_port: str
    destination_port: str
    route_key: str
    date_from: str | None
    date_to: str | None
    shipment_count: int = Field(description="All curated shipments on the route in the window")
    completed_count: int = Field(description="Shipments with an actual arrival (used for delay metrics)")
    cancelled_count: int
    avg_delay_hours: float | None
    on_time_rate: float | None = Field(description="Share of completed shipments with delay <= 24h")


class Port(BaseModel):
    port_code: str
    port_name: str
    country: str | None
    region: str | None


class PredictRequest(BaseModel):
    """Either a known shipment_id, or the booking-time fields of a new booking."""

    shipment_id: str | None = Field(default=None, examples=["SHP-00421"])
    origin_port: str | None = Field(default=None, examples=["CNSHA"])
    destination_port: str | None = Field(default=None, examples=["NLRTM"])
    cargo_type: str | None = Field(default=None, examples=["Electronics"])
    container_count: int | None = Field(default=None, ge=1, le=100)
    weight_tons: float | None = Field(default=None, ge=0)
    booking_date: datetime | None = None
    planned_departure: datetime | None = None
    planned_arrival: datetime | None = None

    @model_validator(mode="after")
    def _either_id_or_booking(self) -> PredictRequest:
        if self.shipment_id:
            return self
        required = [
            "origin_port",
            "destination_port",
            "cargo_type",
            "booking_date",
            "planned_departure",
            "planned_arrival",
        ]
        missing = [f for f in required if getattr(self, f) in (None, "")]
        if missing:
            raise ValueError(f"Provide shipment_id, or all booking fields. Missing: {missing}")
        if not (self.booking_date <= self.planned_departure < self.planned_arrival):
            raise ValueError("Require booking_date <= planned_departure < planned_arrival")
        cargo = normalize_cargo_type(self.cargo_type)
        allowed = CANONICAL_CARGO_TYPES | {UNKNOWN_CARGO}
        if cargo not in allowed:
            raise ValueError(f"cargo_type must be one of {sorted(allowed)}")
        self.cargo_type = cargo
        self.origin_port = self.origin_port.strip().upper()
        self.destination_port = self.destination_port.strip().upper()
        return self


class PredictResponse(BaseModel):
    shipment_id: str | None
    delay_probability: float
    predicted_delayed: bool
    risk_band: str
    threshold: float
    model_version: str
    model_test_roc_auc: float | None = Field(
        description="Held-out AUC of the served model, so callers can judge how much to trust the score"
    )


class HealthResponse(BaseModel):
    status: str
    version: str
    git_sha: str
    checks: dict[str, str]
    data: dict | None = None

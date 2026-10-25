"""Inference wrapper: loads the committed artefact once and scores booking-time inputs."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from sci.ml.features import build_features

logger = logging.getLogger("sci.ml.predict")


def risk_band(probability: float) -> str:
    if probability >= 0.6:
        return "high"
    if probability >= 0.4:
        return "medium"
    return "low"


@dataclass
class Prediction:
    delay_probability: float
    predicted_delayed: bool
    risk_band: str
    threshold: float
    model_version: str


class DelayPredictor:
    def __init__(self, model, card: dict):
        self.model = model
        self.card = card
        self.threshold = float(card.get("threshold", 0.5))
        self.version = card.get("model_version", "unknown")

    @classmethod
    def load(cls, model_path: Path, card_path: Path) -> DelayPredictor:
        card = json.loads(card_path.read_text())
        trained_with = card.get("sklearn_version")
        if trained_with and trained_with != sklearn.__version__:
            logger.warning(
                "sklearn_version_mismatch", extra={"trained_with": trained_with, "running": sklearn.__version__}
            )
        return cls(joblib.load(model_path), card)

    def predict(self, booking: dict, ports: pd.DataFrame) -> Prediction:
        features = build_features(pd.DataFrame([booking]), ports)
        probability = float(self.model.predict_proba(features)[0, 1])
        return Prediction(
            delay_probability=round(probability, 4),
            predicted_delayed=probability >= self.threshold,
            risk_band=risk_band(probability),
            threshold=self.threshold,
            model_version=self.version,
        )

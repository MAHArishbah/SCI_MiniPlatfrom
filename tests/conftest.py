"""Shared fixtures. Unit tests run on tiny hand-made CSVs (tests/fixtures/raw) where every
planted data-quality issue is known, so assertions can be exact. No Docker, no network."""

import json
from pathlib import Path

import pytest

from sci.config import Settings
from sci.ml.predict import Prediction

FIXTURE_RAW = Path(__file__).parent / "fixtures" / "raw"


@pytest.fixture(scope="session")
def raw_dir() -> Path:
    return FIXTURE_RAW


@pytest.fixture(scope="session")
def raw_frames(raw_dir):
    from sci.dq.runner import load_raw

    frames, _ = load_raw(raw_dir)
    return frames


@pytest.fixture(scope="session")
def settings(tmp_path_factory, raw_dir) -> Settings:
    """A fully built environment (DuckDB + DQ report) in a temp dir, from the fixture CSVs."""
    from sci.dq.runner import build_report, load_raw
    from sci.pipeline.build import build

    root = tmp_path_factory.mktemp("env")
    db_path = root / "sci.duckdb"
    build(raw_dir, db_path)
    frames, paths = load_raw(raw_dir)
    report_path = root / "dq_report.json"
    report_path.write_text(json.dumps(build_report(frames, paths)))
    return Settings(
        data_dir=raw_dir,
        db_path=db_path,
        dq_report_path=report_path,
        model_path=root / "not-used.joblib",
        model_card_path=root / "not-used.json",
        log_level="WARNING",
    )


class StubPredictor:
    """Stands in for the real model so API tests do not depend on the trained artefact."""

    version = "stub-1"
    card = {"metrics_test": {"logistic_regression": {"roc_auc": 0.51}}}

    def __init__(self):
        self.calls = []

    def predict(self, booking, ports):
        self.calls.append(booking)
        return Prediction(
            delay_probability=0.42, predicted_delayed=False, risk_band="medium", threshold=0.5, model_version="stub-1"
        )


@pytest.fixture
def stub_predictor():
    return StubPredictor()


@pytest.fixture
def client(settings, stub_predictor):
    from fastapi.testclient import TestClient

    from sci.api.main import create_app

    with TestClient(create_app(settings, predictor=stub_predictor)) as c:
        yield c

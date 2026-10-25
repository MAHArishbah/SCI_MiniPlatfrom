import numpy as np
import pandas as pd
import pytest

from sci.ml.features import FEATURE_COLUMNS, FORBIDDEN_COLUMNS, build_features, build_label
from sci.ml.predict import DelayPredictor, risk_band
from sci.pipeline import transform


@pytest.fixture(scope="module")
def curated(raw_frames):
    return transform.clean_shipments(raw_frames)[0]


@pytest.fixture(scope="module")
def ports(raw_frames):
    return transform.clean_ports(raw_frames.ports)


def test_feature_matrix_has_no_post_booking_columns(curated, ports):
    x = build_features(curated, ports)  # curated contains actuals/status; they must not leak
    assert list(x.columns) == FEATURE_COLUMNS
    assert not FORBIDDEN_COLUMNS & set(x.columns)


def test_features_do_not_change_when_actuals_change(curated, ports):
    tampered = curated.assign(actual_arrival=pd.NaT, actual_departure=pd.NaT, status="DELAYED", actual_delay_hours=999)
    pd.testing.assert_frame_equal(build_features(curated, ports), build_features(tampered, ports))


def test_missing_inputs_raise(ports):
    with pytest.raises(ValueError, match="Missing booking-time inputs"):
        build_features(pd.DataFrame({"origin_port": ["CNSHA"]}), ports)


def test_feature_values(curated, ports):
    row = build_features(curated[curated["shipment_id"] == "SHP-00001"], ports).iloc[0]
    assert row["booking_lead_days"] == pytest.approx(7)
    assert row["planned_transit_days"] == pytest.approx(31)
    assert row["origin_region"] == "APAC" and row["destination_region"] == "EUR"
    assert row["origin_congestion"] == pytest.approx(0.85)
    assert row["log_weight_tons"] == pytest.approx(np.log1p(500.5))


def test_negative_weight_becomes_nan(ports):
    booking = {
        "origin_port": "CNSHA", "destination_port": "NLRTM", "cargo_type": "Apparel", "container_count": 3,
        "weight_tons": -1, "booking_date": "2025-01-01", "planned_departure": "2025-01-05",
        "planned_arrival": "2025-02-01",
    }  # fmt: skip
    assert np.isnan(build_features(pd.DataFrame([booking]), ports)["log_weight_tons"].iloc[0])


def test_label_threshold():
    labels = build_label(pd.DataFrame({"actual_delay_hours": [0, 24, 24.01, 100]}))
    assert labels.tolist() == [0, 0, 1, 1]


@pytest.mark.parametrize("p,band", [(0.1, "low"), (0.39, "low"), (0.4, "medium"), (0.59, "medium"), (0.6, "high")])
def test_risk_band(p, band):
    assert risk_band(p) == band


def test_predictor_uses_threshold_from_model_card(curated, ports):
    class FakeModel:
        def predict_proba(self, x):
            assert list(x.columns) == FEATURE_COLUMNS
            return np.array([[0.3, 0.7]])

    predictor = DelayPredictor(FakeModel(), {"threshold": 0.8, "model_version": "v-test"})
    booking = curated.iloc[0].to_dict()
    result = predictor.predict(booking, ports)
    assert result.delay_probability == 0.7 and result.risk_band == "high"
    assert result.predicted_delayed is False  # 0.7 < threshold 0.8
    assert result.model_version == "v-test"

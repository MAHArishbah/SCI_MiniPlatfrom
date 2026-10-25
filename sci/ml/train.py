"""Train the booking-time delay-risk model and write the artefact + model card.

    python -m sci.ml.train --db warehouse/sci.duckdb --out artifacts/

Evaluation design:
  * Time-based split on booking_date (train on the past, test on the most recent 20%):
    this mirrors production, where the model always scores bookings from the future.
  * Baselines: a prior-only classifier (AUC = 0.5 by construction) to anchor the numbers.
  * Logistic regression is the served model; gradient boosting is trained only for comparison.
  * Bootstrap 95% CI on test AUC, and TimeSeriesSplit CV on the training window.
  * Leakage reference: the same model with departure delay added, to show what a
    leaky feature does (reported only, never served).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sci.ml.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, build_features, build_label

logger = logging.getLogger("sci.ml.train")

THRESHOLD = 0.5
TEST_FRACTION = 0.2
SEED = 42


def make_preprocessor(categorical: list[str], numeric: list[str], dense: bool = False) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)),
                    ]
                ),
                categorical,
            ),
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ]
    )


def make_logreg() -> Pipeline:
    return Pipeline(
        [
            ("prep", make_preprocessor(CATEGORICAL_FEATURES, NUMERIC_FEATURES)),
            ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
        ]
    )


def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float = THRESHOLD) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "brier": round(float(brier_score_loss(y_true, proba)), 4),
        "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def bootstrap_auc_ci(y_true: np.ndarray, proba: np.ndarray, n: int = 1000, seed: int = SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) == 2:
            scores.append(roc_auc_score(y_true[idx], proba[idx]))
    return [round(float(np.percentile(scores, 2.5)), 4), round(float(np.percentile(scores, 97.5)), 4)]


def load_training_data(db_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        shipments = con.execute("SELECT * FROM curated.shipments WHERE is_labelled ORDER BY booking_date").df()
        ports = con.execute("SELECT * FROM curated.ports").df()
    finally:
        con.close()
    return shipments, ports


def time_split(df: pd.DataFrame, test_fraction: float = TEST_FRACTION) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["booking_date", "shipment_id"]).reset_index(drop=True)
    cut = int(len(df) * (1 - test_fraction))
    return df.iloc[:cut], df.iloc[cut:]


def train(db_path: Path, out_dir: Path) -> dict:
    shipments, ports = load_training_data(db_path)
    train_df, test_df = time_split(shipments)
    x_train, x_test = build_features(train_df, ports), build_features(test_df, ports)
    y_train, y_test = build_label(train_df).to_numpy(), build_label(test_df).to_numpy()

    results: dict[str, dict] = {}

    prior = DummyClassifier(strategy="prior").fit(x_train, y_train)
    results["baseline_prior"] = evaluate(y_test, prior.predict_proba(x_test)[:, 1])

    logreg = make_logreg().fit(x_train, y_train)
    lr_proba = logreg.predict_proba(x_test)[:, 1]
    results["logistic_regression"] = evaluate(y_test, lr_proba)
    results["logistic_regression"]["roc_auc_95ci"] = bootstrap_auc_ci(y_test, lr_proba)
    cv = cross_val_score(make_logreg(), x_train, y_train, cv=TimeSeriesSplit(n_splits=5), scoring="roc_auc")
    results["logistic_regression"]["train_timeseries_cv_auc"] = [round(float(v), 4) for v in cv]

    hgb = Pipeline(
        [
            ("prep", make_preprocessor(CATEGORICAL_FEATURES, NUMERIC_FEATURES, dense=True)),
            ("clf", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED)),
        ]
    ).fit(x_train, y_train)
    results["gradient_boosting_comparison"] = evaluate(y_test, hgb.predict_proba(x_test)[:, 1])

    # Leakage reference: what happens if departure delay (known only after sailing) sneaks in.
    leak_train = x_train.assign(departure_delay_hours=train_df["departure_delay_hours"].to_numpy())
    leak_test = x_test.assign(departure_delay_hours=test_df["departure_delay_hours"].to_numpy())
    leaky = Pipeline(
        [
            ("prep", make_preprocessor(CATEGORICAL_FEATURES, NUMERIC_FEATURES + ["departure_delay_hours"])),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000)),
        ]
    ).fit(leak_train, y_train)
    leak_auc = round(float(roc_auc_score(y_test, leaky.predict_proba(leak_test)[:, 1])), 4)

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.joblib"
    joblib.dump(logreg, model_path)
    artefact_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]

    card = {
        "model_name": "delay_risk_logreg",
        "model_version": f"{datetime.now(UTC):%Y%m%d}-{artefact_hash}",
        "trained_at": datetime.now(UTC).isoformat(),
        "sklearn_version": sklearn.__version__,
        "target": "actual_arrival - planned_arrival > 24 hours",
        "prediction_time": "booking",
        "features": FEATURE_COLUMNS,
        "threshold": THRESHOLD,
        "split": {
            "method": "time-based on booking_date (no shuffling)",
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_window": [str(train_df["booking_date"].min()), str(train_df["booking_date"].max())],
            "test_window": [str(test_df["booking_date"].min()), str(test_df["booking_date"].max())],
            "train_positive_rate": round(float(y_train.mean()), 4),
            "test_positive_rate": round(float(y_test.mean()), 4),
        },
        "metrics_test": results,
        "leakage_reference": {
            "description": "Same model plus departure_delay_hours (post-booking). Never served.",
            "roc_auc": leak_auc,
        },
    }
    (out_dir / "model_card.json").write_text(json.dumps(card, indent=2))
    logger.info(
        "model_trained",
        extra={"model_version": card["model_version"], "test_auc": results["logistic_regression"]["roc_auc"]},
    )
    return card


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the delay-risk model.")
    parser.add_argument("--db", type=Path, default=Path("warehouse/sci.duckdb"))
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    args = parser.parse_args(argv)

    from sci.logging_config import configure_logging

    configure_logging()
    card = train(args.db, args.out)
    print(json.dumps(card["metrics_test"], indent=2))
    print(json.dumps(card["leakage_reference"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

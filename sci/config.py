"""Runtime configuration, read from environment variables with sane local defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from sci import __version__

REPO_ROOT = Path(__file__).resolve().parent.parent


def _path(env: str, default: Path) -> Path:
    return Path(os.getenv(env, str(default)))


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: _path("SCI_DATA_DIR", REPO_ROOT / "data" / "raw"))
    db_path: Path = field(default_factory=lambda: _path("SCI_DB_PATH", REPO_ROOT / "warehouse" / "sci.duckdb"))
    dq_report_path: Path = field(
        default_factory=lambda: _path("SCI_DQ_REPORT_PATH", REPO_ROOT / "artifacts" / "dq_report.json")
    )
    model_path: Path = field(default_factory=lambda: _path("SCI_MODEL_PATH", REPO_ROOT / "artifacts" / "model.joblib"))
    model_card_path: Path = field(
        default_factory=lambda: _path("SCI_MODEL_CARD_PATH", REPO_ROOT / "artifacts" / "model_card.json")
    )
    log_level: str = field(default_factory=lambda: os.getenv("SCI_LOG_LEVEL", "INFO"))
    version: str = __version__
    git_sha: str = field(default_factory=lambda: os.getenv("GIT_SHA", "local"))


def get_settings() -> Settings:
    return Settings()

"""Central configuration + path helpers: the single source of truth shared by every pipeline stage."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

DEFAULT_CFG = dict(
    seed=42,
    # --- simulator (users + interactions are synthetic, grounded in the real Goodreads star histograms)
    n_users=4000, catalog_size=3000, min_num_ratings=2000,
    start_date="2019-01-01", end_date="2024-06-30",
    fault_rate=0.02,
    # --- modelling
    test_fraction=0.20,        # last 20 % of interactions (by time) -> test
    val_fraction=0.15,         # last 15 % of TRAIN (by time) -> validation fold used for tuning
    positive_threshold=4,      # rating >= 4  => "relevant"
    ks=[5, 10],
    bayes_prior=20,            # shrinkage strength of the Bayesian item average
    # --- orchestration
    max_reject_ratio=0.05,     # quality gate: stop the pipeline if > 5 % of interaction rows are rejected
    source_csv=None,           # path of the real Goodreads items file (filled in by the notebook / CLI)
)

SUBDIRS = ["data/raw", "data/processed", "data/features", "data/errors", "models",
           "reports", "reports/metrics", "reports/orchestration", "configs", "dags"]


class Paths:
    """All folders of the suggested project structure, relative to one base directory."""

    def __init__(self, base):
        self.base = Path(base).resolve()
        self.raw = self.base / "data/raw"
        self.processed = self.base / "data/processed"
        self.features = self.base / "data/features"
        self.errors = self.base / "data/errors"
        self.models = self.base / "models"
        self.reports = self.base / "reports"
        self.metrics = self.base / "reports/metrics"
        self.orchestration = self.base / "reports/orchestration"
        self.configs = self.base / "configs"
        self.dags = self.base / "dags"
        self.mlflow_db = self.base / "mlflow.db"
        self.mlruns = self.base / "mlruns"

    def ensure(self):
        for d in SUBDIRS:
            (self.base / d).mkdir(parents=True, exist_ok=True)
        return self


def save_config(base, cfg) -> Path:
    P = Paths(base).ensure()
    path = P.configs / "pipeline_config.json"
    path.write_text(json.dumps(cfg, indent=2))
    return path


def load_config(base) -> dict:
    return json.loads((Paths(base).configs / "pipeline_config.json").read_text())


def config_hash(cfg) -> str:
    """Short fingerprint of a configuration (used by the orchestrator's resume logic)."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]

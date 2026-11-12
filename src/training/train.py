"""Stage 6 - Model training: one callable per model (= one orchestrator task each).

For each model
  1. hyper-parameters are tuned on a TEMPORAL validation fold (train_fit -> val); the test set is never touched while tuning,
  2. the model is re-fitted on the full training window and evaluated ONCE on the held-out test window,
  3. parameters, metrics (validation trials as nested runs) and the model artifact are logged to MLflow,
  4. metrics + per-user scores are saved to reports/metrics/ for the evaluation stage.
"""
from __future__ import annotations

import itertools
import json
import os
import re
import time

import joblib
import mlflow
import numpy as np
import pandas as pd

from src.common.log import get_logger
from src.config import Paths
from src.evaluation.metrics import RecData, build_vocab, evaluate
from src.training.models import (ALSModel, ContentModel, HybridModel, ItemKNNModel, PopularityModel, RandomModel, SEED, SVDModel)

log = get_logger("training")
EXPERIMENT = "goodreads-recsys"

# key -> (class, tuning grid, selection rule). The Hybrid reuses the tuned ALS + ItemKNN parameters and only tunes the blend weights.
REGISTRY = {
    "random":     dict(cls=RandomModel, grid=[], select=None),
    "popularity": dict(cls=PopularityModel, grid=[], select=None),
    "als":        dict(cls=ALSModel, select=("NDCG", "max"),
                       grid=[dict(factors=f, regularization=r, alpha=a) for f, r, a in itertools.product([32, 64], [0.01, 0.1], [10, 40])]),
    "svd":        dict(cls=SVDModel, select=("RMSE", "min"),      # SVD is selected on rating error (its natural objective)
                       grid=[dict(n_factors=f, reg_all=r) for f, r in itertools.product([20, 50], [0.02, 0.08])]),
    "itemknn":    dict(cls=ItemKNNModel, select=("NDCG", "max"),
                       grid=[dict(k=k, shrink=s) for k, s in itertools.product([30, 100], [10.0, 50.0])]),
    "content":    dict(cls=ContentModel, grid=[], select=None),
    "hybrid":     dict(cls=HybridModel, select=("NDCG", "max"),
                       grid=[dict(w_als=wa, w_content=wc, w_pop=wp) for wa, wc, wp in itertools.product([0.5, 1.0, 2.0], [0.0, 0.25], [0.0, 0.15])]),
}
ORDER = list(REGISTRY)
mlkey = lambda k: k.replace("@", "_at_").lower()


def setup_mlflow(P: Paths):
    mlflow.set_tracking_uri(f"sqlite:///{P.mlflow_db}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=P.mlruns.resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)


def _load(P: Paths):
    items_clean = pd.read_parquet(P.processed / "items_clean.parquet")
    train = pd.read_parquet(P.processed / "train.parquet")
    test = pd.read_parquet(P.processed / "test.parquet")
    info = json.loads((P.processed / "split_info.json").read_text())
    return items_clean, train, test, info


def run_model(base, cfg, key: str) -> dict:
    P = Paths(base).ensure()
    spec = REGISTRY[key]
    cls, ks, pos = spec["cls"], cfg["ks"], cfg["positive_threshold"]
    Kmax = max(ks)
    items_clean, train, test, info = _load(P)
    vocab = build_vocab(items_clean)
    d_full = RecData(train, test, items_clean, pos, vocab)
    setup_mlflow(P)
    np.random.seed(SEED)

    base_params = {}
    if key == "hybrid":                                    # depends on the ALS and ItemKNN tasks: reuse their tuned parameters
        for dep in ("als", "itemknn"):
            base_params.update(json.loads((P.metrics / f"{dep}.json").read_text())["best_params"])
    tags = {"pipeline_mode": os.environ.get("RECSYS_RUN_MODE", "interactive"), "pipeline_run_id": os.environ.get("RECSYS_RUN_ID", "-"), "model_key": key}

    with mlflow.start_run(run_name=f"train_{key}", tags=tags) as parent:
        tuning, best = [], {}
        if spec["grid"]:
            val_cut = pd.Timestamp(info["val_cut"])
            train_fit, val = train[train["timestamp"] < val_cut], train[train["timestamp"] >= val_cut]
            d_fit = RecData(train_fit, val, items_clean, pos, vocab)
            mlflow.log_params({"tuning_fold": "temporal validation (last 15% of train)", "val_cut": str(val_cut), "n_val_eval_users": d_fit.info["n_eval_users"]})
            for p in spec["grid"]:
                with mlflow.start_run(run_name=f"{key}_" + "_".join(f"{k}{v}" for k, v in p.items()), nested=True, tags={"model_key": key}):
                    m = cls(**{**base_params, **p}).fit(d_fit)
                    res = evaluate(m, d_fit, ks)
                    mlflow.log_params({"model": key, **m.params})
                    mlflow.log_metrics({mlkey(k): v for k, v in res.items() if not np.isnan(v)})
                    tuning.append({**p, **{f"val_{k}": v for k, v in res.items() if k in (f"NDCG@{Kmax}", f"Recall@{Kmax}", f"Precision@{Kmax}", "RMSE")}})
            tdf = pd.DataFrame(tuning)
            metric, how = spec["select"]
            col = "val_RMSE" if metric == "RMSE" else f"val_NDCG@{Kmax}"
            row = tdf.sort_values(col, ascending=(how == "min")).iloc[0]
            best = {k: (type(v)(row[k]) if not isinstance(v, str) else v) for k, v in spec["grid"][0].items()}
            log.info(f"{key}: best on validation ({col}) -> {best}")

        final_params = {**base_params, **best}
        model = cls(**final_params)
        t0 = time.time(); model.fit(d_full); fit_s = time.time() - t0
        res, pu = evaluate(model, d_full, ks, per_user=True)
        mlflow.log_params({"model": model.name, **model.params, "split_cutoff": info["cutoff"], "positive_threshold": pos, "ks": str(ks),
                           **{f"data_{k}": v for k, v in d_full.info.items()}})
        mlflow.log_metrics({mlkey(k): v for k, v in res.items() if not np.isnan(v)})
        mlflow.log_metric("fit_seconds", fit_s)

        path = P.models / f"{re.sub(r'[^a-z0-9]+', '_', model.name.lower()).strip('_')}.joblib"
        joblib.dump(model, path)
        mlflow.log_artifact(str(path), artifact_path="model")
        out = dict(key=key, name=model.name, params=model.params, best_params=final_params if key != "hybrid" else {**base_params, **best},
                   metrics=res, fit_seconds=fit_s, tuning=tuning, mlflow_run_id=parent.info.run_id, model_path=str(path.relative_to(P.base)),
                   data_info=d_full.info)
        (P.metrics / f"{key}.json").write_text(json.dumps(out, indent=2, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else float(o)))
        np.savez_compressed(P.metrics / f"{key}_per_user.npz", user_ids=d_full.users[d_full.eval_users], **{k: v for k, v in pu.items()})
        log.info(f"{model.name:36s} fit {fit_s:5.1f}s | NDCG@{Kmax}={res[f'NDCG@{Kmax}']:.4f}  Recall@{Kmax}={res[f'Recall@{Kmax}']:.4f}  RMSE={res['RMSE']:.4f}")
    return {"model": model.name, f"NDCG@{Kmax}": res[f"NDCG@{Kmax}"], "fit_seconds": round(fit_s, 2), "best_params": best}

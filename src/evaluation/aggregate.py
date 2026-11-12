"""Stage 7 - Model evaluation: compare all trained models in one results table (+ uncertainty + significance tests)."""
from __future__ import annotations

import json

import mlflow
import numpy as np
import pandas as pd

from src.common.log import get_logger
from src.config import Paths
from src.evaluation.metrics import bootstrap_ci, paired_bootstrap
from src.training.train import EXPERIMENT, ORDER, mlkey, setup_mlflow

log = get_logger("evaluation")
BASELINE = "Popularity (baseline)"


def result_columns(ks):
    K = max(ks)
    return [f"{m}@{k}" for k in ks for m in ["Precision", "Recall", "NDCG"]] + [f"Coverage@{K}", f"Diversity@{K}", "RMSE", "MAE"]


def load_results(P: Paths):
    runs = [json.loads((P.metrics / f"{k}.json").read_text()) for k in ORDER if (P.metrics / f"{k}.json").exists()]
    return runs


def run(base, cfg) -> dict:
    P = Paths(base).ensure()
    ks = cfg["ks"]; K = max(ks)
    runs = load_results(P)
    names = {r["key"]: r["name"] for r in runs}
    res_df = pd.DataFrame({r["name"]: {k: (np.nan if v is None else v) for k, v in r["metrics"].items()} for r in runs}).T[result_columns(ks)]
    res_df.index.name = "Model"
    res_df.to_csv(P.reports / "model_results.csv")

    # ---- uncertainty: bootstrap CI over evaluated users + paired tests vs. the popularity baseline / ALS
    pu = {r["name"]: np.load(P.metrics / f"{r['key']}_per_user.npz", allow_pickle=True) for r in runs}
    ci = []
    for n, z in pu.items():
        for m in ("precision", "recall", "ndcg"):
            mean, lo, hi = bootstrap_ci(z[f"{m}@{K}"])
            ci.append({"Model": n, "metric": f"{m.upper() if m == 'ndcg' else m.capitalize()}@{K}", "mean": mean, "ci_low": lo, "ci_high": hi})
    ci = pd.DataFrame(ci)
    ci.to_csv(P.reports / "model_results_ci.csv", index=False)
    sig = []
    pairs = [(n, BASELINE) for n in pu if n not in (BASELINE, names.get("random"))]
    if "hybrid" in names:
        pairs += [(names["hybrid"], names[k]) for k in ("itemknn", "als") if k in names]
    for a, b in pairs:
        diff, lo, hi, p = paired_bootstrap(pu[a][f"ndcg@{K}"], pu[b][f"ndcg@{K}"])
        sig.append({"model": a, "versus": b, f"delta_NDCG@{K}": diff, "ci_low": lo, "ci_high": hi, "bootstrap_p(delta<=0)": p, "significant_95": bool(lo > 0 or hi < 0)})
    sig = pd.DataFrame(sig)
    sig.to_csv(P.reports / "significance_tests.csv", index=False)

    # ---- accuracy by user-activity segment
    seg = pd.read_parquet(P.features / "user_features.parquet").set_index("user_id")["activity_segment"]
    some = next(iter(pu.values()))
    seg_df = (pd.DataFrame({n: z[f"ndcg@{K}"] for n, z in pu.items()}, index=some["user_ids"]).join(seg).groupby("activity_segment").mean()
              .reindex(["light", "medium", "active", "power"]))
    seg_df.to_csv(P.reports / "ndcg_by_segment.csv")

    personalised = [n for n in res_df.index if n not in (names.get("random"), BASELINE) and not np.isnan(res_df.loc[n, f"NDCG@{K}"])]
    best = res_df.loc[personalised, f"NDCG@{K}"].idxmax()

    # ---- MLflow summary run (artifacts of the whole pipeline)
    setup_mlflow(P)
    import os
    with mlflow.start_run(run_name="evaluation_summary", tags={"pipeline_mode": os.environ.get("RECSYS_RUN_MODE", "interactive"), "pipeline_run_id": os.environ.get("RECSYS_RUN_ID", "-")}):
        mlflow.log_params({"best_model": best, "n_models": len(res_df), "ks": str(ks)})
        mlflow.log_metrics({f"best_{mlkey(f'NDCG@{K}')}": float(res_df.loc[best, f"NDCG@{K}"])})
        for f in ["reports/validation_report.json", "reports/feature_catalog.csv", "reports/model_results.csv", "reports/model_results_ci.csv",
                  "reports/significance_tests.csv", "configs/pipeline_config.json", "reports/data_validation_report.html"]:
            if (P.base / f).exists():
                mlflow.log_artifact(str(P.base / f))
    log.info(f"results table -> reports/model_results.csv | best personalised model by NDCG@{K}: {best} ({res_df.loc[best, f'NDCG@{K}']:.4f})")
    return {"best_model": best, f"best_NDCG@{K}": float(res_df.loc[best, f"NDCG@{K}"]), "n_models": len(res_df)}

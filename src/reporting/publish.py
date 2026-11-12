"""Final pipeline task: regenerate figures + PDF report + run manifest."""
from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from src.common.log import get_logger
from src.config import Paths
from src.reporting import architecture, figures, final_report

log = get_logger("reporting")


def write_manifest(P: Paths, cfg: dict):
    split = json.loads((P.processed / "split_info.json").read_text())
    res = pd.read_csv(P.reports / "model_results.csv", index_col=0)
    best = {k: json.loads((P.metrics / f"{k}.json").read_text())["best_params"] for k in ["als", "svd", "itemknn", "hybrid"] if (P.metrics / f"{k}.json").exists()}
    manifest = {"created": datetime.now().isoformat(timespec="seconds"), "config": cfg, "split": split, "best_params": best,
                "results": res.round(5).where(res.notna(), None).to_dict(orient="index")}
    (P.reports / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def run(base, cfg) -> dict:
    P = Paths(base).ensure()
    architecture.draw(base)
    figures.results_figure(base)
    figures.validation_figure(base)
    write_manifest(P, cfg)
    pdf = final_report.build(base)
    log.info(f"figures, run manifest and final report written -> {pdf.relative_to(P.base)}")
    return {"report": str(pdf.relative_to(P.base))}

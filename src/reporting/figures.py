"""Result figures saved to reports/ (used by the PDF report and displayed in the notebook)."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import Paths

PALETTE = ["#cfcfcf", "#8c8c8c", "#4C72B0", "#C44E52", "#55A868", "#CCB974", "#8172B2"]


def results_figure(base, show=False):
    P = Paths(base)
    cfg = json.loads((P.configs / "pipeline_config.json").read_text()); K = max(cfg["ks"])
    res = pd.read_csv(P.reports / "model_results.csv", index_col=0)
    ci = pd.read_csv(P.reports / "model_results_ci.csv")
    short = [n.split(" (")[0] for n in res.index]
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.8))
    for ax, metric in zip(axes[:3], [f"Precision@{K}", f"Recall@{K}", f"NDCG@{K}"]):
        c = ci[ci.metric == metric].set_index("Model").loc[res.index]
        ax.bar(short, c["mean"], yerr=[c["mean"] - c["ci_low"], c["ci_high"] - c["mean"]], color=PALETTE[:len(res)], capsize=3, ecolor="#333")
        ax.set_title(f"{metric} (95% bootstrap CI)"); ax.tick_params(axis="x", rotation=35)
        for i, v in enumerate(c["mean"]):
            ax.text(i, c["ci_high"].iloc[i], f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax, x, w = axes[3], np.arange(len(res)), 0.38
    ax.bar(x - w / 2, res[f"Coverage@{K}"], w, label="Coverage", color="#55A868"); ax.bar(x + w / 2, res[f"Diversity@{K}"], w, label="Diversity", color="#CCB974")
    ax.set_xticks(x); ax.set_xticklabels(short, rotation=35); ax.set_title(f"Coverage & Diversity @{K}"); ax.legend()
    plt.tight_layout(); fig.savefig(P.reports / "results_overview.png", dpi=140)
    if not show:
        plt.close(fig)
    return P.reports / "results_overview.png"


def validation_figure(base, show=False):
    P = Paths(base)
    log = pd.read_csv(P.errors / "validation_error_log.csv")
    plot = log.groupby(["table", "rule", "severity"]).size().rename("rows").reset_index()
    plot["label"] = plot["table"] + " | " + plot["rule"]
    plot = plot.sort_values("rows")
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.barh(plot["label"], plot["rows"], color=plot["severity"].map({"reject": "#C44E52", "warning": "#CCB974"}))
    ax.set(title="Validation findings per rule (red = rejected, yellow = repaired warning)", xlabel="rows affected (log scale)", xscale="log")
    plt.tight_layout(); fig.savefig(P.reports / "validation_findings.png", dpi=140)
    if not show:
        plt.close(fig)
    return P.reports / "validation_findings.png"

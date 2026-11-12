"""Architecture diagram of the pipeline (saved to reports/architecture_diagram.png)."""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.config import Paths


def draw(base, show=False):
    P = Paths(base).ensure()
    fig, ax = plt.subplots(figsize=(16, 7.4)); ax.axis("off"); ax.set_xlim(0, 16); ax.set_ylim(0, 7.4)

    def box(x, y, w, h, title, sub, color, fs=10):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=color, ec="#333", lw=1.1))
        ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=fs, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center", fontsize=7.4, color="#222")

    def arrow(x1, y1, x2, y2, ls="-", c="#333"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, lw=1.3, color=c, ls=ls))

    row = [("1 Ingestion", "Goodreads items (real)\nusers + interactions\n(documented simulator)", "#dbe9f6"),
           ("2 Validation", "Pandera schemas, dup /\nref-integrity / temporal\nrules -> error dataset", "#fde0dc"),
           ("3 Preparation", "clean, impute, standardise\ngenre parsing", "#e2f0d9"),
           ("4 Transformation", "time-based split\nactivity, popularity,\nuser-item matrix", "#e2f0d9"),
           ("5 Feature mgmt", "user / item / interaction\nfeatures -> Parquet\n+ catalog + FeatureStore", "#fff2cc")]
    for k, (t, s, c) in enumerate(row):
        box(0.3 + k * 3.1, 5.0, 2.5, 1.75, t, s, c)
        if k:
            arrow(0.3 + k * 3.1 - 0.6, 5.88, 0.3 + k * 3.1, 5.88)
    box(0.3, 2.55, 4.6, 1.75, "6 Model training", "Random | Popularity | ALS | SVD\nItemKNN | Content | Hybrid\ntuned on a temporal validation fold", "#e4dcf1")
    box(5.7, 2.55, 4.0, 1.75, "7 Evaluation", "P@K  R@K  NDCG@K  Coverage\nDiversity  RMSE  MAE\nbootstrap CIs + significance tests", "#e4dcf1")
    box(10.5, 2.55, 5.2, 1.75, "Serving / reporting", "top-10 batch lists + cold-start fallback\nfeature catalog, validation report,\nresults table, final PDF report", "#d9ead3")
    arrow(4.9, 3.42, 5.7, 3.42); arrow(9.7, 3.42, 10.5, 3.42)
    arrow(13.4, 5.0, 2.6, 4.3)
    ax.add_patch(FancyBboxPatch((0.3, 0.35), 15.4, 1.45, boxstyle="round,pad=0.04,rounding_size=0.12", fc="#eeeeee", ec="#333", lw=1.1, ls="--"))
    ax.text(8.0, 1.52, "8 Orchestration - dependency-aware Python workflow engine (retries, quality gate, resume, run log)  |  same task registry exported as an Apache Airflow DAG",
            ha="center", fontsize=9.6, fontweight="bold")
    ax.text(8.0, 0.86, "ingest -> validate -> QUALITY GATE -> prepare -> transform -> features -> train x7 -> evaluate -> batch recommend -> publish reports\n"
            "MLflow (SQLite backend): params | metrics | model artifacts  -  data/{raw,processed,features,errors}/ as Parquet / CSV",
            ha="center", fontsize=8.2, color="#222")
    for x, t in [(3.5, "data/errors/\n(rejected rows + log)"), (9.7, "data/processed/"), (12.8, "data/features/")]:
        ax.text(x, 4.78, t, ha="center", va="top", fontsize=7.2, color="#555")
    ax.set_title("Recommendation data-management pipeline - architecture", fontweight="bold", fontsize=13)
    plt.tight_layout()
    fig.savefig(P.reports / "architecture_diagram.png", dpi=150)
    if not show:
        plt.close(fig)
    return P.reports / "architecture_diagram.png"

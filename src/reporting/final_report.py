"""Builds the short final report (2-3 pages, PDF) from the artefacts of the latest run, so every number is traceable."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.config import Paths

_FONTS = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DJ", str(_FONTS / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DJ-B", str(_FONTS / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DJ-I", str(_FONTS / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFontFamily("DJ", normal="DJ", bold="DJ-B", italic="DJ-I", boldItalic="DJ-B")

NAVY = colors.HexColor("#1f3b73")
BODY = ParagraphStyle("body", fontName="DJ", fontSize=8.1, leading=10.6, alignment=TA_JUSTIFY, spaceAfter=3)
H1 = ParagraphStyle("h1", fontName="DJ-B", fontSize=15, leading=18, textColor=NAVY, spaceAfter=2)
H2 = ParagraphStyle("h2", fontName="DJ-B", fontSize=10.2, leading=12.5, textColor=NAVY, spaceBefore=6, spaceAfter=3)
SUB = ParagraphStyle("sub", fontName="DJ", fontSize=8, leading=10, textColor=colors.HexColor("#555555"), spaceAfter=4)
CELL = ParagraphStyle("cell", fontName="DJ", fontSize=6.8, leading=8.2)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="DJ-B")
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=9, bulletIndent=0, spaceAfter=1.5)


def _table(rows, widths, header=True, align_right_from=None, zebra=True):
    data = [[Paragraph(str(c), CELLB if (header and i == 0) else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#bbbbbb")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6), ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6ecf7")))
    if zebra:
        st += [("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f7f7f7")) for i in range(2, len(rows), 2)]
    t.setStyle(TableStyle(st))
    return t


def _img(path, width):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    return Image(str(path), width=width, height=width * h / w)


def build(base) -> Path:
    P = Paths(base)
    cfg = json.loads((P.configs / "pipeline_config.json").read_text()); K = max(cfg["ks"])
    val = json.loads((P.reports / "validation_report.json").read_text())
    split = json.loads((P.processed / "split_info.json").read_text())
    res = pd.read_csv(P.reports / "model_results.csv", index_col=0)
    sig = pd.read_csv(P.reports / "significance_tests.csv")
    cat = pd.read_csv(P.reports / "feature_catalog.csv")
    det = pd.read_csv(P.reports / "injection_detection_check.csv", index_col=0) if (P.reports / "injection_detection_check.csv").exists() else None
    # the most recent run that actually executed work (a pure cache-hit re-run says nothing about the pipeline's cost)
    run = None
    for rp in sorted(P.orchestration.glob("run_*.json")):
        r_ = json.loads(rp.read_text())
        if any(t_["status"] == "success" for t_ in r_["tasks"]):
            run = r_
    n_eval = json.loads((P.metrics / "popularity.json").read_text())["data_info"]["n_eval_users"]

    personalised = [n for n in res.index if not n.startswith(("Random", "Popularity")) and not np.isnan(res.loc[n, f"NDCG@{K}"])]
    best = res.loc[personalised, f"NDCG@{K}"].idxmax()
    base_n = "Popularity (baseline)"
    lift = res.loc[best, f"NDCG@{K}"] / res.loc[base_n, f"NDCG@{K}"] - 1
    hyb = next((n for n in res.index if n.startswith("Hybrid")), None)
    knn = next((n for n in res.index if n.startswith("ItemKNN")), None)
    out = P.reports / "final_report.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.3 * cm, bottomMargin=1.2 * cm,
                            title="End-to-End Data Management Pipeline for a Recommendation System", author="DM4ML Assignment 2")
    W = A4[0] - 3.2 * cm
    S = []
    S.append(Paragraph("End-to-End Data Management Pipeline for a Book Recommendation System", H1))
    S.append(Paragraph("Data Management for Machine Learning - Assignment 2 - Final report", SUB))

    # ------------------------------------------------------------------ 1
    S.append(Paragraph("1. Problem and data", H2))
    S.append(Paragraph(
        f"An online book platform wants to recommend books a reader is likely to rate &ge; {cfg['positive_threshold']} in the <b>future</b>. The pipeline ingests user, item and "
        f"interaction data, validates it, prepares and transforms it, manages reusable features, trains and evaluates models and automates the whole workflow. "
        f"<b>Items</b> are real: the Goodreads books catalogue ({val['items']['rows']:,} rows). Because the file has no reader-level data, <b>users</b> "
        f"({val['users']['rows']:,} rows) and <b>interactions</b> ({val['interactions']['rows']:,} rows, timestamped 2019-2024) are <b>synthetic and documented</b>: "
        f"books are chosen proportionally to real popularity and taste match, and every rating is drawn from the book's <i>real</i> 1-5 star histogram. "
        f"About {cfg['fault_rate']:.0%} of the rows carry injected faults (duplicates, invalid ratings, orphan keys, bad timestamps) so that validation can be demonstrated "
        f"and verified against a ledger. Model scores therefore describe the simulator, <b>not</b> real Goodreads behaviour; the engineering is what transfers.", BODY))

    # ------------------------------------------------------------------ 2
    S.append(Paragraph("2. Pipeline design and tools", H2))
    rows = [["Stage", "Tool / module", "Key design decision"],
            ["Ingestion", "pandas, src/ingestion", "Raw extracts persisted as untouched CSV (like an upstream export); everything read as text so nothing is silently coerced."],
            ["Validation", "Pandera + custom rules, src/validation", "Rules have severity: <i>reject</i> (-> error dataset) or <i>warning</i> (repaired later). Users -> items -> interactions order enables referential-integrity checks."],
            ["Preparation", "pandas, src/preparation", "Category-median imputation with <i>_imputed</i> flags; genre boilerplate removed; repeated (user,item) pairs keep the latest."],
            ["Transformation", "scipy.sparse, src/transformation", f"Time-based split at the 80th time-percentile ({split['cutoff'][:10]}), asserted leak-free; 15 % temporal validation fold for tuning."],
            ["Feature mgmt", "Parquet + FeatureStore class", "Features computed from TRAIN only at a fixed REF_DATE; one class builds training and prediction features (train/serve consistency)."],
            ["Training", "implicit, Surprise, NumPy, MLflow", "7 models incl. a popularity baseline; every trial is a nested MLflow run; models are plain-array objects."],
            ["Evaluation", "src/evaluation", "P/R/NDCG@5,10, Coverage, Diversity, RMSE, MAE; bootstrap CIs and paired significance tests over users."],
            ["Orchestration", "Python DAG engine + Airflow DAG", "One task registry drives both; retries, quality gate, resume/caching, run log and DAG-run graph."]]
    S.append(_table(rows, [2.3 * cm, 4.3 * cm, W - 6.6 * cm]))
    arch = P.reports / "architecture_diagram.png"
    if arch.exists():
        S.append(Spacer(1, 4)); S.append(_img(arch, W * 0.86))

    # ------------------------------------------------------------------ 3
    S.append(Paragraph("3. Data validation and preparation", H2))
    r = [["Table", "Rows", "Rejected", "Warning rows", "Main findings"],
         ["users", f"{val['users']['rows']:,}", f"{val['users']['rejected_rows']:,}", f"{val['users']['warning_rows']:,}", "duplicate user_id, future / unparsable signup_date; missing country, invalid age_group repaired"],
         ["items (real)", f"{val['items']['rows']:,}", f"{val['items']['rejected_rows']:,}", f"{val['items']['warning_rows']:,}", "near-empty rows, duplicate bookId; missing genres / pages / star histogram"],
         ["interactions", f"{val['interactions']['rows']:,}", f"{val['interactions']['rejected_rows']:,}", f"{val['interactions']['warning_rows']:,}", "bad dtype / range / null ratings, bad or future timestamps, orphan keys, duplicates, pre-signup events"]]
    S.append(_table(r, [2.1 * cm, 1.4 * cm, 1.6 * cm, 1.9 * cm, W - 7.0 * cm]))
    ok_txt = ""
    if det is not None:
        ok_txt = f" The detection check confirms that <b>{int((det['match'] == 'OK').sum())} of {len(det)}</b> injected fault families were found."
    S.append(Paragraph(f"Rejected rows are written to <i>data/errors/</i> (CSV + Parquet) with the violated rules; the full log and an HTML report are kept in <i>reports/</i>.{ok_txt} "
                       f"Validation also exposed genuine problems of the real file. Referential integrity cascades: rejecting a user orphans all of that user's interactions, which is why validation order matters.", BODY))

    # ------------------------------------------------------------------ 4
    n = cat.groupby("entity").size().to_dict()
    S.append(Paragraph("4. Feature management", H2))
    S.append(Paragraph(f"The feature catalog ({len(cat)} documented entries: {n.get('user', 0)} user, {n.get('item', 0)} item, {n.get('interaction', 0)} interaction) lists name, type, definition, source and leakage guard "
                       f"(<i>reports/feature_catalog.csv|md</i>). Examples: user average rating, activity count, preferred category, days since last interaction; item popularity, Bayesian average rating, "
                       f"top audience age group; user-category affinity and item-vs-user rating gap. Cold-start users ({split['cold_start_users']} of {split['test_users']:,} test users) "
                       f"have no features by construction and are served a popularity fallback.", BODY))

    # ------------------------------------------------------------------ 5
    S.append(Paragraph("5. Models and results", H2))
    cols = [f"Precision@{K}", f"Recall@{K}", f"NDCG@{K}", f"Coverage@{K}", f"Diversity@{K}", "RMSE", "MAE"]
    t = [["Model"] + [c.replace("@", "@") for c in cols]]
    for nme in res.index:
        t.append([f"<b>{nme}</b>" if nme == best else nme] + ["n/a" if np.isnan(res.loc[nme, c]) else f"{res.loc[nme, c]:.4f}" for c in cols])
    S.append(_table(t, [5.0 * cm] + [(W - 5.0 * cm) / len(cols)] * len(cols)))
    S.append(Paragraph(f"<i>Test window, evaluated on {n_eval:,} users with at least one relevant test item; K = {K}; best personalised model in bold.</i>", SUB))
    txt = (f"<b>{best}</b> is the best ranker (NDCG@{K} {lift:+.0%} vs. the popularity baseline). ")
    if hyb and knn:
        h = sig[(sig.model == hyb) & (sig.versus == knn)]
        if len(h):
            tied = not bool(h["significant_95"].iloc[0])
            txt += (f"The hybrid is {'statistically tied with' if tied else 'significantly different from'} ItemKNN on NDCG@{K} (paired bootstrap 95 % CI "
                    f"[{h['ci_low'].iloc[0]:+.4f}, {h['ci_high'].iloc[0]:+.4f}]) but gives more diverse lists (diversity {res.loc[hyb, f'Diversity@{K}']:.2f} vs. {res.loc[knn, f'Diversity@{K}']:.2f}). ")
    svd = next((nme for nme in res.index if nme.startswith("SVD")), None)
    if svd:
        txt += (f"SVD improves rating error only marginally (RMSE {res.loc[svd, 'RMSE']:.3f} vs. {res.loc[base_n, 'RMSE']:.3f}) and, used as a ranker, falls below random: "
                f"rating accuracy and top-K usefulness are different objectives. Popularity covers {res.loc[base_n, f'Coverage@{K}']:.1%} of the catalogue, personalised models far more.")
    S.append(Paragraph(txt, BODY))
    fig = P.reports / "results_overview.png"
    if fig.exists():
        S.append(_img(fig, W))

    # ------------------------------------------------------------------ 6
    S.append(Paragraph("6. Orchestration and reproducibility", H2))
    if run:
        ok = sum(t_["status"] in ("success", "cached") for t_ in run["tasks"])
        S.append(Paragraph(f"The workflow has {len(run['tasks'])} tasks with explicit dependencies (ingest -> validate -> <b>quality gate</b> -> prepare -> transform -> features -> 7 training tasks -> evaluate -> batch recommend -> publish). "
                           f"Latest run <i>{run['run_id']}</i>: <b>{run['status']}</b>, {ok}/{len(run['tasks'])} tasks successful, {run['seconds']:.0f} s. "
                           f"The gate stops the pipeline (downstream tasks are <i>skipped</i>) if more than {cfg['max_reject_ratio']:.0%} of interaction rows are rejected; failed tasks are retried; "
                           f"<i>resume</i> mode skips tasks that already succeeded with an identical configuration. All seeds are fixed, and the metrics of the orchestrated run are compared with the "
                           f"stage-by-stage run in the notebook. The same task registry is exported as an Apache Airflow DAG (<i>dags/recommendation_pipeline_dag.py</i>).", BODY))
        g = P.reports / "orchestration_run.png"
        if g.exists():
            S.append(_img(g, W))
    else:
        S.append(Paragraph("Run the orchestrated workflow to populate this section.", BODY))

    # ------------------------------------------------------------------ 7
    S.append(Paragraph("7. Limitations and future work", H2))
    for b in ["<b>Simulated users / interactions:</b> conclusions about model ranking hold for this simulator; the pipeline itself transfers to real logs.",
              "<b>Editions:</b> different editions of one book have separate ids (e.g. several Harry Potter editions in the popularity list); entity resolution to a work id is needed in production.",
              "<b>Offline evaluation only:</b> one temporal split, no A/B test; absolute scores are low because the matrix is ~99 % sparse and only the future window is scored.",
              "<b>Content features</b> are used only through the genre profile in the hybrid; richer text / author embeddings, cold-start modelling and a hosted feature store / MLflow server are natural extensions."]:
        S.append(Paragraph(b, BUL, bulletText="\u2022"))
    doc.build(S)
    return out

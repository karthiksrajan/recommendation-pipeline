"""Stage 3 - Data preparation: cleaning, standardisation, imputation (with *_imputed flags), de-duplication."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.genres import GenreParser
from src.common.log import get_logger
from src.config import Paths
from src.validation.validate import AGE_GROUPS, HIST_COLS, parse_interactions

log = get_logger("preparation")


def _good_rows(error_log: pd.DataFrame, table: str, n: int) -> np.ndarray:
    rejected = error_log.loc[(error_log.table == table) & (error_log.severity == "reject"), "row_index"].unique()
    return np.setdiff1d(np.arange(n), rejected)


def clean_items(items_raw, good, parser, end_year):
    ic = items_raw.iloc[good].copy()
    ic["genres_list"] = ic["genres"].map(parser.parse)
    ic["category"] = ic["genres_list"].map(parser.primary).fillna("Unknown")
    ic["genres"] = ic["genres_list"].map("|".join)
    ic["title"] = ic["title"].str.strip()
    ic["primary_author"] = ic["author"].fillna("Unknown").str.split(",").str[0].str.strip()
    ic["language"] = ic["language"].fillna("Unknown").str.strip().str.title()
    ic["num_pages_imputed"] = ic["num_pages"].isna() | (ic["num_pages"] < 1)
    ic["num_pages"] = ic["num_pages"].where(~ic["num_pages_imputed"])
    ic["num_pages"] = ic["num_pages"].fillna(ic.groupby("category")["num_pages"].transform("median")).fillna(ic["num_pages"].median()).round()
    yr = ic["first_publish_date"].str.extract(r"(\d{4})")[0].fillna(ic["publish_date"].str.extract(r"(\d{4})")[0]).astype(float)
    yr = yr.where(yr.between(1000, end_year))
    ic["publish_year_imputed"] = yr.isna()
    ic["publish_year"] = yr.fillna(yr.groupby(ic["category"]).transform("median")).fillna(yr.median()).astype(int)
    ic["has_series"] = ic["series"].notna()
    ic["num_awards"] = ic["awards"].fillna("").map(lambda s: len([a for a in s.split(",") if a.strip()]))
    ic["n_genres"] = ic["genres_list"].map(len)
    out = ic.rename(columns={"bookId": "item_id", "avg_rating": "gr_avg_rating", "num_ratings": "gr_num_ratings"})
    out["gr_log_num_ratings"] = np.log1p(out["gr_num_ratings"])
    keep = ["item_id", "title", "primary_author", "category", "genres", "n_genres", "language", "num_pages", "num_pages_imputed", "publish_year",
            "publish_year_imputed", "has_series", "num_awards", "gr_avg_rating", "gr_num_ratings", "gr_log_num_ratings"] + HIST_COLS
    out = out[keep].reset_index(drop=True)
    out["item_id"] = out["item_id"].astype(int)
    return out


def clean_users(users_raw, good):
    uc = users_raw.iloc[good].copy()
    uc["signup_date"] = pd.to_datetime(uc["signup_date"], errors="coerce", format="%Y-%m-%d")
    uc["country"] = uc["country"].fillna("Unknown")
    uc["age_group"] = uc["age_group"].where(uc["age_group"].isin(AGE_GROUPS), "Unknown")
    return uc.reset_index(drop=True)


def clean_interactions(inter_raw, good):
    xt = parse_interactions(inter_raw)
    ic = xt.iloc[good][["interaction_id", "user_id", "item_id", "rating", "timestamp"]].copy()
    ic["item_id"] = ic["item_id"].astype(int)
    ic["rating"] = ic["rating"].astype(int)
    before = len(ic)
    ic = ic.sort_values("timestamp").drop_duplicates(["user_id", "item_id"], keep="last").reset_index(drop=True)
    return ic, before - len(ic)


def run(base, cfg) -> dict:
    P = Paths(base).ensure()
    items_raw = pd.read_csv(P.raw / "items_raw.csv")
    users_raw = pd.read_csv(P.raw / "users_raw.csv", dtype=str)
    inter_raw = pd.read_csv(P.raw / "interactions_raw.csv", dtype=str)
    error_log = pd.read_csv(P.errors / "validation_error_log.csv")
    parser = GenreParser.fit(items_raw["genres"])

    items_clean = clean_items(items_raw, _good_rows(error_log, "items", len(items_raw)), parser, int(cfg["end_date"][:4]))
    users_clean = clean_users(users_raw, _good_rows(error_log, "users", len(users_raw)))
    inter_clean, n_repeat = clean_interactions(inter_raw, _good_rows(error_log, "interactions", len(inter_raw)))
    log.info(f"repeated (user,item) pairs removed (kept latest): {n_repeat}")
    for name, df in [("items_clean", items_clean), ("users_clean", users_clean), ("interactions_clean", inter_clean)]:
        df.to_parquet(P.processed / f"{name}.parquet", index=False)
        nulls = int(df.drop(columns=[c for c in df.columns if c.startswith("rated_")]).isna().sum().sum())
        log.info(f"{name:20s} {df.shape}  nulls (excl. optional star-histogram cols) = {nulls}")
    return {"items_clean": len(items_clean), "users_clean": len(users_clean), "interactions_clean": len(inter_clean), "repeated_pairs_removed": int(n_repeat)}

"""Stage 4 - Transformation: leakage-free time-based split + derived datasets (all derived from TRAIN only)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.common.log import get_logger
from src.config import Paths

log = get_logger("transformation")


def time_split(inter_clean: pd.DataFrame, test_fraction: float, val_fraction: float):
    """train = before the (1 - test_fraction) time-quantile; validation fold = last val_fraction of train (by time)."""
    cutoff = inter_clean["timestamp"].quantile(1 - test_fraction)
    train = inter_clean[inter_clean["timestamp"] < cutoff].reset_index(drop=True)
    test = inter_clean[inter_clean["timestamp"] >= cutoff].reset_index(drop=True)
    val_cut = train["timestamp"].quantile(1 - val_fraction)
    train_fit = train[train["timestamp"] < val_cut].reset_index(drop=True)
    val = train[train["timestamp"] >= val_cut].reset_index(drop=True)
    assert train.timestamp.max() < test.timestamp.min(), "temporal leakage!"
    return cutoff, val_cut, train, test, train_fit, val


def derive_datasets(train, items_clean, users_clean, pos, prior):
    tr = (train.merge(items_clean[["item_id", "category"]], on="item_id", how="left")
               .merge(users_clean[["user_id", "age_group"]], on="user_id", how="left"))
    tr["is_pos"] = (tr["rating"] >= pos).astype(int)
    global_mean = tr["rating"].mean()

    ua = tr.groupby("user_id").agg(n_interactions=("rating", "size"), avg_rating=("rating", "mean"), rating_std=("rating", "std"),
                                   n_positive=("is_pos", "sum"), n_categories=("category", "nunique"),
                                   first_interaction=("timestamp", "min"), last_interaction=("timestamp", "max")).reset_index()
    ua["positive_ratio"] = ua["n_positive"] / ua["n_interactions"]
    ua["active_span_days"] = (ua["last_interaction"] - ua["first_interaction"]).dt.days
    ua["activity_segment"] = pd.qcut(ua["n_interactions"].rank(method="first"), 4, labels=["light", "medium", "active", "power"])

    ip = tr.groupby("item_id").agg(n_interactions=("rating", "size"), n_users=("user_id", "nunique"), n_positive=("is_pos", "sum"),
                                   avg_rating=("rating", "mean")).reset_index()
    ip["bayes_avg_rating"] = (prior * global_mean + ip["avg_rating"] * ip["n_interactions"]) / (prior + ip["n_interactions"])
    ip["popularity_score"] = np.log1p(ip["n_interactions"]) / np.log1p(ip["n_interactions"].max())
    ip["popularity_rank"] = ip["n_interactions"].rank(ascending=False, method="min").astype(int)
    ip = ip.merge(items_clean[["item_id", "title", "category"]], on="item_id", how="left").sort_values("popularity_rank")

    u_ids, i_ids = np.sort(train["user_id"].unique()), np.sort(train["item_id"].unique())
    u_map, i_map = {u: k for k, u in enumerate(u_ids)}, {i: k for k, i in enumerate(i_ids)}
    R = sp.csr_matrix((train["rating"].astype(float), (train["user_id"].map(u_map), train["item_id"].map(i_map))), shape=(len(u_ids), len(i_ids)))
    return ua, ip, R, u_ids, i_ids


def run(base, cfg) -> dict:
    P = Paths(base).ensure()
    items_clean = pd.read_parquet(P.processed / "items_clean.parquet")
    users_clean = pd.read_parquet(P.processed / "users_clean.parquet")
    inter_clean = pd.read_parquet(P.processed / "interactions_clean.parquet")

    cutoff, val_cut, train, test, train_fit, val = time_split(inter_clean, cfg["test_fraction"], cfg["val_fraction"])
    cold_users, cold_items = set(test.user_id) - set(train.user_id), set(test.item_id) - set(train.item_id)
    info = dict(cutoff=str(cutoff), val_cut=str(val_cut), n_train=len(train), n_test=len(test), n_train_fit=len(train_fit), n_val=len(val),
                cold_start_users=len(cold_users), test_users=int(test.user_id.nunique()), cold_items=len(cold_items), test_items=int(test.item_id.nunique()))
    log.info(f"cutoff {cutoff} | validation cut {val_cut}")
    log.info(f"train {len(train):,} | test {len(test):,} | (train_fit {len(train_fit):,} / val {len(val):,})")
    log.info(f"test users never seen in train (cold-start): {len(cold_users)} of {info['test_users']} | test items unseen in train: {len(cold_items)} of {info['test_items']}")
    train.to_parquet(P.processed / "train.parquet", index=False)
    test.to_parquet(P.processed / "test.parquet", index=False)

    ua, ip, R, u_ids, i_ids = derive_datasets(train, items_clean, users_clean, cfg["positive_threshold"], cfg["bayes_prior"])
    sp.save_npz(P.processed / "user_item_matrix_train.npz", R)
    pd.DataFrame({"user_id": u_ids}).to_parquet(P.processed / "matrix_users.parquet")
    pd.DataFrame({"item_id": i_ids}).to_parquet(P.processed / "matrix_items.parquet")
    ua.to_parquet(P.processed / "user_activity_summary.parquet", index=False)
    ip.to_parquet(P.processed / "item_popularity.parquet", index=False)
    sparsity = 1 - R.nnz / (R.shape[0] * R.shape[1])
    info["sparsity"] = float(sparsity)
    (P.processed / "split_info.json").write_text(json.dumps(info, indent=2))
    log.info(f"user-item matrix: {R.shape[0]:,} users x {R.shape[1]:,} items | nnz={R.nnz:,} | sparsity={sparsity:.2%}")
    return {**info, "matrix_users": R.shape[0], "matrix_items": R.shape[1]}

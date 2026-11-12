"""Evaluation protocol: ranking metrics (Precision / Recall / NDCG @K, Coverage, Diversity), rating metrics (RMSE, MAE)
and bootstrap confidence intervals.

Protocol
* relevant item  = a test-window rating >= 4 for a book that exists in the training catalogue
* candidates     = every book seen in train minus the books the user already rated in train
* evaluated users= users with >= 1 train interaction AND >= 1 relevant test item (cold-start users are reported, not scored)
* Coverage@K     = share of the candidate catalogue appearing in ANY user's top-K list
* Diversity@K    = mean intra-list distance, 1 - cosine(genre vector), over all pairs in a user's top-K list
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.common.genres import NON_CATEGORY


def build_vocab(items_clean: pd.DataFrame, top: int = 80):
    gcount = Counter(g for s in items_clean["genres"] for g in s.split("|") if g)
    vocab = [g for g, _ in gcount.most_common(top) if g not in NON_CATEGORY]
    return vocab


class RecData:
    """Index-mapped view of a (train, eval) pair: sparse matrices, relevance sets and genre vectors."""

    def __init__(self, train_df, eval_df, items_clean, pos=4, vocab=None):
        vocab = vocab or build_vocab(items_clean)
        v_idx = {g: k for k, g in enumerate(vocab)}
        self.vocab = vocab
        self.users, self.items = np.sort(train_df["user_id"].unique()), np.sort(train_df["item_id"].unique())
        self.u_map = {u: k for k, u in enumerate(self.users)}
        self.i_map = {i: k for k, i in enumerate(self.items)}
        self.n_users, self.n_items = len(self.users), len(self.items)
        self.tu, self.ti = train_df["user_id"].map(self.u_map).to_numpy(), train_df["item_id"].map(self.i_map).to_numpy()
        self.tr_rating = train_df["rating"].to_numpy(float)
        shape = (self.n_users, self.n_items)
        self.R = sp.csr_matrix((self.tr_rating, (self.tu, self.ti)), shape=shape)
        self.P = sp.csr_matrix(((self.tr_rating >= pos).astype(float), (self.tu, self.ti)), shape=shape)
        self.seen = sp.csr_matrix((np.ones(len(self.tu)), (self.tu, self.ti)), shape=shape)
        e = eval_df[eval_df["user_id"].isin(self.u_map) & eval_df["item_id"].isin(self.i_map)]
        self.eu, self.ei, self.er = e["user_id"].map(self.u_map).to_numpy(), e["item_id"].map(self.i_map).to_numpy(), e["rating"].to_numpy(float)
        m = self.er >= pos
        self.T = sp.csr_matrix((np.ones(m.sum()), (self.eu[m], self.ei[m])), shape=shape)
        self.eval_users = np.flatnonzero(np.asarray(self.T.sum(axis=1)).ravel() > 0)
        n_pos_all = int((eval_df["rating"] >= pos).sum())
        self.info = dict(n_train=len(train_df), n_users=self.n_users, n_items=self.n_items, n_eval_users=len(self.eval_users),
                         n_eval_rows=len(self.er), eval_positives_in_catalog=int(m.sum()), eval_positives_total=n_pos_all)
        by_id = items_clean.set_index("item_id")["genres"]
        G = np.zeros((self.n_items, len(vocab)), dtype=np.float32)
        for r, iid in enumerate(self.items):
            for g in by_id.at[iid].split("|"):
                if g in v_idx:
                    G[r, v_idx[g]] = 1
        self.G = G / np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-9)


def evaluate(model, d: RecData, ks=(5, 10), per_user=False):
    """Score every evaluated user against every candidate item and compute all metrics."""
    ks = list(ks)
    users = d.eval_users
    S = np.asarray(model.score(users), dtype=np.float64)
    S[d.seen[users].toarray() > 0] = -np.inf                       # never re-recommend seen items
    Kmax = max(ks)
    top = np.argpartition(-S, Kmax, axis=1)[:, :Kmax]
    top = np.take_along_axis(top, np.argsort(-np.take_along_axis(S, top, 1), axis=1), 1)
    rel = d.T[users].toarray() > 0
    n_rel = rel.sum(1)
    out, pu = {}, {}
    for K in ks:
        t = top[:, :K]
        hits = np.take_along_axis(rel, t, 1)
        disc = 1 / np.log2(np.arange(2, K + 2))
        ndcg = (hits * disc).sum(1) / np.cumsum(disc)[np.minimum(n_rel, K) - 1]
        Gt = d.G[t]
        sim = np.einsum("nkv,nlv->nkl", Gt, Gt)
        div = 1 - (sim.sum((1, 2)) - np.trace(sim, axis1=1, axis2=2)) / (K * (K - 1))
        prec, rec = hits.sum(1) / K, hits.sum(1) / n_rel
        out[f"Precision@{K}"], out[f"Recall@{K}"], out[f"NDCG@{K}"] = float(prec.mean()), float(rec.mean()), float(ndcg.mean())
        out[f"Coverage@{K}"] = len(np.unique(t)) / d.n_items
        out[f"Diversity@{K}"] = float(div.mean())
        pu[f"precision@{K}"], pu[f"recall@{K}"], pu[f"ndcg@{K}"] = prec, rec, ndcg
    if hasattr(model, "predict"):
        p = np.clip(model.predict(d.eu, d.ei), 1, 5)
        out["RMSE"] = float(np.sqrt(np.mean((p - d.er) ** 2)))
        out["MAE"] = float(np.mean(np.abs(p - d.er)))
    else:
        out["RMSE"] = out["MAE"] = float("nan")
    return (out, pu) if per_user else out


def bootstrap_ci(x, n_boot=2000, seed=0, alpha=0.05):
    """Mean and percentile CI of a per-user metric (resampling users)."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap(a, b, n_boot=2000, seed=0, alpha=0.05):
    """CI of mean(a - b) over the same users + share of resamples where a <= b (one-sided bootstrap p-value)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return float(d.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)), float((means <= 0).mean())

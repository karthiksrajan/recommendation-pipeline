"""Prediction-time layer: top-K recommendations from the best personalised model, features from the SAME FeatureStore,
and a popularity fallback for cold-start users (users with no training history)."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from src.common.log import get_logger
from src.config import Paths
from src.features.feature_store import FeatureStore
from src.training.train import ORDER, REGISTRY

log = get_logger("serving")


class Recommender:
    def __init__(self, base, model_key: str | None = None):
        self.P = Paths(base)
        self.cfg = json.loads((self.P.configs / "pipeline_config.json").read_text())
        K = max(self.cfg["ks"])
        meta = {k: json.loads((self.P.metrics / f"{k}.json").read_text()) for k in ORDER if (self.P.metrics / f"{k}.json").exists()}
        if model_key is None:                       # best personalised model by NDCG@K on the test window
            cand = {k: m["metrics"][f"NDCG@{K}"] for k, m in meta.items() if k not in ("random", "popularity", "svd")}
            model_key = max(cand, key=cand.get)
        self.model_key, self.model_name = model_key, meta[model_key]["name"]
        self.model = joblib.load(self.P.base / meta[model_key]["model_path"])
        self.popularity = joblib.load(self.P.base / meta["popularity"]["model_path"])
        self.items = pd.read_parquet(self.P.processed / "items_clean.parquet").set_index("item_id")
        train = pd.read_parquet(self.P.processed / "train.parquet")
        self.users, self.item_ids = np.sort(train["user_id"].unique()), np.sort(train["item_id"].unique())
        self.u_map = {u: k for k, u in enumerate(self.users)}
        import scipy.sparse as sp
        self.seen = sp.csr_matrix((np.ones(len(train)), (train["user_id"].map(self.u_map), train["item_id"].map({i: k for k, i in enumerate(self.item_ids)}))),
                                  shape=(len(self.users), len(self.item_ids)))
        self.fs = FeatureStore(self.P.features)

    def _format(self, idx, scores, source):
        iid = self.item_ids[idx]
        meta = self.items.loc[iid, ["title", "primary_author", "category"]].reset_index(drop=True)
        return pd.DataFrame({"rank": np.arange(1, len(iid) + 1), "item_id": iid, **meta.to_dict("list"), "score": scores, "source": source})

    def user_context(self, user_id) -> dict:
        """Prediction-time features of a user (identical definitions to the training features)."""
        f = self.fs.get_user_features([user_id])
        if f["user_avg_rating"].isna().all():
            return {"user_id": user_id, "cold_start": True}
        r = f.iloc[0]
        return {"user_id": user_id, "cold_start": False, "preferred_category": r["user_preferred_category"],
                "activity_count": int(r["user_activity_count"]), "avg_rating": round(float(r["user_avg_rating"]), 2), "segment": r["activity_segment"]}

    def recommend(self, user_id, k: int = 10) -> pd.DataFrame:
        if user_id not in self.u_map:               # cold start -> non-personalised popularity fallback
            s = self.popularity.pop.astype(float)
            top = np.argsort(-s)[:k]
            return self._format(top, s[top], "popularity fallback (cold start)")
        ui = self.u_map[user_id]
        s = self.model.score(np.array([ui]))[0].astype(float)
        s[self.seen[ui].indices] = -np.inf
        top = np.argsort(-s)[:k]
        return self._format(top, s[top], self.model_name)

    def batch(self, k: int = 10, chunk: int = 500) -> pd.DataFrame:
        out = []
        for a in range(0, len(self.users), chunk):
            ui = np.arange(a, min(a + chunk, len(self.users)))
            S = np.asarray(self.model.score(ui), dtype=float)
            S[self.seen[ui].toarray() > 0] = -np.inf
            top = np.argpartition(-S, k, axis=1)[:, :k]
            top = np.take_along_axis(top, np.argsort(-np.take_along_axis(S, top, 1), axis=1), 1)
            for r, u in enumerate(ui):
                out.append(pd.DataFrame({"user_id": self.users[u], "rank": np.arange(1, k + 1), "item_id": self.item_ids[top[r]]}))
        df = pd.concat(out, ignore_index=True)
        return df.merge(self.items[["title", "category"]].reset_index(), on="item_id", how="left")


def run(base, cfg) -> dict:
    """Batch prediction task: top-10 list for every user with a training history."""
    P = Paths(base).ensure()
    rec = Recommender(base)
    df = rec.batch(k=10)
    df.to_parquet(P.reports / "top10_recommendations.parquet", index=False)
    df.head(500).to_csv(P.reports / "top10_recommendations_sample.csv", index=False)
    log.info(f"batch recommendations ({rec.model_name}): {df['user_id'].nunique():,} users x 10 -> reports/top10_recommendations.parquet")
    return {"users": int(df["user_id"].nunique()), "model": rec.model_name}

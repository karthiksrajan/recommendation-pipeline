"""Stage 5 - Feature management.

Features are computed from the TRAINING window only, relative to a fixed reference date (REF_DATE = train/test cut-off),
so there is no look-ahead. They are persisted as Parquet, documented in a catalog and served through `FeatureStore`:
the SAME class and code path produce features for training rows and for prediction-time rows (train/serve consistency).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.common.log import get_logger
from src.config import Paths

log = get_logger("features")

CATALOG_ROWS = [
    ("user_avg_rating", "user", "float", "Mean rating given by the user in the training window", "interactions.rating", "train only"),
    ("user_rating_std", "user", "float", "Std-dev of the user's ratings (0 if one rating)", "interactions.rating", "train only"),
    ("user_activity_count", "user", "int", "Number of training interactions", "interactions", "train only"),
    ("user_positive_ratio", "user", "float", "Share of the user's ratings >= 4", "interactions.rating", "train only"),
    ("user_days_since_last_interaction", "user", "float", "Days between the user's last train interaction and REF_DATE", "interactions.timestamp", "REF_DATE = train/test cut-off"),
    ("user_tenure_days", "user", "int", "Days between signup and REF_DATE", "users.signup_date", "REF_DATE"),
    ("user_n_categories", "user", "int", "# distinct item categories the user interacted with", "interactions x items.category", "train only"),
    ("user_preferred_category", "user", "category", "Most frequent category among the user's positive (>=4) ratings", "interactions x items.category", "train only"),
    ("country / age_group", "user", "category", "Demographics (after cleaning, 'Unknown' when missing)", "users", "static"),
    ("activity_segment", "user", "category", "Quartile of activity count: light/medium/active/power", "interactions", "train only"),
    ("item_avg_rating", "item", "float", "Mean rating received in the training window (global mean if cold)", "interactions.rating", "train only"),
    ("item_bayes_avg_rating", "item", "float", "Shrunk mean: (C*m + sum)/(C + n), C=20", "interactions.rating", "train only"),
    ("item_popularity", "item", "int", "# training interactions (0 for cold items)", "interactions", "train only"),
    ("item_popularity_score", "item", "float", "log1p(count)/log1p(max count) in [0,1]", "interactions", "train only"),
    ("item_positive_rate", "item", "float", "Share of the item's ratings >= 4", "interactions.rating", "train only"),
    ("item_days_since_last_interaction", "item", "float", "Days between the item's last train interaction and REF_DATE", "interactions.timestamp", "REF_DATE"),
    ("item_category", "item", "category", "Primary genre (first non-generic genre tag)", "items.genres", "static"),
    ("item_n_genres", "item", "int", "# distinct genre tags", "items.genres", "static"),
    ("item_top_audience_age_group", "item", "category", "Age group that rates the item most", "interactions x users", "train only"),
    ("item_num_pages / item_publish_year", "item", "int", "Pages and year (median-imputed per category when missing/invalid)", "items", "static"),
    ("item_gr_avg_rating / item_gr_log_num_ratings", "item", "float", "Goodreads global rating and log #ratings (external prior)", "items", "static"),
    ("user_category_affinity", "interaction", "float", "Share of the user's train interactions in the item's category", "user_category_share", "train only"),
    ("is_preferred_category", "interaction", "int", "1 if item category == user preferred category", "user x item features", "train only"),
    ("item_vs_user_avg_gap", "interaction", "float", "item_bayes_avg_rating - user_avg_rating", "user x item features", "train only"),
    ("item_age_years_at_interaction", "interaction", "int", "Interaction year - publication year", "interactions x items", "static"),
]


def _mode_first(df, key, col, name):
    return (df.groupby([key, col]).size().rename("c").reset_index()
              .sort_values([key, "c", col], ascending=[True, False, True]).drop_duplicates(key)
              .set_index(key)[col].rename(name))


def build_user_item_features(train, items_clean, users_clean, ua, ip, ref_date, pos):
    tr = (train.merge(items_clean[["item_id", "category"]], on="item_id", how="left")
               .merge(users_clean[["user_id", "age_group"]], on="user_id", how="left"))
    tr["is_pos"] = (tr["rating"] >= pos).astype(int)
    global_mean = tr["rating"].mean()

    # ---- user features
    pref = _mode_first(tr[tr.is_pos == 1], "user_id", "category", "user_preferred_category")
    uf = ua.set_index("user_id").join(pref)
    uf["user_preferred_category"] = uf["user_preferred_category"].fillna("Unknown")
    uf = uf.rename(columns={"avg_rating": "user_avg_rating", "rating_std": "user_rating_std", "n_interactions": "user_activity_count",
                            "positive_ratio": "user_positive_ratio", "n_categories": "user_n_categories"})
    uf["user_rating_std"] = uf["user_rating_std"].fillna(0)
    uf["user_days_since_last_interaction"] = (ref_date - uf["last_interaction"]).dt.total_seconds() / 86400
    uf = uf.join(users_clean.set_index("user_id")[["signup_date", "country", "age_group"]])
    uf["user_tenure_days"] = (ref_date - uf["signup_date"]).dt.days
    user_features = uf.reset_index()[["user_id", "user_avg_rating", "user_rating_std", "user_activity_count", "user_positive_ratio",
                                      "user_days_since_last_interaction", "user_tenure_days", "user_n_categories",
                                      "user_preferred_category", "country", "age_group", "activity_segment"]]
    user_features["activity_segment"] = user_features["activity_segment"].astype(str)

    # ---- item features
    audience = _mode_first(tr.dropna(subset=["age_group"]), "item_id", "age_group", "item_top_audience_age_group")
    last_seen = tr.groupby("item_id")["timestamp"].max()
    itf = items_clean.set_index("item_id")[["category", "n_genres", "num_pages", "publish_year", "gr_avg_rating", "gr_log_num_ratings"]]
    itf = itf.join(ip.set_index("item_id")[["n_interactions", "avg_rating", "bayes_avg_rating", "popularity_score", "n_positive", "n_users"]])
    itf = itf.join(audience)
    itf["item_days_since_last_interaction"] = (ref_date - last_seen.reindex(itf.index)).dt.total_seconds() / 86400
    itf["item_popularity"] = itf["n_interactions"].fillna(0).astype(int)                 # cold items -> popularity 0
    itf["item_popularity_score"] = itf["popularity_score"].fillna(0.0)
    itf["item_avg_rating"] = itf["avg_rating"].fillna(global_mean)
    itf["item_bayes_avg_rating"] = itf["bayes_avg_rating"].fillna(global_mean)
    itf["item_positive_rate"] = (itf["n_positive"] / itf["n_interactions"]).fillna(0)
    itf["item_top_audience_age_group"] = itf["item_top_audience_age_group"].fillna("Unknown")
    itf = itf.rename(columns={"category": "item_category", "n_genres": "item_n_genres", "num_pages": "item_num_pages",
                              "publish_year": "item_publish_year", "gr_avg_rating": "item_gr_avg_rating", "gr_log_num_ratings": "item_gr_log_num_ratings"})
    item_features = itf.reset_index()[["item_id", "item_avg_rating", "item_bayes_avg_rating", "item_popularity", "item_popularity_score",
                                       "item_positive_rate", "item_days_since_last_interaction", "item_category", "item_n_genres",
                                       "item_top_audience_age_group", "item_num_pages", "item_publish_year", "item_gr_avg_rating", "item_gr_log_num_ratings"]]

    # ---- user x category share (used for interaction features at train AND prediction time)
    cat_share = tr.groupby(["user_id", "category"]).size().rename("n").reset_index()
    cat_share["user_category_share"] = cat_share["n"] / cat_share.groupby("user_id")["n"].transform("sum")
    return user_features, item_features, cat_share.drop(columns="n")


class FeatureStore:
    """Parquet-backed feature layer. Same code path for training frames and online / batch prediction."""

    def __init__(self, root):
        root = Path(root)
        self.users = pd.read_parquet(root / "user_features.parquet").set_index("user_id")
        self.items = pd.read_parquet(root / "item_features.parquet").set_index("item_id")
        self.cat_share = pd.read_parquet(root / "user_category_share.parquet")

    def get_user_features(self, user_ids, cols=None):
        out = self.users.reindex(user_ids)
        return out if cols is None else out[cols]

    def get_item_features(self, item_ids, cols=None):
        out = self.items.reindex(item_ids)
        return out if cols is None else out[cols]

    def build_interaction_features(self, inter):
        """`inter` needs user_id, item_id (and optionally timestamp)."""
        df = inter[["user_id", "item_id"] + (["timestamp"] if "timestamp" in inter else [])].copy()
        u = self.get_user_features(df["user_id"]).reset_index(drop=True)
        i = self.get_item_features(df["item_id"]).reset_index(drop=True)
        df = pd.concat([df.reset_index(drop=True), u, i], axis=1)
        df = df.merge(self.cat_share.rename(columns={"category": "item_category"}), on=["user_id", "item_category"], how="left")
        df["user_category_affinity"] = df["user_category_share"].fillna(0.0)
        df["is_preferred_category"] = (df["item_category"] == df["user_preferred_category"]).astype(int)
        df["item_vs_user_avg_gap"] = df["item_bayes_avg_rating"] - df["user_avg_rating"]
        if "timestamp" in df:
            df["item_age_years_at_interaction"] = df["timestamp"].dt.year - df["item_publish_year"]
        return df.drop(columns=["user_category_share"])


def feature_catalog() -> pd.DataFrame:
    return pd.DataFrame(CATALOG_ROWS, columns=["feature", "entity", "dtype", "description", "source", "leakage_guard"])


def write_catalog(P):
    cat = feature_catalog()
    cat.to_csv(P.reports / "feature_catalog.csv", index=False)
    with open(P.reports / "feature_catalog.md", "w") as f:
        f.write("| " + " | ".join(cat.columns) + " |\n|" + "---|" * cat.shape[1] + "\n")
        for r in cat.itertuples(index=False):
            f.write("| " + " | ".join(map(str, r)) + " |\n")
    return cat


def run(base, cfg) -> dict:
    P = Paths(base).ensure()
    items_clean = pd.read_parquet(P.processed / "items_clean.parquet")
    users_clean = pd.read_parquet(P.processed / "users_clean.parquet")
    train = pd.read_parquet(P.processed / "train.parquet")
    test = pd.read_parquet(P.processed / "test.parquet")
    ua = pd.read_parquet(P.processed / "user_activity_summary.parquet")
    ip = pd.read_parquet(P.processed / "item_popularity.parquet")
    ref_date = pd.Timestamp(json.loads((P.processed / "split_info.json").read_text())["cutoff"])

    uf, itf, cat_share = build_user_item_features(train, items_clean, users_clean, ua, ip, ref_date, cfg["positive_threshold"])
    uf.to_parquet(P.features / "user_features.parquet", index=False)
    itf.to_parquet(P.features / "item_features.parquet", index=False)
    cat_share.to_parquet(P.features / "user_category_share.parquet", index=False)
    log.info(f"user_features {uf.shape} | item_features {itf.shape} | user_category_share {cat_share.shape}")

    fs = FeatureStore(P.features)
    train_feat = fs.build_interaction_features(train).assign(rating=train["rating"].to_numpy())
    test_feat = fs.build_interaction_features(test).assign(rating=test["rating"].to_numpy())
    train_feat.to_parquet(P.features / "interaction_features_train.parquet", index=False)
    test_feat.to_parquet(P.features / "interaction_features_test.parquet", index=False)
    cold = int(test_feat["user_avg_rating"].isna().sum())
    log.info(f"training frame {train_feat.shape} | test frame {test_feat.shape} (same code path, train-derived stats) | cold-start test rows: {cold}")
    cat = write_catalog(P)
    return {"n_user_features": uf.shape[1] - 1, "n_item_features": itf.shape[1] - 1, "catalog_rows": len(cat), "cold_start_test_rows": cold}

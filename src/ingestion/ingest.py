"""Stage 1 - Data ingestion.

* loads the REAL Goodreads items file,
* simulates the `users` and `interactions` source extracts (documented simulator, ratings drawn from each
  book's real 1-5 star histogram),
* injects a small, reproducible fraction of faults so that the validation layer can be demonstrated,
* writes the three messy extracts to data/raw/ exactly like an upstream system export.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.genres import GenreParser
from src.common.log import get_logger
from src.config import Paths

log = get_logger("ingestion")


def load_items(csv_path) -> pd.DataFrame:
    return pd.read_csv(csv_path).drop(columns=["Unnamed: 0"], errors="ignore")


def simulate_users_and_interactions(items: pd.DataFrame, cfg: dict, parser: GenreParser):
    rng = np.random.default_rng(cfg["seed"])
    it = items.copy()
    it["genres_list"] = it["genres"].map(parser.parse)
    it["category"] = it["genres_list"].map(parser.primary)
    hist_cols = [f"rated_{k}" for k in range(1, 6)]
    ok = (it[hist_cols].notna().all(axis=1) & (it["num_ratings"] >= cfg["min_num_ratings"])
          & it["category"].notna() & ~it["bookId"].duplicated())
    cat = it[ok].sort_values("num_ratings", ascending=False).head(cfg["catalog_size"]).reset_index(drop=True)
    n_items, n_users = len(cat), cfg["n_users"]

    # keep the 14 biggest categories, everything else -> "Other"
    top_cats = cat["category"].value_counts().head(14).index
    cat["sim_cat"] = np.where(cat["category"].isin(top_cats), cat["category"], "Other")
    cats = sorted(cat["sim_cat"].unique()); K = len(cats)
    cat_idx = cat["sim_cat"].map({c: i for i, c in enumerate(cats)}).to_numpy()

    # latent taste space: items cluster around their category centroid, users mix 1-3 favourite categories
    D = 8
    centroids = rng.normal(size=(K, D))
    item_vec = centroids[cat_idx] + 0.7 * rng.normal(size=(n_items, D))
    taste = rng.dirichlet(np.full(K, 0.25), size=n_users)
    user_vec = taste @ centroids + 0.5 * rng.normal(size=(n_users, D))
    item_vec /= np.linalg.norm(item_vec, axis=1, keepdims=True)
    user_vec /= np.linalg.norm(user_vec, axis=1, keepdims=True)
    aff = user_vec @ item_vec.T
    aff_z = (aff - aff.mean()) / aff.std()
    pop_z = np.log(cat["num_ratings"].to_numpy()); pop_z = (pop_z - pop_z.mean()) / pop_z.std()
    choice_score = 0.8 * pop_z[None, :] + 1.6 * aff_z                      # popularity + taste

    # users
    start, end = pd.Timestamp(cfg["start_date"]), pd.Timestamp(cfg["end_date"])
    span_days = (end - start).days - 90
    signup = start + pd.to_timedelta(rng.integers(0, span_days, n_users), unit="D")
    users = pd.DataFrame({
        "user_id": [f"U{i+1:05d}" for i in range(n_users)],
        "signup_date": signup.strftime("%Y-%m-%d"),
        "country": rng.choice(["IN", "US", "GB", "CA", "AU", "DE", "SG", "AE"], n_users,
                              p=[.30, .28, .12, .08, .06, .06, .05, .05]),
        "age_group": rng.choice(["18-24", "25-34", "35-44", "45-54", "55+"], n_users, p=[.22, .36, .22, .12, .08]),
    })

    # interactions: Gumbel-top-n sampling = weighted sampling without replacement
    n_per_user = np.clip(rng.lognormal(np.log(24), 0.75, n_users).round().astype(int), 6, 250)
    rows_u, rows_i = [], []
    for u in range(n_users):
        g = choice_score[u] + rng.gumbel(size=n_items)
        top = np.argpartition(-g, n_per_user[u])[: n_per_user[u]]
        rows_u.append(np.full(len(top), u)); rows_i.append(top)
    u_idx, i_idx = np.concatenate(rows_u), np.concatenate(rows_i)

    # rating = draw from the book's REAL star histogram, nudged by taste match + user bias
    H = cat[hist_cols].to_numpy(float); H = H / H.sum(axis=1, keepdims=True)
    cdf = np.cumsum(H, axis=1)[i_idx]
    base = (rng.random(len(i_idx))[:, None] > cdf).sum(axis=1) + 1
    user_bias = rng.normal(0, 0.30, n_users)
    rating = np.clip(np.rint(base + 0.45 * aff_z[u_idx, i_idx] + user_bias[u_idx] + rng.normal(0, 0.3, len(base))), 1, 5).astype(int)

    # timestamp: between the user's signup and end_date
    s_ts = pd.to_datetime(users["signup_date"]).to_numpy()[u_idx].astype("datetime64[s]").astype("int64")
    e_ts = np.int64(end.value // 10**9)
    ts = (s_ts + (e_ts - s_ts) * rng.beta(1.0, 1.25, len(u_idx))).astype("int64")
    inter = pd.DataFrame({
        "user_id": users["user_id"].to_numpy()[u_idx],
        "item_id": cat["bookId"].to_numpy()[i_idx],
        "rating": rating,
        "timestamp": pd.to_datetime(ts, unit="s").strftime("%Y-%m-%d %H:%M:%S"),
    }).sample(frac=1, random_state=cfg["seed"]).reset_index(drop=True)
    inter.insert(0, "interaction_id", [f"I{i+1:07d}" for i in range(len(inter))])
    return users, inter


def inject_faults(users: pd.DataFrame, inter: pd.DataFrame, cfg: dict):
    """Documented + reproducible fault injection. Returns (users, interactions, ledger)."""
    rng = np.random.default_rng(cfg["seed"] + 1)
    u, x = users.astype(object).copy(), inter.astype(object).copy()
    n = len(x); nf = int(n * cfg["fault_rate"] / 10)
    perm = rng.permutation(n); slot = lambda k: perm[k * nf:(k + 1) * nf]
    led = {}

    x.loc[slot(0), "rating"] = rng.choice(["0", "6", "-1", "11", "7"], nf);               led["interactions: rating out of range"] = nf
    x.loc[slot(1), "rating"] = rng.choice(["five", "4 stars", "excellent", "good"], nf);   led["interactions: rating wrong dtype"] = nf
    x.loc[slot(2), "rating"] = np.nan;                                                     led["interactions: rating missing"] = nf
    x.loc[slot(3), "user_id"] = np.nan;                                                    led["interactions: user_id missing"] = nf
    x.loc[slot(4), "timestamp"] = rng.choice(["not_a_date", "32/13/2020", "0000-00-00"], nf); led["interactions: timestamp invalid"] = nf
    x.loc[slot(5), "timestamp"] = [f"{y}-0{m}-15 10:00:00" for y, m in zip(rng.integers(2026, 2032, nf), rng.integers(1, 9, nf))]; led["interactions: timestamp in future"] = nf
    x.loc[slot(6), "user_id"] = [f"U9{k:04d}" for k in rng.integers(0, 9999, nf)];          led["interactions: orphan user_id"] = nf
    x.loc[slot(7), "item_id"] = rng.integers(9_000_000, 9_999_999, nf).astype(str);        led["interactions: orphan item_id"] = nf
    # timestamp before the user's signup
    sd = pd.to_datetime(u.set_index("user_id")["signup_date"])
    b = slot(8); bad_ts = (sd.reindex(x.loc[b, "user_id"]).to_numpy() - pd.to_timedelta(rng.integers(5, 400, nf), unit="D"))
    x.loc[b, "timestamp"] = pd.Series(bad_ts).dt.strftime("%Y-%m-%d 12:00:00").to_numpy();   led["interactions: timestamp before signup"] = nf
    dups = x.iloc[slot(9)].copy(); x = pd.concat([x, dups], ignore_index=True);            led["interactions: exact duplicate rows"] = nf
    x = x.sample(frac=1, random_state=cfg["seed"]).reset_index(drop=True)

    nu = len(u)
    u.loc[rng.choice(nu, 40, replace=False), "country"] = np.nan;                           led["users: country missing"] = 40
    u.loc[rng.choice(nu, 30, replace=False), "age_group"] = rng.choice(["unknown", "150", "-"], 30); led["users: invalid age_group"] = 30
    u.loc[rng.choice(nu, 5, replace=False), "signup_date"] = ["2031-05-01", "unknown", "2030-01-01", "2045-12-31", "00-00-0000"]; led["users: invalid/future signup_date"] = 5
    u = pd.concat([u, u.iloc[rng.choice(nu, 15, replace=False)]], ignore_index=True);       led["users: duplicate user_id rows"] = 15
    return u.sample(frac=1, random_state=cfg["seed"]).reset_index(drop=True), x, pd.Series(led, name="injected").to_frame()


def run(base, cfg) -> dict:
    P = Paths(base).ensure()
    if not cfg.get("source_csv"):
        raise ValueError("cfg['source_csv'] must point to the Goodreads items CSV")
    items = load_items(cfg["source_csv"])
    parser = GenreParser.fit(items["genres"])
    log.info(f"items loaded: {items.shape} | boilerplate genre prefix in {parser.prefix_share(items['genres']):.1%} of genre strings")
    users_ok, inter_ok = simulate_users_and_interactions(items, cfg, parser)
    users_src, inter_src, ledger = inject_faults(users_ok, inter_ok, cfg)
    items.to_csv(P.raw / "items_raw.csv", index=False)
    users_src.to_csv(P.raw / "users_raw.csv", index=False)
    inter_src.to_csv(P.raw / "interactions_raw.csv", index=False)
    ledger.to_csv(P.raw / "injected_fault_ledger.csv")
    log.info(f"source extracts written to data/raw/: items={items.shape} users={users_src.shape} interactions={inter_src.shape}")
    return {"items": len(items), "users": len(users_src), "interactions": len(inter_src), "faults_injected": int(ledger["injected"].sum())}

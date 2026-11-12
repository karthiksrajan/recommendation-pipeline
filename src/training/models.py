"""Recommender models. Every model stores only NumPy / SciPy arrays, so it is trivially serialisable and portable.

Interface:  fit(d) -> self   |   score(user_idx) -> (n_users x n_items) scores   |   optional predict(user_idx, item_idx) -> ratings
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.cpu.als import AlternatingLeastSquares
from surprise import SVD, Dataset, Reader

SEED = 42


def _z(S):
    """Row-wise standardisation so that scores of different models can be blended."""
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)


class RandomModel:
    name = "Random (sanity floor)"

    def __init__(self, seed=SEED):
        self.params = {"seed": seed}

    def fit(self, d):
        self.n_items = d.n_items
        self.rs = np.random.RandomState(self.params["seed"])
        return self

    def score(self, u):
        return self.rs.random_sample((len(u), self.n_items))


class PopularityModel:
    name = "Popularity (baseline)"

    def __init__(self, prior_strength=20):
        self.params = {"prior_strength": prior_strength}

    def fit(self, d):
        pos = np.asarray(d.P.sum(axis=0)).ravel()
        cnt = np.asarray(d.seen.sum(axis=0)).ravel()
        self.pop = pos + 1e-6 * cnt                                   # rank by # positive ratings, tie-break by total interactions
        mu, C = d.tr_rating.mean(), self.params["prior_strength"]
        self.item_mean = (C * mu + np.asarray(d.R.sum(axis=0)).ravel()) / (C + cnt)   # shrunk item mean -> rating prediction
        return self

    def score(self, u):
        return np.tile(self.pop, (len(u), 1))

    def predict(self, u, i):
        return self.item_mean[i]


class ALSModel:
    name = "ALS (implicit MF)"

    def __init__(self, factors=64, regularization=0.1, alpha=20.0, iterations=20):
        self.params = dict(factors=factors, regularization=regularization, alpha=alpha, iterations=iterations)

    def fit(self, d):
        m = AlternatingLeastSquares(factors=self.params["factors"], regularization=self.params["regularization"], alpha=self.params["alpha"],
                                    iterations=self.params["iterations"], random_state=SEED, calculate_training_loss=False)
        m.fit(d.P, show_progress=False)
        self.U, self.V = np.asarray(m.user_factors), np.asarray(m.item_factors)
        return self

    def score(self, u):
        return self.U[u] @ self.V.T


class SVDModel:
    name = "SVD (Surprise MF)"

    def __init__(self, n_factors=50, n_epochs=30, lr_all=0.005, reg_all=0.05):
        self.params = dict(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)

    def fit(self, d):
        df = pd.DataFrame({"u": d.users[d.tu], "i": d.items[d.ti], "r": d.tr_rating})
        ts = Dataset.load_from_df(df, Reader(rating_scale=(1, 5))).build_full_trainset()
        algo = SVD(random_state=SEED, **self.params)
        algo.fit(ts)
        ui = np.array([ts.to_inner_uid(u) for u in d.users])
        ii = np.array([ts.to_inner_iid(i) for i in d.items])
        self.pu, self.qi, self.bu, self.bi, self.mu = algo.pu[ui], algo.qi[ii], algo.bu[ui], algo.bi[ii], ts.global_mean
        return self

    def score(self, u):
        return self.mu + self.bu[u][:, None] + self.bi[None, :] + self.pu[u] @ self.qi.T

    def predict(self, u, i):
        return self.mu + self.bu[u] + self.bi[i] + (self.pu[u] * self.qi[i]).sum(1)


class ItemKNNModel:
    """Neighbourhood collaborative filtering: cosine similarity between books on positive feedback, top-k neighbours kept."""
    name = "ItemKNN (cosine CF)"

    def __init__(self, k=50, shrink=10.0):
        self.params = dict(k=k, shrink=shrink)

    def fit(self, d):
        P = d.P.astype(np.float64)
        co = (P.T @ P).toarray()
        norms = np.sqrt(np.diag(co))
        sim = co / (np.outer(norms, norms) + self.params["shrink"])
        np.fill_diagonal(sim, 0.0)
        k = min(self.params["k"], sim.shape[1] - 1)
        drop = np.argpartition(-sim, k, axis=1)[:, k:]                # everything except the k nearest neighbours
        np.put_along_axis(sim, drop, 0.0, axis=1)
        self.S = sp.csr_matrix(sim.astype(np.float32))
        self.P = d.P.tocsr()
        return self

    def score(self, u):
        return np.asarray((self.P[u] @ self.S).todense())


class ContentModel:
    """Content-based filtering: user genre profile (from positively rated books) vs. book genre vectors, IDF-weighted."""
    name = "Content-based (genres)"

    def __init__(self, tie_break=1e-3):
        self.params = {"tie_break": tie_break}

    def fit(self, d):
        self.G = d.G.astype(np.float32)
        df_g = (self.G > 0).sum(0)
        self.idf = np.log((self.G.shape[0] + 1) / (df_g + 1)).astype(np.float32) + 1.0
        prof = (d.P @ self.G) * self.idf
        self.prof = np.asarray(prof / np.maximum(np.linalg.norm(prof, axis=1, keepdims=True), 1e-9), dtype=np.float32)
        pop = np.log1p(np.asarray(d.seen.sum(0)).ravel())
        self.pop_z = (pop - pop.mean()) / (pop.std() + 1e-9)
        return self

    def score(self, u):
        return self.prof[u] @ (self.G * self.idf).T + self.params["tie_break"] * self.pop_z[None, :]


class HybridModel:
    """Hybrid = standardised ItemKNN score + w_als * ALS + w_content * content score + w_pop * popularity prior.

    Combines two collaborative signals (neighbourhood + latent factors) with the engineered genre content features and a
    popularity prior; blend weights are tuned on the temporal validation fold.
    """
    name = "Hybrid (ItemKNN + ALS + content)"

    def __init__(self, factors=32, regularization=0.1, alpha=10.0, k=100, shrink=50.0, w_als=1.0, w_content=0.25, w_pop=0.15):
        self.params = dict(factors=factors, regularization=regularization, alpha=alpha, k=k, shrink=shrink,
                           w_als=w_als, w_content=w_content, w_pop=w_pop)

    def fit(self, d):
        p = self.params
        self.als = ALSModel(p["factors"], p["regularization"], p["alpha"]).fit(d)
        self.knn = ItemKNNModel(p["k"], p["shrink"]).fit(d)
        self.content = ContentModel().fit(d)
        return self

    def score(self, u):
        p = self.params
        return (_z(self.knn.score(u)) + p["w_als"] * _z(self.als.score(u))
                + p["w_content"] * _z(self.content.score(u)) + p["w_pop"] * self.content.pop_z[None, :])

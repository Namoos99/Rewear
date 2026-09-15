"""Phase 1 recommenders. Every model exposes the same interface:

    model.fit(train_df)
    model.recommend(customer_ids, k=12) -> {customer_id: [article_id, ...]}

so the evaluation script can treat them interchangeably.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from .data import build_interaction_matrix


class PopularityRecommender:
    """Recommend the best-selling items from the most recent `recent_days`.

    This is the floor. It's also embarrassingly strong on H&M because the
    validation week is dominated by whatever was new and heavily stocked.
    Beating it with anything personalised is the first real milestone.
    """

    def __init__(self, recent_days: int = 7):
        self.recent_days = recent_days
        self.top_items: list[str] = []

    def fit(self, train: pd.DataFrame) -> PopularityRecommender:
        cutoff = train["t_dat"].max() - pd.Timedelta(days=self.recent_days - 1)
        recent = train[train["t_dat"] >= cutoff]
        self.top_items = recent["article_id"].value_counts().index.tolist()
        return self

    def recommend(self, customer_ids, k: int = 12) -> dict[str, list[str]]:
        top = self.top_items[:k]
        return {cid: list(top) for cid in customer_ids}


class ItemKNNRecommender:
    """Item-based collaborative filtering with cosine similarity.

    Score(user, item) = sum over items the user bought of sim(bought_item, item)
                        + pop_blend * recent_popularity(item)
    Recently bought items are up-weighted with exponential decay so a purchase
    last week counts more than one from a year ago. Falls back to popularity
    for customers with no training history (cold start).

    exclude_owned: if True, never recommend an item the customer already bought.
        Off by default — repurchases are a large share of real purchases and
        this is the *relevance* model. "Show me new pieces" is a product
        decision that belongs in the Rewear reranking layer, not hidden here.
    pop_blend: small additive weight on recent popularity. Breaks ties for
        customers with thin histories and stops the model from wandering into
        pure noise for items with two co-purchases.
    """

    def __init__(
        self,
        n_neighbors: int = 50,
        decay_days: float = 30.0,
        recent_days: int = 7,
        exclude_owned: bool = False,
        pop_blend: float = 0.1,
    ):
        self.n_neighbors = n_neighbors
        self.decay_days = decay_days
        self.exclude_owned = exclude_owned
        self.pop_blend = pop_blend
        self._pop_scores: np.ndarray | None = None
        self.fallback = PopularityRecommender(recent_days=recent_days)
        self.sim: csr_matrix | None = None
        self.customer_index: pd.Index | None = None
        self.article_index: pd.Index | None = None
        self.user_item: csr_matrix | None = None

    def fit(self, train: pd.DataFrame) -> ItemKNNRecommender:
        self.fallback.fit(train)

        # Time-decayed interaction weights instead of raw counts.
        last = train["t_dat"].max()
        age = (last - train["t_dat"]).dt.days.to_numpy(dtype=np.float32)
        weights = np.exp(-age / self.decay_days)

        mat, self.customer_index, self.article_index = build_interaction_matrix(train)
        rows = self.customer_index.get_indexer(train["customer_id"])
        cols = self.article_index.get_indexer(train["article_id"])
        weighted = csr_matrix((weights, (rows, cols)), shape=mat.shape)
        weighted.sum_duplicates()
        self.user_item = weighted

        # Item-item cosine similarity, keep top-n neighbours per item to stay sparse.
        item_vecs = normalize(weighted.T.tocsr(), axis=1)
        sim = (item_vecs @ item_vecs.T).tocsr()
        sim.setdiag(0)
        sim.eliminate_zeros()
        self.sim = _keep_top_n_per_row(sim, self.n_neighbors)

        # Recent-popularity prior aligned to article_index, scaled to [0, 1].
        cutoff = last - pd.Timedelta(days=self.fallback.recent_days - 1)
        recent_counts = train.loc[train["t_dat"] >= cutoff, "article_id"].value_counts()
        pop = recent_counts.reindex(self.article_index).fillna(0.0).to_numpy(dtype=np.float32)
        self._pop_scores = pop / pop.max() if pop.max() > 0 else pop
        return self

    def recommend(self, customer_ids, k: int = 12) -> dict[str, list[str]]:
        assert self.sim is not None, "call fit() first"
        out: dict[str, list[str]] = {}
        fallback = self.fallback.recommend(customer_ids, k)

        known = [c for c in customer_ids if c in self.customer_index]
        if known:
            rows = self.customer_index.get_indexer(known)
            scores = self.user_item[rows] @ self.sim  # (n_known, n_items)
            scores = scores.toarray()
            # Row-normalise so the popularity prior is on a comparable scale per user.
            row_max = scores.max(axis=1, keepdims=True)
            row_max[row_max == 0] = 1.0
            scores = scores / row_max + self.pop_blend * self._pop_scores
            if self.exclude_owned:
                owned = self.user_item[rows].toarray() > 0
                scores[owned] = -np.inf
            top = np.argsort(-scores, axis=1)[:, :k]
            for cid, row_scores, row_top in zip(known, scores, top):
                recs = [self.article_index[j] for j in row_top if np.isfinite(row_scores[j]) and row_scores[j] > 0]
                # Pad with popularity if the user's neighbourhood was too thin.
                if len(recs) < k:
                    recs += [a for a in fallback[cid] if a not in recs][: k - len(recs)]
                out[cid] = recs

        for cid in customer_ids:
            out.setdefault(cid, fallback[cid])
        return out


def _keep_top_n_per_row(mat: csr_matrix, n: int) -> csr_matrix:
    """Zero out everything except the n largest values in each row."""
    mat = mat.tocsr()
    for i in range(mat.shape[0]):
        start, end = mat.indptr[i], mat.indptr[i + 1]
        if end - start > n:
            row = mat.data[start:end]
            drop = np.argpartition(row, -n)[:-n]
            row[drop] = 0
    mat.eliminate_zeros()
    return mat

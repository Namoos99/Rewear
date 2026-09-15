"""Phase 3, step 1: multi-source candidate generation.

The Phase 1 result — popularity beats item-kNN — is really a *recall* problem.
No single retriever sees the whole picture, so we union several and let a
learned ranker decide. Each source is cheap and recall-oriented; precision is
the ranker's job.

Sources (each tags its candidates so the ranker can learn source reliability):

  repurchase   items the customer bought before, most recent first
  knn          time-decayed item-kNN (Phase 1)
  content      nearest items to the customer's content taste vector (Phase 2)
  popularity   best sellers of the final training week

Output is a long DataFrame: one row per (customer_id, article_id) with
per-source rank/score columns (NaN when a source didn't propose the item).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .baseline import ItemKNNRecommender, PopularityRecommender
from .content import ContentEmbedder

SOURCES = ["repurchase", "knn", "content", "popularity"]


class CandidateGenerator:
    def __init__(self, n_knn: int = 50, n_content: int = 30, n_pop: int = 30, n_repurchase: int = 20, content_backend: str = "tfidf"):
        self.n_knn = n_knn
        self.n_content = n_content
        self.n_pop = n_pop
        self.n_repurchase = n_repurchase
        self.knn = ItemKNNRecommender()
        self.pop = PopularityRecommender()
        self.emb = ContentEmbedder(backend=content_backend)
        self.history: dict[str, list[str]] | None = None

    def fit(self, train: pd.DataFrame, articles: pd.DataFrame) -> CandidateGenerator:
        self.knn.fit(train)
        self.pop.fit(train)
        self.emb.fit(articles)
        # Most-recent-first purchase history per customer, deduplicated.
        ordered = train.sort_values("t_dat", ascending=False)
        self.history = ordered.groupby("customer_id")["article_id"].apply(lambda s: list(dict.fromkeys(s))).to_dict()
        return self

    def generate(self, customer_ids: list[str]) -> pd.DataFrame:
        assert self.history is not None, "call fit() first"
        frames = []

        # repurchase
        rows = []
        for cid in customer_ids:
            for r, aid in enumerate(self.history.get(cid, [])[: self.n_repurchase]):
                rows.append((cid, aid, r))
        frames.append(pd.DataFrame(rows, columns=["customer_id", "article_id", "repurchase_rank"]))

        # knn
        knn_recs = self.knn.recommend(customer_ids, k=self.n_knn)
        rows = [(cid, aid, r) for cid, recs in knn_recs.items() for r, aid in enumerate(recs)]
        frames.append(pd.DataFrame(rows, columns=["customer_id", "article_id", "knn_rank"]))

        # content
        rows = []
        for cid in customer_ids:
            hist = self.history.get(cid, [])
            if not hist:
                continue
            weights = np.exp(-np.arange(len(hist)) / 10.0)  # most recent purchases dominate taste
            u = self.emb.user_vector(hist, weights)
            sims = self.emb.vectors @ u
            top = np.argpartition(-sims, min(self.n_content, len(sims) - 1))[: self.n_content]
            top = top[np.argsort(-sims[top])]
            for r, j in enumerate(top):
                rows.append((cid, self.emb.index[j], r, float(sims[j])))
        frames.append(pd.DataFrame(rows, columns=["customer_id", "article_id", "content_rank", "content_sim"]))

        # popularity
        top_pop = self.pop.top_items[: self.n_pop]
        rows = [(cid, aid, r) for cid in customer_ids for r, aid in enumerate(top_pop)]
        frames.append(pd.DataFrame(rows, columns=["customer_id", "article_id", "pop_rank"]))

        out = frames[0]
        for f in frames[1:]:
            out = out.merge(f, on=["customer_id", "article_id"], how="outer")
        for s, col in zip(SOURCES, ["repurchase_rank", "knn_rank", "content_rank", "pop_rank"]):
            out[f"src_{s}"] = out[col].notna().astype(np.int8)
        out["n_sources"] = out[[f"src_{s}" for s in SOURCES]].sum(axis=1).astype(np.int8)
        return out.reset_index(drop=True)

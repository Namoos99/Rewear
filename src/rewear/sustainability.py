"""Sustainability signals and the reranker that uses them.

None of these are perfect proxies, and the DECISIONS.md is honest about that.
The point is to make the trade-off *explicit and tunable* rather than pretending
the engagement-optimal ranking is the only one that exists.

Three signals per article:

  material_score   [0,1]  parsed from detail_desc. Natural / recycled fibres score
                          high, virgin synthetics score low. Unknown = 0.5.
  wear_again_score [0,1]  how often customers who bought this *category* came back
                          for the same category. Proxy for "this becomes a staple"
                          rather than a one-off. (We can't observe wear directly.)
  popularity       [0,1]  purchase count, normalised. Penalised in the reranker
                          because chasing the head of the curve is what fast
                          fashion already does very well without us.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Material keywords -> impact score. Rough, deliberately simple, and documented.
# Sources for the ordering: Higg MSI-style rankings and the Fashion Transparency
# Index's fibre guidance. Treat as a first pass, not a life-cycle assessment.
MATERIAL_SCORES: dict[str, float] = {
    "recycled": 0.95,
    "organic cotton": 0.9,
    "linen": 0.9,
    "hemp": 0.9,
    "tencel": 0.85,
    "lyocell": 0.85,
    "wool": 0.75,
    "cotton": 0.65,
    "viscose": 0.45,
    "modal": 0.5,
    "leather": 0.35,
    "polyamide": 0.3,
    "nylon": 0.3,
    "acrylic": 0.25,
    "polyester": 0.25,
    "elastane": 0.2,
    "spandex": 0.2,
}


def material_score(desc: str | float) -> float:
    """Score a product description by the materials it mentions.

    If several materials appear, take the mean. Longer keys ("organic cotton")
    are matched before shorter ones ("cotton") so they don't double count.
    """
    if not isinstance(desc, str) or not desc.strip():
        return 0.5
    text = desc.lower()
    found: list[float] = []
    for key in sorted(MATERIAL_SCORES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(key)}\b", text):
            found.append(MATERIAL_SCORES[key])
            text = text.replace(key, " ")
    return float(np.mean(found)) if found else 0.5


def wear_again_scores(train: pd.DataFrame, articles: pd.DataFrame, group_col: str = "product_type_name") -> pd.Series:
    """Per-article 'wear-again' proxy from category repurchase behaviour.

    For each category: fraction of customers who bought in it more than once.
    Every article inherits its category's rate. Categories with too few buyers
    fall back to the global rate so tiny categories don't get extreme scores.
    """
    merged = train.merge(articles[["article_id", group_col]], on="article_id", how="left")
    per_cust_cat = merged.groupby(["customer_id", group_col]).size().rename("n").reset_index()
    cat_stats = per_cust_cat.groupby(group_col)["n"].agg(buyers="size", repeat=lambda s: (s > 1).sum())
    global_rate = cat_stats["repeat"].sum() / max(cat_stats["buyers"].sum(), 1)
    min_buyers = 20
    cat_stats["rate"] = np.where(
        cat_stats["buyers"] >= min_buyers,
        cat_stats["repeat"] / cat_stats["buyers"],
        global_rate,
    )
    rate_by_cat = cat_stats["rate"]
    scores = articles.set_index("article_id")[group_col].map(rate_by_cat).fillna(global_rate)
    # Normalise to [0, 1] so it's on the same scale as the other signals.
    lo, hi = scores.min(), scores.max()
    return (scores - lo) / (hi - lo) if hi > lo else scores * 0 + 0.5


def build_article_features(train: pd.DataFrame, articles: pd.DataFrame) -> pd.DataFrame:
    """One row per article with the three sustainability signals + popularity."""
    feats = articles[["article_id"]].copy().set_index("article_id")
    feats["material_score"] = articles.set_index("article_id")["detail_desc"].map(material_score)
    feats["wear_again_score"] = wear_again_scores(train, articles)
    counts = train["article_id"].value_counts()
    feats["popularity"] = (counts / counts.max()).reindex(feats.index).fillna(0.0)
    return feats


class SustainabilityReranker:
    """Blend a base recommender's ranking with the sustainability signals.

    final = (1 - lam) * relevance
          + lam * (w_mat * material + w_wear * wear_again - w_pop * popularity)

    lam=0 reproduces the base model exactly. lam=1 ignores relevance entirely
    (and MAP@12 collapses — which is the point of the sweep in the eval script).
    """

    def __init__(
        self,
        article_features: pd.DataFrame,
        lam: float = 0.3,
        w_material: float = 1.0,
        w_wear: float = 1.0,
        w_pop: float = 0.5,
        candidate_multiplier: int = 5,
    ):
        self.f = article_features
        self.lam = lam
        self.w_material = w_material
        self.w_wear = w_wear
        self.w_pop = w_pop
        self.candidate_multiplier = candidate_multiplier

    def sustainability(self, article_ids: list[str]) -> np.ndarray:
        f = self.f.reindex(article_ids)
        return (
            self.w_material * f["material_score"].fillna(0.5).to_numpy()
            + self.w_wear * f["wear_again_score"].fillna(0.5).to_numpy()
            - self.w_pop * f["popularity"].fillna(0.0).to_numpy()
        )

    def rerank(self, base_recs: dict[str, list[str]], k: int = 12) -> dict[str, list[str]]:
        """Take the base model's top (k * candidate_multiplier), rerank, cut to k.

        Relevance is the base model's rank position mapped to [1, 0] — we only
        assume the base model's *ordering* is meaningful, not its raw scores.
        """
        out = {}
        for cid, recs in base_recs.items():
            cands = recs[: k * self.candidate_multiplier]
            if not cands:
                out[cid] = []
                continue
            n = len(cands)
            relevance = 1.0 - np.arange(n) / max(n - 1, 1)
            sust = self.sustainability(cands)
            # Scale sustainability to [0,1] within the candidate set so lam is comparable across users.
            lo, hi = sust.min(), sust.max()
            sust = (sust - lo) / (hi - lo) if hi > lo else np.full(n, 0.5)
            score = (1 - self.lam) * relevance + self.lam * sust
            order = np.argsort(-score, kind="stable")
            out[cid] = [cands[i] for i in order[:k]]
        return out

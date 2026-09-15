"""Evaluation metrics.

MAP@K is the Kaggle competition metric, so it's the number everyone recognises.
The rest exist because accuracy alone rewards recommending the same popular
items to everyone, which is exactly the behaviour Rewear is trying to move away from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def apk(actual: list[str], predicted: list[str], k: int = 12) -> float:
    """Average precision at k for a single user."""
    if not actual:
        return 0.0
    predicted = predicted[:k]
    actual_set = set(actual)
    hits = 0
    score = 0.0
    seen = set()
    for i, p in enumerate(predicted):
        if p in actual_set and p not in seen:
            hits += 1
            score += hits / (i + 1.0)
            seen.add(p)
    return score / min(len(actual), k)


def mapk(ground_truth: dict[str, list[str]], predictions: dict[str, list[str]], k: int = 12) -> float:
    """Mean average precision at k across every user in ground_truth.

    Users with no prediction score 0 — that's deliberate. Silently dropping them
    would let a model look good by only answering the easy cases.
    """
    scores = [apk(actual, predictions.get(cid, []), k) for cid, actual in ground_truth.items()]
    return float(np.mean(scores)) if scores else 0.0


def catalog_coverage(predictions: dict[str, list[str]], n_catalog: int, k: int = 12) -> float:
    """Fraction of the catalogue that appears in at least one user's top-k."""
    recommended = {a for recs in predictions.values() for a in recs[:k]}
    return len(recommended) / n_catalog if n_catalog else 0.0


def item_popularity(train: pd.DataFrame) -> pd.Series:
    """Purchase count per article, normalised to [0, 1]."""
    counts = train["article_id"].value_counts()
    return counts / counts.max()


def novelty(predictions: dict[str, list[str]], popularity: pd.Series, k: int = 12) -> float:
    """Mean self-information of recommended items: -log2(p(item)).

    Higher = the system recommends things fewer people have bought.
    Items never seen in training get the minimum popularity (max novelty).
    """
    floor = popularity.min() if len(popularity) else 1e-6
    vals = []
    for recs in predictions.values():
        for a in recs[:k]:
            p = popularity.get(a, floor)
            vals.append(-np.log2(max(p, 1e-9)))
    return float(np.mean(vals)) if vals else 0.0


def long_tail_exposure(
    predictions: dict[str, list[str]],
    train: pd.DataFrame,
    head_fraction: float = 0.2,
    k: int = 12,
) -> float:
    """Share of recommendation slots given to items outside the top `head_fraction` by sales.

    With head_fraction=0.2 this is "what fraction of recommendations are NOT
    among the best-selling 20% of items". A pure popularity model scores ~0.
    """
    counts = train["article_id"].value_counts()
    n_head = max(1, int(len(counts) * head_fraction))
    head = set(counts.index[:n_head])
    slots = 0
    tail = 0
    for recs in predictions.values():
        for a in recs[:k]:
            slots += 1
            if a not in head:
                tail += 1
    return tail / slots if slots else 0.0


def evaluate(
    name: str,
    predictions: dict[str, list[str]],
    ground_truth: dict[str, list[str]],
    train: pd.DataFrame,
    n_catalog: int,
    k: int = 12,
) -> dict[str, float | str]:
    """One row of the results table."""
    pop = item_popularity(train)
    return {
        "model": name,
        f"map@{k}": round(mapk(ground_truth, predictions, k), 5),
        "coverage": round(catalog_coverage(predictions, n_catalog, k), 4),
        "novelty": round(novelty(predictions, pop, k), 3),
        "long_tail_exposure": round(long_tail_exposure(predictions, train, k=k), 4),
    }

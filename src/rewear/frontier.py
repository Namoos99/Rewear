"""Phase 4: the accuracy-vs-sustainability frontier, with uncertainty.

A single MAP@12 number on 2,590 validation customers is noisy. We bootstrap
over customers (resample with replacement, recompute the metric) to get a 95%
interval for every point on the λ sweep. If two λ values have overlapping
intervals, the honest statement is "we can't tell them apart", not "λ=0.2 wins".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import apk, item_popularity
from .sustainability import SustainabilityReranker


def bootstrap_map(
    ground_truth: dict[str, list[str]],
    predictions: dict[str, list[str]],
    k: int = 12,
    n_boot: int = 500,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(mean, ci_low, ci_high) of MAP@k via customer-level bootstrap."""
    cids = list(ground_truth)
    per_user = np.array([apk(ground_truth[c], predictions.get(c, []), k) for c in cids])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_user), size=(n_boot, len(per_user)))
    boots = per_user[idx].mean(axis=1)
    return float(per_user.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def mean_sustainability(predictions: dict[str, list[str]], article_features: pd.DataFrame, k: int = 12) -> dict[str, float]:
    ids = [a for recs in predictions.values() for a in recs[:k]]
    f = article_features.reindex(ids)
    return {
        "mean_material": float(f["material_score"].fillna(0.5).mean()),
        "mean_wear_again": float(f["wear_again_score"].fillna(0.5).mean()),
        "mean_popularity": float(f["popularity"].fillna(0).mean()),
    }


def sweep(
    base_recs: dict[str, list[str]],
    ground_truth: dict[str, list[str]],
    train: pd.DataFrame,
    article_features: pd.DataFrame,
    lambdas: list[float],
    label: str,
    k: int = 12,
    n_boot: int = 500,
) -> pd.DataFrame:
    """One row per λ: MAP@k with CI, plus the sustainability profile of what got shown."""
    from .metrics import catalog_coverage, long_tail_exposure, novelty

    pop = item_popularity(train)
    n_catalog = train["article_id"].nunique()
    rows = []
    for lam in lambdas:
        preds = SustainabilityReranker(article_features, lam=lam).rerank(base_recs, k=k)
        m, lo, hi = bootstrap_map(ground_truth, preds, k, n_boot)
        rows.append(
            {
                "model": label,
                "lambda": lam,
                f"map@{k}": round(m, 5),
                "ci_low": round(lo, 5),
                "ci_high": round(hi, 5),
                "coverage": round(catalog_coverage(preds, n_catalog, k), 4),
                "novelty": round(novelty(preds, pop, k), 3),
                "long_tail_exposure": round(long_tail_exposure(preds, train, k=k), 4),
                **{kk: round(v, 4) for kk, v in mean_sustainability(preds, article_features, k).items()},
            }
        )
    return pd.DataFrame(rows)


def plot_frontier(df: pd.DataFrame, path: str, k: int = 12) -> None:
    """MAP@k (with CI band) against mean material score, one line per model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    for label, g in df.groupby("model"):
        g = g.sort_values("lambda")
        ax.plot(g["mean_material"], g[f"map@{k}"], marker="o", label=label)
        ax.fill_between(g["mean_material"], g["ci_low"], g["ci_high"], alpha=0.15)
        for _, r in g.iterrows():
            ax.annotate(f"λ={r['lambda']:g}", (r["mean_material"], r[f"map@{k}"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("mean material score of what gets shown  →  greener")
    ax.set_ylabel(f"MAP@{k}  (95% bootstrap CI)")
    ax.set_title("Rewear: the accuracy-vs-sustainability frontier")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

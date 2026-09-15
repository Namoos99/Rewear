"""Phases 3 + 4: train the LightGBM reranker, then trace the frontier with bootstrap CIs.

    python scripts/train_ranker.py --data data/sample
    python scripts/train_ranker.py --data data/raw --sample-customers 50000

Writes:
    results/phase3_results.csv    popularity / item-kNN / ranker, all four metrics
    results/frontier.csv          λ sweep for item-kNN and ranker, MAP@12 with 95% CI
    results/feature_importance.csv
    docs/frontier.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rewear.baseline import ItemKNNRecommender, PopularityRecommender
from rewear.data import load_raw
from rewear.frontier import plot_frontier, sweep
from rewear.metrics import evaluate
from rewear.ranker import train_and_evaluate_pipeline
from rewear.sustainability import build_article_features


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sample")
    ap.add_argument("--sample-customers", type=int, default=None)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--lambdas", default="0.0,0.1,0.2,0.3,0.5,0.7,1.0")
    args = ap.parse_args()
    lambdas = [float(x) for x in args.lambdas.split(",")]

    t0 = time.time()
    ds = load_raw(args.data, sample_customers=args.sample_customers)
    ranker, result, train, truth = train_and_evaluate_pipeline(ds.transactions, ds.articles, ds.customers, k=args.k * 5, n_rounds=args.rounds)
    targets = list(truth)
    n_catalog = train["article_id"].nunique()
    print(f"ranker trained + scored in {time.time() - t0:.0f}s  |  val customers={len(targets):,}")

    # --- Phase 3 table ---
    rows = []
    rows.append(evaluate("popularity", PopularityRecommender().fit(train).recommend(targets, args.k), truth, train, n_catalog, args.k))
    knn_recs = ItemKNNRecommender().fit(train).recommend(targets, k=args.k * 5)
    rows.append(evaluate("item_knn", {c: r[: args.k] for c, r in knn_recs.items()}, truth, train, n_catalog, args.k))
    ranker_recs = result.predictions
    rows.append(evaluate("lgbm_ranker", {c: r[: args.k] for c, r in ranker_recs.items()}, truth, train, n_catalog, args.k))
    phase3 = pd.DataFrame(rows)

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    phase3.to_csv(out / "phase3_results.csv", index=False)
    ranker.feature_importance_.rename("gain").to_csv(out / "feature_importance.csv")
    print("\n== Phase 3 ==")
    print(phase3.to_string(index=False))
    print("\ntop features by gain:")
    print(ranker.feature_importance_.head(10).round(0).to_string())

    # --- Phase 4 frontier ---
    feats = build_article_features(train, ds.articles)
    fr = pd.concat(
        [
            sweep(knn_recs, truth, train, feats, lambdas, "item_knn", args.k, args.n_boot),
            sweep(ranker_recs, truth, train, feats, lambdas, "lgbm_ranker", args.k, args.n_boot),
        ]
    )
    fr.to_csv(out / "frontier.csv", index=False)
    (ROOT / "docs").mkdir(exist_ok=True)
    plot_frontier(fr, str(ROOT / "docs" / "frontier.png"), args.k)
    print("\n== Phase 4: frontier (95% bootstrap CI) ==")
    print(fr.to_string(index=False))
    print(f"\nplot -> docs/frontier.png   total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

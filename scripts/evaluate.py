"""Phase 1 evaluation: popularity vs item-kNN, then the sustainability lambda sweep.

    python scripts/evaluate.py --data data/sample
    python scripts/evaluate.py --data data/raw --sample-customers 50000

Writes results/phase1_results.csv and prints the table.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rewear.baseline import ItemKNNRecommender, PopularityRecommender
from rewear.data import load_raw, temporal_split
from rewear.metrics import evaluate
from rewear.sustainability import SustainabilityReranker, build_article_features


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sample")
    ap.add_argument("--sample-customers", type=int, default=None)
    ap.add_argument("--val-days", type=int, default=7)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--lambdas", default="0.0,0.1,0.2,0.3,0.5,0.7,1.0")
    args = ap.parse_args()

    ds = load_raw(args.data, sample_customers=args.sample_customers)
    split = temporal_split(ds.transactions, val_days=args.val_days)
    truth = split.ground_truth()
    targets = list(truth)
    n_catalog = split.train["article_id"].nunique()
    print(f"train rows={len(split.train):,}  val rows={len(split.val):,}  val customers={len(targets):,}  catalog={n_catalog:,}")

    rows = []

    pop = PopularityRecommender().fit(split.train)
    rows.append(evaluate("popularity", pop.recommend(targets, args.k), truth, split.train, n_catalog, args.k))

    knn = ItemKNNRecommender().fit(split.train)
    knn_recs = knn.recommend(targets, k=args.k * 5)  # wide candidate list for the reranker
    rows.append(evaluate("item_knn", {c: r[: args.k] for c, r in knn_recs.items()}, truth, split.train, n_catalog, args.k))

    feats = build_article_features(split.train, ds.articles)
    for lam in [float(x) for x in args.lambdas.split(",")]:
        rr = SustainabilityReranker(feats, lam=lam)
        rows.append(evaluate(f"item_knn+rewear(λ={lam})", rr.rerank(knn_recs, args.k), truth, split.train, n_catalog, args.k))

    results = pd.DataFrame(rows)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    results.to_csv(out / "phase1_results.csv", index=False)
    print()
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

"""Phase 5: export a static demo.

Trains the Phase 3 pipeline, picks a handful of validation customers with
enough history to be interesting, precomputes their top-12 at every λ, and
inlines it all into docs/demo/index.html so the demo is a single static page
that GitHub Pages can serve. No backend, no inference at request time.

    python scripts/export_demo.py --data data/sample
    python scripts/export_demo.py --data data/raw --sample-customers 50000 --n-customers 24
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rewear.data import load_raw
from rewear.frontier import sweep
from rewear.ranker import train_and_evaluate_pipeline
from rewear.sustainability import SustainabilityReranker, build_article_features

LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def article_card(aid: str, articles: pd.DataFrame, feats: pd.DataFrame) -> dict:
    a = articles.loc[aid]
    f = feats.loc[aid]
    return {
        "id": aid,
        "name": str(a.get("prod_name", "")),
        "type": str(a.get("product_type_name", "")),
        "colour": str(a.get("colour_group_name", "")),
        "material": round(float(f["material_score"]), 2),
        "wear_again": round(float(f["wear_again_score"]), 2),
        "popularity": round(float(f["popularity"]), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sample")
    ap.add_argument("--sample-customers", type=int, default=None)
    ap.add_argument("--n-customers", type=int, default=16)
    ap.add_argument("--min-history", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    ds = load_raw(args.data, sample_customers=args.sample_customers)
    _, result, train, truth = train_and_evaluate_pipeline(ds.transactions, ds.articles, ds.customers, k=60, n_rounds=args.rounds)
    feats = build_article_features(train, ds.articles)
    articles = ds.articles.set_index("article_id")

    # Metrics per λ for the ranker, so the page can show the cost of the dial.
    frontier = sweep(result.predictions, truth, train, feats, LAMBDAS, "lgbm_ranker", 12, n_boot=200)
    metrics = {
        f"{r['lambda']:.1f}": {
            "map12": r["map@12"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
            "material": r["mean_material"], "long_tail": r["long_tail_exposure"], "coverage": r["coverage"],
        }
        for _, r in frontier.iterrows()
    }

    # Pick customers with enough history.
    hist = train.sort_values("t_dat", ascending=False).groupby("customer_id")["article_id"].apply(lambda s: list(dict.fromkeys(s)))
    eligible = [c for c in result.predictions if len(hist.get(c, [])) >= args.min_history]
    rng = np.random.default_rng(args.seed)
    chosen = list(rng.choice(eligible, size=min(args.n_customers, len(eligible)), replace=False))

    customers = []
    for i, cid in enumerate(chosen):
        base = {cid: result.predictions[cid]}
        recs = {}
        for lam in LAMBDAS:
            top = SustainabilityReranker(feats, lam=lam).rerank(base, k=12)[cid]
            recs[f"{lam:.1f}"] = [article_card(a, articles, feats) for a in top]
        customers.append(
            {
                "label": f"Wardrobe {i + 1}",
                "history": [article_card(a, articles, feats) for a in hist[cid][:8]],
                "bought_next": [article_card(a, articles, feats) for a in truth[cid][:6] if a in articles.index],
                "recs": recs,
            }
        )

    payload = {"lambdas": [f"{x:.1f}" for x in LAMBDAS], "metrics": metrics, "customers": customers}
    out_dir = ROOT / "docs" / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.json").write_text(json.dumps(payload))

    template = (ROOT / "docs" / "demo" / "template.html").read_text()
    html = template.replace("/*__REWEAR_DATA__*/null", json.dumps(payload))
    (out_dir / "index.html").write_text(html)
    print(f"exported {len(customers)} wardrobes x {len(LAMBDAS)} λ -> docs/demo/index.html")


if __name__ == "__main__":
    main()

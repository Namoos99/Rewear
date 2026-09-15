"""Generate a small synthetic dataset with the exact H&M schema.

Used by the test suite and by `make sample` so the whole pipeline can be
exercised before downloading the real ~30GB Kaggle archive. Purchases are
drawn from latent customer "tastes" so collaborative filtering has real
structure to find (and a pure-random model can't cheat).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PRODUCT_TYPES = ["Trousers", "Dress", "Sweater", "T-shirt", "Jacket", "Skirt", "Blouse", "Shorts", "Coat", "Cardigan"]
COLOURS = ["Black", "White", "Beige", "Dark Blue", "Light Pink", "Green", "Grey", "Red"]
MATERIAL_PHRASES = [
    "in organic cotton",
    "in recycled polyester",
    "in linen",
    "in viscose",
    "in polyester",
    "in cotton with elastane",
    "in acrylic",
    "in wool blend",
]


def make_synthetic(
    out_dir: str | Path,
    n_customers: int = 400,
    n_articles: int = 300,
    n_days: int = 120,
    n_tastes: int = 6,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    article_ids = [f"{700000000 + i:010d}" for i in range(n_articles)]
    article_taste = rng.integers(0, n_tastes, size=n_articles)
    # Popularity is long-tailed, like the real thing.
    base_pop = rng.pareto(1.5, size=n_articles) + 1
    base_pop /= base_pop.sum()

    articles = pd.DataFrame(
        {
            "article_id": article_ids,
            "prod_name": [f"Item {i}" for i in range(n_articles)],
            "product_type_name": rng.choice(PRODUCT_TYPES, size=n_articles),
            "product_group_name": "Garment",
            "graphical_appearance_name": "Solid",
            "colour_group_name": rng.choice(COLOURS, size=n_articles),
            "department_name": "Ladieswear",
            "index_group_name": "Ladieswear",
            "garment_group_name": "Jersey Basic",
            "detail_desc": [
                f"{t} {rng.choice(MATERIAL_PHRASES)} with a regular fit."
                for t in rng.choice(PRODUCT_TYPES, size=n_articles)
            ],
        }
    )

    customer_ids = [f"c{i:06d}" for i in range(n_customers)]
    customer_taste = rng.integers(0, n_tastes, size=n_customers)
    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "FN": np.nan,
            "Active": np.nan,
            "club_member_status": "ACTIVE",
            "fashion_news_frequency": "NONE",
            "age": rng.integers(18, 65, size=n_customers),
            "postal_code": "x",
        }
    )

    start = pd.Timestamp("2020-06-01")
    rows = []
    for ci, cid in enumerate(customer_ids):
        n_purchases = rng.poisson(12) + 2
        # 70% of purchases come from the customer's own taste cluster.
        in_taste = article_taste == customer_taste[ci]
        p = base_pop * np.where(in_taste, 7.0, 1.0)
        p /= p.sum()
        chosen = rng.choice(n_articles, size=n_purchases, p=p)
        days = np.sort(rng.integers(0, n_days, size=n_purchases))
        for a, d in zip(chosen, days):
            rows.append((start + pd.Timedelta(days=int(d)), cid, article_ids[a], round(float(rng.uniform(0.01, 0.1)), 4), 2))

    transactions = pd.DataFrame(rows, columns=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"])
    transactions = transactions.sort_values("t_dat").reset_index(drop=True)

    articles.to_csv(out_dir / "articles.csv", index=False)
    customers.to_csv(out_dir / "customers.csv", index=False)
    transactions.to_csv(out_dir / "transactions_train.csv", index=False, date_format="%Y-%m-%d")

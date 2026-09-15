"""Data loading, sampling, and temporal splitting for the H&M dataset.

Schema (Kaggle: h-and-m-personalized-fashion-recommendations)
    articles.csv      one row per product (article_id, product_type_name, colour_group_name,
                      detail_desc, index_group_name, garment_group_name, ...)
    customers.csv     one row per customer (customer_id, age, club_member_status, ...)
    transactions_train.csv  one row per purchase (t_dat, customer_id, article_id, price, sales_channel_id)

The full transactions file is ~31M rows. Every function here accepts an optional
`sample_customers` so the whole pipeline runs on a laptop in minutes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ARTICLE_COLS = [
    "article_id",
    "prod_name",
    "product_type_name",
    "product_group_name",
    "graphical_appearance_name",
    "colour_group_name",
    "department_name",
    "index_group_name",
    "garment_group_name",
    "detail_desc",
]


@dataclass
class Dataset:
    articles: pd.DataFrame
    customers: pd.DataFrame
    transactions: pd.DataFrame

    @property
    def n_customers(self) -> int:
        return self.transactions["customer_id"].nunique()

    @property
    def n_articles(self) -> int:
        return self.transactions["article_id"].nunique()


def load_raw(raw_dir: str | Path, sample_customers: int | None = None, seed: int = 42) -> Dataset:
    """Load the three CSVs. Optionally keep only a random subset of customers.

    article_id is read as string because the leading zero is significant
    (Kaggle's images are named by the 10-digit zero-padded id).
    """
    raw_dir = Path(raw_dir)
    articles = pd.read_csv(
        raw_dir / "articles.csv",
        dtype={"article_id": str},
        usecols=lambda c: c in ARTICLE_COLS,
    )
    customers = pd.read_csv(raw_dir / "customers.csv", dtype={"customer_id": str})
    transactions = pd.read_csv(
        raw_dir / "transactions_train.csv",
        dtype={"article_id": str, "customer_id": str},
        parse_dates=["t_dat"],
    )

    if sample_customers is not None:
        rng = np.random.default_rng(seed)
        keep = rng.choice(customers["customer_id"].unique(), size=sample_customers, replace=False)
        keep = set(keep)
        customers = customers[customers["customer_id"].isin(keep)].reset_index(drop=True)
        transactions = transactions[transactions["customer_id"].isin(keep)].reset_index(drop=True)
        articles = articles[articles["article_id"].isin(transactions["article_id"].unique())].reset_index(drop=True)

    return Dataset(articles=articles, customers=customers, transactions=transactions)


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    val_start: pd.Timestamp

    def ground_truth(self) -> dict[str, list[str]]:
        """customer_id -> unique articles bought in the validation window."""
        return (
            self.val.groupby("customer_id")["article_id"]
            .apply(lambda s: list(dict.fromkeys(s)))
            .to_dict()
        )


def temporal_split(transactions: pd.DataFrame, val_days: int = 7) -> Split:
    """Hold out the final `val_days` of transactions as validation.

    This mirrors the Kaggle competition: predict the week after the training data ends.
    A random split would leak future purchases into training and inflate every metric.
    """
    last = transactions["t_dat"].max()
    val_start = last - pd.Timedelta(days=val_days - 1)
    train = transactions[transactions["t_dat"] < val_start].reset_index(drop=True)
    val = transactions[transactions["t_dat"] >= val_start].reset_index(drop=True)
    return Split(train=train, val=val, val_start=val_start)


def build_interaction_matrix(train: pd.DataFrame):
    """Sparse customer x article matrix of purchase counts, with index maps.

    Returns (matrix, customer_index, article_index) where the indexes are
    pandas Index objects so you can go id -> position via .get_loc and back via [pos].
    """
    from scipy.sparse import csr_matrix

    customer_index = pd.Index(train["customer_id"].unique(), name="customer_id")
    article_index = pd.Index(train["article_id"].unique(), name="article_id")

    rows = customer_index.get_indexer(train["customer_id"])
    cols = article_index.get_indexer(train["article_id"])
    data = np.ones(len(train), dtype=np.float32)

    mat = csr_matrix((data, (rows, cols)), shape=(len(customer_index), len(article_index)))
    mat.sum_duplicates()
    return mat, customer_index, article_index

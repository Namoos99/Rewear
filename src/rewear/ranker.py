"""Phase 3, step 2: a learned reranker over the candidate set.

Training scheme (no leakage):

    |-------- fit generators on this --------|-- label week --|-- validation week --|
                                              ^ ranker trains  ^ everything is
                                                on candidates    re-fit on train+label
                                                labelled here    and scored here

So the ranker learns "given candidates from these sources with these
features, which get bought next week?" on a week it never sees at eval time.

Features
  source flags + per-source ranks (learned source reliability)
  content similarity to the customer's taste vector
  article: popularity, recency of last sale, mean price, material_score, wear_again_score
  customer: history length, days since last purchase, mean spend, age
  interaction: customer has bought this product type / colour before

The sustainability signals are *features* here — the ranker decides how much
they matter for purchase probability. The λ dial from Phase 1 still sits on
top, so we can trace the accuracy-vs-sustainability frontier of a model
that already knows about sustainability. That's the Phase 4 plot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from .candidates import CandidateGenerator
from .sustainability import build_article_features

FEATURES = [
    "src_repurchase", "src_knn", "src_content", "src_popularity", "n_sources",
    "repurchase_rank", "knn_rank", "content_rank", "pop_rank", "content_sim",
    "art_popularity", "art_days_since_sale", "art_mean_price", "material_score", "wear_again_score",
    "cust_n_purchases", "cust_days_since", "cust_mean_price", "cust_age",
    "bought_type_before", "bought_colour_before",
]


def _article_side(train: pd.DataFrame, articles: pd.DataFrame) -> pd.DataFrame:
    last = train["t_dat"].max()
    g = train.groupby("article_id")
    art = pd.DataFrame(
        {
            "art_days_since_sale": (last - g["t_dat"].max()).dt.days,
            "art_mean_price": g["price"].mean(),
        }
    )
    sust = build_article_features(train, articles)
    art = art.join(sust, how="outer")
    art = art.rename(columns={"popularity": "art_popularity"})
    meta = articles.set_index("article_id")[["product_type_name", "colour_group_name"]]
    return art.join(meta, how="left")


def _customer_side(train: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    last = train["t_dat"].max()
    g = train.groupby("customer_id")
    cust = pd.DataFrame(
        {
            "cust_n_purchases": g.size(),
            "cust_days_since": (last - g["t_dat"].max()).dt.days,
            "cust_mean_price": g["price"].mean(),
        }
    )
    age = customers.set_index("customer_id")["age"] if "age" in customers.columns else None
    cust["cust_age"] = age.reindex(cust.index) if age is not None else np.nan
    return cust


def _history_sets(train: pd.DataFrame, articles: pd.DataFrame) -> tuple[dict, dict]:
    m = train.merge(articles[["article_id", "product_type_name", "colour_group_name"]], on="article_id", how="left")
    types = m.groupby("customer_id")["product_type_name"].apply(set).to_dict()
    colours = m.groupby("customer_id")["colour_group_name"].apply(set).to_dict()
    return types, colours


def build_features(cands: pd.DataFrame, train: pd.DataFrame, articles: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    art = _article_side(train, articles)
    cust = _customer_side(train, customers)
    types, colours = _history_sets(train, articles)

    df = cands.merge(art, left_on="article_id", right_index=True, how="left")
    df = df.merge(cust, left_on="customer_id", right_index=True, how="left")
    df["bought_type_before"] = [
        int(t in types.get(c, ())) for c, t in zip(df["customer_id"], df["product_type_name"])
    ]
    df["bought_colour_before"] = [
        int(k in colours.get(c, ())) for c, k in zip(df["customer_id"], df["colour_group_name"])
    ]
    for col in FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    return df


@dataclass
class RankerResult:
    predictions: dict[str, list[str]]
    scored: pd.DataFrame  # customer_id, article_id, score — the full candidate set, for λ-reranking


@dataclass
class LGBMRanker:
    """LambdaRank over candidates, grouped by customer."""

    params: dict = field(
        default_factory=lambda: {
            "objective": "lambdarank",
            "metric": "map",
            "eval_at": [12],
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambdarank_truncation_level": 12,
            "verbose": -1,
            "seed": 0,
        }
    )
    n_rounds: int = 300
    neg_per_pos: int = 30
    model: lgb.Booster | None = None
    feature_importance_: pd.Series | None = None

    def fit(self, feats: pd.DataFrame, labels: pd.Series, seed: int = 0) -> LGBMRanker:
        df = feats.assign(label=labels.to_numpy())
        # Only customers with at least one positive can teach a ranker anything.
        has_pos = df.groupby("customer_id")["label"].transform("max") > 0
        df = df[has_pos]
        # Downsample negatives per customer to keep training fast and balanced-ish.
        rng = np.random.default_rng(seed)
        pos = df[df["label"] == 1]
        neg = df[df["label"] == 0]
        n_pos = pos.groupby("customer_id").size()
        keep = []
        for cid, grp in neg.groupby("customer_id"):
            n = min(len(grp), int(n_pos.get(cid, 1)) * self.neg_per_pos)
            keep.append(grp.sample(n=n, random_state=int(rng.integers(1 << 30))))
        df = pd.concat([pos] + keep).sort_values("customer_id")
        group = df.groupby("customer_id", sort=False).size().to_numpy()

        dtrain = lgb.Dataset(df[FEATURES], label=df["label"], group=group)
        self.model = lgb.train(self.params, dtrain, num_boost_round=self.n_rounds)
        self.feature_importance_ = pd.Series(self.model.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
        return self

    def rank(self, feats: pd.DataFrame, k: int = 12) -> RankerResult:
        assert self.model is not None, "call fit() first"
        scored = feats[["customer_id", "article_id"]].copy()
        scored["score"] = self.model.predict(feats[FEATURES])
        scored = scored.sort_values(["customer_id", "score"], ascending=[True, False])
        preds = scored.groupby("customer_id")["article_id"].apply(lambda s: s.head(k).tolist()).to_dict()
        return RankerResult(predictions=preds, scored=scored.reset_index(drop=True))


def label_candidates(cands: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    """1 if (customer, article) was bought in the target window, else 0."""
    bought = set(zip(target["customer_id"], target["article_id"]))
    return pd.Series([int((c, a) in bought) for c, a in zip(cands["customer_id"], cands["article_id"])], index=cands.index)


def train_and_evaluate_pipeline(
    transactions: pd.DataFrame,
    articles: pd.DataFrame,
    customers: pd.DataFrame,
    val_days: int = 7,
    label_days: int = 7,
    k: int = 12,
    n_rounds: int = 300,
) -> tuple[LGBMRanker, RankerResult, pd.DataFrame, dict[str, list[str]]]:
    """Full Phase 3 run. Returns (ranker, val_result, val_train_df, ground_truth)."""
    from .data import temporal_split

    outer = temporal_split(transactions, val_days=val_days)   # train | val
    inner = temporal_split(outer.train, val_days=label_days)  # fit | label

    # --- train the ranker on the label week ---
    label_customers = inner.val["customer_id"].unique().tolist()
    gen = CandidateGenerator().fit(inner.train, articles)
    cands = gen.generate(label_customers)
    feats = build_features(cands, inner.train, articles, customers)
    labels = label_candidates(cands, inner.val)
    ranker = LGBMRanker(n_rounds=n_rounds).fit(feats, labels)

    # --- score the validation week with everything re-fit on the full train ---
    truth = outer.ground_truth()
    targets = list(truth)
    gen = CandidateGenerator().fit(outer.train, articles)
    cands = gen.generate(targets)
    feats = build_features(cands, outer.train, articles, customers)
    result = ranker.rank(feats, k=k)
    return ranker, result, outer.train, truth

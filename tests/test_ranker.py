import numpy as np

from rewear.candidates import SOURCES, CandidateGenerator
from rewear.frontier import bootstrap_map, sweep
from rewear.metrics import evaluate
from rewear.ranker import FEATURES, build_features, label_candidates, train_and_evaluate_pipeline
from rewear.sustainability import build_article_features


def test_candidates_union_all_sources(dataset, split):
    gen = CandidateGenerator(n_knn=10, n_content=10, n_pop=10).fit(split.train, dataset.articles)
    cids = split.train["customer_id"].unique()[:20].tolist()
    c = gen.generate(cids)
    assert set(c["customer_id"]) == set(cids)
    assert all(c[f"src_{s}"].sum() > 0 for s in SOURCES)
    assert not c.duplicated(["customer_id", "article_id"]).any()
    assert c["n_sources"].max() >= 2  # sources overlap, that's the point


def test_features_complete_and_labels_binary(dataset, split):
    gen = CandidateGenerator(n_knn=10, n_content=10, n_pop=10).fit(split.train, dataset.articles)
    cids = split.val["customer_id"].unique().tolist()
    c = gen.generate(cids)
    f = build_features(c, split.train, dataset.articles, dataset.customers)
    assert all(col in f.columns for col in FEATURES)
    y = label_candidates(c, split.val)
    assert set(y.unique()) <= {0, 1} and y.sum() > 0


def test_ranker_beats_popularity_and_knn(dataset, split):
    ranker, result, train, truth = train_and_evaluate_pipeline(
        dataset.transactions, dataset.articles, dataset.customers, k=12, n_rounds=60
    )
    from rewear.baseline import ItemKNNRecommender, PopularityRecommender

    n_catalog = train["article_id"].nunique()
    targets = list(truth)
    pop = evaluate("pop", PopularityRecommender().fit(train).recommend(targets), truth, train, n_catalog)
    knn = evaluate("knn", ItemKNNRecommender().fit(train).recommend(targets), truth, train, n_catalog)
    lgb = evaluate("lgb", result.predictions, truth, train, n_catalog)
    assert lgb["map@12"] > pop["map@12"]
    assert lgb["map@12"] > knn["map@12"]
    assert ranker.feature_importance_.sum() > 0


def test_bootstrap_ci_contains_point_estimate():
    truth = {f"u{i}": ["a"] for i in range(50)}
    preds = {f"u{i}": (["a"] if i % 2 else ["b"]) for i in range(50)}
    m, lo, hi = bootstrap_map(truth, preds, n_boot=200)
    assert abs(m - 0.5) < 1e-9 and lo <= m <= hi


def test_sweep_lambda_zero_matches_base(dataset, split):
    from rewear.baseline import ItemKNNRecommender

    truth = split.ground_truth()
    recs = ItemKNNRecommender().fit(split.train).recommend(list(truth), k=60)
    feats = build_article_features(split.train, dataset.articles)
    df = sweep(recs, truth, split.train, feats, [0.0, 0.5], "knn", n_boot=50)
    base = evaluate("knn", {c: r[:12] for c, r in recs.items()}, truth, split.train, split.train["article_id"].nunique())
    assert np.isclose(df.iloc[0]["map@12"], base["map@12"], atol=1e-5)
    assert df.iloc[1]["mean_material"] >= df.iloc[0]["mean_material"]

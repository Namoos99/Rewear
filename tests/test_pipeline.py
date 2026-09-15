from rewear.baseline import ItemKNNRecommender, PopularityRecommender
from rewear.metrics import evaluate
from rewear.sustainability import SustainabilityReranker, build_article_features, material_score


def test_temporal_split_has_no_leakage(split):
    assert split.train["t_dat"].max() < split.val_start
    assert split.val["t_dat"].min() >= split.val_start


def test_popularity_returns_k_items(split):
    truth = split.ground_truth()
    recs = PopularityRecommender().fit(split.train).recommend(list(truth), k=12)
    assert all(len(r) == 12 for r in recs.values())


def test_item_knn_beats_popularity_on_structured_data(split):
    truth = split.ground_truth()
    targets = list(truth)
    n_catalog = split.train["article_id"].nunique()
    pop = evaluate("pop", PopularityRecommender().fit(split.train).recommend(targets), truth, split.train, n_catalog)
    knn = evaluate("knn", ItemKNNRecommender().fit(split.train).recommend(targets), truth, split.train, n_catalog)
    # Synthetic data has taste clusters, so personalisation must win on both accuracy and diversity.
    assert knn["map@12"] > pop["map@12"]
    assert knn["coverage"] > pop["coverage"]


def test_knn_never_recommends_owned_items(split):
    model = ItemKNNRecommender(exclude_owned=True).fit(split.train)
    cid = split.train["customer_id"].iloc[0]
    owned = set(split.train.loc[split.train["customer_id"] == cid, "article_id"])
    assert not owned.intersection(model.recommend([cid])[cid])


def test_material_score_ordering():
    assert material_score("Top in organic cotton") > material_score("Top in cotton") > material_score("Top in polyester")
    assert material_score("") == 0.5
    assert material_score(float("nan")) == 0.5


def test_reranker_lambda_zero_is_identity(dataset, split):
    truth = split.ground_truth()
    knn = ItemKNNRecommender().fit(split.train).recommend(list(truth), k=60)
    feats = build_article_features(split.train, dataset.articles)
    same = SustainabilityReranker(feats, lam=0.0).rerank(knn, k=12)
    assert all(same[c] == knn[c][:12] for c in knn)


def test_reranker_raises_material_score(dataset, split):
    truth = split.ground_truth()
    knn = ItemKNNRecommender().fit(split.train).recommend(list(truth), k=60)
    feats = build_article_features(split.train, dataset.articles)

    def mean_material(recs):
        ids = [a for r in recs.values() for a in r]
        return feats.loc[ids, "material_score"].mean()

    base = mean_material({c: r[:12] for c, r in knn.items()})
    green = mean_material(SustainabilityReranker(feats, lam=0.7).rerank(knn, k=12))
    assert green > base

import numpy as np

from rewear.content import ContentEmbedder


def test_tfidf_embedder_finds_similar_items(dataset):
    emb = ContentEmbedder(backend="tfidf", dim=32).fit(dataset.articles)
    assert emb.vectors.shape[0] == len(dataset.articles)
    aid = dataset.articles["article_id"].iloc[0]
    sims = emb.most_similar(aid, k=5)
    assert len(sims) == 5 and all(a != aid for a, _ in sims)
    # Nearest neighbour should share the product type more often than not.
    ptype = dataset.articles.set_index("article_id")["product_type_name"]
    assert ptype[sims[0][0]] == ptype[aid]


def test_content_recommendations_exclude_history(dataset):
    emb = ContentEmbedder(backend="tfidf", dim=32).fit(dataset.articles)
    history = dataset.articles["article_id"].iloc[:3].tolist()
    recs = emb.recommend_for_history(history, k=12)
    assert len(recs) == 12 and not set(recs) & set(history)


def test_empty_history_gives_zero_vector(dataset):
    emb = ContentEmbedder(backend="tfidf", dim=32).fit(dataset.articles)
    assert np.allclose(emb.user_vector([]), 0)

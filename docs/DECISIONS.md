# Decisions

Each entry is written the way I'd answer the question in an interview: the choice, the alternatives, why, and what I'd revisit.

---

### Why the H&M dataset instead of scraping ASOS or using the Spotify API?

Data access decides feasibility before anything else. ASOS has no public API and scraping violates their terms. Spotify removed the recommendations and audio-features endpoints for new developer apps in late 2024, which guts the "discover new artists" idea unless you fall back to stale archives. H&M is the opposite case: real transaction logs at production scale, product images, product text, and a public benchmark (the Kaggle competition) so my numbers are comparable to thousands of other attempts. A rare combination.

### Why a temporal split and not a random one?

A random split leaks the future into training. If a customer buys a coat in week 40 and I train on weeks 1–52 minus a random 10%, the model has seen their week-41 purchases while predicting week 40. MAP@12 goes up and means nothing. Holding out the *final* week exactly matches how the system would be deployed: train on everything up to today, predict next week.

### Why does popularity beat item-kNN in Phase 1, and why did I leave that in?

Because it's true, and it's the most-repeated finding from the Kaggle competition. Fashion demand is dominated by new-in and heavily stocked items; a customer's history from three months ago is a weak predictor of what's on the rail this week. Personalised models win once you (a) generate candidates from several sources including popularity and repurchase, and (b) learn to rank them. That's Phase 3. Hiding the Phase 1 result would have made the roadmap look arbitrary.

### Why isn't "don't recommend items the customer already owns" the default?

It was, originally. It cut item-kNN's MAP@12 roughly in half on the sample, because repurchases (same item, or a size/colour variant) are a large fraction of real transactions. The relevance model shouldn't quietly encode a product opinion. "Show me new pieces" is a Rewear product decision, so it lives in the reranking layer as an explicit switch, where the cost is measured.

### Why a λ-blend reranker instead of a multi-objective model?

Explainability to the people who'd actually decide. A blend with one dial produces a curve — MAP@12 versus sustainability — that a merchandising lead can point at and say "we'll take the 4% accuracy hit for that much coverage." A jointly-trained multi-objective model gives you a better frontier but buries the trade-off inside the weights. For a first version the legible option is the right one. Phase 3 moves the sustainability signals *into* the learned ranker as features, which gets the best of both.

### Why rerank the top 60 rather than re-score the whole catalogue?

Reranking a candidate set is how every production recommender works: a cheap retrieval stage that's recall-oriented, then an expensive ranker over a few dozen items. Re-scoring 105k articles per user with a sustainability blend would be slow *and* would let a terrible-fit organic-linen dress outrank a great-fit polyester one purely on material. The 5× multiplier means the base model's relevance still bounds what can be shown.

### How do you score sustainability when you can't measure it?

Honestly and with proxies I can defend:

- **Material** — keyword parsing on `detail_desc` against a fibre-impact table informed by Higg MSI-style rankings and the Fashion Transparency Index. Crude, but H&M's descriptions consistently name fibres, and I unit-test the ordering (organic cotton > cotton > polyester).
- **Wear-again** — category repurchase rate. If people who buy in a category keep buying in it, that category tends to be staples (basics, knitwear) rather than occasion pieces. It is not garment durability. It is the best proxy available in a transaction log.
- **Popularity penalty** — the only signal I'm confident in. The head of the sales curve is fast-fashion's core loop; a system that just amplifies it is not a sustainability system.

What I'd add with more data: return rates (a strong "didn't work out" signal), material composition percentages, and any care/durability metadata.

### Why TF-IDF as the default text backend?

Because CI has no GPU, no model cache, and shouldn't need either to prove the pipeline works. `sentence-transformers` is one flag away and better; TF-IDF+SVD is surprisingly competitive because the descriptions are formulaic marketing copy. Keeping a zero-dependency backend is the same instinct as offline fallbacks in a product: the demo should never break because a download failed.

### Why a synthetic dataset generator?

The real archive is ~30GB with images, which nobody should need to download to run `pytest`. The generator writes the exact three-file schema with latent taste clusters and a Pareto popularity curve, so collaborative filtering has real structure to find and the tests can assert *behaviour* (kNN beats popularity on clustered data, λ=0 is the identity, the reranker raises material score) rather than just "it didn't crash."

### What would you do differently at scale?

Approximate nearest neighbours (FAISS/ScaNN) for the content tower; incremental item-item updates instead of a full recompute; per-user λ learned from behaviour (some customers clearly *want* the green option, most won't take a big accuracy hit for it); and an A/B test against return rate, not just clicks, because return rate is the closest online proxy to "kept wearing it."

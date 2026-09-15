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

### Why train the ranker on the week *before* validation?

Because a ranker needs labels, and the only labels that don't leak are from a window the evaluation never sees. The scheme is: fit the candidate generators on everything up to week N-1, generate candidates for the customers who bought in week N-1, label those candidates by what they actually bought, train LightGBM. Then re-fit the generators on everything up to week N and score week N with the trained model. The ranker learns "which candidates from these sources convert", never "what was bought in the week I'm graded on". It's the same discipline as the temporal split, one level down.

### Why downsample negatives, and why 30 per positive?

A customer with 2 purchases and 150 candidates is 148 negatives to 2 positives. LambdaRank handles imbalance better than a classifier, but training on every negative is slow and the extra ones are mostly uninformative (the 120th popularity candidate nobody bought). 30:1 keeps the hard negatives — items several sources agreed on that still didn't convert — which is where the learning is. It's a speed/quality knob, not a principle; I'd sweep it with more compute.

### Why are the sustainability signals features in the ranker *and* a dial on top?

Two different questions. As features, they let the model learn whether material or wear-again *predicts purchase* — on the Phase 1 data, a light nudge helped, and the ranker can find that on its own. As a dial, they let a product owner push *past* what predicts purchase toward what they value, and see the cost. The frontier plot is the ranker's own accuracy traded against its own sustainability profile; that's a stronger statement than the Phase 1 version, because this model already knows everything the dial knows.

### Why bootstrap confidence intervals?

Because 2,590 validation customers and MAP@12 values around 0.01 produce a noisy estimate, and a table with five decimals invites overreading. Resampling customers with replacement 500 times gives a 95% interval per λ. If two points' intervals overlap, I say they're indistinguishable. This is the same instinct as reporting that r=0.37 explains only 14% of variance: put the uncertainty next to the number.

### Why a static demo and not a Streamlit app?

Because the demo has to still work in two years when someone clicks it from a resume. A Streamlit app needs a running server, an environment, and the dataset; a static page needs a CDN. Precomputing 24 wardrobes × 11 λ values × 12 items is ~300KB of JSON, which is nothing. The interaction — drag the dial, tiles reorder, metrics update — is the whole story, and it doesn't need inference at request time. Offline resilience as a design default.

### What would you do differently at scale?

Approximate nearest neighbours (FAISS/ScaNN) for the content tower; incremental item-item updates instead of a full recompute; per-user λ learned from behaviour (some customers clearly *want* the green option, most won't take a big accuracy hit for it); and an A/B test against return rate, not just clicks, because return rate is the closest online proxy to "kept wearing it."

---

### Why multi-source candidates instead of a better single model?

Because the Phase 1 loss to popularity was a recall problem wearing an accuracy costume. Item-kNN can only propose items co-purchased with the customer's history; popularity can only propose the head; content can only propose look-alikes. Each misses what the others catch. Unioning them and letting a ranker sort it out is the standard production pattern (retrieval → ranking), and it's also what every strong Kaggle solution on this dataset did. I log candidate recall separately so I always know whether a weak MAP@12 is a retrieval ceiling or a ranking failure.

### How do you train the ranker without leaking the validation week?

Two temporal splits, nested. The outer split holds out the final week for evaluation. Inside the training portion, a second split holds out the *previous* week as the label week: generators are fit on everything before it, candidates are generated for customers active in it, and the ranker learns which candidates got bought. At evaluation time everything is refit on the full training portion and scored on the final week — a week the ranker has never seen a label from. The alternative (train and evaluate the ranker on the same week) is a common mistake that produces spectacular, meaningless numbers.

### Why LambdaRank rather than a binary classifier?

The objective matches the metric. MAP@12 cares about the *order* within a customer's list and only the top 12 of it; a classifier optimises pointwise probabilities across all customers at once and can spend its capacity separating easy negatives nobody would have shown anyway. LambdaRank with truncation at 12 focuses gradient where the metric is measured. I downsample negatives per customer (30 per positive) so training stays fast and groups stay balanced-ish.

### Why put the sustainability signals into the ranker *and* keep the λ dial?

They answer different questions. As features, the ranker learns whether material and wear-again *predict purchase* — on synthetic data they don't rank highly, which is honest: people don't buy on fibre content. The λ dial on top is the product lever: given a purchase-optimal ranking, how far do we deliberately move toward what lasts, and what does it cost? Phase 4 traces that curve for a model that already knows about sustainability, which is the fair version of the question.

### Why bootstrap confidence intervals?

Because the Phase 1 "λ=0.2 beats λ=0" finding on real data was a 15% relative gain on ~2,600 customers, and I couldn't tell from a point estimate whether that was signal. Resampling customers with replacement and recomputing MAP@12 gives a 95% interval for every point on the sweep. If intervals overlap, I say so. Overclaiming a regularisation effect that's actually noise would be worse than not finding it.

### Why a static demo instead of a live model?

Because a portfolio demo has to still work in two years with zero maintenance. The export script trains the pipeline, picks sixteen wardrobes, precomputes their top-12 at eleven λ settings, and inlines everything into one HTML file GitHub Pages serves for free. No backend, no cold starts, no API keys expiring. The trade-off — you can't type in your own wardrobe — is the right one for this artefact; the code to do that is all in the repo.

### Why colour swatches instead of product images?

Two reasons. The Kaggle image licence is for competition use, not for rehosting on my site. And a demo that depends on 105k images is a demo that breaks. The colour group is real data from the catalogue, it makes the reorder legible at a glance, and the material bar carries the sustainability signal — which is the thing the demo is actually about.

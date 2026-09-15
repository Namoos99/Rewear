# Rewear 

**A fashion recommender that optimizes for what you'll keep wearing, not just what you'll buy.**

Most recommendation systems in fashion have one objective: the next purchase. Rewear asks a different question — *which pieces will earn a place in your rotation?* — and makes the trade-off between engagement and sustainability explicit, measurable, and tunable, instead of pretending the engagement-optimal ranking is the only one that exists.

Built on the [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) dataset: ~1.3M customers, ~105k articles with images and descriptions, ~31M real transactions.

> "The goal is to turn data into information, and information into insight." — Carly Fiorina

## What makes it different

Standard pipeline: candidates → rank by predicted purchase probability → show top 12.

Rewear pipeline: candidates → rank by relevance → **rerank by a sustainability score with a single dial (λ)** → show top 12, and *report what that dial cost you in accuracy.*

The sustainability score blends three signals per item:

| Signal | What it measures | How it's derived |
|---|---|---|
| `material_score` | Fibre impact | Parsed from product descriptions (organic cotton, linen, recycled ↑ — virgin polyester, acrylic ↓) |
| `wear_again_score` | "Becomes a staple" | Category-level repurchase rate: how often buyers in this category come back for more of it |
| `popularity` (penalised) | Fast-fashion churn | Purchase count — chasing the head of the curve is what the industry already does well without help |

Every model is evaluated on **accuracy and beyond**: MAP@12 (the Kaggle metric), catalogue coverage, novelty, and long-tail exposure. The interesting output isn't a single score — it's the curve showing how much MAP@12 you give up for each unit of sustainability, so a product team can choose a point on it deliberately.

## Results

*Phase 1 on 50,000 sampled H&M customers (real data). Validation = final week held out; 2,590 customers with purchases in that week; catalogue of 71,940 articles.*

| model | MAP@12 | coverage | novelty | long-tail exposure |
|---|---|---|---|---|
| popularity | 0.00810 | 0.0002 | 4.98 | 0.17 |
| item-kNN | 0.00522 | 0.2501 | 8.02 | 0.74 |
| item-kNN + Rewear (λ=0.2) | **0.00602** | 0.2473 | 7.98 | 0.73 |
| item-kNN + Rewear (λ=0.5) | 0.00448 | 0.2199 | 7.94 | 0.72 |

Two findings. First, popularity wins on raw accuracy — the same result the Kaggle leaderboard produced, now confirmed here, and the reason Phase 3 exists. Second, and more interesting: a *light* sustainability rerank (λ=0.2) doesn't just cost accuracy, it **improves** MAP@12 by 15% over plain item-kNN while giving up almost nothing in coverage or novelty. The wear-again and material signals appear to regularise a noisy collaborative-filtering signal. Above λ≈0.3 the expected trade-off takes over and accuracy falls. That knee in the curve is the product decision.

## Quickstart

```bash
git clone https://github.com/Namoos99/Rewear && cd Rewear
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

make sample      # synthetic data with the exact H&M schema — no download needed
make test        # 14 tests, ~2 seconds
make evaluate    # Phase 1 table + λ sweep → results/phase1_results.csv
```

To run on the real data: download the Kaggle archive into `data/raw/` (you need `articles.csv`, `customers.csv`, `transactions_train.csv`; images are optional until Phase 2), then:

```bash
python scripts/evaluate.py --data data/raw --sample-customers 50000
```

## Project layout

```
src/rewear/
  data.py            loading, customer sampling, temporal split (no leakage)
  baseline.py        popularity + time-decayed item-kNN
  sustainability.py  material parsing, wear-again proxy, λ-reranker
  content.py         text embeddings (TF-IDF or sentence-transformers), CLIP image embeddings
  metrics.py         MAP@K, coverage, novelty, long-tail exposure
  synthetic.py       schema-faithful synthetic generator for tests and CI
scripts/             make_sample.py, evaluate.py
tests/               pytest suite (runs in CI on every push)
docs/DECISIONS.md    architecture decisions, written as interview answers
```

## Roadmap

- [x] **Phase 1 — Baselines.** Temporal split, popularity, item-kNN, full metric suite, λ sweep.
- [x] **Phase 2 — Content tower.** Text embeddings with a no-download fallback; CLIP image embeddings for cold-start and the demo.
- [ ] **Phase 3 — Learned reranker.** Multi-source candidate generation (repurchase, kNN, content, popularity) → LightGBM ranker with sustainability features. This is where popularity finally loses.
- [ ] **Phase 4 — Evaluation write-up.** The accuracy-vs-sustainability frontier on real data, with confidence intervals.
- [ ] **Phase 5 — Demo.** Interactive front-end: pick a customer, drag the λ slider, watch the recommendations shift.

## Honest limitations

- We can't observe "wear" — `wear_again_score` is a category-level repurchase proxy, not garment durability.
- `material_score` is keyword-based on marketing copy. It's directionally right and easy to fool.
- A single λ is a blunt instrument. A production system would learn per-user or per-context weights.

These are written up properly in [`docs/DECISIONS.md`](docs/DECISIONS.md).

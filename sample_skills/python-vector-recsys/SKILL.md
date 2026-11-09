---
name: python-vector-recsys
description: Building embedding-based recommender systems in Python with two-tower models, ANN indexes (FAISS/HNSW), and offline evaluation
tags: [python, recsys, recommendations, embeddings, faiss, hnsw, numpy, ann]
scope: global
---

# Python Vector Recommender Systems

Patterns for candidate generation and ranking with dense vectors. Keep retrieval cheap
(approximate nearest neighbours over normalized embeddings) and spend compute on the
re-ranking stage.

## Architecture: Retrieval then Ranking

1. **Candidate generation** — a two-tower model embeds users and items into the same
   space. Retrieve the top 200-1000 items per user via ANN search.
2. **Ranking** — a heavier model (gradient-boosted trees or a small MLP) scores candidates
   with rich features: recency, price, user-item interaction counts, context.
3. **Re-ranking / business rules** — diversity (MMR), freshness boosts, de-duplication,
   filtering out already purchased items.

## Embedding Hygiene

- L2-normalize all vectors so dot product equals cosine similarity:
  `vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)`.
- Store embeddings as `float32`; `float64` doubles memory with no ranking benefit.
- Version embeddings together with the model checkpoint; never mix vectors from
  different model versions in one index.
- Cold-start items: fall back to a content tower embedding (title + category text) until
  enough interactions accumulate.

## ANN Index Selection

### FAISS

- `IndexFlatIP` for < 100k items: exact, no training, good baseline.
- `IndexIVFFlat` / `IndexIVFPQ` for millions of items; train on a representative sample,
  tune `nprobe` for the recall/latency trade-off.
- `IndexHNSWFlat` when you need high recall with low latency and can afford RAM.

### HNSW parameters

`M` (graph degree, 16-48) controls memory and recall; `efConstruction` (100-400) controls
build quality; `efSearch` is tuned at query time and is the main latency knob.

```python
import faiss, numpy as np

index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = 200
index.add(item_vecs.astype(np.float32))
index.hnsw.efSearch = 128
scores, ids = index.search(user_vecs, 500)
```

## Training Two-Tower Models

- Use in-batch negatives with a sampled-softmax loss; correct for popularity bias with
  logQ correction so popular items are not over-recommended.
- Mine hard negatives from the ANN index after a warm-up epoch.
- Temperature (0.05-0.1) on the logits matters more than tower depth.

## Offline Evaluation

- Split by time, never randomly: train on weeks 1-N, evaluate on week N+1.
- Report Recall@K for candidate generation and NDCG@K / MAP for ranking.
- Track coverage and popularity bias (Gini) alongside accuracy.
- Always compare against a "most popular in the last 7 days" baseline.

## Serving

- Precompute item embeddings in batch; compute user embeddings online from recent events.
- Keep the ANN index in memory per replica; rebuild nightly and hot-swap atomically.
- Cache top-K results per user for a few minutes to absorb traffic spikes.

# vecnorm

Fast bulk vector ops on f32 matrices. Rust core, Python frontend.

## What it does

The four operations every retrieval/re-ranking codebase rebuilds badly:

- **L2 normalize** an `(n, d)` matrix, in-place or copy
- **Cosine similarity** between two vectors (or two matrices)
- **Top-k argmax** over a 1-D score vector (partial heap, O(n log k))
- **Batch top-k argmax** over an `(n_queries, n_docs)` score matrix, parallelized via rayon

Pure NumPy can do all of this. The wedge is **batch top-k**: a partial heap
in Rust is materially faster than `np.argpartition` for small `k` over many
rows, and it returns scores alongside indices in a single pass.

## Install

```bash
pip install vecnorm
```

## Quickstart

```python
import numpy as np
from vecnorm import l2_normalize, cosine_similarity, top_k, batch_top_k

# L2-normalize a corpus of embeddings (in-place).
embeddings = np.random.randn(10_000, 768).astype(np.float32)
l2_normalize(embeddings)

# Cosine between two normalized vectors == dot product.
s = cosine_similarity(embeddings[0], embeddings[1])

# Top-10 nearest docs to a query.
query = embeddings[0]
scores = embeddings @ query                        # (n,)
hits = top_k(scores, k=10)                          # [(idx, score), ...] desc

# 1000 queries x 10000 docs, top-10 per query, parallel.
queries = np.random.randn(1_000, 768).astype(np.float32)
l2_normalize(queries)
all_scores = queries @ embeddings.T                 # (1000, 10000)
results = batch_top_k(all_scores, k=10, parallel=True)
```

## License

Dual-licensed under MIT or Apache-2.0.

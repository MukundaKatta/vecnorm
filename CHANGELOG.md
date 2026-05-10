# Changelog

## [0.1.1] - 2026-05-10

### Added
- `dot_product(a, b)` — inner product of two 1-D f32 vectors. No
  normalization, errors on dim mismatch.
- `argmax(scores)` — single-element top-1: returns `(index, score)` of
  the largest element, ties broken by ascending index. Useful when
  callers don't need full top-k machinery.

## [0.1.0] - 2026-05-09

### Added
- `l2_normalize` (in-place) and `l2_normalize_copy` for `(n, d)` f32 matrices.
- `cosine_similarity` for two 1-D f32 vectors.
- `cosine_distances` between two `(n, d)` matrices.
- `top_k_argmax` and parallel `batch_top_k_argmax` using a partial heap (O(n log k)).
- abi3-py310 wheel: one wheel for CPython 3.10 through 3.13.

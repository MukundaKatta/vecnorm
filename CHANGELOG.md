# Changelog

## [0.1.0] - 2026-05-09

### Added
- `l2_normalize` (in-place) and `l2_normalize_copy` for `(n, d)` f32 matrices.
- `cosine_similarity` for two 1-D f32 vectors.
- `cosine_distances` between two `(n, d)` matrices.
- `top_k_argmax` and parallel `batch_top_k_argmax` using a partial heap (O(n log k)).
- abi3-py310 wheel: one wheel for CPython 3.10 through 3.13.

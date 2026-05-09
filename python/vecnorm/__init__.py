"""Fast bulk vector ops on f32 matrices.

The native module ``vecnorm._native`` does the heavy lifting in Rust
(ndarray + rayon). This module is a thin re-export with f32-coercion and
keyword-only parameter polish.
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib import metadata
from typing import Final

import numpy as np
from numpy.typing import NDArray

from vecnorm._native import VecnormError
from vecnorm._native import batch_top_k as _batch_top_k
from vecnorm._native import cosine_distances as _cosine_distances
from vecnorm._native import cosine_similarity as _cosine_similarity
from vecnorm._native import l2_normalize as _l2_normalize
from vecnorm._native import l2_normalize_copy as _l2_normalize_copy
from vecnorm._native import top_k as _top_k


def _read_version() -> str:
    try:
        return metadata.version("vecnorm")
    except metadata.PackageNotFoundError:
        return "0.0.0"


__version__: Final[str] = _read_version()

__all__ = [
    "VecnormError",
    "__version__",
    "batch_top_k",
    "cosine_distances",
    "cosine_similarity",
    "l2_normalize",
    "l2_normalize_copy",
    "top_k",
]


def _as_f32_2d(name: str, arr: NDArray[np.float32]) -> NDArray[np.float32]:
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {arr.shape}")
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return np.ascontiguousarray(arr)


def _as_f32_1d(name: str, arr: NDArray[np.float32]) -> NDArray[np.float32]:
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return np.ascontiguousarray(arr)


def l2_normalize(matrix: NDArray[np.float32]) -> None:
    """L2-normalize each row in place. Zero rows stay zero (no NaN)."""
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got shape {matrix.shape}")
    if matrix.dtype != np.float32:
        raise ValueError(
            f"in-place l2_normalize requires float32 input, got {matrix.dtype}; "
            "use l2_normalize_copy to convert"
        )
    _l2_normalize(np.ascontiguousarray(matrix))


def l2_normalize_copy(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return an L2-normalized copy. Coerces to float32 if needed."""
    return _l2_normalize_copy(_as_f32_2d("matrix", matrix))


def cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Cosine similarity between two 1-D vectors. 0 for any zero input."""
    return float(_cosine_similarity(_as_f32_1d("a", a), _as_f32_1d("b", b)))


def cosine_distances(a: NDArray[np.float32], b: NDArray[np.float32]) -> NDArray[np.float32]:
    """`(n_a, n_b)` matrix of `1 - cos(a_i, b_j)`. Inputs auto-normalized."""
    return _cosine_distances(_as_f32_2d("a", a), _as_f32_2d("b", b))


def top_k(scores: NDArray[np.float32], k: int) -> list[tuple[int, float]]:
    """Top-k `(index, score)` pairs in descending order."""
    raw: Sequence[tuple[int, float]] = _top_k(_as_f32_1d("scores", scores), k)
    return [(int(i), float(s)) for i, s in raw]


def batch_top_k(
    scores: NDArray[np.float32], k: int, *, parallel: bool = False
) -> list[list[tuple[int, float]]]:
    """Per-row top-k over an `(n_rows, n_cols)` score matrix."""
    raw: Sequence[Sequence[tuple[int, float]]] = _batch_top_k(
        _as_f32_2d("scores", scores), k, parallel
    )
    return [[(int(i), float(s)) for i, s in row] for row in raw]

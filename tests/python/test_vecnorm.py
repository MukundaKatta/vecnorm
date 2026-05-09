"""End-to-end tests."""

from __future__ import annotations

import numpy as np
import pytest
from vecnorm import (
    VecnormError,
    __version__,
    batch_top_k,
    cosine_distances,
    cosine_similarity,
    l2_normalize,
    l2_normalize_copy,
    top_k,
)


def test_version_present() -> None:
    assert isinstance(__version__, str) and __version__ != ""


def test_l2_normalize_basic() -> None:
    a = np.array([[3.0, 4.0]], dtype=np.float32)
    l2_normalize(a)
    np.testing.assert_allclose(a, [[0.6, 0.8]], atol=1e-6)


def test_l2_normalize_zero_row() -> None:
    a = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    l2_normalize(a)
    np.testing.assert_array_equal(a[0], [0.0, 0.0])
    np.testing.assert_allclose(a[1], [0.6, 0.8], atol=1e-6)


def test_l2_normalize_requires_float32() -> None:
    a = np.array([[3.0, 4.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="float32"):
        l2_normalize(a)


def test_l2_normalize_copy_coerces_dtype() -> None:
    a = np.array([[3.0, 4.0]], dtype=np.float64)
    out = l2_normalize_copy(a)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [[0.6, 0.8]], atol=1e-6)
    # Source unchanged.
    np.testing.assert_array_equal(a, [[3.0, 4.0]])


def test_cosine_similarity_orthogonal() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert abs(cosine_similarity(a, b)) < 1e-6


def test_cosine_similarity_identical() -> None:
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert abs(cosine_similarity(a, a) - 1.0) < 1e-6


def test_cosine_similarity_zero_vector() -> None:
    a = np.zeros(3, dtype=np.float32)
    b = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    assert cosine_similarity(a, b) == 0.0


def test_cosine_similarity_dim_mismatch() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    with pytest.raises(ValueError):
        cosine_similarity(a, b)


def test_top_k_basic() -> None:
    s = np.array([1.0, 5.0, 3.0, 4.0, 2.0], dtype=np.float32)
    r = top_k(s, 3)
    assert r == [(1, 5.0), (3, 4.0), (2, 3.0)]


def test_top_k_full_length_returns_full_sort() -> None:
    s = np.array([1.0, 5.0, 3.0], dtype=np.float32)
    r = top_k(s, 3)
    assert r == [(1, 5.0), (2, 3.0), (0, 1.0)]


def test_top_k_zero_rejected() -> None:
    s = np.array([1.0, 2.0], dtype=np.float32)
    with pytest.raises(ValueError):
        top_k(s, 0)


def test_top_k_too_large_rejected() -> None:
    s = np.array([1.0, 2.0], dtype=np.float32)
    with pytest.raises(ValueError):
        top_k(s, 3)


def test_batch_top_k_serial_and_parallel_match() -> None:
    rng = np.random.default_rng(0)
    m = rng.standard_normal((20, 50)).astype(np.float32)
    s = batch_top_k(m, 5)
    p = batch_top_k(m, 5, parallel=True)
    assert s == p
    assert len(s) == 20 and len(s[0]) == 5


def test_cosine_distances_zero_diagonal() -> None:
    a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    d = cosine_distances(a, a)
    assert d.shape == (2, 2)
    assert abs(d[0, 0]) < 1e-6
    assert abs(d[1, 1]) < 1e-6
    assert abs(d[0, 1] - 1.0) < 1e-6


def test_native_error_class_exposed() -> None:
    assert issubclass(VecnormError, Exception)


def test_l2_normalize_2d_required() -> None:
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(ValueError, match="2-D"):
        l2_normalize(a)


def test_top_k_rejects_2d_input() -> None:
    s = np.zeros((3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="1-D"):
        top_k(s, 2)

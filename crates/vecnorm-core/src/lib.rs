//! Pure-Rust core for `vecnorm`. Bulk f32 matrix operations:
//!
//! - [`l2_normalize`] / [`l2_normalize_copy`] — row-wise unit-length scaling.
//!   Rows whose norm is below `EPS` are left at zero rather than dividing
//!   by zero.
//! - [`cosine_similarity`] — single pair on 1-D vectors. Returns 0 for
//!   any pair where either side has zero norm.
//! - [`top_k_argmax`] / [`batch_top_k_argmax`] — partial-heap top-k that
//!   runs in `O(n log k)`. Tied scores are broken by the original index
//!   ascending (deterministic).

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use ndarray::{ArrayView1, ArrayView2, ArrayViewMut2, Axis};
use rayon::prelude::*;
use thiserror::Error;

/// Tiny norm below which a row is considered all-zero and left unscaled.
pub const EPS: f32 = 1e-12;

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, VecNormError>;

/// All errors surfaced by `vecnorm-core`.
#[derive(Error, Debug)]
pub enum VecNormError {
    /// Two arrays had incompatible shapes.
    #[error("dimension mismatch: a={a:?}, b={b:?}")]
    DimensionMismatch {
        /// Shape of the first input.
        a: Vec<usize>,
        /// Shape of the second input.
        b: Vec<usize>,
    },
    /// Caller asked for more elements than the input has.
    #[error("k ({k}) must be <= len ({len})")]
    KTooLarge {
        /// Requested k.
        k: usize,
        /// Available length.
        len: usize,
    },
    /// Caller passed `k = 0`.
    #[error("k must be > 0")]
    KZero,
}

/// L2-normalize `matrix` in place, row by row. Rows with norm below `EPS`
/// are zeroed out (i.e. left unchanged at all-zero) to avoid NaN.
pub fn l2_normalize(matrix: &mut ArrayViewMut2<'_, f32>) {
    matrix
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut row| {
            let mut sum_sq = 0.0_f32;
            for &x in row.iter() {
                sum_sq += x * x;
            }
            let norm = sum_sq.sqrt();
            if norm > EPS {
                for x in row.iter_mut() {
                    *x /= norm;
                }
            } else {
                for x in row.iter_mut() {
                    *x = 0.0;
                }
            }
        });
}

/// L2-normalize a copy. Same semantics as [`l2_normalize`].
pub fn l2_normalize_copy(matrix: &ArrayView2<'_, f32>) -> ndarray::Array2<f32> {
    let mut out = matrix.to_owned();
    l2_normalize(&mut out.view_mut());
    out
}

/// Cosine similarity between two 1-D vectors. Returns 0 if either side is
/// all-zero.
pub fn cosine_similarity(a: &ArrayView1<'_, f32>, b: &ArrayView1<'_, f32>) -> Result<f32> {
    if a.len() != b.len() {
        return Err(VecNormError::DimensionMismatch {
            a: a.shape().to_vec(),
            b: b.shape().to_vec(),
        });
    }
    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom <= EPS {
        return Ok(0.0);
    }
    Ok(dot / denom)
}

/// Inner product (dot product) of two 1-D vectors. No normalization.
/// Errors on dim mismatch.
pub fn dot_product(a: &ArrayView1<'_, f32>, b: &ArrayView1<'_, f32>) -> Result<f32> {
    if a.len() != b.len() {
        return Err(VecNormError::DimensionMismatch {
            a: a.shape().to_vec(),
            b: b.shape().to_vec(),
        });
    }
    let mut s = 0.0_f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        s += x * y;
    }
    Ok(s)
}

/// Single argmax: returns `(index, score)` of the largest element. Ties
/// broken by ascending index. Errors on empty input.
pub fn argmax(scores: &ArrayView1<'_, f32>) -> Result<(usize, f32)> {
    if scores.is_empty() {
        return Err(VecNormError::KZero);
    }
    let mut best_i = 0usize;
    let mut best_v = scores[0];
    for (i, &v) in scores.iter().enumerate().skip(1) {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    Ok((best_i, best_v))
}

/// Top-k argmax over a 1-D score vector. Returns `(index, score)` pairs in
/// descending order. Ties broken by ascending index.
pub fn top_k_argmax(scores: &ArrayView1<'_, f32>, k: usize) -> Result<Vec<(usize, f32)>> {
    if k == 0 {
        return Err(VecNormError::KZero);
    }
    if k > scores.len() {
        return Err(VecNormError::KTooLarge {
            k,
            len: scores.len(),
        });
    }
    // Maintain a min-heap of size k. The smallest element on the heap is
    // the threshold to beat. We compare on `(Reverse(score), idx)` so equal
    // scores order ascending by index, which matches the stable convention.
    let mut heap: BinaryHeap<(Reverse<OrdFloat>, usize)> = BinaryHeap::with_capacity(k);
    for (i, &s) in scores.iter().enumerate() {
        let entry = (Reverse(OrdFloat(s)), i);
        if heap.len() < k {
            heap.push(entry);
        } else if let Some(top) = heap.peek() {
            // Heap is a min-heap on score (because of Reverse); the *largest*
            // Reverse-key is the smallest score on the heap.
            if entry.0 < top.0 {
                heap.pop();
                heap.push(entry);
            }
        }
    }
    // Drain heap and sort descending.
    let mut out: Vec<(usize, f32)> = heap.into_iter().map(|(rs, i)| (i, rs.0 .0)).collect();
    out.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    Ok(out)
}

/// Batch top-k argmax over an `(n_rows, n_cols)` matrix. With `parallel = true`
/// distributes rows across rayon's pool.
pub fn batch_top_k_argmax(
    scores: &ArrayView2<'_, f32>,
    k: usize,
    parallel: bool,
) -> Result<Vec<Vec<(usize, f32)>>> {
    if k == 0 {
        return Err(VecNormError::KZero);
    }
    if k > scores.ncols() {
        return Err(VecNormError::KTooLarge {
            k,
            len: scores.ncols(),
        });
    }
    if parallel {
        scores
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|row| top_k_argmax(&row, k))
            .collect()
    } else {
        scores
            .axis_iter(Axis(0))
            .map(|row| top_k_argmax(&row, k))
            .collect()
    }
}

/// Cosine distance matrix between two `(n_a, d)` and `(n_b, d)` matrices.
/// Returns an `(n_a, n_b)` matrix where `out[i, j]` is the cosine distance
/// `1 - cos(a_i, b_j)`. Inputs are not modified; this normalizes copies
/// internally so accuracy is preserved on un-normalized inputs.
pub fn cosine_distances(
    a: &ArrayView2<'_, f32>,
    b: &ArrayView2<'_, f32>,
) -> Result<ndarray::Array2<f32>> {
    if a.ncols() != b.ncols() {
        return Err(VecNormError::DimensionMismatch {
            a: a.shape().to_vec(),
            b: b.shape().to_vec(),
        });
    }
    let an = l2_normalize_copy(a);
    let bn = l2_normalize_copy(b);
    let n_a = an.nrows();
    let n_b = bn.nrows();
    let mut out = ndarray::Array2::<f32>::zeros((n_a, n_b));
    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            for (j, cell) in row.iter_mut().enumerate() {
                let mut dot = 0.0_f32;
                for (&x, &y) in an.row(i).iter().zip(bn.row(j).iter()) {
                    dot += x * y;
                }
                *cell = 1.0 - dot;
            }
        });
    Ok(out)
}

// ---- internal: Ord-able f32 wrapper ----

#[derive(Debug, Clone, Copy, PartialEq)]
struct OrdFloat(f32);

impl Eq for OrdFloat {}

impl Ord for OrdFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // NaN sorts as smallest; we don't expect NaN in scores but tolerate.
        match self.0.partial_cmp(&other.0) {
            Some(o) => o,
            None => {
                let s = self.0.is_nan();
                let o = other.0.is_nan();
                match (s, o) {
                    (true, true) => std::cmp::Ordering::Equal,
                    (true, false) => std::cmp::Ordering::Less,
                    (false, true) => std::cmp::Ordering::Greater,
                    (false, false) => std::cmp::Ordering::Equal,
                }
            }
        }
    }
}

impl PartialOrd for OrdFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2, Array1, Array2};

    #[test]
    fn l2_normalize_basic() {
        let mut a = arr2(&[[3.0_f32, 4.0], [1.0, 0.0]]);
        l2_normalize(&mut a.view_mut());
        // Row 0 norm 5 -> [0.6, 0.8]
        assert!((a[[0, 0]] - 0.6).abs() < 1e-6);
        assert!((a[[0, 1]] - 0.8).abs() < 1e-6);
        // Row 1 norm 1 -> [1.0, 0.0]
        assert!((a[[1, 0]] - 1.0).abs() < 1e-6);
        assert!((a[[1, 1]] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_zero_row_left_zero() {
        let mut a = arr2(&[[0.0_f32, 0.0], [3.0, 4.0]]);
        l2_normalize(&mut a.view_mut());
        assert_eq!(a[[0, 0]], 0.0);
        assert_eq!(a[[0, 1]], 0.0);
        assert!(!a[[0, 0]].is_nan());
    }

    #[test]
    fn l2_normalize_copy_does_not_mutate_input() {
        let a = arr2(&[[3.0_f32, 4.0]]);
        let _ = l2_normalize_copy(&a.view());
        assert_eq!(a[[0, 0]], 3.0);
        assert_eq!(a[[0, 1]], 4.0);
    }

    #[test]
    fn cosine_basic() {
        let a = arr1(&[1.0_f32, 0.0]);
        let b = arr1(&[1.0_f32, 0.0]);
        let c = arr1(&[0.0_f32, 1.0]);
        assert!((cosine_similarity(&a.view(), &b.view()).unwrap() - 1.0).abs() < 1e-6);
        assert!(cosine_similarity(&a.view(), &c.view()).unwrap().abs() < 1e-6);
    }

    #[test]
    fn dot_product_basic() {
        let a = arr1(&[1.0_f32, 2.0, 3.0]);
        let b = arr1(&[4.0_f32, -5.0, 6.0]);
        // 1*4 + 2*(-5) + 3*6 = 4 - 10 + 18 = 12.
        assert!((dot_product(&a.view(), &b.view()).unwrap() - 12.0).abs() < 1e-6);
    }

    #[test]
    fn dot_product_dim_mismatch() {
        let a = arr1(&[1.0_f32, 0.0]);
        let b = arr1(&[1.0_f32]);
        assert!(dot_product(&a.view(), &b.view()).is_err());
    }

    #[test]
    fn argmax_picks_largest() {
        let s = arr1(&[1.0_f32, 5.0, 3.0, 4.0, 2.0]);
        let (i, v) = argmax(&s.view()).unwrap();
        assert_eq!(i, 1);
        assert!((v - 5.0).abs() < 1e-6);
    }

    #[test]
    fn argmax_ties_pick_lowest_index() {
        let s = arr1(&[3.0_f32, 3.0, 3.0]);
        assert_eq!(argmax(&s.view()).unwrap().0, 0);
    }

    #[test]
    fn argmax_empty_rejected() {
        let s: ndarray::Array1<f32> = arr1(&[]);
        assert!(argmax(&s.view()).is_err());
    }

    #[test]
    fn cosine_zero_for_zero_vector() {
        let a = arr1(&[0.0_f32, 0.0]);
        let b = arr1(&[1.0_f32, 1.0]);
        assert_eq!(cosine_similarity(&a.view(), &b.view()).unwrap(), 0.0);
    }

    #[test]
    fn cosine_dim_mismatch() {
        let a = arr1(&[1.0_f32, 0.0]);
        let b = arr1(&[1.0_f32, 0.0, 1.0]);
        assert!(cosine_similarity(&a.view(), &b.view()).is_err());
    }

    #[test]
    fn top_k_correct_order() {
        let s = arr1(&[1.0, 5.0, 3.0, 4.0, 2.0]);
        let r = top_k_argmax(&s.view(), 3).unwrap();
        assert_eq!(r, vec![(1, 5.0), (3, 4.0), (2, 3.0)]);
    }

    #[test]
    fn top_k_full_length_returns_full_sort() {
        let s = arr1(&[1.0, 5.0, 3.0]);
        let r = top_k_argmax(&s.view(), 3).unwrap();
        assert_eq!(r, vec![(1, 5.0), (2, 3.0), (0, 1.0)]);
    }

    #[test]
    fn top_k_ties_broken_by_lower_index() {
        let s = arr1(&[1.0, 1.0, 1.0]);
        let r = top_k_argmax(&s.view(), 2).unwrap();
        assert_eq!(r, vec![(0, 1.0), (1, 1.0)]);
    }

    #[test]
    fn top_k_zero_rejected() {
        let s = arr1(&[1.0, 2.0]);
        assert!(top_k_argmax(&s.view(), 0).is_err());
    }

    #[test]
    fn top_k_too_large_rejected() {
        let s = arr1(&[1.0, 2.0]);
        assert!(top_k_argmax(&s.view(), 3).is_err());
    }

    #[test]
    fn batch_top_k_serial_and_parallel_match() {
        let m = Array2::from_shape_fn((10, 50), |(i, j)| (i * 50 + j) as f32);
        let s = batch_top_k_argmax(&m.view(), 5, false).unwrap();
        let p = batch_top_k_argmax(&m.view(), 5, true).unwrap();
        assert_eq!(s, p);
        assert_eq!(s.len(), 10);
        // First row: top-5 of [0..50) is [49, 48, 47, 46, 45].
        assert_eq!(s[0][0], (49, 49.0));
    }

    #[test]
    fn cosine_distances_zero_diagonal() {
        let a = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
        let d = cosine_distances(&a.view(), &a.view()).unwrap();
        // Diagonal is cosine to self == 0 distance.
        assert!(d[[0, 0]].abs() < 1e-6);
        assert!(d[[1, 1]].abs() < 1e-6);
        // Off-diagonal: orthogonal == 1 distance.
        assert!((d[[0, 1]] - 1.0).abs() < 1e-6);
        assert!((d[[1, 0]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_distances_dim_mismatch() {
        let a = Array2::<f32>::zeros((4, 3));
        let b = Array2::<f32>::zeros((4, 5));
        assert!(cosine_distances(&a.view(), &b.view()).is_err());
    }

    #[test]
    fn nan_in_top_k_does_not_panic() {
        let s = Array1::from(vec![1.0_f32, f32::NAN, 3.0]);
        // We don't promise NaN handling, but we promise no panic.
        let r = top_k_argmax(&s.view(), 2);
        assert!(r.is_ok());
    }
}

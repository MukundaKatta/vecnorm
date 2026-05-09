//! PyO3 bindings exposing `vecnorm_core` as `vecnorm._native`.

use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use vecnorm_core::{
    batch_top_k_argmax, cosine_distances, cosine_similarity, l2_normalize, l2_normalize_copy,
    top_k_argmax, VecNormError,
};

pyo3::create_exception!(_native, VecnormError, pyo3::exceptions::PyException);

fn map_err(e: VecNormError) -> PyErr {
    match e {
        VecNormError::DimensionMismatch { .. }
        | VecNormError::KTooLarge { .. }
        | VecNormError::KZero => PyValueError::new_err(e.to_string()),
    }
}

#[pyfunction]
#[pyo3(name = "l2_normalize")]
fn py_l2_normalize(py: Python<'_>, matrix: Bound<'_, PyArray2<f32>>) -> PyResult<()> {
    // Borrow the array mutably and run the in-place normalization while
    // releasing the GIL. The numpy crate's RW guard keeps the underlying
    // buffer pinned for the duration.
    let mut rw = matrix.readwrite();
    let mut view = rw.as_array_mut();
    py.allow_threads(move || l2_normalize(&mut view));
    Ok(())
}

#[pyfunction]
#[pyo3(name = "l2_normalize_copy")]
fn py_l2_normalize_copy<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'_, f32>,
) -> Bound<'py, PyArray2<f32>> {
    let owned = matrix.as_array().to_owned();
    let out = py.allow_threads(move || l2_normalize_copy(&owned.view()));
    out.to_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "cosine_similarity")]
fn py_cosine_similarity(
    py: Python<'_>,
    a: PyReadonlyArray1<'_, f32>,
    b: PyReadonlyArray1<'_, f32>,
) -> PyResult<f32> {
    let aa = a.as_array().to_owned();
    let bb = b.as_array().to_owned();
    py.allow_threads(move || cosine_similarity(&aa.view(), &bb.view()))
        .map_err(map_err)
}

#[pyfunction]
#[pyo3(name = "top_k", signature = (scores, k))]
fn py_top_k(
    py: Python<'_>,
    scores: PyReadonlyArray1<'_, f32>,
    k: usize,
) -> PyResult<Vec<(usize, f32)>> {
    let owned = scores.as_array().to_owned();
    py.allow_threads(move || top_k_argmax(&owned.view(), k))
        .map_err(map_err)
}

#[pyfunction]
#[pyo3(name = "batch_top_k", signature = (scores, k, parallel=false))]
fn py_batch_top_k(
    py: Python<'_>,
    scores: PyReadonlyArray2<'_, f32>,
    k: usize,
    parallel: bool,
) -> PyResult<Vec<Vec<(usize, f32)>>> {
    let owned = scores.as_array().to_owned();
    py.allow_threads(move || batch_top_k_argmax(&owned.view(), k, parallel))
        .map_err(map_err)
}

#[pyfunction]
#[pyo3(name = "cosine_distances")]
fn py_cosine_distances<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f32>,
    b: PyReadonlyArray2<'_, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let aa = a.as_array().to_owned();
    let bb = b.as_array().to_owned();
    let out = py
        .allow_threads(move || cosine_distances(&aa.view(), &bb.view()))
        .map_err(map_err)?;
    Ok(out.to_pyarray(py))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("VecnormError", m.py().get_type::<VecnormError>())?;
    m.add_function(wrap_pyfunction!(py_l2_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(py_l2_normalize_copy, m)?)?;
    m.add_function(wrap_pyfunction!(py_cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(py_top_k, m)?)?;
    m.add_function(wrap_pyfunction!(py_batch_top_k, m)?)?;
    m.add_function(wrap_pyfunction!(py_cosine_distances, m)?)?;
    Ok(())
}

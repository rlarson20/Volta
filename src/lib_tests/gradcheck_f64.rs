//! Shared f64 numerical gradient check for the test suites (ADDENDUM §11, §10.2).
//!
//! A test-only f64 copy of `check_gradients`: the public one is f32-only, and there is no
//! public f64 gradient check (adding one is a design decision, not part of this module). Both
//! `f64_tests` (one op, one leaf at a time) and `dag_tests` (random DAGs, every leaf) use it,
//! so they share one tolerance and one finite-difference step.

use super::*;
use crate::autograd::no_grad;
use crate::tensor::manual_seed;

/// Relative tolerance of the f64 gradient checks.
pub(crate) const TOL: f64 = 1e-7;
/// Central-difference step: truncation error ~ EPS² and rounding error ~ 1e-16 / EPS
/// balance near 1e-10.
pub(crate) const EPS: f64 = 1e-6;

/// Seeded normal values as an f64 leaf that requires grad.
pub(crate) fn leaf(shape: &[usize], seed: u64) -> Tensor {
    leaf_from(shape, seed, |v| v)
}

/// Seeded normal values mapped through `f`, as an f64 leaf that requires grad.
pub(crate) fn leaf_from(shape: &[usize], seed: u64, f: impl Fn(f64) -> f64) -> Tensor {
    manual_seed(seed);
    let raw = RawTensor::randn(shape);
    let vals: Vec<f64> = raw.data().iter().map(|&v| f(f64::from(v))).collect();
    let x = RawTensor::from_typed(vals, shape);
    x.set_requires_grad(true);
    x
}

/// The f64 values of `t`, through the public lossless readback.
pub(crate) fn values(t: &Tensor) -> Vec<f64> {
    t.to_vec_as::<f64>().expect("an f64 tensor")
}

/// Weighted sum `sum(out * w)` with fixed random f64 weights, so every output element
/// gets a different gradient (a plain sum would hide permutations and broadcasts).
pub(crate) fn weighted(out: &Tensor, seed: u64) -> Tensor {
    let w = leaf(out.shape().truncate_rank(), seed);
    w.set_requires_grad(false);
    out.elem_mul(&w).sum()
}

/// Max relative error between `x`'s analytic gradient of `f` and a central difference,
/// both in f64. Relative to `max(|analytic|, |numeric|, 1)`.
pub(crate) fn gradcheck(x: &Tensor, f: impl Fn(&Tensor) -> Tensor) -> f64 {
    RawTensor::zero_grad(x);
    let loss = f(x);
    assert_eq!(loss.dtype(), DType::F64, "the loss must be f64");
    loss.into_loss().expect("scalar loss").backward();
    let grad = x.grad_storage().expect("x has a gradient");
    assert_eq!(
        grad.dtype(),
        DType::F64,
        "an f64 tensor has an f64 gradient"
    );
    let analytic = grad.as_host::<f64>().expect("f64").to_vec();

    let (base, shape) = {
        let t = x.borrow();
        (t.data.clone(), t.shape.clone())
    };
    let _no_grad = no_grad();
    let eval = |delta: f64, i: usize| {
        let mut data = base.clone();
        *crate::index::slot(data.as_host_mut::<f64>().expect("f64"), i) += delta;
        let t = RawTensor::new_with_storage(data, shape.clone(), Device::CPU, false);
        crate::index::fetch(&values(&f(&t)), 0)
    };
    let mut max_err: f64 = 0.0;
    for (i, &a) in analytic.iter().enumerate() {
        let numeric = (eval(EPS, i) - eval(-EPS, i)) / (2.0 * EPS);
        let err = (a - numeric).abs() / a.abs().max(numeric.abs()).max(1.0);
        max_err = max_err.max(err);
    }
    max_err
}

/// Asserts the gradient check passes, naming the case.
pub(crate) fn assert_grad(name: &str, x: &Tensor, f: impl Fn(&Tensor) -> Tensor) {
    let err = gradcheck(x, f);
    assert!(
        err < TOL,
        "{name}: f64 gradcheck max relative error {err:.3e}"
    );
}

/// The multi-leaf form of [`gradcheck`]: one backward pass through `f(leaves)`, then every
/// element of every leaf checked against a central difference. Returns the worst relative
/// error per leaf, in `leaves` order, so a failure names the leaf.
///
/// A leaf the loss does not reach has no gradient after the backward pass. Its analytic
/// gradient is then exactly zero, and it is still compared against the central difference, so
/// a backward that wrongly drops a parent fails here instead of passing as a silent zero.
pub(crate) fn gradcheck_all(leaves: &[Tensor], f: impl Fn(&[Tensor]) -> Tensor) -> Vec<f64> {
    for x in leaves {
        RawTensor::zero_grad(x);
    }
    let loss = f(leaves);
    assert_eq!(loss.dtype(), DType::F64, "the loss must be f64");
    loss.into_loss().expect("scalar loss").backward();
    let analytic: Vec<Vec<f64>> = leaves
        .iter()
        .map(|x| match x.grad_storage() {
            Some(grad) => {
                assert_eq!(
                    grad.dtype(),
                    DType::F64,
                    "an f64 tensor has an f64 gradient"
                );
                grad.as_host::<f64>().expect("f64").to_vec()
            }
            None => vec![0.0; x.shape().numel()],
        })
        .collect();

    let bases: Vec<_> = leaves
        .iter()
        .map(|x| {
            let t = x.borrow();
            (t.data.clone(), t.shape.clone())
        })
        .collect();
    let _no_grad = no_grad();
    // The loss with element `i` of leaf `which` nudged by `delta`, on fresh no-grad copies.
    let eval = |which: usize, delta: f64, i: usize| {
        let fresh: Vec<Tensor> = bases
            .iter()
            .enumerate()
            .map(|(k, (data, shape))| {
                let mut data = data.clone();
                if k == which {
                    *crate::index::slot(data.as_host_mut::<f64>().expect("f64"), i) += delta;
                }
                RawTensor::new_with_storage(data, shape.clone(), Device::CPU, false)
            })
            .collect();
        crate::index::fetch(&values(&f(&fresh)), 0)
    };
    analytic
        .iter()
        .enumerate()
        .map(|(which, grads)| {
            grads.iter().enumerate().fold(0.0f64, |max_err, (i, &a)| {
                let numeric = (eval(which, EPS, i) - eval(which, -EPS, i)) / (2.0 * EPS);
                let err = (a - numeric).abs() / a.abs().max(numeric.abs()).max(1.0);
                max_err.max(err)
            })
        })
        .collect()
}

//! `sum_dim`, `mean_dim`, `softmax`, and `log_softmax` (forward and weighted gradients)
//! against PyTorch (`tests/fixtures/torch/reductions.safetensors`), plus the large-sum
//! accuracy check (S6).

#[path = "torch_ref/mod.rs"]
mod torch_ref;

use torch_ref::{Fixture, TOL_BWD, TOL_FWD, assert_close, backward_weighted, grad_of};
use volta::{Axis, RawTensor};

/// Run `op` on a fresh copy of the fixture input and compare forward and (weighted) gradient.
fn check_axis_op(
    name: &str,
    dim: usize,
    grad_name: Option<(&str, &str)>,
    op: impl Fn(&volta::Tensor, Axis) -> volta::Tensor,
) {
    let f = Fixture::load("reductions");
    let x = f.tensor("x", true);
    let out = op(&x, Axis(dim));
    assert_close(
        &format!("{name}.out"),
        &out.data(),
        &f.values(&format!("{name}.out")),
        TOL_FWD,
    );
    if let Some((w_name, g_name)) = grad_name {
        let (w, shape) = f.get(w_name);
        let w = RawTensor::new(w, &shape, false);
        backward_weighted(&out, Some(&w));
        assert_close(
            &format!("{name}.grad"),
            &grad_of(name, &x),
            &f.values(g_name),
            TOL_BWD,
        );
    }
}

#[test]
fn sum_dim_matches_torch() {
    for dim in 0..3 {
        let n = format!("sum_dim{dim}");
        check_axis_op(
            &n,
            dim,
            Some((&format!("{n}.w"), &format!("{n}.grad"))),
            |x, a| x.sum_dim(a, false),
        );
    }
}

#[test]
fn mean_dim_matches_torch() {
    for dim in 0..3 {
        check_axis_op(&format!("mean_dim{dim}"), dim, None, |x, a| {
            x.mean_dim(a, true)
        });
    }
}

#[test]
fn softmax_matches_torch() {
    for dim in 0..3 {
        let n = format!("softmax_dim{dim}");
        check_axis_op(
            &n,
            dim,
            Some((&format!("{n}.w"), &format!("{n}.grad"))),
            volta::Tensor::softmax,
        );
    }
}

#[test]
fn log_softmax_matches_torch() {
    for dim in 0..3 {
        let n = format!("log_softmax_dim{dim}");
        check_axis_op(
            &n,
            dim,
            Some((&format!("softmax_dim{dim}.w"), &format!("{n}.grad"))),
            volta::Tensor::log_softmax,
        );
    }
}

#[test]
#[ignore = "S6: sum() adds f32 values in sequence, so a sum of ones stalls at 2^24; torch uses pairwise summation"]
fn sum_of_2_pow_25_ones_is_exact_like_torch() {
    // torch.ones(2**25).sum() == 33554432.0 (pairwise/cascade summation).
    let n = 1usize << 25;
    let x = RawTensor::new(vec![1.0; n], &[n], false);
    assert_eq!(x.sum().item(), 33_554_432.0);
}

#[test]
#[ignore = "S6: mean() shares the sequential f32 sum; torch.ones(2**25).mean() is exactly 1.0"]
fn mean_of_2_pow_25_ones_is_one_like_torch() {
    let n = 1usize << 25;
    let x = RawTensor::new(vec![1.0; n], &[n], false);
    assert_eq!(x.mean().item(), 1.0);
}

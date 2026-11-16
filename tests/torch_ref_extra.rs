//! #56: `bce_with_logits_loss`, `kl_divergence_gaussian`, the unary ops over wide input
//! ranges, and `stride_op`, checked against PyTorch reference values
//! (`tests/fixtures/torch/extra.safetensors`, group `extra` of
//! `scripts/torch_ref/gen_fixtures.py`). Each gradient is of `sum(out * w)` for a fixed
//! random `w` stored in the fixture.

#[path = "torch_ref/mod.rs"]
mod torch_ref;

use torch_ref::{Fixture, TOL_BWD, TOL_FWD, assert_close, backward_weighted, grad_of};
use volta::{Tensor, bce_with_logits_loss, kl_divergence_gaussian};

#[test]
fn bce_with_logits_matches_torch_over_wide_logits() {
    let f = Fixture::load("extra");
    let x = f.tensor("bce_logits.x", true);
    let y = f.tensor("bce_logits.y", false);
    let out = bce_with_logits_loss(&x, &y);
    assert_close(
        "bce_logits.out",
        &out.data(),
        &f.values("bce_logits.out"),
        TOL_FWD,
    );
    backward_weighted(&out, None);
    assert_close(
        "bce_logits.grad",
        &grad_of("x", &x),
        &f.values("bce_logits.grad"),
        TOL_BWD,
    );
}

#[test]
fn kl_divergence_gaussian_matches_torch() {
    let f = Fixture::load("extra");
    let mu = f.tensor("kl.mu", true);
    let logvar = f.tensor("kl.logvar", true);
    let out = kl_divergence_gaussian(&mu, &logvar);
    assert_close("kl.out", &out.data(), &f.values("kl.out"), TOL_FWD);
    backward_weighted(&out, None);
    assert_close(
        "kl.grad_mu",
        &grad_of("mu", &mu),
        &f.values("kl.grad_mu"),
        TOL_BWD,
    );
    assert_close(
        "kl.grad_logvar",
        &grad_of("logvar", &logvar),
        &f.values("kl.grad_logvar"),
        TOL_BWD,
    );
}

/// Forward and weighted backward of one unary op against `unary.<name>.*`.
fn check_unary(name: &str, op: fn(&Tensor) -> Tensor) {
    let f = Fixture::load("extra");
    let x = f.tensor(&format!("unary.{name}.x"), true);
    let w = f.tensor(&format!("unary.{name}.w"), false);
    let out = op(&x);
    assert_close(
        &format!("unary.{name}.out"),
        &out.data(),
        &f.values(&format!("unary.{name}.out")),
        TOL_FWD,
    );
    backward_weighted(&out, Some(&w));
    assert_close(
        &format!("unary.{name}.grad_x"),
        &grad_of(name, &x),
        &f.values(&format!("unary.{name}.grad_x")),
        TOL_BWD,
    );
}

#[test]
fn unary_ops_match_torch_over_wide_ranges() {
    let ops: [(&str, fn(&Tensor) -> Tensor); 13] = [
        ("neg", Tensor::neg),
        ("exp", Tensor::exp),
        ("exp2", Tensor::exp2),
        ("tanh", Tensor::tanh),
        ("sigmoid", Tensor::sigmoid),
        ("relu", Tensor::relu),
        ("erf", Tensor::erf),
        ("sin", Tensor::sin),
        ("cos", Tensor::cos),
        ("log", Tensor::log),
        ("log2", Tensor::log2),
        ("sqrt", Tensor::sqrt),
        ("reciprocal", Tensor::recip),
    ];
    for (name, op) in ops {
        check_unary(name, op);
    }
}

#[test]
fn stride_op_matches_torch_strided_slicing() {
    let f = Fixture::load("extra");
    for name in ["s23", "s14", "s51"] {
        let x = f.tensor(&format!("stride.{name}.x"), true);
        let w = f.tensor(&format!("stride.{name}.w"), false);
        let strides: Vec<usize> = f
            .values(&format!("stride.{name}.strides"))
            .iter()
            .map(|&s| s as usize)
            .collect();
        let out = x.stride_op(&strides);
        let (want, want_shape) = f.get(&format!("stride.{name}.out"));
        assert_eq!(out.shape(), want_shape, "stride.{name} shape");
        assert_close(&format!("stride.{name}.out"), &out.data(), &want, TOL_FWD);
        backward_weighted(&out, Some(&w));
        assert_close(
            &format!("stride.{name}.grad_x"),
            &grad_of("x", &x),
            &f.values(&format!("stride.{name}.grad_x")),
            TOL_BWD,
        );
    }
}

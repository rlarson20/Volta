//! Volta's elementwise ops, ties, NaN handling, remainder, and loss edge cases, checked
//! against PyTorch reference values (`tests/fixtures/torch/ops.safetensors`).
//!
//! A test marked `#[ignore = "S<n>: ..."]` asserts PyTorch's answer and fails today: it
//! documents a confirmed divergence in `docs/audits/2026-11-sweep.md`. Run them with
//! `cargo test --no-default-features --test torch_ref_ops -- --ignored`.

#[path = "torch_ref/mod.rs"]
mod torch_ref;

use torch_ref::{Fixture, TOL_BWD, TOL_FWD, assert_close, backward_weighted, grad_of};
use volta::{Axis, RawTensor};

#[test]
#[ignore = "S4: max_elem gives the whole gradient to a on a tie; torch.maximum splits it 0.5/0.5"]
fn max_elem_tie_splits_gradient_like_torch_maximum() {
    // S4
    let f = Fixture::load("ops");
    let (a, b) = (f.tensor("max_elem.a", true), f.tensor("max_elem.b", true));
    let out = a.max_elem(&b);
    assert_close(
        "max_elem.out",
        &out.data(),
        &f.values("max_elem.out"),
        TOL_FWD,
    );
    backward_weighted(&out, None);
    assert_close(
        "grad_a",
        &grad_of("a", &a),
        &f.values("max_elem.grad_a"),
        TOL_BWD,
    );
    assert_close(
        "grad_b",
        &grad_of("b", &b),
        &f.values("max_elem.grad_b"),
        TOL_BWD,
    );
}

#[test]
#[ignore = "S4: f32::max drops NaN; torch.maximum returns NaN"]
fn max_elem_propagates_nan_like_torch_maximum() {
    // S4
    let f = Fixture::load("ops");
    let (a, b) = (
        f.tensor("max_elem_nan.a", false),
        f.tensor("max_elem_nan.b", false),
    );
    assert_close(
        "max_elem_nan.out",
        &a.max_elem(&b).data(),
        &f.values("max_elem_nan.out"),
        TOL_FWD,
    );
}

#[test]
#[ignore = "S4: max_reduce sends the gradient to the last tied element; torch splits it evenly"]
fn max_reduce_tie_splits_gradient_like_torch_max() {
    // S4
    let f = Fixture::load("ops");
    let x = f.tensor("max_all_tie.x", true);
    let out = x.max_reduce();
    assert_close("out", &out.data(), &f.values("max_all_tie.out"), TOL_FWD);
    backward_weighted(&out, None);
    assert_close(
        "grad",
        &grad_of("x", &x),
        &f.values("max_all_tie.grad"),
        TOL_BWD,
    );
}

#[test]
#[ignore = "S4: partial_cmp().unwrap() panics on NaN; torch.max returns NaN"]
fn max_reduce_propagates_nan_like_torch_max() {
    // S4: panics today (partial_cmp().unwrap()).
    let f = Fixture::load("ops");
    let x = f.tensor("max_all_nan.x", false);
    assert_close(
        "out",
        &x.max_reduce().data(),
        &f.values("max_all_nan.out"),
        TOL_FWD,
    );
}

#[test]
#[ignore = "S4: relu is x.max(0.0), which maps NaN to 0; torch.relu(NaN) is NaN"]
fn relu_propagates_nan_like_torch() {
    // S4
    let f = Fixture::load("ops");
    let x = f.tensor("relu_nan.x", false);
    assert_close("out", &x.relu().data(), &f.values("relu_nan.out"), TOL_FWD);
}

#[test]
#[ignore = "S5: CPU modulo is truncated (-1 mod 3 = -1; GPU and torch give 2) and non-differentiable"]
fn remainder_matches_torch_forward_and_gradient() {
    // S5
    let f = Fixture::load("ops");
    let (a, b) = (f.tensor("remainder.a", true), f.tensor("remainder.b", true));
    let out = a.modulo(&b);
    assert_close("out", &out.data(), &f.values("remainder.out"), TOL_FWD);
    backward_weighted(&out, None);
    assert_close(
        "grad_a",
        &grad_of("a", &a),
        &f.values("remainder.grad_a"),
        TOL_BWD,
    );
    assert_close(
        "grad_b",
        &grad_of("b", &b),
        &f.values("remainder.grad_b"),
        TOL_BWD,
    );
}

#[test]
#[ignore = "S2: a row with no winner keeps flat index 0 of the whole tensor, so its gradient lands in row 0"]
fn max_dim_all_neg_inf_row_keeps_gradient_in_its_row() {
    // S2
    let f = Fixture::load("ops");
    let x = f.tensor("max_dim_neginf.x", true);
    let out = x.max_dim(Axis(1), false);
    assert_close("out", &out.data(), &f.values("max_dim_neginf.out"), TOL_FWD);
    let w = RawTensor::new(vec![1.0, 10.0], &[2], false);
    backward_weighted(&out, Some(&w));
    assert_close(
        "grad",
        &grad_of("x", &x),
        &f.values("max_dim_neginf.grad"),
        TOL_BWD,
    );
}

#[test]
#[ignore = "S2: > skips NaN, so max_dim([NaN, 3]) is 3; torch returns NaN"]
fn max_dim_propagates_nan_like_torch() {
    // S2/S4
    let f = Fixture::load("ops");
    let x = f.tensor("max_dim_nan.x", false);
    assert_close(
        "out",
        &x.max_dim(Axis(1), false).data(),
        &f.values("max_dim_nan.out"),
        TOL_FWD,
    );
}

#[test]
#[ignore = "S7: bce_loss has no log clamp; pred 0 or 1 gives NaN where torch gives a finite value"]
fn bce_matches_torch_including_the_log_clamp() {
    // S7
    let f = Fixture::load("ops");
    let (p, y) = (f.tensor("bce.pred", true), f.tensor("bce.target", false));
    let out = RawTensor::bce_loss(&p, &y);
    assert_close("out", &out.data(), &f.values("bce.out"), TOL_FWD);
    backward_weighted(&out, None);
    assert_close("grad", &grad_of("pred", &p), &f.values("bce.grad"), TOL_BWD);
}

/// A class index stored as an integer-valued f32 in the fixture.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the fixture stores small non-negative class indices as f32"
)]
fn class_index(c: f32) -> usize {
    c.round() as usize
}

#[test]
#[ignore = "S8: one-hot * log_softmax gives 0 * -inf = NaN for the whole loss"]
fn cross_entropy_with_neg_inf_off_target_logit_is_finite_like_torch() {
    // S8: Volta multiplies one-hot targets by log-probs, so 0 * -inf is NaN.
    let f = Fixture::load("ops");
    let (logits, dims) = f.get("ce_neginf.logits");
    let z = RawTensor::new(logits.clone(), &dims, false);
    let classes = f.values("ce_neginf.classes");
    let mut onehot = vec![0.0f32; logits.len()];
    let width = dims[1];
    for (row, c) in classes.iter().enumerate() {
        // class indices are small non-negative integers stored as f32
        onehot[row * width + class_index(*c)] = 1.0;
    }
    let y = RawTensor::new(onehot, &dims, false);
    let out = RawTensor::cross_entropy_loss(&z, &y);
    assert_close("out", &out.data(), &f.values("ce_neginf.out"), TOL_FWD);
}

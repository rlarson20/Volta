//! CPU/GPU parity for every movement backward with a result past the first workgroup (#301).
//!
//! The movement backward kernels run one thread per element of the buffer they write: the
//! input's gradient, shaped like the forward input. `shrink` and `stride` make the forward
//! output smaller than the input, so a launch sized from the incoming gradient left every
//! input-gradient element past `ceil(out/256)*256` at zero. Each test here uses an input of
//! more than 256 elements whose window or stride grid lands past the first workgroup, and
//! squares the moved tensor so each gradient element is distinct (`2 * y`), not just a one.

#![cfg(feature = "gpu")]

#[path = "common/mod.rs"]
mod common;

use common::{TOL_BWD_GPU, TOL_FWD_GPU, assert_unary_parity_backward};
use volta::{Axis, PadValue, Tensor};

/// `op(x)`, squared elementwise, so the loss `sum(op(x)^2)` gives `x` a gradient of
/// `2 * op(x)` routed back through the movement.
fn squared(op: impl Fn(&Tensor) -> Tensor) -> impl Fn(&Tensor) -> Tensor {
    move |x| {
        let y = op(x);
        y.elem_mul(&y)
    }
}

#[test]
fn shrink_backward_past_first_workgroup_1d() {
    // 1000 elements, window 900..1000: 100 gradient elements, all of them past index 256.
    assert_unary_parity_backward(
        "shrink 1d",
        squared(|x| x.shrink(&[(900, 1000)])),
        &[1000],
        (-2.0, 2.0),
        TOL_FWD_GPU,
        TOL_BWD_GPU,
    );
}

#[test]
fn shrink_backward_past_first_workgroup_2d() {
    // 40x40, rows 30..40: the window starts at element 1200.
    assert_unary_parity_backward(
        "shrink 2d",
        squared(|x| x.shrink(&[(30, 40), (0, 40)])),
        &[40, 40],
        (-2.0, 2.0),
        TOL_FWD_GPU,
        TOL_BWD_GPU,
    );
}

#[test]
fn stride_backward_past_first_workgroup_1d() {
    // 1024 elements, stride 2: 512 gradient elements scatter over all 1024.
    assert_unary_parity_backward(
        "stride 1d",
        squared(|x| x.stride_op(&[2])),
        &[1024],
        (-2.0, 2.0),
        TOL_FWD_GPU,
        TOL_BWD_GPU,
    );
}

#[test]
fn stride_backward_past_first_workgroup_2d() {
    // 40x40, stride (2, 2): 400 gradient elements scatter over 1600.
    assert_unary_parity_backward(
        "stride 2d",
        squared(|x| x.stride_op(&[2, 2])),
        &[40, 40],
        (-2.0, 2.0),
        TOL_FWD_GPU,
        TOL_BWD_GPU,
    );
}

#[test]
fn pad_backward_past_first_workgroup() {
    assert_unary_parity_backward(
        "pad",
        squared(|x| x.pad_with(&[(3, 5), (2, 1)], PadValue::ZERO)),
        &[30, 30],
        (-2.0, 2.0),
        TOL_FWD_GPU,
        TOL_BWD_GPU,
    );
}

#[test]
fn expand_backward_past_first_workgroup() {
    // [300, 1] -> [300, 4]: each of the 300 input-gradient elements sums 4 values.
    assert_unary_parity_backward(
        "expand",
        squared(|x| x.expand(&[300, 4])),
        &[300, 1],
        (-2.0, 2.0),
        TOL_FWD_GPU,
        TOL_BWD_GPU,
    );
}

#[test]
fn permute_backward_past_first_workgroup() {
    assert_unary_parity_backward(
        "permute",
        squared(|x| x.permute(&[Axis(2), Axis(0), Axis(1)])),
        &[6, 10, 9],
        (-2.0, 2.0),
        TOL_FWD_GPU,
        TOL_BWD_GPU,
    );
}

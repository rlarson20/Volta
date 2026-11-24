//! Tests that kill mutants `cargo mutants` found surviving in the autograd core on the CPU
//! lane (`just mutants-core`, #389). Each pins behavior a mutant changed without any test
//! noticing: op names in error messages, which inputs a backward reads (S9), the
//! `to_device` backward, and the broadcast error naming the bad dimension. The mutants that
//! are equivalent or reachable only on a GPU are excluded, with reasons, in
//! `.cargo/mutants.toml`.

use super::*;
use crate::autograd::{GradFn, ToDeviceGradFn, edge};
use crate::shape::Shape;
use crate::{DTypePolicy, with_dtype_policy};

/// The name the output's grad fn reports in device errors, fallback events and S9 panics.
fn op_name_of(t: &Tensor) -> String {
    t.borrow()
        .grad_fn
        .as_ref()
        .expect("an op on a requires-grad input attaches a grad fn")
        .op_name()
}

/// A leaf that requires grad.
fn leaf(values: &[f32]) -> Tensor {
    RawTensor::new(values.to_vec(), &[values.len()], true)
}

#[test]
fn op_names_name_the_variant_and_the_inner_op() {
    let x = leaf(&[0.5, -1.0]);
    assert_eq!(op_name_of(&x.exp()), "Unary(Exp)");
    assert_eq!(op_name_of(&x.elem_mul(&x)), "Binary(Mul)");
    let m = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
    assert_eq!(
        op_name_of(&m.permute(&[Axis(1), Axis(0)])),
        "Movement(Permute)"
    );
}

/// A comparison has no grad fn (its output is Bool), so its name shows only in errors.
#[test]
#[should_panic(expected = "Compare(Lt): ")]
fn compare_names_itself_in_a_strict_dtype_panic() {
    let _strict = with_dtype_policy(DTypePolicy::Strict);
    let a = RawTensor::new(vec![1.0, 2.0], &[2], false);
    let b = a.to_dtype(DType::F64);
    let _ = a.lt(&b);
}

/// The S9 panic names the op whose saved input changed.
#[test]
#[should_panic(expected = "Unary(Exp): input 0 was modified by an inplace operation")]
fn s9_panic_names_the_op() {
    let x = leaf(&[0.5, -1.0]);
    let loss = x.exp().sum();
    x.update_data(|d| d.as_host_f32_mut().expect("CPU f32")[0] = 2.0);
    loss.into_loss().expect("scalar").backward();
}

/// `Neg`'s derivative is the constant -1, so it saves nothing: changing its input between
/// forward and backward is fine, as in PyTorch.
#[test]
fn neg_does_not_save_its_input() {
    let x = leaf(&[0.5, -1.0]);
    let loss = x.neg().sum();
    x.update_data(|d| d.as_host_f32_mut().expect("CPU f32")[0] = 2.0);
    loss.into_loss().expect("scalar").backward();
    assert_eq!(x.grad(), Some(vec![-1.0, -1.0]));
}

/// `mulacc(x, y, z) = x * y + z` reads `x` and `y` in backward but not the addend `z`
/// (parent 2), so an in-place write to `z` is allowed and its gradient is `out_grad`.
#[test]
fn mulacc_does_not_save_its_addend() {
    let x = leaf(&[1.0, 2.0]);
    let y = leaf(&[3.0, -4.0]);
    let z = leaf(&[0.5, 0.25]);
    let loss = x.mulacc(&y, &z).sum();
    z.update_data(|d| d.as_host_f32_mut().expect("CPU f32")[0] = 9.0);
    loss.into_loss().expect("scalar").backward();
    assert_eq!(z.grad(), Some(vec![1.0, 1.0]));
    assert_eq!(x.grad(), Some(vec![3.0, -4.0]));
}

/// ...but it does read `x`.
#[test]
#[should_panic(expected = "MulAcc: input 0 was modified by an inplace operation")]
fn mulacc_saves_its_factors() {
    let x = leaf(&[1.0, 2.0]);
    let y = leaf(&[3.0, -4.0]);
    let z = leaf(&[0.5, 0.25]);
    let loss = x.mulacc(&y, &z).sum();
    x.update_data(|d| d.as_host_f32_mut().expect("CPU f32")[0] = 9.0);
    loss.into_loss().expect("scalar").backward();
}

/// `to_device`'s backward hands the incoming gradient to its one parent, on the source
/// device. A real transfer needs a GPU, so this builds the edge directly with a CPU source;
/// the GPU lane covers the transfer itself.
#[test]
fn to_device_backward_returns_the_gradient_on_the_source_device() {
    let x = leaf(&[1.0, 2.0, 3.0]);
    let moved = edge(
        [x.clone()],
        x.borrow().data.clone(),
        Shape::derived([3]),
        Device::CPU,
        GradFn::ToDevice(ToDeviceGradFn {
            source: Device::CPU,
        }),
    );
    let w = RawTensor::new(vec![2.0, -1.0, 0.5], &[3], false);
    moved
        .elem_mul(&w)
        .sum()
        .into_loss()
        .expect("scalar")
        .backward();
    assert_eq!(x.grad(), Some(vec![2.0, -1.0, 0.5]));
    assert_eq!(x.device(), Device::CPU);
}

/// The broadcast panic names the first incompatible dimension, counted from the left of
/// the longer shape, whichever operand is longer.
#[test]
#[should_panic(expected = "Cannot broadcast shapes [2, 3] and [4, 3] at dimension 0")]
fn broadcast_panic_names_the_leading_dimension() {
    let _ = RawTensor::broadcast_shape(&[2, 3], &[4, 3]);
}

#[test]
#[should_panic(expected = "Cannot broadcast shapes [2, 3, 4] and [5, 4] at dimension 1")]
fn broadcast_panic_names_the_dimension_when_the_left_operand_is_longer() {
    let _ = RawTensor::broadcast_shape(&[2, 3, 4], &[5, 4]);
}

#[test]
#[should_panic(expected = "Cannot broadcast shapes [5, 4] and [2, 3, 4] at dimension 1")]
fn broadcast_panic_names_the_dimension_when_the_right_operand_is_longer() {
    let _ = RawTensor::broadcast_shape(&[5, 4], &[2, 3, 4]);
}

#[test]
#[should_panic(expected = "Cannot broadcast shapes [3] and [2, 4] at dimension 1")]
fn broadcast_panic_names_the_trailing_dimension_of_a_rank_1_operand() {
    let _ = RawTensor::broadcast_shape(&[3], &[2, 4]);
}

/// Size-1 axes on either side broadcast, so a mismatch past them is still found.
#[test]
#[should_panic(expected = "Cannot broadcast shapes [2, 3, 1] and [4, 5] at dimension 1")]
fn broadcast_panic_skips_size_one_axes() {
    let _ = RawTensor::broadcast_shape(&[2, 3, 1], &[4, 5]);
}

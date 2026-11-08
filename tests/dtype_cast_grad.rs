//! `Tensor::to_dtype` as a differentiable op (ADDENDUM §11, P2): values follow
//! `Storage::cast` (checked against torch in `torch_ref_casts.rs`), backward casts the
//! gradient back to the input's dtype, and casts to integer or bool carry no gradient.

use half::f16;
use volta::{DType, RawTensor};

/// Leaf `x` with seeded random values, requiring grad.
fn leaf(n: usize) -> volta::Tensor {
    volta::manual_seed(7);
    let x = RawTensor::randn(&[n]);
    x.set_requires_grad(true);
    x
}

/// f32 -> f64 -> f32 keeps the values exactly and passes the gradient through unchanged.
#[test]
fn f32_f64_f32_round_trip_is_exact_with_unit_gradient() {
    let x = leaf(6);
    let y = x.to_dtype(DType::F64);
    assert_eq!(y.dtype(), DType::F64);
    assert!(y.requires_grad());
    let z = y.to_dtype(DType::F32);
    assert_eq!(z.data(), x.data());
    z.sum().into_loss().expect("scalar").backward();
    assert_eq!(x.grad().expect("x has a gradient"), vec![1.0; 6]);
    let g = x.grad_storage().expect("grad");
    assert_eq!(g.dtype(), DType::F32, "the gradient has x's dtype");
}

/// Numerical gradient check through the casts, with a nonlinear loss around them.
#[test]
fn gradcheck_through_f64_round_trip() {
    let x = leaf(5);
    let (max_err, _, passed) = RawTensor::check_gradients(
        &x,
        |x| {
            let back = x.to_dtype(DType::F64).to_dtype(DType::F32);
            back.elem_mul(x).tanh().sum()
        },
        1e-2,
        1e-2,
    );
    assert!(passed, "max relative error {max_err}");
}

/// Rounding to f16 is treated as identity in backward (PyTorch's `.to()` backward), while
/// the forward value is rounded.
#[test]
fn f16_round_trip_rounds_forward_and_passes_gradient() {
    let x = RawTensor::new(vec![0.1, 1.0 / 3.0, 70000.0], &[3], true);
    let h = x.to_dtype(DType::F16);
    let back = h.to_dtype(DType::F32);
    let expected: Vec<f32> = [0.1f32, 1.0 / 3.0, 70000.0]
        .iter()
        .map(|&v| f16::from_f32(v).to_f32())
        .collect();
    assert_eq!(back.data(), expected);
    assert_eq!(back.data()[2], f32::INFINITY, "70000 overflows f16");
    back.sum().into_loss().expect("scalar").backward();
    assert_eq!(x.grad().expect("grad"), vec![1.0; 3]);
}

/// Two paths meet at an f64 node, so its two gradient contributions are added in f64
/// before being cast back (before P2, accumulation was f32-only and panicked here):
/// d/dx [sum(y) + 3 sum(y)] = 4 for y = f32(f64(x)).
#[test]
fn gradient_accumulates_in_f64_at_a_shared_cast() {
    let x = leaf(4);
    let wide = x.to_dtype(DType::F64);
    let a = wide.to_dtype(DType::F32).sum();
    let b = wide
        .to_dtype(DType::F32)
        .elem_mul(&RawTensor::new(vec![3.0; 4], &[4], false))
        .sum();
    a.add(&b).into_loss().expect("scalar").backward();
    assert_eq!(x.grad().expect("grad"), vec![4.0; 4]);
}

/// A cast to an integer dtype carries no gradient, even from an input that requires one.
#[test]
fn cast_to_int_carries_no_gradient() {
    let x = RawTensor::new(vec![1.7, -2.2], &[2], true);
    let i = x.to_dtype(DType::I64);
    assert_eq!(i.dtype(), DType::I64);
    assert!(!i.requires_grad());
    assert_eq!(i.data(), vec![1.0, -2.0], "truncates toward zero");
}

/// A cast to bool carries no gradient either.
#[test]
fn cast_to_bool_carries_no_gradient() {
    let x = RawTensor::new(vec![0.0, f32::NAN, -0.5], &[3], true);
    let b = x.to_dtype(DType::Bool);
    assert!(!b.requires_grad());
    assert_eq!(b.data(), vec![0.0, 1.0, 1.0]);
}

/// An integer tensor cast to float is a leaf-like value that can then be made trainable.
#[test]
fn int_to_float_cast_has_values_and_no_gradient() {
    let i = RawTensor::from_typed(vec![3i32, -4], &[2]);
    let f = i.to_dtype(DType::F32);
    assert_eq!(f.dtype(), DType::F32);
    assert!(!f.requires_grad());
    assert_eq!(f.data(), vec![3.0, -4.0]);
}

/// The same dtype returns the tensor itself: a gradient set through the result is the
/// original's.
#[test]
fn same_dtype_returns_the_same_tensor() {
    let x = RawTensor::new(vec![1.0, 2.0], &[2], true);
    let same = x.to_dtype(DType::F32);
    same.set_grad(volta::Storage::cpu(vec![5.0, 6.0]));
    assert_eq!(x.grad().expect("shared"), vec![5.0, 6.0]);
}

/// A NaN cast to an integer is refused loudly (torch leaves it undefined).
#[test]
#[should_panic(expected = "element 1 is NaN, outside i32's range")]
fn nan_to_int_cast_panics() {
    let _ = RawTensor::new(vec![1.0, f32::NAN], &[2], false).to_dtype(DType::I32);
}

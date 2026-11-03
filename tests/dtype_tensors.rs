//! Tensors of non-f32 dtypes (ADDENDUM §11, P1): they can be built and read back, integer
//! and bool tensors cannot require gradients, and every op refuses a non-f32 operand
//! loudly instead of converting it, until the kernels learn other dtypes (P3, P4).

use half::{bf16, f16};
use volta::{DType, RawTensor, Storage, Tensor, VoltaError};

/// Each dtype's tensor reports its dtype, and `data()` decodes it to f32.
#[test]
fn from_typed_keeps_the_dtype_and_data_decodes_it() {
    let cases: Vec<(Tensor, DType, Vec<f32>)> = vec![
        (
            RawTensor::from_typed(vec![f16::from_f32(1.5), f16::from_f32(-2.0)], &[2]),
            DType::F16,
            vec![1.5, -2.0],
        ),
        (
            RawTensor::from_typed(vec![bf16::from_f32(0.5), bf16::from_f32(4.0)], &[2]),
            DType::BF16,
            vec![0.5, 4.0],
        ),
        (
            RawTensor::from_typed(vec![1.5f32, -2.0], &[2]),
            DType::F32,
            vec![1.5, -2.0],
        ),
        (
            RawTensor::from_typed(vec![0.25f64, -8.0], &[2, 1]),
            DType::F64,
            vec![0.25, -8.0],
        ),
        (
            RawTensor::from_typed(vec![-3i32, 7], &[2]),
            DType::I32,
            vec![-3.0, 7.0],
        ),
        (
            RawTensor::from_typed(vec![-3i64, 7], &[1, 2]),
            DType::I64,
            vec![-3.0, 7.0],
        ),
        (
            RawTensor::from_typed(vec![0u8, 255], &[2]),
            DType::U8,
            vec![0.0, 255.0],
        ),
        (
            RawTensor::from_typed(vec![true, false], &[2]),
            DType::Bool,
            vec![1.0, 0.0],
        ),
    ];
    for (t, dtype, expected) in cases {
        assert_eq!(t.dtype(), dtype);
        assert_eq!(t.data(), expected, "{dtype}");
        assert!(!t.requires_grad());
    }
}

/// Existing f32 constructors still make f32 tensors. `from_vec` with unsuffixed literals
/// is the case that matters: a generic `from_vec` would infer f64 from `vec![1.0]`.
#[test]
fn f32_constructors_are_f32() {
    assert_eq!(
        RawTensor::from_vec(vec![1.0, 2.0], &[2]).dtype(),
        DType::F32
    );
    assert_eq!(RawTensor::new(vec![1.0], &[1], true).dtype(), DType::F32);
    assert_eq!(RawTensor::zeros(&[2, 2]).dtype(), DType::F32);
    assert_eq!(RawTensor::randn(&[3]).dtype(), DType::F32);
}

/// A float tensor of any width can require gradients.
#[test]
fn float_tensors_can_require_grad() {
    let t = RawTensor::from_typed(vec![1.0f64, 2.0], &[2]);
    t.set_requires_grad(true);
    assert!(t.requires_grad());
}

/// An integer tensor has no gradient; asking for one is refused (PyTorch raises too).
#[test]
#[should_panic(expected = "Only floating-point tensors can require gradients, got i64")]
fn int_tensor_cannot_require_grad() {
    RawTensor::from_typed(vec![1i64, 2], &[2]).set_requires_grad(true);
}

/// A bool tensor has no gradient either.
#[test]
#[should_panic(expected = "Only floating-point tensors can require gradients, got bool")]
fn bool_tensor_cannot_require_grad() {
    RawTensor::from_typed(vec![true], &[1]).set_requires_grad(true);
}

/// Turning gradients off is always allowed.
#[test]
fn int_tensor_can_turn_grad_off() {
    let t = RawTensor::from_typed(vec![1i32], &[1]);
    t.set_requires_grad(false);
    assert!(!t.requires_grad());
}

/// The fallible constructor reports shape errors for typed data like for f32.
#[test]
fn try_from_typed_checks_the_length() {
    let err = RawTensor::try_from_typed(vec![1i64, 2, 3], &[2, 2]).expect_err("3 for [2, 2]");
    assert!(matches!(err, VoltaError::ShapeDataMismatch { .. }), "{err}");
}

/// A mixed-dtype op fails loudly; before P1 the f64 operand was silently read as f32.
#[test]
#[should_panic(expected = "DType mismatch: expected F32, got F64")]
fn mixed_dtype_binary_op_panics() {
    let a = RawTensor::new(vec![1.0, 2.0], &[2], false);
    let b = RawTensor::from_typed(vec![1.0f64, 2.0], &[2]);
    let _ = a.add(&b);
}

/// Two f64 operands agree, but no kernel computes in f64 yet (P3): still a loud failure.
#[test]
#[should_panic(expected = "DType mismatch: expected F32, got F64")]
fn f64_binary_op_panics_until_kernels_support_it() {
    let a = RawTensor::from_typed(vec![1.0f64, 2.0], &[2]);
    let _ = a.add(&a);
}

/// Unary ops read through the same check.
#[test]
#[should_panic(expected = "DType mismatch: expected F32, got F16")]
fn f16_unary_op_panics_until_kernels_support_it() {
    let _ = RawTensor::from_typed(vec![f16::from_f32(1.0)], &[1]).exp();
}

/// Matmul reads through the same check.
#[test]
#[should_panic(expected = "DType mismatch: expected F32, got I64")]
fn int_matmul_panics() {
    let a = RawTensor::from_typed(vec![1i64, 2, 3, 4], &[2, 2]);
    let _ = a.matmul(&a);
}

/// A seeded gradient must have the tensor's dtype.
#[test]
#[should_panic(expected = "set_grad: DType mismatch: expected F32, got F64")]
fn set_grad_rejects_another_dtype() {
    let t = RawTensor::new(vec![1.0, 2.0], &[2], true);
    t.set_grad(Storage::from_typed(vec![1.0f64, 1.0]));
}

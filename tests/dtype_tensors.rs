//! Tensors of non-f32 dtypes (ADDENDUM §11): they can be built and read back, integer and
//! bool tensors cannot require gradients, half ops compute in f32 and round each output,
//! and an op refuses an operand it has no kernel for (integers and bool) or mixed dtypes
//! (until P6) loudly, naming the op, instead of converting. f64 values are checked in
//! `src/lib_tests/f64_tests.rs`, f16/bf16 against torch in `tests/torch_ref_half.rs`.

use half::{bf16, f16};
use volta::{Conv2d, DType, RawTensor, SGD, Storage, Tensor, VoltaError};

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
#[should_panic(expected = "Binary(Add): DType mismatch: expected F32, got F64 (input 1)")]
fn mixed_dtype_binary_op_panics() {
    let a = RawTensor::new(vec![1.0, 2.0], &[2], false);
    let b = RawTensor::from_typed(vec![1.0f64, 2.0], &[2]);
    let _ = a.add(&b);
}

/// A half op decodes to f32, computes there, and rounds the output once (PyTorch CPU).
/// `x * y` for these bf16 values differs from the bf16 product rounded twice.
#[test]
fn half_ops_compute_in_f32_and_round_the_output() {
    let xs = [1.007_812_5_f32, 3.0, -0.3, 1e-3];
    let ys = [1.007_812_5_f32, 0.1, 7.0, 3e4];
    let x = RawTensor::from_typed(xs.map(bf16::from_f32).to_vec(), &[4]);
    let y = RawTensor::from_typed(ys.map(bf16::from_f32).to_vec(), &[4]);
    let got = x.elem_mul(&y).to_vec_as::<bf16>().expect("bf16");
    let want: Vec<bf16> = xs
        .iter()
        .zip(&ys)
        .map(|(&a, &b)| bf16::from_f32(bf16::from_f32(a).to_f32() * bf16::from_f32(b).to_f32()))
        .collect();
    assert_eq!(got, want);

    let h = RawTensor::from_typed(xs.map(f16::from_f32).to_vec(), &[4]);
    let got = h.exp().to_vec_as::<f16>().expect("f16");
    let want: Vec<f16> = xs
        .iter()
        .map(|&a| f16::from_f32(f16::from_f32(a).to_f32().exp()))
        .collect();
    assert_eq!(got, want);
}

/// A half tensor's gradient is half too, and a second contribution is added in f32 and
/// rounded.
#[test]
fn half_gradient_has_the_tensor_dtype() {
    let x = RawTensor::from_typed(vec![bf16::from_f32(1.5), bf16::from_f32(-2.0)], &[2]);
    x.set_requires_grad(true);
    x.elem_mul(&x).sum().into_loss().expect("scalar").backward();
    let g = x.grad_storage().expect("a gradient");
    assert_eq!(g.dtype(), DType::BF16);
    assert_eq!(
        g.as_host::<bf16>().expect("bf16"),
        &[bf16::from_f32(3.0), bf16::from_f32(-4.0)]
    );
}

/// Matmul reads through the same check.
#[test]
#[should_panic(expected = "MatMul: no i64 kernel")]
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

/// Movement ops check at entry too: integer tensors have no kernels until P5.
#[test]
#[should_panic(expected = "Reshape: no i64 kernel")]
fn int_reshape_panics_naming_the_op() {
    let _ = RawTensor::from_typed(vec![1i64, 2, 3, 4], &[4]).reshape(&[2, 2]);
}

/// An f64 input to an f32 layer is a mismatch, named by the layer: `to_dtype` the layer
/// (or the input) first.
#[test]
#[should_panic(expected = "Conv2d: DType mismatch: expected F64, got F32 (input 1)")]
fn f64_input_to_f32_conv_panics_naming_the_layer() {
    let conv = Conv2d::new(1, 1, 1, 1, 0, false);
    let x = RawTensor::from_typed(vec![1.0f64; 4], &[1, 1, 2, 2]);
    let _ = conv.forward(&x);
}

/// An optimizer steps each parameter in its own dtype: a mixed f32/f64 list trains both.
#[test]
fn sgd_steps_mixed_f32_and_f64_params() {
    let a = RawTensor::new(vec![1.0], &[1], true);
    let b = RawTensor::from_typed(vec![1.0f64], &[1]);
    b.set_requires_grad(true);
    a.set_grad(Storage::cpu(vec![0.5]));
    b.set_grad(Storage::from_typed(vec![0.25f64]));
    let mut opt = SGD::new(vec![a.clone(), b.clone()], 0.5, 0.0, 0.0);
    opt.step();
    assert_eq!(a.to_vec_as::<f32>().expect("f32"), vec![0.75]);
    assert_eq!(b.to_vec_as::<f64>().expect("f64"), vec![0.875]);
}

/// An f16 parameter steps in f32 and rounds once: the update `1 - 0.1 * g` is the f32
/// result rounded to f16, next to an f32 parameter in the same optimizer.
#[test]
fn sgd_steps_an_f16_param_in_f32_and_rounds_once() {
    let a = RawTensor::new(vec![1.0], &[1], true);
    let b = RawTensor::from_typed(vec![f16::from_f32(1.0)], &[1]);
    b.set_requires_grad(true);
    let g = f16::from_f32(0.3);
    a.set_grad(Storage::cpu(vec![0.3]));
    b.set_grad(Storage::from_typed(vec![g]));
    let mut opt = SGD::new(vec![a.clone(), b.clone()], 0.1, 0.0, 0.0);
    opt.step();
    assert_eq!(a.dtype(), DType::F32);
    assert_eq!(
        b.to_vec_as::<f16>().expect("f16"),
        vec![f16::from_f32(0.1f32.mul_add(-g.to_f32(), 1.0))]
    );
}

/// `to_vec_as` returns the values in their own dtype, exactly: no f32 decode.
#[test]
fn to_vec_as_reads_every_dtype_losslessly() {
    let f64s = vec![0.1f64, 1.0 + 1e-12, -3.5e300];
    assert_eq!(
        RawTensor::from_typed(f64s.clone(), &[3])
            .to_vec_as::<f64>()
            .expect("f64"),
        f64s
    );
    let i64s = vec![i64::MIN, 0, (1 << 53) + 1];
    assert_eq!(
        RawTensor::from_typed(i64s.clone(), &[3])
            .to_vec_as::<i64>()
            .expect("i64"),
        i64s
    );
    assert_eq!(
        RawTensor::from_typed(vec![true, false], &[2])
            .to_vec_as::<bool>()
            .expect("bool"),
        vec![true, false]
    );
    assert_eq!(
        RawTensor::new(vec![1.5, -2.0], &[2], false)
            .to_vec_as::<f32>()
            .expect("f32"),
        vec![1.5, -2.0]
    );
}

/// Asking for another dtype is an error, not a conversion.
#[test]
fn to_vec_as_refuses_another_dtype() {
    let err = RawTensor::new(vec![1.0], &[1], false)
        .to_vec_as::<f64>()
        .expect_err("f32 tensor read as f64");
    assert!(
        matches!(
            err,
            VoltaError::DTypeMismatch {
                expected: DType::F64,
                actual: DType::F32
            }
        ),
        "{err}"
    );
}

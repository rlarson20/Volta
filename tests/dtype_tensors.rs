//! Tensors of non-f32 dtypes (ADDENDUM §11): they can be built and read back, integer and
//! bool tensors cannot require gradients, half ops compute in f32 and round each output,
//! mixed operands of elementwise ops promote (or panic under `DTypePolicy::Strict`), and an
//! op refuses an operand it has no kernel for (integer arithmetic) or mixed dtypes where
//! PyTorch does too (matmul, conv, layers) loudly, naming the op. f64 values are checked in
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

/// A mixed f32/f64 product promotes to f64 (PyTorch), and each operand's gradient comes
/// back in its own dtype through the inserted cast. Before P1 the f64 operand was silently
/// read as f32; since P6 it is cast, in the graph, to the wider dtype.
#[test]
fn mixed_dtype_binary_op_promotes_and_grads_keep_their_dtype() {
    let a = RawTensor::new(vec![0.1, 2.0], &[2], true);
    let b = RawTensor::from_typed(vec![1.0f64 + 1e-12, 3.0], &[2]);
    b.set_requires_grad(true);
    let out = a.elem_mul(&b);
    assert_eq!(out.dtype(), DType::F64);
    assert_eq!(
        out.to_vec_as::<f64>().expect("f64"),
        vec![f64::from(0.1f32) * (1.0 + 1e-12), 6.0]
    );
    out.sum().into_loss().expect("scalar").backward();
    let ga = a.grad_storage().expect("grad");
    let gb = b.grad_storage().expect("grad");
    assert_eq!(ga.dtype(), DType::F32);
    assert_eq!(gb.dtype(), DType::F64);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the f64 gradient rounds to f32"
    )]
    let want_a = (1.0f64 + 1e-12) as f32;
    assert_eq!(ga.as_host::<f32>().expect("f32"), &[want_a, 3.0]);
    assert_eq!(gb.as_host::<f64>().expect("f64"), &[f64::from(0.1f32), 2.0]);
}

/// Under `DTypePolicy::Strict` a mixed-dtype op panics instead of casting, and the guard
/// restores `Promote` when it drops.
#[test]
fn strict_dtype_policy_refuses_mixed_operands() {
    let a = RawTensor::new(vec![1.0, 2.0], &[2], false);
    let b = RawTensor::from_typed(vec![1.0f64, 2.0], &[2]);
    {
        let _strict = volta::with_dtype_policy(volta::DTypePolicy::Strict);
        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| a.add(&b)))
            .expect_err("Strict refuses a mixed add");
        let msg = err
            .downcast_ref::<String>()
            .map_or("", String::as_str)
            .to_string();
        assert!(
            msg.contains("Binary(Add): DType mismatch: expected F32, got F64 (input 1) under DTypePolicy::Strict"),
            "{msg}"
        );
        assert_eq!(
            a.add(&a).dtype(),
            DType::F32,
            "one dtype is fine under Strict"
        );
    }
    assert_eq!(volta::dtype_policy(), volta::DTypePolicy::Promote);
    assert_eq!(a.add(&b).dtype(), DType::F64);
}

/// Integer division is true division, in f32 (PyTorch's `div`); integer `add` still has no
/// kernel (data-only integers, `notes/TODO.md`).
#[test]
fn int_division_is_true_division_in_f32() {
    let a = RawTensor::from_typed(vec![1i64, 7, -3], &[3]);
    let b = RawTensor::from_typed(vec![3i64, 2, 2], &[3]);
    let q = a.div(&b);
    assert_eq!(q.dtype(), DType::F32);
    assert_eq!(
        q.to_vec_as::<f32>().expect("f32"),
        vec![1.0 / 3.0, 3.5, -1.5]
    );
}

/// `i64 + i64` promotes to i64, which has no arithmetic kernel: it panics, naming the op.
#[test]
#[should_panic(expected = "Binary(Add): no i64 kernel")]
fn int_add_panics_naming_the_op() {
    let a = RawTensor::from_typed(vec![1i64], &[1]);
    let _ = a.add(&a);
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

/// Movement ops are data copies, so they work on every dtype and move values exactly:
/// i64 beyond 2^53 and the bool pattern survive permute, expand, shrink, pad and transpose.
#[test]
fn int_and_bool_movement_ops_copy_exactly() {
    let big = 1i64 << 60;
    let x = RawTensor::from_typed(vec![big, -1, big + 1, 7, i64::MIN, 0], &[2, 3]);
    assert_eq!(
        x.reshape(&[3, 2]).to_vec_as::<i64>().expect("i64"),
        x.to_vec_as::<i64>().expect("i64")
    );
    assert_eq!(
        x.permute(&[volta::Axis(1), volta::Axis(0)])
            .to_vec_as::<i64>()
            .expect("i64"),
        vec![big, 7, -1, i64::MIN, big + 1, 0]
    );
    assert_eq!(
        x.transpose().to_vec_as::<i64>().expect("i64"),
        vec![big, 7, -1, i64::MIN, big + 1, 0]
    );
    assert_eq!(
        x.shrink(&[(1, 2), (0, 2)]).to_vec_as::<i64>().expect("i64"),
        vec![7, i64::MIN]
    );
    assert_eq!(
        x.shrink(&[(0, 1), (0, 3)])
            .expand(&[2, 3])
            .to_vec_as::<i64>()
            .expect("i64"),
        vec![big, -1, big + 1, big, -1, big + 1]
    );

    let m = RawTensor::from_typed(vec![true, false], &[2]);
    let padded = m.pad_with(&[(1, 1)], volta::PadValue::ZERO);
    assert_eq!(padded.dtype(), DType::Bool);
    assert_eq!(
        padded.to_vec_as::<bool>().expect("bool"),
        vec![false, true, false, false]
    );
}

/// Max pooling's `-inf` border has no integer value: padding an integer tensor with it
/// panics instead of wrapping.
#[test]
#[should_panic(expected = "Pad: a -inf fill cannot be stored in a i32 tensor")]
fn int_pad_with_neg_inf_panics() {
    let _ =
        RawTensor::from_typed(vec![1i32, 2], &[2]).pad_with(&[(1, 0)], volta::PadValue::NEG_INF);
}

/// Comparisons return Bool on every dtype, with broadcasting; NaN compares false except
/// under `elem_ne`.
#[test]
fn comparisons_return_bool_on_every_dtype() {
    let a = RawTensor::from_typed(vec![1i64, 5, -3], &[3]);
    let b = RawTensor::from_typed(vec![2i64], &[1]);
    let get = |t: &Tensor| {
        assert_eq!(t.dtype(), DType::Bool);
        t.to_vec_as::<bool>().expect("bool")
    };
    assert_eq!(get(&a.lt(&b)), vec![true, false, true]);
    assert_eq!(get(&a.ge(&b)), vec![false, true, false]);
    assert_eq!(get(&a.elem_eq(&a)), vec![true, true, true]);

    let h = RawTensor::from_typed(vec![bf16::from_f32(1.0), bf16::NAN], &[2]);
    let one = RawTensor::from_typed(vec![bf16::from_f32(1.0)], &[1]);
    assert_eq!(get(&h.le(&one)), vec![true, false]);
    assert_eq!(get(&h.elem_ne(&h)), vec![false, true]);

    let p = RawTensor::from_typed(vec![true, false], &[2]);
    let q = RawTensor::from_typed(vec![false], &[1]);
    assert_eq!(get(&p.gt(&q)), vec![true, false]);
}

/// Mixed comparisons compare in the promoted dtype: i64 against f32 compares in f32, so
/// 2^24 + 1 (rounded to 2^24) equals 2^24, as in PyTorch.
#[test]
fn comparison_of_mixed_dtypes_promotes() {
    let a = RawTensor::from_typed(vec![(1i64 << 24) + 1, 3], &[2]);
    let b = RawTensor::new(vec![16_777_216.0, 2.5], &[2], false);
    assert_eq!(
        a.elem_eq(&b).to_vec_as::<bool>().expect("bool"),
        vec![true, false]
    );
    assert_eq!(
        a.gt(&b).to_vec_as::<bool>().expect("bool"),
        vec![false, true]
    );
}

/// `where` selects from branches of any dtype under a Bool condition.
#[test]
fn where_selects_int_branches_under_a_bool_condition() {
    let cond = RawTensor::from_typed(vec![true, false, true], &[3]);
    let x = RawTensor::from_typed(vec![1i64 << 60, 2, 3], &[3]);
    let y = RawTensor::from_typed(vec![-1i64], &[1]);
    let out = cond.where_op(&x, &y);
    assert_eq!(out.to_vec_as::<i64>().expect("i64"), vec![1 << 60, -1, 3]);
}

/// A float condition is refused (PyTorch's `where` takes bool), naming the fix.
#[test]
#[should_panic(expected = "Where: the condition must be Bool, got f32")]
fn where_with_float_condition_panics() {
    let cond = RawTensor::new(vec![1.0, 0.0], &[2], false);
    let x = RawTensor::new(vec![1.0, 2.0], &[2], false);
    let _ = cond.where_op(&x, &x);
}

/// An f64 input to an f32 layer is a mismatch, named by the layer: `to_dtype` the layer
/// (or the input) first.
#[test]
#[should_panic(
    expected = "Conv2d: DType mismatch: expected F64, got F32 (input 1); this op needs one dtype, as in PyTorch: cast with to_dtype (a layer with Module::to_dtype)"
)]
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

/// `Embedding` is a `Layer`: in a `Sequential` with a `Linear` head, trained with
/// class-index (I64) cross-entropy, it learns an arbitrary token-to-class map, and the
/// gradient reaches the embedding table.
#[test]
fn embedding_in_sequential_trains_with_class_index_targets() {
    use volta::{Adam, Embedding, Layer, Linear, Module, Sequential};
    volta::manual_seed(7);
    let model = Sequential::new(vec![
        Box::new(Embedding::new(8, 6)),
        Box::new(Linear::new(6, 3, true)),
    ]);
    let tokens = RawTensor::from_typed((0..8i64).collect(), &[8]);
    let classes: Vec<i64> = (0..8i64).map(|i| (i * 5 + 1) % 3).collect();
    let targets = RawTensor::from_typed(classes.clone(), &[8]);
    let mut opt = Adam::new(model.parameters(), 0.05, (0.9, 0.999), 1e-8, 0.0);

    let loss_of = || RawTensor::cross_entropy_loss(&model.forward(&tokens), &targets);
    let first = loss_of().item();
    for _ in 0..150 {
        opt.zero_grad();
        loss_of().into_loss().expect("scalar").backward();
        opt.step();
    }
    let last = loss_of().item();
    assert!(first > 0.9, "starts near ln 3, got {first}");
    assert!(last < 0.05, "loss {first} -> {last}");

    let logits = model.forward(&tokens).data();
    for (i, &c) in classes.iter().enumerate() {
        let row = &logits[i * 3..i * 3 + 3];
        let best = (0..3)
            .max_by(|&a, &b| row[a].total_cmp(&row[b]))
            .expect("three classes");
        assert_eq!(i64::try_from(best).expect("small"), c, "token {i}");
    }
    let emb_grad_seen = model.parameters()[0].grad_storage().is_some();
    assert!(emb_grad_seen, "the embedding table received a gradient");
}

/// Class-index targets give exactly the one-hot loss and gradient (I64 and I32), and an
/// out-of-range target panics instead of indexing past the row.
#[test]
fn cross_entropy_class_indices_match_one_hot() {
    let logits = [0.3f32, -1.2, 2.0, 0.5, 0.1, -0.7];
    let run = |targets: &Tensor| {
        let x = RawTensor::new(logits.to_vec(), &[2, 3], true);
        let loss = RawTensor::cross_entropy_loss(&x, targets);
        let v = loss.item();
        loss.into_loss().expect("scalar").backward();
        (v, x.grad().expect("a gradient"))
    };
    let one_hot = RawTensor::new(vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0], &[2, 3], false);
    let want = run(&one_hot);
    assert_eq!(run(&RawTensor::from_typed(vec![2i64, 0], &[2])), want);
    assert_eq!(run(&RawTensor::from_typed(vec![2i32, 0], &[2])), want);
}

/// A class index outside `0..C` is refused, naming the loss.
#[test]
#[should_panic(expected = "cross_entropy_loss: target 3 out of range for 3 classes")]
fn cross_entropy_out_of_range_target_panics() {
    let x = RawTensor::new(vec![0.0; 6], &[2, 3], false);
    let _ = RawTensor::cross_entropy_loss(&x, &RawTensor::from_typed(vec![0i64, 3], &[2]));
}

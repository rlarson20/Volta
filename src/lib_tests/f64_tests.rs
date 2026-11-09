//! f64 end to end (ADDENDUM §11, P3a): every core op family computes in f64 and passes a
//! numerical gradient check in f64 at a tolerance (1e-7) five orders of magnitude tighter
//! than the f32 checks use (1e-2), and f64 values keep f64 precision.
//!
//! The check is a test-only f64 copy of `check_gradients`: there is no public f64 gradient
//! check yet.

use super::*;
use crate::autograd::no_grad;
use crate::ops::ReduceOp;
use crate::tensor::manual_seed;

/// Relative tolerance of the f64 gradient checks.
const TOL: f64 = 1e-7;
/// Central-difference step: truncation error ~ EPS² and rounding error ~ 1e-16 / EPS
/// balance near 1e-10.
const EPS: f64 = 1e-6;

/// Seeded normal values as an f64 leaf that requires grad.
fn leaf(shape: &[usize], seed: u64) -> Tensor {
    leaf_from(shape, seed, |v| v)
}

/// Seeded normal values mapped through `f`, as an f64 leaf that requires grad.
fn leaf_from(shape: &[usize], seed: u64, f: impl Fn(f64) -> f64) -> Tensor {
    manual_seed(seed);
    let raw = RawTensor::randn(shape);
    let vals: Vec<f64> = raw.data().iter().map(|&v| f(f64::from(v))).collect();
    let x = RawTensor::from_typed(vals, shape);
    x.set_requires_grad(true);
    x
}

/// The f64 values of `t`, through the public lossless readback.
fn values(t: &Tensor) -> Vec<f64> {
    t.to_vec_as::<f64>().expect("an f64 tensor")
}

/// Weighted sum `sum(out * w)` with fixed random f64 weights, so every output element
/// gets a different gradient (a plain sum would hide permutations and broadcasts).
fn weighted(out: &Tensor, seed: u64) -> Tensor {
    let w = leaf(out.shape().truncate_rank(), seed);
    w.set_requires_grad(false);
    out.elem_mul(&w).sum()
}

/// Max relative error between `x`'s analytic gradient of `f` and a central difference,
/// both in f64. Relative to `max(|analytic|, |numeric|, 1)`.
fn gradcheck(x: &Tensor, f: impl Fn(&Tensor) -> Tensor) -> f64 {
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
fn assert_grad(name: &str, x: &Tensor, f: impl Fn(&Tensor) -> Tensor) {
    let err = gradcheck(x, f);
    assert!(
        err < TOL,
        "{name}: f64 gradcheck max relative error {err:.3e}"
    );
}

/// Sums in f64 round like f64, not like f32: ten 0.1s.
#[test]
fn f64_sum_keeps_f64_rounding() {
    let t = RawTensor::from_typed(vec![0.1f64; 10], &[10]);
    let s = t.sum();
    assert_eq!(s.dtype(), DType::F64);
    let expected: f64 = std::iter::repeat_n(0.1f64, 10).sum();
    assert_eq!(values(&s), vec![expected]);
    assert_ne!(
        expected,
        f64::from(10.0f32 * 0.1f32),
        "the case distinguishes f32"
    );
}

/// 1 + 1e-12 survives in f64 (it is 1.0 in f32).
#[test]
fn f64_add_keeps_small_terms() {
    let a = RawTensor::from_typed(vec![1.0f64], &[1]);
    let b = RawTensor::from_typed(vec![1e-12f64], &[1]);
    assert_eq!(values(&a.add(&b)), vec![1.0 + 1e-12]);
}

/// Every unary op, on inputs inside its domain.
#[test]
fn unary_ops_gradcheck_in_f64() {
    for &op in UnaryOp::ALL {
        let x = leaf_from(&[3, 4], 11, |v| match op {
            UnaryOp::Log | UnaryOp::Log2 | UnaryOp::Sqrt | UnaryOp::Recip => 0.5 + v.abs(),
            UnaryOp::ReLU => (0.3 + v.abs()).copysign(v),
            UnaryOp::Neg
            | UnaryOp::Exp2
            | UnaryOp::Exp
            | UnaryOp::Sin
            | UnaryOp::Cos
            | UnaryOp::Tanh
            | UnaryOp::Sigmoid
            | UnaryOp::Erf => v,
        });
        assert_grad(&format!("{op:?}"), &x, |t| {
            weighted(&RawTensor::unary_op(t, op), 12)
        });
    }
}

/// Every differentiable binary op, same shape and broadcast, for both operands.
#[test]
fn binary_ops_gradcheck_in_f64() {
    let shapes: [(&[usize], &[usize]); 3] =
        [(&[2, 3], &[2, 3]), (&[3, 1], &[1, 4]), (&[2, 3], &[3])];
    for &op in BinaryOp::ALL.iter().filter(|op| !op.is_trivial()) {
        for (sa, sb) in shapes {
            let a = leaf(sa, 21);
            // Div's divisor away from 0; Max's operands away from ties.
            let b = leaf_from(sb, 22, |v| match op {
                BinaryOp::Div => 0.5 + v.abs(),
                BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Max => v + 0.25,
                BinaryOp::Mod => unreachable!("filtered: not differentiable"),
            });
            let name = format!("{op:?} {sa:?} {sb:?}");
            let b2 = b.clone();
            assert_grad(&format!("{name} lhs"), &a, |t| {
                weighted(&RawTensor::binary_op(t, &b2, op), 23)
            });
            let a2 = a.clone();
            assert_grad(&format!("{name} rhs"), &b, |t| {
                weighted(&RawTensor::binary_op(&a2, t, op), 23)
            });
        }
    }
}

/// Whole-tensor reductions.
#[test]
fn reductions_gradcheck_in_f64() {
    let x = leaf(&[3, 4], 31);
    assert_grad("Sum", &x, |t| {
        RawTensor::reduce_op(t, ReduceOp::Sum).elem_mul(t).sum()
    });
    assert_grad("Mean", &x, |t| {
        RawTensor::reduce_op(t, ReduceOp::Mean).elem_mul(t).sum()
    });
    assert_grad("Max", &x, |t| RawTensor::reduce_op(t, ReduceOp::Max));
}

/// Axis reductions and the softmaxes built on them.
#[test]
fn axis_reductions_gradcheck_in_f64() {
    let x = leaf(&[2, 3, 4], 41);
    for axis in 0..3 {
        let ax = Axis(axis);
        for keepdim in [false, true] {
            assert_grad(&format!("sum_dim {axis} {keepdim}"), &x, |t| {
                weighted(&RawTensor::sum_dim(t, ax, keepdim), 42)
            });
            assert_grad(&format!("max_dim {axis} {keepdim}"), &x, |t| {
                weighted(&RawTensor::max_dim(t, ax, keepdim), 42)
            });
            assert_grad(&format!("mean_dim {axis} {keepdim}"), &x, |t| {
                weighted(&RawTensor::mean_dim(t, ax, keepdim), 42)
            });
        }
        assert_grad(&format!("softmax {axis}"), &x, |t| {
            weighted(&RawTensor::softmax(t, ax), 43)
        });
        assert_grad(&format!("log_softmax {axis}"), &x, |t| {
            weighted(&RawTensor::log_softmax(t, ax), 43)
        });
    }
}

/// `mulacc` for each input, and `where` for both branches with a broadcast condition.
#[test]
fn ternary_ops_gradcheck_in_f64() {
    let (x, y, z) = (leaf(&[2, 3], 51), leaf(&[2, 3], 52), leaf(&[2, 3], 53));
    let (y2, z2) = (y.clone(), z.clone());
    assert_grad("MulAcc x", &x, |t| weighted(&t.mulacc(&y2, &z2), 54));
    let (x2, z3) = (x.clone(), z.clone());
    assert_grad("MulAcc y", &y, |t| weighted(&x2.mulacc(t, &z3), 54));
    let (x3, y3) = (x.clone(), y.clone());
    assert_grad("MulAcc z", &z, |t| weighted(&x3.mulacc(&y3, t), 54));

    let cond = RawTensor::from_typed(vec![true, false, true], &[3]);
    let a = leaf(&[2, 3], 55);
    let b = leaf(&[2, 1], 56);
    let (c2, b2) = (cond.clone(), b.clone());
    assert_grad("Where true", &a, |t| weighted(&c2.where_op(t, &b2), 57));
    let (c3, a3) = (cond.clone(), a.clone());
    assert_grad("Where false", &b, |t| weighted(&c3.where_op(&a3, t), 57));
}

/// Every movement op.
#[test]
fn movement_ops_gradcheck_in_f64() {
    let x = leaf(&[2, 3, 4], 61);
    assert_grad("reshape", &x, |t| weighted(&t.reshape(&[4, 6]), 62));
    assert_grad("permute", &x, |t| {
        weighted(&t.permute(&[Axis(2), Axis(0), Axis(1)]), 62)
    });
    assert_grad("pad", &x, |t| {
        weighted(&t.pad_with(&[(1, 0), (0, 2), (1, 1)], PadValue::ZERO), 62)
    });
    assert_grad("shrink", &x, |t| {
        weighted(&t.shrink(&[(0, 2), (1, 3), (1, 4)]), 62)
    });
    assert_grad("stride", &x, |t| weighted(&t.stride_op(&[1, 2, 2]), 62));
    let small = leaf(&[1, 3, 1], 63);
    assert_grad("expand", &small, |t| weighted(&t.expand(&[2, 3, 4]), 62));
}

/// Every matmul form, both operands, and transpose.
#[test]
fn matmul_gradcheck_in_f64() {
    let cases: [(&[usize], &[usize]); 5] = [
        (&[3, 4], &[4, 2]),
        (&[2, 1, 3, 4], &[1, 3, 4, 2]),
        (&[3, 4], &[4]),
        (&[4], &[4, 2]),
        (&[4], &[4]),
    ];
    for (sa, sb) in cases {
        let (a, b) = (leaf(sa, 71), leaf(sb, 72));
        let b2 = b.clone();
        assert_grad(&format!("matmul {sa:?} {sb:?} lhs"), &a, |t| {
            weighted(&t.matmul(&b2), 73)
        });
        let a2 = a.clone();
        assert_grad(&format!("matmul {sa:?} {sb:?} rhs"), &b, |t| {
            weighted(&a2.matmul(t), 73)
        });
    }
    let x = leaf(&[3, 5], 74);
    assert_grad("transpose", &x, |t| weighted(&t.transpose(), 75));
}

/// f64 matmul agrees with a plain f64 triple loop (dgemm is not the f32 kernel).
#[test]
fn f64_matmul_matches_a_reference_loop() {
    let (m, k, n) = (5, 7, 3);
    let a = leaf(&[m, k], 81);
    let b = leaf(&[k, n], 82);
    let (av, bv) = (values(&a), values(&b));
    let got = values(&a.matmul(&b));
    for i in 0..m {
        for j in 0..n {
            let want: f64 = (0..k).map(|p| av[i * k + p] * bv[p * n + j]).sum();
            let g = got[i * n + j];
            assert!(
                (g - want).abs() <= 1e-14 * want.abs().max(1.0),
                "({i},{j}): {g} vs {want}"
            );
        }
    }
}

/// An f32 parameter can be trained through an f64 section: its gradient comes back in f32.
#[test]
fn f32_leaf_through_f64_ops_gets_an_f32_gradient() {
    manual_seed(91);
    let x = RawTensor::randn(&[4]);
    x.set_requires_grad(true);
    let y = x.to_dtype(DType::F64).exp().sum();
    y.into_loss().expect("scalar").backward();
    let g = x.grad_storage().expect("grad");
    assert_eq!(g.dtype(), DType::F32);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the f64 gradient is cast back to f32, which is what the test checks"
    )]
    let expected: Vec<f32> = x
        .data()
        .iter()
        .map(|v| f64::from(*v).exp() as f32)
        .collect();
    assert_eq!(x.grad().expect("grad"), expected);
}

/// Unary ops compute in f64: exp(1e-12) is 1 + 1e-12 in f64 and exactly 1 in f32, and
/// erf goes through libm's f64 `erf`, not `erff`.
#[test]
fn f64_unary_values_are_f64() {
    let x = RawTensor::from_typed(vec![1e-12f64, 0.5], &[2]);
    let e = values(&x.exp());
    assert_eq!(crate::index::fetch(&e, 0), 1e-12f64.exp());
    assert_ne!(crate::index::fetch(&e, 0), 1.0);
    assert_eq!(crate::index::fetch(&values(&x.erf()), 1), libm::erf(0.5));
}

// ===== Layers and optimizers in f64 (P3b) =====

use crate::nn::layers::lstm::LSTMCell;
use crate::nn::{
    BatchNorm1d, BatchNorm2d, Conv2d, ConvAlgo, ConvTranspose2d, Embedding, FeedForward, GELU,
    Layer, LayerNorm, Linear, MaxPool2d, Module, MultiHeadAttention, PixelShuffle,
    PositionalEncoding, PositionalEncodingType, RecurrentLayer, TransformerBlock,
};

/// Max relative error of `param`'s analytic gradient of `loss` against a central
/// difference taken by perturbing `param` in place (f64). `param` is restored after.
fn param_gradcheck(param: &Tensor, loss: impl Fn() -> Tensor) -> f64 {
    RawTensor::zero_grad(param);
    loss().into_loss().expect("scalar loss").backward();
    let grad = param.grad_storage().expect("param has a gradient");
    assert_eq!(
        grad.dtype(),
        DType::F64,
        "an f64 parameter has an f64 gradient"
    );
    let analytic = grad.as_host::<f64>().expect("f64").to_vec();
    let _no_grad = no_grad();
    let nudge = |i: usize, delta: f64| {
        let mut p = param.borrow_mut();
        *crate::index::slot(p.data.as_host_mut::<f64>().expect("f64 param"), i) += delta;
    };
    let mut max_err: f64 = 0.0;
    for (i, &a) in analytic.iter().enumerate() {
        nudge(i, EPS);
        let plus = crate::index::fetch(&values(&loss()), 0);
        nudge(i, -2.0 * EPS);
        let minus = crate::index::fetch(&values(&loss()), 0);
        nudge(i, EPS);
        let numeric = (plus - minus) / (2.0 * EPS);
        let err = (a - numeric).abs() / a.abs().max(numeric.abs()).max(1.0);
        max_err = max_err.max(err);
    }
    max_err
}

/// Converts `layer` to f64, then gradchecks the input and every parameter of `sum(f(x) * w)`.
fn check_layer(name: &str, layer: &mut dyn Layer, x_shape: &[usize]) {
    layer.to_dtype(DType::F64);
    for p in layer.parameters() {
        assert_eq!(
            p.dtype(),
            DType::F64,
            "{name}: to_dtype converted every parameter"
        );
    }
    let x = leaf(x_shape, 101);
    let out = layer.forward(&x);
    assert_eq!(out.dtype(), DType::F64, "{name}: output dtype");
    let layer: &dyn Layer = layer;
    assert_grad(&format!("{name} input"), &x, |t| {
        weighted(&layer.forward(t), 102)
    });
    for (k, p) in layer.parameters().iter().enumerate() {
        let x2 = x.clone();
        let err = param_gradcheck(p, || weighted(&layer.forward(&x2), 102));
        assert!(
            err < TOL,
            "{name} parameter {k}: f64 gradcheck max relative error {err:.3e}"
        );
    }
}

/// Every `Layer` with a CPU kernel, in f64, input and parameters.
#[test]
fn layers_gradcheck_in_f64() {
    manual_seed(1);
    check_layer("Linear", &mut Linear::new(4, 3, true), &[2, 4]);
    for algo in [ConvAlgo::Direct, ConvAlgo::Im2col, ConvAlgo::IGEMM] {
        let conv = Conv2d::new(2, 3, 3, 2, 1, true);
        conv.set_algo(algo);
        let mut conv = conv;
        check_layer(&format!("Conv2d {algo:?}"), &mut conv, &[2, 2, 5, 5]);
    }
    check_layer(
        "ConvTranspose2d",
        &mut ConvTranspose2d::new(2, 3, 3, 2, 1, true),
        &[2, 2, 3, 3],
    );
    check_layer("MaxPool2d", &mut MaxPool2d::new(2, 2, 0), &[2, 2, 4, 4]);
    check_layer("BatchNorm1d", &mut BatchNorm1d::new(3), &[4, 3]);
    check_layer("BatchNorm2d", &mut BatchNorm2d::new(2), &[3, 2, 2, 2]);
    check_layer("LayerNorm", &mut LayerNorm::new(vec![4]), &[3, 4]);
    check_layer("GELU", &mut GELU::new(), &[2, 5]);
    check_layer("PixelShuffle", &mut PixelShuffle::new(2), &[1, 4, 2, 2]);
    check_layer("FeedForward", &mut FeedForward::new(4, 2, 0.0), &[2, 3, 4]);
    check_layer(
        "MultiHeadAttention",
        &mut MultiHeadAttention::new(4, 2, 0.0),
        &[2, 3, 4],
    );
    check_layer(
        "TransformerBlock",
        &mut TransformerBlock::new(4, 2, 2, 0.0),
        &[2, 3, 4],
    );
    check_layer(
        "PositionalEncoding learned",
        &mut PositionalEncoding::new(4, 5, PositionalEncodingType::Learned),
        &[2, 3, 4],
    );
    check_layer(
        "PositionalEncoding sinusoidal",
        &mut PositionalEncoding::new(4, 5, PositionalEncodingType::Sinusoidal),
        &[2, 3, 4],
    );
}

/// `Embedding` (I64 indices): the weight's gradient in f64, with a repeated index.
#[test]
fn embedding_gradcheck_in_f64() {
    manual_seed(2);
    let mut emb = Embedding::new(5, 3);
    emb.to_dtype(DType::F64);
    let w = emb.parameters()[0].clone();
    let err = param_gradcheck(&w, || {
        weighted(
            &emb.forward(&RawTensor::from_typed(vec![1i64, 3, 1], &[3])),
            103,
        )
    });
    assert!(
        err < TOL,
        "Embedding weight: f64 gradcheck max relative error {err:.3e}"
    );
}

/// `LSTMCell` over two steps from zero state (one step leaves `weight_hh` without a
/// gradient), input and every parameter.
#[test]
fn lstm_cell_gradcheck_in_f64() {
    manual_seed(3);
    let mut cell = LSTMCell::new(3, 4, true);
    cell.to_dtype(DType::F64);
    let two_steps = |x: &Tensor| {
        let mut state = cell.initial_state(x.shape().dim_at(0));
        let _ = cell.step(x, &mut state);
        cell.step(x, &mut state)
    };
    let x = leaf(&[2, 3], 104);
    assert_grad("LSTMCell input", &x, |t| weighted(&two_steps(t), 105));
    for (k, p) in cell.parameters().iter().enumerate() {
        let x2 = x.clone();
        let err = param_gradcheck(p, || weighted(&two_steps(&x2), 105));
        assert!(err < TOL, "LSTMCell parameter {k}: {err:.3e}");
    }
}

/// `to_dtype` reaches buffers: `BatchNorm`'s running statistics, and through `Sequential`.
#[test]
fn to_dtype_converts_buffers_and_children() {
    let mut seq = crate::nn::Sequential::new(vec![
        Box::new(Linear::new(3, 2, true)),
        Box::new(BatchNorm1d::new(2)),
    ]);
    seq.to_dtype(DType::F64);
    let x = leaf(&[4, 3], 106);
    // Training mode updates the running stats, which must be f64 by now.
    let out = seq.forward(&x);
    assert_eq!(out.dtype(), DType::F64);
    seq.eval();
    assert_eq!(
        seq.forward(&x).dtype(),
        DType::F64,
        "eval reads f64 running stats"
    );
}

/// One Adam step in f64 matches the update written out in f64, to rounding.
#[test]
fn adam_step_in_f64_matches_the_formula() {
    let w = leaf(&[3], 107);
    let g = vec![0.5f64, -1e-9, 2.0];
    w.set_grad(Storage::from_typed(g.clone()));
    let before = values(&w);
    let mut opt = crate::nn::Adam::new(vec![w.clone()], 1e-3, (0.9, 0.999), 1e-8, 0.0);
    opt.step();
    let (lr, b1, b2, eps) = (
        f64::from(1e-3f32),
        f64::from(0.9f32),
        f64::from(0.999f32),
        f64::from(1e-8f32),
    );
    for ((&w0, &gi), &w1) in before.iter().zip(&g).zip(&values(&w)) {
        let m_hat = (1.0 - b1) * gi / (1.0 - b1);
        let v_hat = (1.0 - b2) * gi * gi / (1.0 - b2);
        let want = w0 - lr * m_hat / (v_hat.sqrt() + eps);
        assert!(
            (w1 - want).abs() <= 1e-15 * w0.abs().max(1.0),
            "{w1} vs {want}"
        );
    }
}

/// SGD with momentum and Muon step f64 parameters in f64 (a 1e-12 update survives).
#[test]
fn sgd_and_muon_keep_f64_precision() {
    let w = RawTensor::from_typed(vec![1.0f64, 2.0], &[2]);
    w.set_requires_grad(true);
    w.set_grad(Storage::from_typed(vec![1e-12f64, 0.0]));
    let mut sgd = crate::nn::SGD::new(vec![w.clone()], 1.0, 0.9, 0.0);
    sgd.step();
    assert_eq!(values(&w), vec![1.0 - 1e-12, 2.0]);

    let m = leaf(&[3, 2], 108);
    m.set_grad(Storage::from_typed(vec![0.1f64, -0.2, 0.3, 0.4, -0.5, 0.6]));
    let before = values(&m);
    let mut muon = crate::nn::Muon::new(vec![m.clone()], 0.02, 0.95, true, 5);
    muon.step();
    let after = values(&m);
    assert_eq!(m.dtype(), DType::F64);
    assert!(
        before.iter().zip(&after).any(|(b, a)| b != a),
        "Muon updated the f64 matrix"
    );
}

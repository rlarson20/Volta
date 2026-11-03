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

/// The f64 values of `t`.
fn values(t: &Tensor) -> Vec<f64> {
    t.borrow()
        .data
        .as_host::<f64>()
        .expect("an f64 tensor")
        .to_vec()
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
                BinaryOp::Mod | BinaryOp::Cmplt => unreachable!("filtered: not differentiable"),
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

    let cond = RawTensor::from_typed(vec![1.0f64, 0.0, 1.0], &[3]);
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

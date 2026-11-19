//! Numerical-stability cases against PyTorch (`tests/fixtures/torch/stability.safetensors`,
//! generator group `stability`): `log_softmax` at large logits (#315), and f16
//! `BatchNorm2d` and `LayerNorm` whose sums of squares pass f16's 65504 (#316), plus the
//! f16 `mean_dim` overflow itself.

#[path = "torch_ref/mod.rs"]
mod torch_ref;

use half::f16;
use torch_ref::Fixture;
use volta::io::{StateDict, TensorData};
use volta::{Axis, BatchNorm2d, DType, Layer, LayerNorm, Module, RawTensor, Tensor};

/// f16 machine epsilon.
const F16_EPS: f64 = 1.0 / 1024.0;

/// Asserts `got` matches `want` within `bound` units of `eps`, relative to
/// `max(|want|, rms(want))` (the floor keeps an entry that cancels to near zero from
/// demanding more relative precision than its tensor carries).
fn assert_eps(label: &str, got: &[f32], want: &[f32], eps: f64, bound: f64) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    #[expect(clippy::cast_precision_loss, reason = "small test tensors")]
    let rms = (want.iter().map(|&w| f64::from(w).powi(2)).sum::<f64>() / want.len() as f64).sqrt();
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let (g, w) = (f64::from(g), f64::from(w));
        assert!(g.is_finite(), "{label}[{i}]: volta {g}, torch {w}");
        let k = (g - w).abs() / (eps * w.abs().max(rms).max(f64::MIN_POSITIVE));
        assert!(
            k <= bound,
            "{label}[{i}]: volta {g:e}, torch {w:e} ({k:.2} eps > {bound})"
        );
    }
}

/// An f16 leaf holding the fixture tensor `name` (stored as f16, so exact).
fn f16_leaf(f: &Fixture, name: &str) -> Tensor {
    let (v, shape) = f.get(name);
    let t = RawTensor::from_typed(v.into_iter().map(f16::from_f32).collect(), &shape);
    t.set_requires_grad(true);
    t
}

/// `gamma` and `beta` from the fixture's `<name>.gamma` and `<name>.beta`.
fn affine(f: &Fixture, name: &str) -> StateDict {
    ["gamma", "beta"]
        .into_iter()
        .map(|k| {
            let (v, shape) = f.get(&format!("{name}.{k}"));
            (k.to_string(), TensorData::from_f32(v, shape))
        })
        .collect()
}

/// Bound, in f16 epsilons, against the float64 reference. Every output and gradient is
/// computed in f32 and rounded once, so it is within half an epsilon (measured: 0.5).
const F64_BOUND: f64 = 1.0;

/// Backprop `sum(out * <name>.w)`, then compare `out` with torch's f16 result within
/// `fwd` f16 epsilons, and `out` and each gradient with the float64 reference
/// (`<name>.f64.*`) within [`F64_BOUND`]. The gradients are not compared with torch's
/// f16 ones: torch rounds the normalized input to f16 before the gamma gradient, which
/// puts its `layernorm` gamma gradient 1% (5 eps) from float64, where Volta's is within
/// one rounding.
fn check_f16(f: &Fixture, name: &str, out: &Tensor, grads: &[(&str, &Tensor)], fwd: f64) {
    assert_eq!(out.dtype(), DType::F16, "{name}: output dtype");
    let got = out.data();
    let torch = format!("{name}.out");
    assert_eps(&torch, &got, &f.values(&torch), F16_EPS, fwd);
    let reference = format!("{name}.f64.out");
    assert_eps(&reference, &got, &f.values(&reference), F16_EPS, F64_BOUND);
    let (w, shape) = f.get(&format!("{name}.w"));
    let w = RawTensor::new(w, &shape, false).to_dtype(DType::F16);
    out.elem_mul(&w)
        .sum()
        .into_loss()
        .expect("scalar")
        .backward();
    for (g, t) in grads {
        let got = t
            .grad()
            .unwrap_or_else(|| panic!("{name}: no gradient for {g}"));
        let want = format!("{name}.f64.grad_{g}");
        assert_eps(&want, &got, &f.values(&want), F16_EPS, F64_BOUND);
    }
}

/// #315: `log_softmax` keeps f32 accuracy where the largest logit's ulp is 6e-5 to 1e-3.
/// Adding the max back before subtracting was off by up to half that ulp (-0.69317627
/// for `[1000, 1000]`).
#[test]
fn log_softmax_large_logits_match_torch() {
    let f = Fixture::load("stability");
    for i in 0..3 {
        let name = format!("log_softmax{i}");
        let x = f.tensor(&format!("{name}.x"), true);
        let out = x.log_softmax(Axis(1));
        // Two f32 epsilons of the result: the error no longer scales with the logits.
        assert_eps(
            &format!("{name}.out"),
            &out.data(),
            &f.values(&format!("{name}.out")),
            f64::from(f32::EPSILON),
            2.0,
        );
        let (w, shape) = f.get(&format!("{name}.w"));
        out.elem_mul(&RawTensor::new(w, &shape, false))
            .sum()
            .into_loss()
            .expect("scalar")
            .backward();
        assert_eps(
            &format!("{name}.grad_x"),
            &x.grad().expect("gradient"),
            &f.values(&format!("{name}.grad_x")),
            f64::from(f32::EPSILON),
            2.0,
        );
    }
}

/// #316: an f16 `mean_dim` divides before it rounds, as `mean` does, so 1024 x 100.0
/// is 100.0, not inf (the sum, 102400, is past f16's 65504).
#[test]
fn f16_mean_dim_does_not_overflow_where_the_sum_would() {
    let x = RawTensor::from_typed(vec![f16::from_f32(100.0); 1024], &[1, 1024]);
    x.set_requires_grad(true);
    let m = x.mean_dim(Axis(1), false);
    assert_eq!(m.dtype(), DType::F16);
    assert_eq!(m.data(), vec![100.0]);
    m.sum().into_loss().expect("scalar").backward();
    let g = x.grad().expect("gradient");
    assert!(g.iter().all(|&v| v == 1.0 / 1024.0), "{:?}", &g[..4]);
    // The sum itself still overflows f16, as it does in PyTorch.
    assert_eq!(x.sum_dim(Axis(1), false).data(), vec![f32::INFINITY]);
}

/// #316: f16 `BatchNorm2d` in training mode on 64x1x32x32 with variance about 3, where
/// count x variance passes 65504. It returned NaN; now the statistics are taken in f32.
#[test]
fn f16_batchnorm2d_large_batch_matches_torch() {
    let f = Fixture::load("stability");
    let mut bn = BatchNorm2d::new(1);
    bn.to_dtype(DType::F16);
    bn.load_state_dict(&affine(&f, "batchnorm")).expect("load");
    let x = f16_leaf(&f, "batchnorm.x");
    let out = bn.forward(&x);
    let params = bn.parameters();
    let [gamma, beta] = params.as_slice() else {
        panic!("two parameters")
    };
    assert_eq!(gamma.dtype(), DType::F16);
    check_f16(
        &f,
        "batchnorm",
        &out,
        &[("x", &x), ("gamma", gamma), ("beta", beta)],
        1.5,
    );
}

/// #316: f16 `LayerNorm(1024)` on rows `100 + 0.01 i`, whose sum is about 105000. It
/// returned NaN; now the statistics are taken in f32.
#[test]
fn f16_layernorm_large_row_matches_torch() {
    let f = Fixture::load("stability");
    let mut ln = LayerNorm::new(vec![1024]);
    ln.to_dtype(DType::F16);
    ln.load_state_dict(&affine(&f, "layernorm")).expect("load");
    let x = f16_leaf(&f, "layernorm.x");
    let out = ln.forward(&x);
    let params = ln.parameters();
    let [gamma, beta] = params.as_slice() else {
        panic!("two parameters")
    };
    check_f16(
        &f,
        "layernorm",
        &out,
        &[("x", &x), ("gamma", gamma), ("beta", beta)],
        1.5,
    );
}

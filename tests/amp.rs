//! Automatic mixed precision (ADDENDUM §11 P7): `autocast` follows PyTorch's CPU op lists
//! (low precision for matmul, `Linear` and the convolutions, f32 for losses, the input's
//! dtype for everything else), keeps f32 parameters with f32 gradients, nests, and leaves
//! f64 alone; `GradScaler` rescues f16 gradients that would underflow, skips a step with
//! an inf gradient, and adapts its scale with PyTorch's rules. Values against torch are in
//! `tests/torch_ref_autocast.rs`.

use volta::amp::{GradScaler, autocast, autocast_dtype};
use volta::{
    Axis, Conv2d, ConvTranspose2d, DType, Layer, Linear, MaxPool2d, Module, Optimizer, RawTensor,
    ReLU, SGD, Storage, Tensor,
};

/// A seeded f32 tensor of `shape`.
fn randn(shape: &[usize]) -> Tensor {
    RawTensor::randn(shape)
}

/// Output dtypes under `autocast(low)`, for each family in torch 2.15's CPU behavior
/// (measured with `torch.autocast("cpu", dtype=..)`).
fn op_dtypes(low: DType) {
    volta::manual_seed(3);
    let x = randn(&[4, 8]);
    let lin = Linear::new(8, 3, true);
    let conv = Conv2d::new(1, 2, 3, 1, 0, true);
    let convt = ConvTranspose2d::new(1, 2, 3, 1, 0, true);
    let img = randn(&[2, 1, 5, 5]);
    let _ac = autocast(low);
    assert_eq!(autocast_dtype(), Some(low));
    let h = x.to_dtype(low);
    let targets = RawTensor::from_typed(vec![0i64, 1, 2, 0], &[4]);
    let cases: Vec<(&str, Tensor, DType)> = vec![
        ("matmul f32 @ f32", x.matmul(&randn(&[8, 3])), low),
        ("Linear f32", lin.forward(&x), low),
        ("Conv2d f32", conv.forward(&img), low),
        ("ConvTranspose2d f32", convt.forward(&img), low),
        ("softmax(half)", h.softmax(Axis(1)), low),
        ("log_softmax(half)", h.log_softmax(Axis(1)), low),
        ("sum(half)", h.sum(), low),
        ("mean(half)", h.mean(), low),
        ("relu(half)", h.relu(), low),
        ("exp(half)", h.exp(), low),
        ("half + f32", h.add(&x), DType::F32),
        (
            "MaxPool2d(half)",
            MaxPool2d::new(2, 2, 0).forward(&img.to_dtype(low)),
            low,
        ),
        (
            "cross_entropy(half, i64)",
            RawTensor::cross_entropy_loss(&h.shrink(&[(0, 4), (0, 3)]), &targets),
            DType::F32,
        ),
        (
            "mse_loss(half, half)",
            RawTensor::mse_loss(&h, &h),
            DType::F32,
        ),
        (
            "nll_loss(half, i64)",
            RawTensor::nll_loss(&h.shrink(&[(0, 4), (0, 3)]), &targets),
            DType::F32,
        ),
    ];
    for (name, t, want) in cases {
        assert_eq!(t.dtype(), want, "{low}: {name}");
    }
}

/// The op lists under bf16.
#[test]
fn bf16_autocast_op_dtypes_match_torch_cpu() {
    op_dtypes(DType::BF16);
}

/// The op lists under f16.
#[test]
fn f16_autocast_op_dtypes_match_torch_cpu() {
    op_dtypes(DType::F16);
}

/// Parameters stay f32 under autocast and get f32 gradients through the casts; no
/// master weights are needed.
#[test]
fn f32_parameters_get_f32_gradients_under_autocast() {
    volta::manual_seed(4);
    let model = volta::Sequential::new(vec![
        Box::new(Linear::new(8, 16, true)),
        Box::new(ReLU),
        Box::new(Linear::new(16, 3, true)),
    ]);
    let x = randn(&[5, 8]);
    let y = RawTensor::from_typed(vec![0i64, 1, 2, 1, 0], &[5]);
    let loss = {
        let _ac = autocast(DType::BF16);
        RawTensor::cross_entropy_loss(&model.forward(&x), &y)
    };
    assert_eq!(loss.dtype(), DType::F32);
    loss.into_loss().expect("scalar").backward();
    for p in model.parameters() {
        assert_eq!(p.dtype(), DType::F32, "parameter dtype");
        let g = p.grad_storage().expect("a gradient");
        assert_eq!(g.dtype(), DType::F32, "gradient dtype");
        assert!(
            g.as_host::<f32>().expect("f32").iter().any(|&v| v != 0.0),
            "a non-zero gradient"
        );
    }
}

/// Scopes nest and restore; outside every scope nothing is cast; f64 is never cast.
#[test]
fn autocast_scopes_nest_and_leave_f64_alone() {
    let x = randn(&[2, 2]);
    assert_eq!(autocast_dtype(), None);
    {
        let _outer = autocast(DType::BF16);
        {
            let _inner = autocast(DType::F16);
            assert_eq!(x.matmul(&x).dtype(), DType::F16);
        }
        assert_eq!(x.matmul(&x).dtype(), DType::BF16);
        let d = x.to_dtype(DType::F64);
        assert_eq!(d.matmul(&d).dtype(), DType::F64, "f64 is not eligible");
    }
    assert_eq!(autocast_dtype(), None);
    assert_eq!(x.matmul(&x).dtype(), DType::F32);
}

/// Autocast's dtype must be a half type.
#[test]
#[should_panic(expected = "autocast: the low-precision dtype must be F16 or BF16, got f32")]
fn autocast_rejects_f32() {
    let _ac = autocast(DType::F32);
}

/// A model, its loss under f16 autocast scaled by `tiny`, and backward through `scaler`
/// (or unscaled if `None`). Returns the model's parameters.
fn tiny_loss_grads(scaler: Option<&GradScaler>, tiny: f32) -> Vec<Tensor> {
    volta::manual_seed(9);
    let model = Linear::new(4, 2, true);
    let x = randn(&[8, 4]);
    let y = randn(&[8, 2]);
    let loss = {
        let _ac = autocast(DType::F16);
        RawTensor::mse_loss(&model.forward(&x), &y)
    };
    let loss = loss.elem_mul(&RawTensor::new(vec![tiny], &[1], false));
    let loss = scaler.map_or(loss.clone(), |s| s.scale(&loss));
    loss.into_loss().expect("scalar").backward();
    model.parameters()
}

/// Why the scaler exists: with the loss scaled by 2e-8, the output gradient (order 1e-8)
/// rounds to zero in f16 (smallest subnormal 6e-8), so f16 autocast loses every weight
/// gradient; scaled by 2^16 it lands in f16's normal range, and unscaling gives it back in
/// f32 within f16's precision of the f32 run.
#[test]
fn grad_scaler_rescues_f16_gradients_from_underflow() {
    let tiny = 2e-8;
    let lost = tiny_loss_grads(None, tiny);
    let w = lost[0].grad_storage().expect("grad");
    assert!(
        w.as_host::<f32>().expect("f32").iter().all(|&v| v == 0.0),
        "without scaling the f16 weight gradient underflows to zero"
    );

    let mut scaler = GradScaler::new();
    let params = tiny_loss_grads(Some(&scaler), tiny);
    let opt = SGD::new(params.clone(), 0.1, 0.0, 0.0);
    scaler.unscale(&opt);

    // Reference: the same loss in f32, no autocast.
    volta::manual_seed(9);
    let model = Linear::new(4, 2, true);
    let (x, y) = (randn(&[8, 4]), randn(&[8, 2]));
    RawTensor::mse_loss(&model.forward(&x), &y)
        .elem_mul(&RawTensor::new(vec![tiny], &[1], false))
        .into_loss()
        .expect("scalar")
        .backward();
    for (p, r) in params.iter().zip(model.parameters()) {
        let got = p.grad().expect("grad");
        let want = r.grad().expect("grad");
        assert_eq!(got.len(), want.len());
        for (&g, &w) in got.iter().zip(&want) {
            assert!(g != 0.0, "the scaled gradient survived");
            assert!(
                (g - w).abs() <= 4e-3 * w.abs().max(1e-12),
                "unscaled {g:e} vs f32 {w:e}"
            );
        }
    }
}

/// An inf gradient skips the step (parameters unchanged), and `update` halves the scale.
#[test]
fn grad_scaler_skips_a_step_with_an_inf_gradient() {
    let p = RawTensor::new(vec![1.0, 2.0], &[2], true);
    let mut opt = SGD::new(vec![p.clone()], 0.1, 0.0, 0.0);
    let mut scaler = GradScaler::new();
    p.set_grad(Storage::cpu(vec![f32::INFINITY, 1.0]));
    assert!(!scaler.step(&mut opt), "the step is skipped");
    assert_eq!(p.data(), vec![1.0, 2.0], "parameters unchanged");
    scaler.update();
    assert_eq!(scaler.get_scale(), 32768.0);

    // A NaN skips too.
    p.set_grad(Storage::cpu(vec![f32::NAN, 1.0]));
    assert!(!scaler.step(&mut opt));
    scaler.update();
    assert_eq!(scaler.get_scale(), 16384.0);

    // A finite gradient steps, unscaled: 0.1 * (16384 / 16384).
    p.set_grad(Storage::cpu(vec![16384.0, 0.0]));
    assert!(scaler.step(&mut opt));
    assert_eq!(p.data(), vec![0.9, 2.0]);
    scaler.update();
}

/// `growth_interval` clean steps double the scale; a skip resets the count.
#[test]
fn grad_scaler_grows_after_the_interval_and_a_skip_resets_it() {
    let p = RawTensor::new(vec![0.0], &[1], true);
    let mut opt = SGD::new(vec![p.clone()], 0.0, 0.0, 0.0);
    let mut scaler = GradScaler::with_params(8.0, 2.0, 0.5, 3);
    let mut run = |g: f32| {
        p.set_grad(Storage::cpu(vec![g]));
        let _ = scaler.step(&mut opt);
        scaler.update();
        scaler.get_scale()
    };
    assert_eq!([run(1.0), run(1.0), run(f32::INFINITY)], [8.0, 8.0, 4.0]);
    assert_eq!([run(1.0), run(1.0), run(1.0)], [4.0, 4.0, 8.0]);
}

/// Unscaling twice would divide twice: refused.
#[test]
#[should_panic(expected = "GradScaler::unscale: already unscaled since the last update()")]
fn grad_scaler_refuses_a_second_unscale() {
    let p = RawTensor::new(vec![0.0], &[1], true);
    let opt = SGD::new(vec![p], 0.1, 0.0, 0.0);
    let mut scaler = GradScaler::new();
    scaler.unscale(&opt);
    scaler.unscale(&opt);
}

/// `update` needs a `step` first.
#[test]
#[should_panic(expected = "GradScaler::update: call step() first")]
fn grad_scaler_update_needs_a_step() {
    GradScaler::new().update();
}

/// The `Optimizer` trait drives any optimizer: the parameters are the ones given.
#[test]
fn optimizer_trait_exposes_parameters() {
    let p = RawTensor::new(vec![1.0], &[1], true);
    let opt = volta::Adam::new(vec![p.clone()], 0.1, (0.9, 0.999), 1e-8, 0.0);
    let params = Optimizer::parameters(&opt);
    assert_eq!(params.len(), 1);
    assert_eq!(params[0].data(), p.data());
}

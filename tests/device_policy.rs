//! `DevicePolicy` on a real GPU: device labels, the forward and backward gap checks, and
//! the fallback event log (ADDENDUM §2.1).
//!
//! Each test skips cleanly when no GPU adapter is available.

#![cfg(feature = "gpu")]

#[path = "common/mod.rs"]
mod common;

use common::{TOL_MATMUL_FWD, assert_close, gpu_device, skip_if_no_gpu};
use volta::Axis;
use volta::nn::{MultiHeadAttention, PositionalEncoding, PositionalEncodingType};
use volta::{
    BatchNorm1d, BatchNorm2d, Device, DevicePolicy, Dropout, Embedding, FallbackEvent, GELU,
    IndexedLayer, LSTMCell, Layer, LayerNorm, Linear, Module, Pass, RawTensor, RecurrentLayer,
    Storage, Tensor, bce_loss, bce_with_logits_loss, cross_entropy_loss, kl_divergence_gaussian,
    manual_seed, mse_loss, nll_loss, take_fallback_events, with_device_policy,
};

/// `to_device` moves the gradient with the data.
#[test]
fn to_device_moves_grad() {
    if skip_if_no_gpu() {
        return;
    }
    let t = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4], true);
    t.set_grad(Storage::cpu(vec![1.0; 4]));
    let g = t.to_device(gpu_device());
    assert!(g.grad_storage().expect("grad was set").is_gpu());
    assert_eq!(g.grad().expect("grad was set"), vec![1.0; 4]);
}

/// A GPU name that isn't the adapter's is rejected, like `cuda:7` on a one-GPU machine.
#[test]
fn unknown_gpu_name_panics() {
    if skip_if_no_gpu() {
        return;
    }
    let msg = panic_message(|| {
        let _ = RawTensor::new(vec![1.0], &[1], false).to_device(Device::GPU("ParityTest".into()));
    });
    assert!(msg.contains("no such GPU"), "{msg}");
}

/// A seeded random CPU tensor and its GPU copy.
fn pair(shape: &[usize]) -> (Tensor, Tensor) {
    manual_seed(7);
    let cpu = RawTensor::randn(shape);
    let gpu = cpu.to_device(gpu_device());
    (cpu, gpu)
}

/// Shorthand for one expected event.
fn event(op: &str, pass: Pass) -> FallbackEvent {
    FallbackEvent {
        op: op.to_string(),
        pass,
    }
}

/// Under `Strict`, a GPU binary op whose operands broadcast (no GPU kernel) panics.
#[test]
fn gap_strict_panics() {
    if skip_if_no_gpu() {
        return;
    }
    let (_, a) = pair(&[2, 3]);
    let (_, b) = pair(&[3]);
    let _strict = with_device_policy(DevicePolicy::Strict);
    let msg = panic_message(|| {
        let _ = a.add(&b);
    });
    assert!(
        msg.contains("Binary(Add): no GPU kernel for this case"),
        "{msg}"
    );
}

/// Under `PreferGpu`, the same gap runs on the CPU, lands on the GPU, and is recorded.
#[test]
fn gap_prefer_gpu_runs_and_records() {
    if skip_if_no_gpu() {
        return;
    }
    let (a_cpu, a) = pair(&[2, 3]);
    let (b_cpu, b) = pair(&[3]);
    let _prefer = with_device_policy(DevicePolicy::PreferGpu);
    let _ = take_fallback_events();
    let out = a.add(&b);
    assert!(out.device().is_gpu());
    assert!(out.grad_storage().is_none());
    assert_close("add", &a_cpu.add(&b_cpu).data(), &out.data(), 1e-6);
    assert_eq!(
        take_fallback_events(),
        vec![event("Binary(Add)", Pass::Forward)]
    );
}

/// Batched matmul has no GPU kernel: under `PreferGpu` it runs on the CPU and lands on
/// the GPU.
#[test]
fn batched_matmul_gap_prefer_gpu() {
    if skip_if_no_gpu() {
        return;
    }
    let (a_cpu, a) = pair(&[2, 3, 4]);
    let (b_cpu, b) = pair(&[2, 4, 5]);
    let _prefer = with_device_policy(DevicePolicy::PreferGpu);
    let _ = take_fallback_events();
    let out = a.matmul(&b);
    assert!(out.device().is_gpu());
    assert_close(
        "batched matmul",
        &a_cpu.matmul(&b_cpu).data(),
        &out.data(),
        TOL_MATMUL_FWD,
    );
    assert_eq!(take_fallback_events(), vec![event("MatMul", Pass::Forward)]);
}

/// Mixed devices panic even under `PreferGpu`.
#[test]
fn mixed_devices_panic_under_prefer_gpu() {
    if skip_if_no_gpu() {
        return;
    }
    let (a_cpu, a) = pair(&[3]);
    let _prefer = with_device_policy(DevicePolicy::PreferGpu);
    let msg = panic_message(|| {
        let _ = a_cpu.add(&a);
    });
    assert!(msg.contains("expected all inputs on one device"), "{msg}");
}

/// The forward check runs under `no_grad()` too, when no graph edge is attached.
#[test]
fn gap_under_no_grad_still_placed() {
    if skip_if_no_gpu() {
        return;
    }
    let (_, a) = pair(&[2, 3]);
    let (_, b) = pair(&[3]);
    let _prefer = with_device_policy(DevicePolicy::PreferGpu);
    let _ = take_fallback_events();
    let out = {
        let _ng = volta::autograd::no_grad();
        a.add(&b)
    };
    assert!(out.device().is_gpu());
    assert_eq!(
        take_fallback_events(),
        vec![event("Binary(Add)", Pass::Forward)]
    );
}

/// Put `layer`'s parameters on the GPU.
fn on_gpu<L: Module>(mut layer: L) -> L {
    layer.to_device(gpu_device());
    layer
}

/// GPU layers build their internal constants on the GPU: no mixed-device panic, and
/// every output stays on the GPU.
#[test]
fn gpu_layers_keep_constants_on_device() {
    if skip_if_no_gpu() {
        return;
    }
    let _prefer = with_device_policy(DevicePolicy::PreferGpu);
    let (_, x2) = pair(&[3, 4]);
    let (_, x3) = pair(&[2, 4, 8]);
    let (_, x4) = pair(&[2, 2, 3, 3]);

    let outputs: Vec<(&str, Tensor)> = vec![
        ("Linear", on_gpu(Linear::new(4, 5, true)).forward(&x2)),
        ("GELU", GELU.forward(&x2)),
        ("LayerNorm", on_gpu(LayerNorm::new(vec![4])).forward(&x2)),
        ("BatchNorm1d", on_gpu(BatchNorm1d::new(4)).forward(&x2)),
        ("BatchNorm2d", on_gpu(BatchNorm2d::new(2)).forward(&x4)),
        ("Dropout", Dropout::new(0.5).forward(&x2)),
        (
            "Embedding",
            on_gpu(Embedding::new(10, 4)).forward_indices(&[1, 3, 3]),
        ),
        (
            "MultiHeadAttention",
            on_gpu(MultiHeadAttention::new(8, 2, 0.0)).forward(&x3, true, None),
        ),
    ];
    for (name, out) in outputs {
        assert!(out.device().is_gpu(), "{name} output left the GPU");
    }

    let mut bn_eval = on_gpu(BatchNorm1d::new(4));
    bn_eval.eval();
    assert!(bn_eval.forward(&x2).device().is_gpu(), "BatchNorm1d (eval)");
}

/// Losses and `mean_dim` build their constants on the input's device.
#[test]
fn gpu_losses_stay_on_device() {
    if skip_if_no_gpu() {
        return;
    }
    let _prefer = with_device_policy(DevicePolicy::PreferGpu);
    let (_, x) = pair(&[3, 4]);
    let (_, y) = pair(&[3, 4]);
    let probs = x.sigmoid();
    let outputs: Vec<(&str, Tensor)> = vec![
        ("mean_dim", x.mean_dim(Axis(1), false)),
        ("mse_loss", mse_loss(&x, &y)),
        ("cross_entropy_loss", cross_entropy_loss(&x, &probs)),
        ("nll_loss", nll_loss(&x.log_softmax(Axis(1)), &probs)),
        ("bce_loss", bce_loss(&probs, &y.sigmoid())),
        ("bce_with_logits_loss", bce_with_logits_loss(&x, &probs)),
        ("kl_divergence_gaussian", kl_divergence_gaussian(&x, &y)),
    ];
    for (name, out) in outputs {
        assert!(out.device().is_gpu(), "{name} output left the GPU");
    }
}

/// `PositionalEncoding` keeps a GPU input on the GPU.
#[test]
#[ignore = "B3: PositionalEncoding::forward rebuilds its output from raw host data on the CPU, and the layer has no to_device"]
fn gpu_positional_encoding_stays_on_device() {
    if skip_if_no_gpu() {
        return;
    }
    let _prefer = with_device_policy(DevicePolicy::PreferGpu);
    let (_, x3) = pair(&[2, 4, 8]);
    for kind in [
        PositionalEncodingType::Sinusoidal,
        PositionalEncodingType::Learned,
    ] {
        let pe = PositionalEncoding::new(8, 16, kind);
        assert!(pe.forward(&x3).device().is_gpu());
    }
}

/// `LSTMCell` keeps a GPU input and state on the GPU.
#[test]
#[ignore = "B2: LSTMCell::slice_gate copies gate values into a fresh CPU tensor"]
fn gpu_lstm_stays_on_device() {
    if skip_if_no_gpu() {
        return;
    }
    let _prefer = with_device_policy(DevicePolicy::PreferGpu);
    let (_, x) = pair(&[2, 4]);
    let cell = on_gpu(LSTMCell::new(4, 3, true));
    let mut state = cell.initial_state(2);
    let h = cell.step(&x, &mut state);
    assert!(h.device().is_gpu());
}

/// Run `f`, which must panic, and return its panic message.
///
/// GPU tests can't use `#[should_panic]`: without an adapter they return early, which
/// would fail a `should_panic` test.
fn panic_message(f: impl FnOnce()) -> String {
    let err =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).expect_err("expected a panic");
    err.downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(ToString::to_string))
        .unwrap_or_default()
}

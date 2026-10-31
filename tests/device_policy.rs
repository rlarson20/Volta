//! `DevicePolicy` on a real GPU: device labels, the forward and backward gap checks, and
//! the fallback event log (ADDENDUM §2.1).
//!
//! Each test skips cleanly when no GPU adapter is available.

#![cfg(feature = "gpu")]

#[path = "common/mod.rs"]
mod common;

use common::{TOL_MATMUL_BWD, TOL_MATMUL_FWD, assert_close, gpu_device, skip_if_no_gpu};
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

/// `LSTMCell` keeps a GPU input and state on the GPU, and two steps match the CPU in the
/// output and in every parameter gradient (B2: the gate slice used to copy to the CPU and
/// cut the graph).
#[test]
fn gpu_lstm_stays_on_device() {
    if skip_if_no_gpu() {
        return;
    }
    let _prefer = with_device_policy(DevicePolicy::PreferGpu);
    let (x_cpu, x) = pair(&[2, 4]);
    manual_seed(0x157);
    let cpu_cell = LSTMCell::new(4, 3, true);
    let w_cpu = RawTensor::randn(&[2, 3]);
    let w = w_cpu.to_device(gpu_device());
    let mut gpu_cell = LSTMCell::new(4, 3, true);
    gpu_cell.load_state_dict(&cpu_cell.state_dict());
    let gpu_cell = on_gpu(gpu_cell);

    let two_steps = |cell: &LSTMCell, x: &Tensor| {
        let mut state = cell.initial_state(2);
        let _ = cell.step(x, &mut state);
        cell.step(x, &mut state)
    };
    let h_cpu = two_steps(&cpu_cell, &x_cpu);
    let h = two_steps(&gpu_cell, &x);
    assert!(h.device().is_gpu());
    assert_close("h", &h_cpu.data(), &h.data(), TOL_MATMUL_FWD);

    h_cpu.elem_mul(&w_cpu).sum().into_loss().unwrap().backward();
    h.elem_mul(&w).sum().into_loss().unwrap().backward();
    for (i, (cpu, gpu)) in cpu_cell
        .parameters()
        .iter()
        .zip(&gpu_cell.parameters())
        .enumerate()
    {
        let g_cpu = cpu.grad().expect("CPU parameter has a gradient");
        assert!(
            g_cpu.iter().any(|v| *v != 0.0),
            "parameter {i}: zero CPU gradient"
        );
        let g = gpu.grad().expect("GPU parameter has a gradient");
        assert_close(&format!("parameter {i} grad"), &g_cpu, &g, TOL_MATMUL_BWD);
    }
}

/// Seeded random leaves requiring grad: a CPU tensor and its GPU copy.
fn leaf_pair(shape: &[usize]) -> (Tensor, Tensor) {
    let (cpu, _) = pair(shape);
    cpu.set_requires_grad(true);
    let gpu = RawTensor::new(cpu.data(), shape, true).to_device(gpu_device());
    (cpu, gpu)
}

/// A gradient computed on the CPU for a GPU parameter is a backward gap: under
/// `PreferGpu` it lands on the GPU and is recorded.
#[test]
fn backward_gap_prefer_gpu_grads_on_gpu() {
    if skip_if_no_gpu() {
        return;
    }
    let (x_cpu, x) = leaf_pair(&[2, 3, 4]);
    let (w_cpu, w) = leaf_pair(&[2, 4, 5]);
    x_cpu.matmul(&w_cpu).sum().into_loss().unwrap().backward();

    let _prefer = with_device_policy(DevicePolicy::PreferGpu);
    let _ = take_fallback_events();
    x.matmul(&w).sum().into_loss().unwrap().backward();
    for (name, cpu, gpu) in [("x", &x_cpu, &x), ("w", &w_cpu, &w)] {
        assert!(
            gpu.grad_storage().unwrap().is_gpu(),
            "{name} grad left the GPU"
        );
        assert_close(
            name,
            &cpu.grad().unwrap(),
            &gpu.grad().unwrap(),
            TOL_MATMUL_BWD,
        );
    }
    assert!(
        take_fallback_events().contains(&event("MatMul", Pass::Backward)),
        "missing backward MatMul event"
    );
}

/// Under `Strict`, a backward gap panics.
#[test]
fn backward_gap_strict_panics() {
    if skip_if_no_gpu() {
        return;
    }
    let (_, a) = leaf_pair(&[2, 3]);
    let (_, b) = leaf_pair(&[3]);
    // The broadcasting add's forward is a gap too, so build the graph under `PreferGpu`;
    // its backward computes both gradients on the CPU, which `Strict` then rejects.
    let loss = {
        let _prefer = with_device_policy(DevicePolicy::PreferGpu);
        a.add(&b).sum().into_loss().unwrap()
    };
    let _strict = with_device_policy(DevicePolicy::Strict);
    let msg = panic_message(|| loss.backward());
    assert!(
        msg.contains("Binary(Add): no GPU kernel for this case"),
        "{msg}"
    );
}

/// A broadcasting add on the GPU backpropagates without reaching the broken GPU
/// broadcast kernels, and the summed bias gradient is right.
#[test]
fn backward_broadcast_add_prefer_gpu() {
    if skip_if_no_gpu() {
        return;
    }
    let (_, a) = leaf_pair(&[2, 3]);
    let (_, b) = leaf_pair(&[3]);
    let _prefer = with_device_policy(DevicePolicy::PreferGpu);
    let _ = take_fallback_events();
    a.add(&b).sum().into_loss().unwrap().backward();
    assert!(a.grad_storage().unwrap().is_gpu());
    assert!(b.grad_storage().unwrap().is_gpu());
    assert_eq!(b.grad().unwrap(), vec![2.0, 2.0, 2.0]);
    assert!(take_fallback_events().contains(&event("Binary(Add)", Pass::Backward)));
}

/// A gap in the middle of a graph (kernel op, gap op, kernel op) backpropagates to the
/// same gradients as the all-CPU run, and they stay on the GPU.
#[test]
fn gap_mid_graph_backward_matches_cpu() {
    if skip_if_no_gpu() {
        return;
    }
    let (x_cpu, x) = leaf_pair(&[2, 3, 4]);
    let (w_cpu, w) = leaf_pair(&[2, 4, 5]);
    x_cpu
        .exp()
        .matmul(&w_cpu)
        .relu()
        .sum()
        .into_loss()
        .unwrap()
        .backward();

    let _prefer = with_device_policy(DevicePolicy::PreferGpu);
    x.exp()
        .matmul(&w)
        .relu()
        .sum()
        .into_loss()
        .unwrap()
        .backward();
    for (name, cpu, gpu) in [("x", &x_cpu, &x), ("w", &w_cpu, &w)] {
        assert!(
            gpu.grad_storage().unwrap().is_gpu(),
            "{name} grad left the GPU"
        );
        assert_close(
            name,
            &cpu.grad().unwrap(),
            &gpu.grad().unwrap(),
            TOL_MATMUL_BWD,
        );
    }
}

/// Forward events come before backward events, and draining empties the log.
#[test]
fn events_ordered_and_drained() {
    if skip_if_no_gpu() {
        return;
    }
    let (_, x) = leaf_pair(&[2, 3, 4]);
    let (_, w) = leaf_pair(&[2, 4, 5]);
    let _prefer = with_device_policy(DevicePolicy::PreferGpu);
    let _ = take_fallback_events();
    x.matmul(&w).sum().into_loss().unwrap().backward();
    let events = take_fallback_events();
    assert_eq!(events.first(), Some(&event("MatMul", Pass::Forward)));
    let first_backward = events
        .iter()
        .position(|e| e.pass == Pass::Backward)
        .expect("a backward event");
    assert!(
        events[first_backward..]
            .iter()
            .all(|e| e.pass == Pass::Backward)
    );
    assert!(events[first_backward..].contains(&event("MatMul", Pass::Backward)));
    assert!(take_fallback_events().is_empty());
}

/// `sum_dim` has no GPU kernel: it used to compute on the CPU and upload silently.
#[test]
fn sum_dim_is_a_gap() {
    if skip_if_no_gpu() {
        return;
    }
    let (_, x) = pair(&[2, 3]);
    let _strict = with_device_policy(DevicePolicy::Strict);
    let msg = panic_message(|| {
        let _ = x.sum_dim(Axis(1), false);
    });
    assert!(msg.contains("SumDim: no GPU kernel"), "{msg}");
}

/// `max_dim` has no GPU kernel either.
#[test]
fn max_dim_is_a_gap() {
    if skip_if_no_gpu() {
        return;
    }
    let (_, x) = pair(&[2, 3]);
    let _strict = with_device_policy(DevicePolicy::Strict);
    let msg = panic_message(|| {
        let _ = x.max_dim(Axis(1), false);
    });
    assert!(msg.contains("MaxDim: no GPU kernel"), "{msg}");
}

/// The GPU movement kernels hold at most four dims: rank 5 is a gap, not a silently
/// truncated kernel (H5).
#[test]
fn rank5_permute_is_a_gap() {
    if skip_if_no_gpu() {
        return;
    }
    let (_, x) = pair(&[2, 1, 3, 1, 2]);
    let axes = [Axis(4), Axis(3), Axis(2), Axis(1), Axis(0)];
    let _strict = with_device_policy(DevicePolicy::Strict);
    let msg = panic_message(|| {
        let _ = x.permute(&axes);
    });
    assert!(msg.contains("Movement(Permute): no GPU kernel"), "{msg}");
}

/// Under `PreferGpu`, a rank-5 permute on the GPU matches the CPU.
#[test]
fn rank5_permute_prefer_gpu_matches_cpu() {
    if skip_if_no_gpu() {
        return;
    }
    let (x_cpu, x) = pair(&[2, 1, 3, 1, 2]);
    let axes = [Axis(4), Axis(3), Axis(2), Axis(1), Axis(0)];
    let _prefer = with_device_policy(DevicePolicy::PreferGpu);
    let out = x.permute(&axes);
    assert!(out.device().is_gpu());
    assert_eq!(out.data(), x_cpu.permute(&axes).data());
}

/// Moving a tensor between devices mid-graph is a deliberate transfer, not a gap:
/// backward through `to_device` brings the gradient back to the source device under
/// `Strict`, with no fallback recorded.
#[test]
fn backward_through_to_device_is_not_a_gap() {
    if skip_if_no_gpu() {
        return;
    }
    let (x_cpu, x) = leaf_pair(&[2, 3]);
    x_cpu.exp().sum().into_loss().unwrap().backward();

    let _strict = with_device_policy(DevicePolicy::Strict);
    let _ = take_fallback_events();
    let y = x.exp().to_device(Device::CPU);
    y.sum().into_loss().unwrap().backward();
    assert!(x.grad_storage().unwrap().is_gpu());
    assert_close("x", &x_cpu.grad().unwrap(), &x.grad().unwrap(), 1e-5);
    assert!(take_fallback_events().is_empty());
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

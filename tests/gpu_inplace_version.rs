//! GPU parameters and S9: an optimizer step on a GPU parameter bumps its version like a CPU
//! one, so backward refuses the stale graph, and an op that doesn't read the parameter
//! still runs.
//!
//! Each test prints `GPU-RAN <name>` before its assertions, because a GPU test with no
//! adapter returns early and still reports `ok`; check the log for the marker.

#![cfg(feature = "gpu")]

use volta::gpu::is_gpu_available;
use volta::{Adam, Device, Muon, RawTensor, SGD, Storage, Tensor};

/// Skip (returning `true`) when there is no GPU; otherwise print the marker for `name`.
fn skip_or_mark(name: &str) -> bool {
    if is_gpu_available() {
        println!("GPU-RAN {name}");
        false
    } else {
        println!("GPU-SKIPPED {name}");
        true
    }
}

/// An independent 2x2 GPU leaf with distinct values.
fn on_gpu(data: Vec<f32>, requires_grad: bool) -> Tensor {
    let t = RawTensor::new(data, &[2, 2], false).to_device(Device::gpu().expect("gpu"));
    t.set_requires_grad(requires_grad);
    t
}

/// A GPU weight with a gradient set, so every optimizer writes it.
fn weight() -> Tensor {
    let w = on_gpu(vec![1.0, 2.0, 3.0, 4.0], true);
    w.set_grad(Storage::cpu(vec![0.5, -1.0, 0.25, 2.0]));
    w
}

/// Whether `loss.backward()` panics with the in-place message.
fn backward_refuses(loss: &volta::ScalarLoss) -> bool {
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| loss.backward()));
    caught.is_err_and(|p| {
        p.downcast_ref::<String>()
            .is_some_and(|m| m.contains("modified by an inplace operation"))
    })
}

/// One step of the named optimizer on `w`, which must have a gradient.
fn step(optimizer: &str, w: Tensor) {
    match optimizer {
        "SGD" => SGD::new(vec![w], 0.1, 0.0, 0.0).step(),
        "Adam" => Adam::new(vec![w], 0.1, (0.9, 0.999), 1e-8, 0.0).step(),
        "Muon" => Muon::new(vec![w], 0.1, 0.9, false, 5).step(),
        other => panic!("unknown optimizer {other}"),
    }
}

#[test]
fn gpu_optimizer_steps_invalidate_a_saved_weight() {
    if skip_or_mark("gpu_optimizer_steps_invalidate_a_saved_weight") {
        return;
    }
    for name in ["SGD", "Adam", "Muon"] {
        let w = weight();
        let x = on_gpu(vec![0.5, -1.5, 2.0, 3.0], true);
        let loss = w.elem_mul(&x).sum().into_loss().expect("scalar");
        step(name, w);
        assert!(backward_refuses(&loss), "{name}: stale GPU graph accepted");
    }
}

#[test]
fn a_gpu_matmul_operand_is_checked_too() {
    if skip_or_mark("a_gpu_matmul_operand_is_checked_too") {
        return;
    }
    let w = weight();
    let x = on_gpu(vec![0.5, -1.5, 2.0, 3.0], true);
    let loss = x.matmul(&w).sum().into_loss().expect("scalar");
    SGD::new(vec![w], 0.1, 0.0, 0.0).step();
    assert!(backward_refuses(&loss), "stale GPU matmul graph accepted");
}

#[test]
fn a_gpu_add_does_not_read_the_weight_so_a_step_is_allowed() {
    if skip_or_mark("a_gpu_add_does_not_read_the_weight_so_a_step_is_allowed") {
        return;
    }
    let w = weight();
    let x = on_gpu(vec![0.5, -1.5, 2.0, 3.0], true);
    let loss = w.add(&x).sum().into_loss().expect("scalar");
    SGD::new(vec![w], 0.1, 0.0, 0.0).step();
    loss.backward();
    assert_eq!(x.grad().expect("grad"), vec![1.0; 4]);
}

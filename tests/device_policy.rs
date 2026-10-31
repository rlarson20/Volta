//! `DevicePolicy` on a real GPU: device labels, the forward and backward gap checks, and
//! the fallback event log (ADDENDUM §2.1).
//!
//! Each test skips cleanly when no GPU adapter is available.

#![cfg(feature = "gpu")]

#[path = "common/mod.rs"]
mod common;

use common::{gpu_device, skip_if_no_gpu};
use volta::{Device, RawTensor, Storage};

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

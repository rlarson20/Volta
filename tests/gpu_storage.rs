//! GPU storage semantics (H3): clones share one device buffer and one host copy, and the
//! only write path (the optimizer step) copies before it writes.
//!
//! These need a real adapter. Each test returns early without one, so "passed" is only
//! GPU-verified on a machine that has a GPU: run `just test-gpu` there.

#![cfg(feature = "gpu")]

use volta::gpu::is_gpu_available;
use volta::{Adam, Device, RawTensor, SGD, Storage, Tensor};

const N: usize = 37;
const STEPS: usize = 3;

/// A non-constant, sign-mixed vector, distinct for each `salt`.
fn wavy(n: usize, salt: f32) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * 0.37 + salt).sin()).collect()
}

/// A trainable parameter with the given starting values, on the CPU or the GPU.
fn param(gpu: bool) -> Tensor {
    let p = RawTensor::new(wavy(N, 0.1), &[N], true);
    if gpu {
        p.to_device(Device::gpu().expect("adapter checked by the caller"))
    } else {
        p
    }
}

fn assert_close(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() < 1e-5, "{what}[{i}]: gpu={g} cpu={w}");
    }
}

/// Run `STEPS` optimizer steps with a different gradient each step, reading the parameter
/// on the host after every step. Returns the host value after each step.
fn run_steps(gpu: bool, make: impl Fn(Vec<Tensor>) -> Box<dyn FnMut()>) -> Vec<Vec<f32>> {
    let p = param(gpu);
    let mut step = make(vec![p.clone()]);
    (0..STEPS)
        .map(|s| {
            p.set_grad(Storage::cpu(wavy(N, 1.0 + s as f32)));
            step();
            p.data()
        })
        .collect()
}

#[test]
fn sgd_momentum_matches_cpu_and_host_reads_see_every_step() {
    if !is_gpu_available() {
        return;
    }
    let make = |ps: Vec<Tensor>| -> Box<dyn FnMut()> {
        let mut opt = SGD::new(ps, 0.05, 0.9, 0.01);
        Box::new(move || opt.step())
    };
    let cpu = run_steps(false, make);
    let gpu = run_steps(true, make);
    for (s, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
        assert_close(&format!("SGD momentum after step {}", s + 1), g, c);
    }
    // A host read after step 1 must not freeze what step 2 reads.
    assert_ne!(gpu[0], gpu[1], "step 2 did not change the parameter");
}

#[test]
fn adam_matches_cpu_and_host_reads_see_every_step() {
    if !is_gpu_available() {
        return;
    }
    let make = |ps: Vec<Tensor>| -> Box<dyn FnMut()> {
        let mut opt = Adam::new(ps, 0.01, (0.9, 0.999), 1e-8, 0.0);
        Box::new(move || opt.step())
    };
    let cpu = run_steps(false, make);
    let gpu = run_steps(true, make);
    // Three steps compare the first and second moments too: they feed every later update.
    for (s, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
        assert_close(&format!("Adam after step {}", s + 1), g, c);
    }
    assert_ne!(gpu[0], gpu[1], "step 2 did not change the parameter");
}

/// `detach` shares storage, so it is a snapshot. Left unread until after the step, the old
/// code showed the stepped values through it, because the clones shared one buffer.
#[test]
fn detached_snapshot_keeps_its_value_across_an_optimizer_step() {
    if !is_gpu_available() {
        return;
    }
    let p = param(true);
    let before = p.data();
    let snapshot = p.detach();
    let mut opt = SGD::new(vec![p.clone()], 0.5, 0.0, 0.0);
    p.set_grad(Storage::cpu(wavy(N, 2.0)));
    opt.step();

    let after = p.data();
    assert_ne!(after, before, "the step must change the parameter");
    assert_eq!(snapshot.data(), before, "the snapshot must keep its value");
}

/// The same, with the snapshot read first and the parameter read between two steps.
#[test]
fn snapshot_taken_after_a_host_read_still_sees_only_its_own_step() {
    if !is_gpu_available() {
        return;
    }
    let p = param(true);
    let mut opt = SGD::new(vec![p.clone()], 0.5, 0.0, 0.0);
    p.set_grad(Storage::cpu(wavy(N, 2.0)));
    opt.step();
    let after_one = p.data();

    let snapshot = p.detach();
    assert_eq!(snapshot.data(), after_one);
    p.set_grad(Storage::cpu(wavy(N, 3.0)));
    opt.step();

    assert_ne!(p.data(), after_one, "step 2 must change the parameter");
    assert_eq!(
        snapshot.data(),
        after_one,
        "the snapshot must stay at step 1"
    );
}

/// The raw-bytes view of GPU storage is the f32 data, in order.
#[test]
fn gpu_storage_bytes_round_trip_to_the_f32_data() {
    if !is_gpu_available() {
        return;
    }
    let values = wavy(N, 0.5);
    let storage = Storage::gpu(values.clone());
    assert!(storage.is_gpu(), "storage must be on the GPU");

    let bytes = storage.try_as_bytes().expect("GPU bytes");
    assert_eq!(bytes.len(), N * 4);
    let (chunks, rest) = bytes.as_chunks::<4>();
    assert!(rest.is_empty(), "byte length is not a multiple of 4");
    let decoded: Vec<f32> = chunks.iter().map(|c| f32::from_ne_bytes(*c)).collect();
    assert_eq!(decoded, values);
}

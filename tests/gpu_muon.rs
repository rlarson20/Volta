//! GPU Muon: a GPU parameter takes the same steps as its CPU twin, and the write path
//! keeps snapshots intact.
//!
//! The CPU `Muon::step` is the reference (its own tests pin the quintic iteration and the
//! shape scale), so these compare a GPU parameter against a CPU one over several steps.
//! Each test returns early without an adapter, so "passed" is only GPU-verified on a
//! machine that has one: run `just test-gpu` there.

#![cfg(feature = "gpu")]

use volta::gpu::is_gpu_available;
use volta::{Device, Muon, RawTensor, Storage, Tensor, manual_seed, randn};

/// Steps per run; each gets its own gradient, so momentum carries real history.
const STEPS: usize = 3;

/// Parameter shapes: wide, tall, square, a conv weight, a vector (plain momentum), and a
/// single column (also plain momentum).
const SHAPES: [&[usize]; 6] = [&[4, 8], &[8, 4], &[6, 6], &[3, 2, 2, 2], &[37], &[5, 1]];

/// A seeded random vector with the number of elements of `shape`.
fn random_values(shape: &[usize]) -> Vec<f32> {
    randn(shape).data()
}

/// A trainable parameter with the given values, on the CPU or the GPU.
fn param(values: &[f32], shape: &[usize], gpu: bool) -> Tensor {
    let p = RawTensor::new(values.to_vec(), shape, false);
    if gpu {
        let p = p.to_device(Device::gpu().expect("adapter checked by the caller"));
        p.set_requires_grad(true);
        p
    } else {
        p.set_requires_grad(true);
        p
    }
}

/// Take `grads.len()` Muon steps on a parameter of `shape` starting at `start`, reading
/// the parameter on the host after every step. Returns the host value after each step.
fn run(
    shape: &[usize],
    start: &[f32],
    grads: &[Vec<f32>],
    nesterov: bool,
    gpu: bool,
) -> Vec<Vec<f32>> {
    let p = param(start, shape, gpu);
    let mut opt = Muon::new(vec![p.clone()], 0.1, 0.9, nesterov, 5);
    grads
        .iter()
        .map(|g| {
            p.set_grad(Storage::cpu(g.clone()));
            opt.step();
            p.data()
        })
        .collect()
}

#[test]
fn gpu_steps_match_cpu_steps_for_every_shape_and_both_momentum_forms() {
    if !is_gpu_available() {
        return;
    }
    manual_seed(0x6D75);
    for shape in SHAPES {
        let start = random_values(shape);
        let grads: Vec<Vec<f32>> = (0..STEPS).map(|_| random_values(shape)).collect();
        for nesterov in [false, true] {
            let cpu = run(shape, &start, &grads, nesterov, false);
            let gpu = run(shape, &start, &grads, nesterov, true);
            let mut previous = start.clone();
            for (s, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
                assert_eq!(c.len(), g.len(), "{shape:?} step {}: length", s + 1);
                for (i, (cv, gv)) in c.iter().zip(g).enumerate() {
                    assert!(
                        (cv - gv).abs() < 1e-5,
                        "{shape:?} nesterov={nesterov} step {} [{i}]: gpu={gv} cpu={cv}",
                        s + 1
                    );
                }
                assert_ne!(
                    g,
                    &previous,
                    "{shape:?} nesterov={nesterov}: step {} did not move the GPU parameter",
                    s + 1
                );
                previous.clone_from(g);
            }
        }
    }
}

/// `detach` shares storage, so it is a snapshot. The Muon step writes the parameter, so it
/// has to go through the path that copies a shared buffer first.
#[test]
fn detached_snapshot_keeps_its_value_across_a_muon_step() {
    if !is_gpu_available() {
        return;
    }
    manual_seed(0x5A91);
    let shape: &[usize] = &[4, 8];
    let p = param(&random_values(shape), shape, true);
    let before = p.data();
    let snapshot = p.detach();
    let mut opt = Muon::new(vec![p.clone()], 0.1, 0.9, true, 5);
    p.set_grad(Storage::cpu(random_values(shape)));
    opt.step();

    assert_ne!(p.data(), before, "the step must change the parameter");
    assert_eq!(snapshot.data(), before, "the snapshot must keep its value");
}

/// A parameter with no gradient is left alone, on the GPU too.
#[test]
fn gpu_parameter_without_a_gradient_is_untouched() {
    if !is_gpu_available() {
        return;
    }
    manual_seed(0x0FF1);
    let shape: &[usize] = &[3, 5];
    let values = random_values(shape);
    let p = param(&values, shape, true);
    let mut opt = Muon::new(vec![p.clone()], 0.1, 0.9, false, 5);
    opt.step();
    assert_eq!(p.data(), values);
}

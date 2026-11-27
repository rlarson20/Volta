//! The reference workloads for #464: one full training step (forward, MSE loss, backward,
//! `Adam` step) on the CPU and on the GPU, for a matmul-bound MLP and a small CNN.
//!
//! Sized to show the difference on an Apple M2 (24 GB) without long runs: the MLP is
//! `784 -> 2048 -> 2048 -> 10` at batch 256 (about 9 GFLOP a step), the CNN two 3x3 conv
//! blocks with max pooling and two linear layers over 64 MNIST-shaped images. Each GPU
//! iteration ends with `gpu_sync()`, so it times finished work, not queued work.
//!
//! Before timing, one GPU step is checked with `gpu_counters()`: it must not read back or
//! drain the queue (the gate test asserts the same for every layer). The step's counts are
//! printed, so a run shows how many submissions a step costs.
//!
//! Run on the Mac with `just bench-train` (CPU with Accelerate against Metal). Skips the
//! GPU half when no adapter is available.

use criterion::{BenchmarkId, Criterion, criterion_main};
use volta::gpu::{gpu_counters, gpu_sync, is_gpu_available};
use volta::nn::{Adam, Conv2d, Flatten, Layer, Linear, MaxPool2d, Module, ReLU, Sequential};
use volta::{Device, RawTensor, Tensor, manual_seed, mse_loss};

/// One reference workload: a model, its input shape and its output width.
struct Workload {
    /// The name in reports.
    name: &'static str,
    /// Builds the model on the CPU.
    build: fn() -> Sequential,
    /// The input batch's shape.
    input: &'static [usize],
    /// Output features per example.
    outputs: usize,
}

/// The MLP: three linear layers, almost all of the time in matmuls.
fn mlp() -> Sequential {
    Sequential::new(vec![
        Box::new(Linear::new(784, 2048, true)),
        Box::new(ReLU),
        Box::new(Linear::new(2048, 2048, true)),
        Box::new(ReLU),
        Box::new(Linear::new(2048, 10, true)),
    ])
}

/// The CNN: `conv -> relu -> pool` twice, then two linear layers.
fn cnn() -> Sequential {
    Sequential::new(vec![
        Box::new(Conv2d::new(1, 32, 3, 1, 1, true)),
        Box::new(ReLU),
        Box::new(MaxPool2d::new(2, 2, 0)),
        Box::new(Conv2d::new(32, 64, 3, 1, 1, true)),
        Box::new(ReLU),
        Box::new(MaxPool2d::new(2, 2, 0)),
        Box::new(Flatten::new()),
        Box::new(Linear::new(64 * 7 * 7, 128, true)),
        Box::new(ReLU),
        Box::new(Linear::new(128, 10, true)),
    ])
}

/// The workloads, lightest first.
const WORKLOADS: [Workload; 2] = [
    Workload {
        name: "cnn_b64",
        build: cnn,
        input: &[64, 1, 28, 28],
        outputs: 10,
    },
    Workload {
        name: "mlp_b256",
        build: mlp,
        input: &[256, 784],
        outputs: 10,
    },
];

/// A model, its optimizer and one batch, all on one device.
struct Run {
    /// The model.
    model: Sequential,
    /// Adam over the model's parameters.
    opt: Adam,
    /// The input batch.
    x: Tensor,
    /// The regression target.
    y: Tensor,
}

impl Run {
    /// Build `w` on the CPU, then move it and its batch to `device`.
    fn new(w: &Workload, device: &Device) -> Self {
        manual_seed(0x464);
        let mut model = (w.build)();
        let x = RawTensor::randn(w.input);
        let batch = w.input.first().copied().expect("a batch dimension");
        let y = RawTensor::randn(&[batch, w.outputs]);
        let (x, y) = if device.is_gpu() {
            model.to_device(device.clone());
            (x.to_device(device.clone()), y.to_device(device.clone()))
        } else {
            (x, y)
        };
        let opt = Adam::new(model.parameters(), 1e-3, (0.9, 0.999), 1e-8, 0.0);
        Self { model, opt, x, y }
    }

    /// One training step.
    fn step(&mut self) {
        self.opt.zero_grad();
        let out = self.model.forward(&self.x);
        mse_loss(&out, &self.y)
            .into_loss()
            .expect("a scalar loss")
            .backward();
        self.opt.step();
    }
}

/// Check that a GPU step of `w` never waits on the host, and print what it costs.
fn check_gpu_step(w: &Workload, gpu: &Device) {
    let mut run = Run::new(w, gpu);
    run.step(); // the first step allocates Adam's state
    assert!(gpu_sync(), "GPU sync timed out");
    let before = gpu_counters();
    run.step();
    let used = gpu_counters().since(&before);
    assert!(gpu_sync(), "GPU sync timed out");
    println!("{}: one GPU step: {used:?}", w.name);
    assert_eq!(
        (used.readbacks, used.syncs),
        (0, 0),
        "{}: a GPU training step waited on the host",
        w.name
    );
}

/// One training step of each workload, on the CPU and (when available) on the GPU.
fn bench_training_step(c: &mut Criterion) {
    let gpu = if is_gpu_available() {
        Device::gpu()
    } else {
        None
    };
    let mut group = c.benchmark_group("training_step");
    group.sample_size(10);
    for w in &WORKLOADS {
        let mut cpu_run = Run::new(w, &Device::CPU);
        group.bench_function(BenchmarkId::new("cpu", w.name), |b| {
            b.iter(|| cpu_run.step());
        });
        if let Some(gpu) = &gpu {
            check_gpu_step(w, gpu);
            let mut gpu_run = Run::new(w, gpu);
            group.bench_function(BenchmarkId::new("gpu", w.name), |b| {
                b.iter(|| {
                    gpu_run.step();
                    assert!(gpu_sync(), "GPU sync timed out");
                });
            });
        }
    }
    group.finish();
}

// `criterion_group!` (criterion 0.5) generates an undocumented `pub fn benches`. A lint
// attribute on the macro call is ignored, so the call lives in a module that carries the
// `allow`, which keeps the lint active for everything else in this file.
#[allow(
    missing_docs,
    clippy::missing_docs_in_private_items,
    reason = "`criterion_group!` (criterion 0.5) generates an undocumented `pub fn benches`"
)]
mod group {
    use criterion::criterion_group;

    criterion_group!(benches, super::bench_training_step);
}
use group::benches;

criterion_main!(benches);

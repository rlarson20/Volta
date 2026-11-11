//! Neural network layer benchmarks
//!
//! Benchmarks for high-level neural network components including:
//! - Linear layer (forward and backward pass)
//! - Conv2d layer (forward for various kernel sizes, and backward)
//! - `MaxPool2d` layer
//! - Sequential models (end-to-end forward, and forward plus backward)
use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, criterion_main};
use std::hint::black_box;
use volta::{Conv2d, Layer, Linear, MaxPool2d, RawTensor, ReLU, Sequential, Tensor};

/// Generate a random tensor of the given size
fn random_tensor(size: usize) -> Tensor {
    let data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01).sin()).collect();
    RawTensor::new(data, &[size], false)
}

/// Generate a random 2D tensor
fn random_tensor_2d(rows: usize, cols: usize) -> Tensor {
    let size = rows * cols;
    let data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01).sin()).collect();
    RawTensor::new(data, &[rows, cols], false)
}

/// Generate a random 4D tensor (batch, channels, height, width)
fn random_tensor_4d(batch: usize, channels: usize, height: usize, width: usize) -> Tensor {
    let size = batch * channels * height * width;
    let data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01).sin()).collect();
    RawTensor::new(data, &[batch, channels, height, width], false)
}

// ===== LINEAR LAYER =====

/// `Linear` forward pass at MNIST-style (784x128), output (128x10), square (512x512), and rectangular (1024x256) sizes.
fn bench_linear_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_forward");

    // Various layer sizes using bench_function for simplicity
    group.bench_function("mnist_style_784x128", |b| {
        let layer = Linear::new(784, 128, true);
        let input = random_tensor_2d(32, 784);
        b.iter(|| black_box(&layer).forward(black_box(&input)));
    });

    group.bench_function("output_layer_128x10", |b| {
        let layer = Linear::new(128, 10, true);
        let input = random_tensor_2d(32, 128);
        b.iter(|| black_box(&layer).forward(black_box(&input)));
    });

    group.bench_function("square_large_512x512", |b| {
        let layer = Linear::new(512, 512, true);
        let input = random_tensor_2d(16, 512);
        b.iter(|| black_box(&layer).forward(black_box(&input)));
    });

    group.bench_function("rectangular_1024x256", |b| {
        let layer = Linear::new(1024, 256, true);
        let input = random_tensor_2d(16, 1024);
        b.iter(|| black_box(&layer).forward(black_box(&input)));
    });

    group.finish();
}

/// A fixed regression target with the shape of `model`'s output on `input`.
///
/// Taken from a forward pass, not hard-coded, so a shape change in a layer cannot turn
/// the loss into a silent broadcast against a mis-sized target.
fn target_for(model: &dyn Layer, input: &Tensor) -> Tensor {
    let shape = model.forward(input).shape();
    let dims = shape.truncate_rank();
    let size = dims.iter().product();
    let data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.013).cos()).collect();
    RawTensor::new(data, dims, false)
}

/// Benchmark one training step's gradient work on `model`: clear the parameter
/// gradients, forward, MSE against a fixed target, backward.
///
/// `zero_grad` runs inside the measured loop, so every iteration does the same work
/// as a real step instead of accumulating into the previous iteration's `grad`
/// buffers. Its cost is part of a training step, so it belongs in the number.
fn bench_train_step(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    mut model: Box<dyn Layer>,
    input: &Tensor,
) {
    let target = target_for(model.as_ref(), input);
    group.bench_function(name, |b| {
        b.iter(|| {
            model.zero_grad();
            let output = model.forward(black_box(input));
            RawTensor::mse_loss(&output, &target)
                .into_loss()
                .expect("MSE loss is a scalar that requires grad")
                .backward();
        });
    });
}

/// `Linear` forward plus backward at 128x64 and 256x128.
fn bench_linear_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_backward");

    // Larger sizes for backward pass to make gradient computation meaningful
    bench_train_step(
        &mut group,
        "backward_128x64",
        Box::new(Linear::new(128, 64, true)),
        &random_tensor_2d(16, 128),
    );
    bench_train_step(
        &mut group,
        "backward_256x128",
        Box::new(Linear::new(256, 128, true)),
        &random_tensor_2d(16, 256),
    );

    group.finish();
}

// ===== CONV2D LAYER =====

/// `Conv2d` forward pass for 1x1, 3x3, 5x5, and 7x7 kernels, and for small grayscale, small RGB, and medium RGB inputs.
fn bench_conv2d_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv2d_forward");

    // Different kernel sizes
    group.bench_function("kernel_1x1", |b| {
        let layer = Conv2d::new(3, 16, 1, 1, 0, true);
        let input = random_tensor_4d(8, 3, 32, 32);
        b.iter(|| black_box(&layer).forward(black_box(&input)));
    });

    group.bench_function("kernel_3x3", |b| {
        let layer = Conv2d::new(3, 16, 3, 1, 0, true);
        let input = random_tensor_4d(8, 3, 32, 32);
        b.iter(|| black_box(&layer).forward(black_box(&input)));
    });

    group.bench_function("kernel_5x5", |b| {
        let layer = Conv2d::new(3, 16, 5, 1, 0, true);
        let input = random_tensor_4d(8, 3, 32, 32);
        b.iter(|| black_box(&layer).forward(black_box(&input)));
    });

    group.bench_function("kernel_7x7", |b| {
        let layer = Conv2d::new(3, 16, 7, 1, 0, true);
        let input = random_tensor_4d(8, 3, 32, 32);
        b.iter(|| black_box(&layer).forward(black_box(&input)));
    });

    // Different channel configurations
    group.bench_function("channels_grayscale_small", |b| {
        let layer = Conv2d::new(1, 8, 3, 1, 0, true);
        let input = random_tensor_4d(4, 1, 28, 28);
        b.iter(|| black_box(&layer).forward(black_box(&input)));
    });

    group.bench_function("channels_rgb_small", |b| {
        let layer = Conv2d::new(3, 16, 3, 1, 0, true);
        let input = random_tensor_4d(4, 3, 28, 28);
        b.iter(|| black_box(&layer).forward(black_box(&input)));
    });

    group.bench_function("channels_rgb_medium", |b| {
        let layer = Conv2d::new(3, 32, 3, 1, 0, true);
        let input = random_tensor_4d(4, 3, 28, 28);
        b.iter(|| black_box(&layer).forward(black_box(&input)));
    });

    group.finish();
}

/// `Conv2d` forward plus backward for 3x3 kernels on small grayscale and RGB inputs.
fn bench_conv2d_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv2d_backward");

    bench_train_step(
        &mut group,
        "grayscale_1to8_3x3",
        Box::new(Conv2d::new(1, 8, 3, 1, 0, true)),
        &random_tensor_4d(4, 1, 28, 28),
    );
    bench_train_step(
        &mut group,
        "rgb_3to16_3x3",
        Box::new(Conv2d::new(3, 16, 3, 1, 0, true)),
        &random_tensor_4d(8, 3, 32, 32),
    );

    group.finish();
}

// ===== MAXPOOL2D LAYER =====

/// `MaxPool2d` forward pass with a 2x2 window, an overlapping 3x3 window with stride 2, and a 4x4 window.
fn bench_maxpool2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("maxpool2d");

    group.bench_function("standard_2x2", |b| {
        let layer = MaxPool2d::new(2, 2, 0);
        let input = random_tensor_4d(8, 16, 28, 28);
        b.iter(|| black_box(&layer).forward(black_box(&input)));
    });

    group.bench_function("overlap_3x2", |b| {
        let layer = MaxPool2d::new(3, 2, 0);
        let input = random_tensor_4d(8, 16, 28, 28);
        b.iter(|| black_box(&layer).forward(black_box(&input)));
    });

    group.bench_function("large_4x4", |b| {
        let layer = MaxPool2d::new(4, 4, 0);
        let input = random_tensor_4d(8, 16, 28, 28);
        b.iter(|| black_box(&layer).forward(black_box(&input)));
    });

    group.finish();
}

// ===== SEQUENTIAL MODEL =====

/// `Sequential` forward pass end to end: a small `MLP`, a medium `MLP`, and a small conv + `ReLU` + pool model.
fn bench_sequential_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequential_forward");

    // Small MLP
    group.bench_function("small_mlp", |b| {
        let model = Sequential::new(vec![
            Box::new(Linear::new(128, 64, true)),
            Box::new(ReLU),
            Box::new(Linear::new(64, 10, true)),
        ]);
        let input = random_tensor_2d(32, 128);
        b.iter(|| black_box(&model).forward(black_box(&input)));
    });

    // Medium MLP
    group.bench_function("medium_mlp", |b| {
        let model = Sequential::new(vec![
            Box::new(Linear::new(256, 128, true)),
            Box::new(ReLU),
            Box::new(Linear::new(128, 64, true)),
            Box::new(ReLU),
            Box::new(Linear::new(64, 10, true)),
        ]);
        let input = random_tensor_2d(32, 256);
        b.iter(|| black_box(&model).forward(black_box(&input)));
    });

    // CNN-like model (conv + pool + linear)
    group.bench_function("small_cnn", |b| {
        let model = Sequential::new(vec![
            Box::new(Conv2d::new(3, 8, 3, 1, 0, true)),
            Box::new(ReLU),
            Box::new(MaxPool2d::new(2, 2, 0)),
        ]);
        let input = random_tensor_4d(8, 3, 28, 28);
        b.iter(|| black_box(&model).forward(black_box(&input)));
    });

    group.finish();
}

/// `Sequential` forward plus backward end to end: the small `MLP` and the small conv +
/// `ReLU` + pool model from `sequential_forward`.
fn bench_sequential_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequential_backward");

    bench_train_step(
        &mut group,
        "small_mlp",
        Box::new(Sequential::new(vec![
            Box::new(Linear::new(128, 64, true)),
            Box::new(ReLU),
            Box::new(Linear::new(64, 10, true)),
        ])),
        &random_tensor_2d(32, 128),
    );
    bench_train_step(
        &mut group,
        "small_cnn",
        Box::new(Sequential::new(vec![
            Box::new(Conv2d::new(3, 8, 3, 1, 0, true)),
            Box::new(ReLU),
            Box::new(MaxPool2d::new(2, 2, 0)),
        ])),
        &random_tensor_4d(8, 3, 28, 28),
    );

    group.finish();
}

// ===== ACTIVATION FUNCTIONS =====

/// `ReLU` (1024 and 4096 elements), `sigmoid`, and `tanh` (1024 elements).
fn bench_activations(c: &mut Criterion) {
    let mut group = c.benchmark_group("activations");

    group.bench_function("relu_1024", |b| {
        let tensor = random_tensor(1024);
        b.iter(|| black_box(&tensor).relu());
    });

    group.bench_function("relu_4096", |b| {
        let tensor = random_tensor(4096);
        b.iter(|| black_box(&tensor).relu());
    });

    group.bench_function("sigmoid_1024", |b| {
        let tensor = random_tensor(1024);
        b.iter(|| black_box(&tensor).sigmoid());
    });

    group.bench_function("tanh_1024", |b| {
        let tensor = random_tensor(1024);
        b.iter(|| black_box(&tensor).tanh());
    });

    group.finish();
}

// Runs the layer benchmarks: `Linear`, `Conv2d`, `MaxPool2d`, `Sequential`, and activations,
// with backward groups for `Linear`, `Conv2d` and `Sequential`.
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

    criterion_group!(
        benches,
        super::bench_linear_forward,
        super::bench_linear_backward,
        super::bench_conv2d_forward,
        super::bench_conv2d_backward,
        super::bench_maxpool2d,
        super::bench_sequential_forward,
        super::bench_sequential_backward,
        super::bench_activations,
    );
}
use group::benches;

criterion_main!(benches);

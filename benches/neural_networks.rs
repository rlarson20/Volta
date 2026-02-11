//! Neural network layer benchmarks
//!
//! Benchmarks for high-level neural network components including:
//! - Linear layer (forward and backward pass)
//! - Conv2d layer (various kernel sizes)
//! - `MaxPool2d` layer
//! - Sequential models (end-to-end forward pass)

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use volta::{Conv2d, Linear, MaxPool2d, Module, RawTensor, ReLU, Sequential, Tensor, TensorOps};

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

fn bench_linear_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_backward");

    // Larger sizes for backward pass to make gradient computation meaningful
    group.bench_function("backward_128x64", |b| {
        let layer = Linear::new(128, 64, true);
        let input = random_tensor_2d(16, 128);
        b.iter(|| {
            let output = layer.forward(&input);
            output.backward();
        });
    });

    group.bench_function("backward_256x128", |b| {
        let layer = Linear::new(256, 128, true);
        let input = random_tensor_2d(16, 256);
        b.iter(|| {
            let output = layer.forward(&input);
            output.backward();
        });
    });

    group.finish();
}

// ===== CONV2D LAYER =====

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

// ===== MAXPOOL2D LAYER =====

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

// ===== ACTIVATION FUNCTIONS =====

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

criterion_group!(
    benches,
    bench_linear_forward,
    bench_linear_backward,
    bench_conv2d_forward,
    bench_maxpool2d,
    bench_sequential_forward,
    bench_activations
);

criterion_main!(benches);

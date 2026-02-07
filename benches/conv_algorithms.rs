//! Convolution algorithm comparison benchmarks
//!
//! Compares Direct vs im2col+GEMM convolution algorithms across various scenarios.
//! All benchmarks are designed to stay under 16GB memory limit.
//!
//! Memory calculation for im2col:
//! - Input: (B, C, H, W)
//! - Kernel: (K, K)
//! - im2col output: (B * H_out * W_out, C * K * K)
//! - Memory = B * H_out * W_out * C * K * K * 4 bytes (f32)

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use volta::{Conv2d, ConvAlgo, RawTensor, Tensor};

/// Generate a random 4D tensor (batch, channels, height, width)
fn random_tensor_4d(batch: usize, channels: usize, height: usize, width: usize) -> Tensor {
    let size = batch * channels * height * width;
    let data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01).sin()).collect();
    RawTensor::new(data, &[batch, channels, height, width], false)
}

/// Calculate im2col memory usage in MB
fn im2col_memory_mb(
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> f64 {
    let h_padded = height + 2 * padding;
    let w_padded = width + 2 * padding;
    let h_out = (h_padded - kernel) / stride + 1;
    let w_out = (w_padded - kernel) / stride + 1;

    let elements = batch * h_out * w_out * channels * kernel * kernel;
    let bytes = elements * 4; // f32 = 4 bytes
    (bytes as f64) / (1024.0 * 1024.0)
}

/// Benchmark group for comparing algorithms at different scales
fn bench_conv_algorithm_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv_algorithm_comparison");

    // Test scenarios: (name, batch, in_ch, out_ch, h, w, kernel, stride, padding, expected_im2col_mb)
    let scenarios = vec![
        // Small inputs - im2col memory < 1 MB
        ("tiny", 1, 3, 16, 16, 16, 3, 1, 0, 0.02),
        ("small", 4, 3, 16, 32, 32, 3, 1, 1, 0.41),
        ("medium", 8, 64, 128, 28, 28, 3, 1, 1, 67.5),
        ("medium_large", 4, 64, 128, 64, 64, 3, 1, 1, 193.0),
        // Larger inputs - still well under 16GB limit
        ("large", 8, 64, 128, 112, 112, 3, 2, 1, 247.5), // ~250 MB
        ("very_large", 16, 128, 256, 56, 56, 3, 1, 1, 993.0), // ~1 GB
        // Different kernel sizes
        ("kernel_1x1", 8, 64, 128, 56, 56, 1, 1, 0, 44.5), // ~45 MB
        ("kernel_5x5", 4, 32, 64, 56, 56, 5, 1, 2, 124.0), // ~125 MB
        ("kernel_7x7", 2, 32, 64, 56, 56, 7, 1, 3, 103.0), // ~100 MB
        // ResNet-style layers
        ("resnet_block", 8, 64, 64, 56, 56, 3, 1, 1, 123.0), // ~125 MB
        ("resnet_large", 4, 128, 128, 112, 112, 3, 2, 1, 247.5), // ~250 MB
    ];

    for scenario in scenarios {
        let (name, batch, in_ch, out_ch, h, w, kernel, stride, padding, im2col_mb) = scenario;

        // Safety check: ensure we're under 16GB limit
        assert!(
            im2col_mb < 16_000.0,
            "Benchmark exceeds memory limit: {} uses {:.1} MB",
            name,
            im2col_mb
        );

        // Benchmark Direct convolution
        group.bench_with_input(
            BenchmarkId::new("direct", name),
            &(batch, in_ch, out_ch, h, w, kernel, stride, padding),
            |b, &(batch_sz, ic, oc, h, w, k, s, p)| {
                let layer = Conv2d::new(ic, oc, k, s, p, false);
                layer.set_algo(ConvAlgo::Direct);
                let input = random_tensor_4d(batch_sz, ic, h, w);

                b.iter(|| black_box(&layer).forward(black_box(&input)))
            },
        );

        // Benchmark im2col convolution
        group.bench_with_input(
            BenchmarkId::new("im2col", name),
            &(batch, in_ch, out_ch, h, w, kernel, stride, padding),
            |b, &(batch_sz, ic, oc, h, w, k, s, p)| {
                let layer = Conv2d::new(ic, oc, k, s, p, false);
                layer.set_algo(ConvAlgo::Im2col);
                let input = random_tensor_4d(batch_sz, ic, h, w);

                b.iter(|| black_box(&layer).forward(black_box(&input)))
            },
        );

        // Print memory info for this scenario
        println!(
            "{}: B={}, C={}→{}, H=W={}, K={}x{}, im2col memory: {:.1} MB",
            name, batch, in_ch, out_ch, h, kernel, kernel, im2col_mb
        );
    }

    group.finish();
}

/// Benchmark comparing algorithms across different kernel sizes
fn bench_conv_kernel_size_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv_kernel_comparison");

    let kernel_sizes = vec![1, 3, 5, 7];

    for kernel_size in kernel_sizes {
        let padding = kernel_size / 2; // Maintain spatial dimensions

        // Calculate memory
        let im2col_mb = im2col_memory_mb(8, 64, 56, 56, kernel_size, 1, padding);

        group.bench_with_input(
            BenchmarkId::new("direct", kernel_size),
            &kernel_size,
            |bench, &k| {
                let layer = Conv2d::new(64, 64, k, 1, k / 2, false);
                layer.set_algo(ConvAlgo::Direct);
                let input = random_tensor_4d(8, 64, 56, 56);

                bench.iter(|| black_box(&layer).forward(black_box(&input)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("im2col", kernel_size),
            &kernel_size,
            |bench, &k| {
                let layer = Conv2d::new(64, 64, k, 1, k / 2, false);
                layer.set_algo(ConvAlgo::Im2col);
                let input = random_tensor_4d(8, 64, 56, 56);

                bench.iter(|| black_box(&layer).forward(black_box(&input)))
            },
        );

        println!(
            "Kernel {}: im2col memory = {:.1} MB",
            kernel_size, im2col_mb
        );
    }

    group.finish();
}

/// Benchmark comparing algorithms across different batch sizes
fn bench_conv_batch_size_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv_batch_comparison");

    let batch_sizes = vec![1, 2, 4, 8, 16, 32];

    for batch in batch_sizes {
        let im2col_mb = im2col_memory_mb(batch, 64, 56, 56, 3, 1, 1);

        group.bench_with_input(
            BenchmarkId::new("direct", batch),
            &batch,
            |bench, &batch_sz| {
                let layer = Conv2d::new(64, 64, 3, 1, 1, false);
                layer.set_algo(ConvAlgo::Direct);
                let input = random_tensor_4d(batch_sz, 64, 56, 56);

                bench.iter(|| black_box(&layer).forward(black_box(&input)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("im2col", batch),
            &batch,
            |bench, &batch_sz| {
                let layer = Conv2d::new(64, 64, 3, 1, 1, false);
                layer.set_algo(ConvAlgo::Im2col);
                let input = random_tensor_4d(batch_sz, 64, 56, 56);

                bench.iter(|| black_box(&layer).forward(black_box(&input)))
            },
        );

        println!("Batch {}: im2col memory = {:.1} MB", batch, im2col_mb);
    }

    group.finish();
}

/// Benchmark comparing algorithms across different spatial dimensions
fn bench_conv_spatial_size_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv_spatial_comparison");

    // Common spatial sizes in CNNs
    let spatial_sizes = vec![
        (28, 28, "mnist"),
        (32, 32, "cifar"),
        (56, 56, "imagenet_1/4"),
        (112, 112, "imagenet_1/2"),
        (224, 224, "imagenet_full"),
    ];

    for (h, w, label) in spatial_sizes {
        // Use smaller batch for larger spatial sizes to stay under memory limit
        let batch = match (h, w) {
            (224, 224) => 2, // ~785 MB for im2col
            (112, 112) => 4, // ~393 MB for im2col
            _ => 8,
        };

        let im2col_mb = im2col_memory_mb(batch, 64, h, w, 3, 1, 1);

        group.bench_with_input(
            BenchmarkId::new("direct", label),
            &(batch, h, w),
            |bench, &(batch_sz, h, w)| {
                let layer = Conv2d::new(64, 64, 3, 1, 1, false);
                layer.set_algo(ConvAlgo::Direct);
                let input = random_tensor_4d(batch_sz, 64, h, w);

                bench.iter(|| black_box(&layer).forward(black_box(&input)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("im2col", label),
            &(batch, h, w),
            |bench, &(batch_sz, h, w)| {
                let layer = Conv2d::new(64, 64, 3, 1, 1, false);
                layer.set_algo(ConvAlgo::Im2col);
                let input = random_tensor_4d(batch_sz, 64, h, w);

                bench.iter(|| black_box(&layer).forward(black_box(&input)))
            },
        );

        println!(
            "Spatial {} (batch={}): im2col memory = {:.1} MB",
            label, batch, im2col_mb
        );
    }

    group.finish();
}

/// Benchmark Auto mode vs explicit selections
fn bench_conv_auto_mode(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv_auto_mode");

    // Test various scenarios where Auto would make different choices
    let scenarios = vec![
        // (batch, in_ch, out_ch, h, w, kernel, stride, padding, expected_algo, label)
        (1, 3, 16, 8, 8, 3, 1, 0, "direct_small_no_grad"),
        (4, 3, 16, 16, 16, 3, 1, 0, "direct_medium_no_grad"),
        (8, 64, 128, 28, 28, 3, 1, 1, "im2col_medium"),
        (4, 64, 128, 64, 64, 3, 1, 1, "im2col_large"),
    ];

    for (batch, in_ch, out_ch, h, w, kernel, stride, padding, label) in scenarios {
        let im2col_mb = im2col_memory_mb(batch, in_ch, h, w, kernel, stride, padding);

        group.bench_with_input(
            BenchmarkId::new("auto", label),
            &(batch, in_ch, out_ch, h, w, kernel, stride, padding),
            |bench, &(batch_sz, ic, oc, h, w, k, s, p)| {
                let layer = Conv2d::new(ic, oc, k, s, p, false);
                layer.set_algo(ConvAlgo::Auto);
                let input = random_tensor_4d(batch_sz, ic, h, w);

                bench.iter(|| black_box(&layer).forward(black_box(&input)))
            },
        );

        println!(
            "Auto scenario {}: im2col memory = {:.1} MB",
            label, im2col_mb
        );
    }

    group.finish();
}

/// Memory scaling benchmark - how memory usage grows with input size
fn bench_conv_memory_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv_memory_scaling");

    // Show how im2col memory scales with different dimensions
    let scaling_tests = vec![
        ("scale_channels", 8, 32, 64, 28, 28, 3, 1, 1), // Vary channels
        ("scale_channels", 8, 64, 128, 28, 28, 3, 1, 1),
        ("scale_channels", 8, 128, 256, 28, 28, 3, 1, 1),
        ("scale_spatial", 8, 64, 64, 28, 28, 3, 1, 1), // Vary spatial
        ("scale_spatial", 8, 64, 64, 56, 56, 3, 1, 1),
        ("scale_spatial", 4, 64, 64, 112, 112, 3, 1, 1),
        ("scale_batch", 4, 64, 64, 56, 56, 3, 1, 1), // Vary batch
        ("scale_batch", 8, 64, 64, 56, 56, 3, 1, 1),
        ("scale_batch", 16, 64, 64, 56, 56, 3, 1, 1),
    ];

    for (scale_type, batch, in_ch, out_ch, h, w, kernel, stride, padding) in scaling_tests {
        let im2col_mb = im2col_memory_mb(batch, in_ch, h, w, kernel, stride, padding);
        let label = format!("{}_{}MB", scale_type, im2col_mb.floor() as i64);

        group.bench_with_input(
            BenchmarkId::new("direct", &label),
            &(batch, in_ch, out_ch, h, w, kernel, stride, padding),
            |bench, &(batch_sz, ic, oc, h, w, k, s, p)| {
                let layer = Conv2d::new(ic, oc, k, s, p, false);
                layer.set_algo(ConvAlgo::Direct);
                let input = random_tensor_4d(batch_sz, ic, h, w);

                bench.iter(|| black_box(&layer).forward(black_box(&input)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("im2col", &label),
            &(batch, in_ch, out_ch, h, w, kernel, stride, padding),
            |bench, &(batch_sz, ic, oc, h, w, k, s, p)| {
                let layer = Conv2d::new(ic, oc, k, s, p, false);
                layer.set_algo(ConvAlgo::Im2col);
                let input = random_tensor_4d(batch_sz, ic, h, w);

                bench.iter(|| black_box(&layer).forward(black_box(&input)))
            },
        );

        println!(
            "{}: im2col memory = {:.1} MB (under 16GB limit: {})",
            label,
            im2col_mb,
            if im2col_mb < 16_000.0 { "✓" } else { "✗" }
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_conv_algorithm_comparison,
    bench_conv_kernel_size_comparison,
    bench_conv_batch_size_comparison,
    bench_conv_spatial_size_comparison,
    bench_conv_auto_mode,
    bench_conv_memory_scaling
);
criterion_main!(benches);

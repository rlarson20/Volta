//! Convolution algorithm comparison benchmarks
//!
//! Compares the three explicit algorithms (Direct, im2col+GEMM, iGEMM) across various
//! scenarios, plus `Auto`. Each scenario computes its im2col buffer size with
//! [`im2col_memory_mb`] and asserts it is under 16 GB before it runs.
//!
//! Memory calculation for im2col:
//! - Input: (B, C, H, W)
//! - Kernel: (K, K)
//! - im2col output: (B * `H_out` * `W_out`, C * K * K)
//! - Memory = B * `H_out` * `W_out` * C * K * K * 4 bytes (f32)
use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, criterion_main};
use std::hint::black_box;
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

/// The explicit algorithms every comparison runs, with their benchmark-id names.
const ALGOS: [(ConvAlgo, &str); 3] = [
    (ConvAlgo::Direct, "direct"),
    (ConvAlgo::Im2col, "im2col"),
    (ConvAlgo::IGEMM, "igemm"),
];

/// One convolution: `(batch, in_ch, out_ch, h, w, kernel, stride, padding)`.
type ConvCase = (usize, usize, usize, usize, usize, usize, usize, usize);

/// The im2col buffer of `case` in MB, after asserting it is under the 16 GB limit.
fn checked_im2col_mb(label: &str, case: ConvCase) -> f64 {
    let (batch, in_ch, _, h, w, kernel, stride, padding) = case;
    let mb = im2col_memory_mb(batch, in_ch, h, w, kernel, stride, padding);
    assert!(
        mb < 16_000.0,
        "Benchmark exceeds memory limit: {label} uses {mb:.1} MB"
    );
    mb
}

/// Benchmark `case` under `algo`, as `<name>/<param>` in `group`.
fn bench_case(
    group: &mut BenchmarkGroup<'_, WallTime>,
    algo: ConvAlgo,
    name: &str,
    param: &str,
    case: ConvCase,
) {
    group.bench_with_input(
        BenchmarkId::new(name, param),
        &case,
        |b, &(batch_sz, ic, oc, h, w, k, s, p)| {
            let layer = Conv2d::new(ic, oc, k, s, p, false);
            layer.set_algo(algo);
            let input = random_tensor_4d(batch_sz, ic, h, w);

            b.iter(|| black_box(&layer).forward(black_box(&input)));
        },
    );
}

/// Benchmark `case` under each of [`ALGOS`].
fn bench_all_algos(group: &mut BenchmarkGroup<'_, WallTime>, param: &str, case: ConvCase) {
    for (algo, name) in ALGOS {
        bench_case(group, algo, name, param, case);
    }
}

/// Benchmark group for comparing algorithms at different scales
fn bench_conv_algorithm_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv_algorithm_comparison");

    // Test scenarios: (name, batch, in_ch, out_ch, h, w, kernel, stride, padding). The
    // im2col size is computed from these, not typed in (#139).
    let scenarios: Vec<(&str, ConvCase)> = vec![
        ("tiny", (1, 3, 16, 16, 16, 3, 1, 0)),
        ("small", (4, 3, 16, 32, 32, 3, 1, 1)),
        ("medium", (8, 64, 128, 28, 28, 3, 1, 1)),
        ("medium_large", (4, 64, 128, 64, 64, 3, 1, 1)),
        ("large", (8, 64, 128, 112, 112, 3, 2, 1)),
        ("very_large", (16, 128, 256, 56, 56, 3, 1, 1)),
        // Different kernel sizes
        ("kernel_1x1", (8, 64, 128, 56, 56, 1, 1, 0)),
        ("kernel_5x5", (4, 32, 64, 56, 56, 5, 1, 2)),
        ("kernel_7x7", (2, 32, 64, 56, 56, 7, 1, 3)),
        // ResNet-style layers
        ("resnet_block", (8, 64, 64, 56, 56, 3, 1, 1)),
        ("resnet_large", (4, 128, 128, 112, 112, 3, 2, 1)),
    ];

    for (name, case) in scenarios {
        let im2col_mb = checked_im2col_mb(name, case);
        bench_all_algos(&mut group, name, case);

        let (batch, in_ch, out_ch, h, _, kernel, _, _) = case;
        println!(
            "{name}: B={batch}, C={in_ch}→{out_ch}, H=W={h}, K={kernel}x{kernel}, im2col memory: {im2col_mb:.1} MB"
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
        let case = (8, 64, 64, 56, 56, kernel_size, 1, padding);
        let im2col_mb = checked_im2col_mb(&format!("kernel {kernel_size}"), case);
        bench_all_algos(&mut group, &kernel_size.to_string(), case);
        println!("Kernel {kernel_size}: im2col memory = {im2col_mb:.1} MB");
    }

    group.finish();
}

/// Benchmark comparing algorithms across different batch sizes
fn bench_conv_batch_size_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv_batch_comparison");

    let batch_sizes = vec![1, 2, 4, 8, 16, 32];

    for batch in batch_sizes {
        let case = (batch, 64, 64, 56, 56, 3, 1, 1);
        let im2col_mb = checked_im2col_mb(&format!("batch {batch}"), case);
        bench_all_algos(&mut group, &batch.to_string(), case);
        println!("Batch {batch}: im2col memory = {im2col_mb:.1} MB");
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

        let case = (batch, 64, 64, h, w, 3, 1, 1);
        let im2col_mb = checked_im2col_mb(label, case);
        bench_all_algos(&mut group, label, case);
        println!("Spatial {label} (batch={batch}): im2col memory = {im2col_mb:.1} MB");
    }

    group.finish();
}

/// Benchmark Auto mode vs explicit selections
fn bench_conv_auto_mode(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv_auto_mode");

    // Inputs on both sides of `ConvAlgo::auto_select`'s thresholds, named by the inputs:
    // which algorithm Auto picks is pinned by the unit tests in `src/nn/layers/conv.rs`,
    // not claimed here (#139).
    let scenarios: Vec<(&str, ConvCase)> = vec![
        ("b1_c3_8x8_k3", (1, 3, 16, 8, 8, 3, 1, 0)),
        ("b4_c3_16x16_k3", (4, 3, 16, 16, 16, 3, 1, 0)),
        ("b8_c64_28x28_k3", (8, 64, 128, 28, 28, 3, 1, 1)),
        ("b4_c64_64x64_k3", (4, 64, 128, 64, 64, 3, 1, 1)),
    ];

    for (label, case) in scenarios {
        let im2col_mb = checked_im2col_mb(label, case);
        bench_case(&mut group, ConvAlgo::Auto, "auto", label, case);
        println!("Auto scenario {label}: im2col memory = {im2col_mb:.1} MB");
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
        let case = (batch, in_ch, out_ch, h, w, kernel, stride, padding);
        let im2col_mb = checked_im2col_mb(scale_type, case);
        let label = format!("{scale_type}_{}MB", im2col_mb.floor());
        bench_all_algos(&mut group, &label, case);
        println!("{label}: im2col memory = {im2col_mb:.1} MB");
    }

    group.finish();
}

// Runs the convolution algorithm benchmarks: Direct, im2col+GEMM and iGEMM across algorithm, kernel size, batch size, and spatial size, the auto-selected mode, and memory scaling.
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
        super::bench_conv_algorithm_comparison,
        super::bench_conv_kernel_size_comparison,
        super::bench_conv_batch_size_comparison,
        super::bench_conv_spatial_size_comparison,
        super::bench_conv_auto_mode,
        super::bench_conv_memory_scaling,
    );
}
use group::benches;
criterion_main!(benches);

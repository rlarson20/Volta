//! Core tensor operation benchmarks
//!
//! Benchmarks for low-level tensor operations including:
//! - Binary operations (add, sub, mul, div)
//! - Unary operations (exp, log, relu, sigmoid, etc.)
//! - Matrix multiplication
//! - Reduction operations (sum, mean, max)
//! - Movement operations (reshape, transpose, etc.)
use criterion::{BenchmarkId, Criterion, black_box, criterion_main};
use volta::{Axis, PadValue};
use volta::{RawTensor, Tensor};

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

// ===== BINARY OPERATIONS =====

/// `add`, `sub`, elementwise `mul`, and `div` over 64 to 4096 elements, plus a broadcasting add.
fn bench_binary_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_ops");

    for size in [64, 256, 1024, 4096] {
        let size_ref = &size;
        group.bench_with_input(BenchmarkId::new("add", size), size_ref, |b, s| {
            let a = random_tensor(*s);
            let tensor_b = random_tensor(*s);
            b.iter(|| black_box(&a).add(black_box(&tensor_b)));
        });

        group.bench_with_input(BenchmarkId::new("sub", size), size_ref, |b, s| {
            let a = random_tensor(*s);
            let tensor_b = random_tensor(*s);
            b.iter(|| black_box(&a).sub(black_box(&tensor_b)));
        });

        group.bench_with_input(BenchmarkId::new("elem_mul", size), size_ref, |b, s| {
            let a = random_tensor(*s);
            let tensor_b = random_tensor(*s);
            b.iter(|| black_box(&a).elem_mul(black_box(&tensor_b)));
        });

        group.bench_with_input(BenchmarkId::new("div", size), size_ref, |b, s| {
            let a = random_tensor(*s);
            let tensor_b = random_tensor(*s);
            b.iter(|| black_box(&a).div(black_box(&tensor_b)));
        });
    }

    // Broadcasting benchmarks
    group.bench_function("broadcast_add", |b| {
        let a = random_tensor_2d(100, 100); // [100, 100]
        let tensor_b = random_tensor(100); // [100]
        b.iter(|| black_box(&a).add(black_box(&tensor_b)));
    });

    group.finish();
}

// ===== UNARY OPERATIONS =====

/// `exp`, `log`, `relu`, `sigmoid`, `tanh`, and `sqrt` over 1024 and 4096 elements.
fn bench_unary_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("unary_ops");

    for size in [1024, 4096] {
        let size_ref = &size;
        group.bench_with_input(BenchmarkId::new("exp", size), size_ref, |b, s| {
            let a = random_tensor(*s);
            b.iter(|| black_box(&a).exp());
        });

        group.bench_with_input(BenchmarkId::new("log", size), size_ref, |b, s| {
            let a = random_tensor(*s);
            b.iter(|| black_box(&a).log());
        });

        group.bench_with_input(BenchmarkId::new("relu", size), size_ref, |b, s| {
            let a = random_tensor(*s);
            b.iter(|| black_box(&a).relu());
        });

        group.bench_with_input(BenchmarkId::new("sigmoid", size), size_ref, |b, s| {
            let a = random_tensor(*s);
            b.iter(|| black_box(&a).sigmoid());
        });

        group.bench_with_input(BenchmarkId::new("tanh", size), size_ref, |b, s| {
            let a = random_tensor(*s);
            b.iter(|| black_box(&a).tanh());
        });

        group.bench_with_input(BenchmarkId::new("sqrt", size), size_ref, |b, s| {
            let a = random_tensor(*s);
            b.iter(|| black_box(&a).sqrt());
        });
    }

    group.finish();
}

// ===== MATRIX MULTIPLICATION =====

/// Square matmul for `n` in 64..=1024, plus 128x256 and 256x128 rectangular cases.
fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");

    // Square matrices
    for n in [64, 128, 256, 512, 1024] {
        let n_ref = &n;
        group.bench_with_input(BenchmarkId::new("square", n), n_ref, |b, s| {
            let a = random_tensor_2d(*s, *s);
            let tensor_b = random_tensor_2d(*s, *s);
            b.iter(|| black_box(&a).matmul(black_box(&tensor_b)));
        });
    }

    // Rectangular matrices
    group.bench_function("rectangular_128x256", |b| {
        let a = random_tensor_2d(128, 256);
        let tensor_b = random_tensor_2d(256, 128);
        b.iter(|| black_box(&a).matmul(black_box(&tensor_b)));
    });

    group.bench_function("rectangular_256x128", |b| {
        let a = random_tensor_2d(256, 128);
        let tensor_b = random_tensor_2d(128, 256);
        b.iter(|| black_box(&a).matmul(black_box(&tensor_b)));
    });

    group.finish();
}

// ===== REDUCTION OPERATIONS =====

/// `sum`, `mean`, and `max_reduce` over 1024 to 16384 elements, plus a 2-D `sum`.
fn bench_reduce_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduce_ops");

    for size in [1024, 4096, 16384] {
        let size_ref = &size;
        group.bench_with_input(BenchmarkId::new("sum", size), size_ref, |b, s| {
            let a = random_tensor(*s);
            b.iter(|| black_box(&a).sum());
        });

        group.bench_with_input(BenchmarkId::new("mean", size), size_ref, |b, s| {
            let a = random_tensor(*s);
            b.iter(|| black_box(&a).mean());
        });

        group.bench_with_input(BenchmarkId::new("max_reduce", size), size_ref, |b, s| {
            let a = random_tensor(*s);
            b.iter(|| black_box(&a).max_reduce());
        });
    }

    // 2D reduction
    group.bench_function("sum_2d", |b| {
        let a = random_tensor_2d(100, 100);
        b.iter(|| black_box(&a).sum());
    });

    group.finish();
}

// ===== MOVEMENT OPERATIONS =====

/// `reshape`, 2-D `transpose`, `expand`, `permute`, and `pad_with` (zero padding).
fn bench_movement_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("movement_ops");

    // Reshape
    group.bench_function("reshape_large", |b| {
        let a = random_tensor_2d(100, 100); // [100, 100]
        b.iter(|| black_box(&a).reshape(&[250, 40]));
    });

    // Transpose (2D)
    group.bench_function("transpose_2d", |b| {
        let a = random_tensor_2d(256, 256);
        b.iter(|| black_box(&a).transpose());
    });

    // Expand (broadcasting)
    group.bench_function("expand", |b| {
        let a = random_tensor_2d(1, 100); // [1, 100]
        b.iter(|| black_box(&a).expand(&[50, 100]));
    });

    // Permute (multi-dimensional)
    group.bench_function("permute", |b| {
        let data: Vec<f32> = (0..10000).map(|i| i as f32 * 0.01).collect();
        let a = RawTensor::new(data, &[10, 10, 10, 10], false);
        b.iter(|| black_box(&a).permute(&[Axis(2), Axis(3), Axis(1), Axis(0)]));
    });

    // Pad
    group.bench_function("pad", |b| {
        let a = random_tensor_2d(50, 50);
        b.iter(|| black_box(&a).pad_with(&[(0, 0), (5, 5)], PadValue::ZERO));
    });

    group.finish();
}

// ===== TERNARY OPERATIONS =====

/// `mulacc` (fused multiply-accumulate) over 512 and 2048 elements.
fn bench_ternary_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("ternary_ops");

    group.bench_function("mulacc_512", |b| {
        let x = random_tensor(512);
        let y = random_tensor(512);
        let z = random_tensor(512);
        b.iter(|| black_box(&x).mulacc(black_box(&y), black_box(&z)));
    });

    group.bench_function("mulacc_2048", |b| {
        let x = random_tensor(2048);
        let y = random_tensor(2048);
        let z = random_tensor(2048);
        b.iter(|| black_box(&x).mulacc(black_box(&y), black_box(&z)));
    });

    group.finish();
}

// Runs the core tensor-op benchmarks: binary, unary, matmul, reduce, movement, and ternary ops.
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
        super::bench_binary_ops,
        super::bench_unary_ops,
        super::bench_matmul,
        super::bench_reduce_ops,
        super::bench_movement_ops,
        super::bench_ternary_ops,
    );
}
use group::benches;

criterion_main!(benches);

//! GPU vs CPU performance comparison benchmarks
//!
//! These benchmarks are only available when the `gpu` feature is enabled.
//! They compare CPU and GPU performance for key operations.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use volta::{Device, RawTensor, Tensor, TensorOps};

#[cfg(feature = "gpu")]
use volta::{get_gpu_context, gpu_sync};

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

// ===== GPU AVAILABILITY CHECK =====

#[cfg(feature = "gpu")]
fn is_gpu_enabled() -> bool {
    get_gpu_context().is_some()
}

#[cfg(not(feature = "gpu"))]
fn is_gpu_enabled() -> bool {
    false
}

// ===== MATMUL COMPARISON =====

fn bench_matmul_cpu_vs_gpu(c: &mut Criterion) {
    if !is_gpu_enabled() {
        return; // Skip if GPU not available
    }

    let mut group = c.benchmark_group("matmul_cpu_vs_gpu");

    // Large matrices where GPU should shine
    for n in [256, 512, 1024, 2048] {
        let n_ref = &n;
        // CPU version
        group.bench_with_input(BenchmarkId::new("cpu", n), n_ref, |b, s| {
            let a = random_tensor_2d(*s, *s);
            let tensor_b = random_tensor_2d(*s, *s);
            b.iter(|| black_box(&a).matmul(black_box(&tensor_b)))
        });

        // GPU version
        group.bench_with_input(BenchmarkId::new("gpu", n), n_ref, |b, s| {
            let a = random_tensor_2d(*s, *s).to_device(Device::gpu().unwrap());
            let tensor_b = random_tensor_2d(*s, *s).to_device(Device::gpu().unwrap());
            b.iter(|| {
                let result = black_box(&a).matmul(black_box(&tensor_b));
                gpu_sync();
                result
            })
        });
    }

    group.finish();
}

// ===== BINARY OPERATIONS COMPARISON =====

fn bench_binary_ops_cpu_vs_gpu(c: &mut Criterion) {
    if !is_gpu_enabled() {
        return;
    }

    let mut group = c.benchmark_group("binary_ops_cpu_vs_gpu");

    for size in [4096, 16384, 65536] {
        let size_ref = &size;
        // CPU addition
        group.bench_with_input(BenchmarkId::new("cpu_add", size), size_ref, |b, s| {
            let a = random_tensor(*s);
            let tensor_b = random_tensor(*s);
            b.iter(|| black_box(&a).add(black_box(&tensor_b)))
        });

        // GPU addition
        group.bench_with_input(BenchmarkId::new("gpu_add", size), size_ref, |b, s| {
            let a = random_tensor(*s).to_device(Device::gpu().unwrap());
            let tensor_b = random_tensor(*s).to_device(Device::gpu().unwrap());
            b.iter(|| {
                let result = black_box(&a).add(black_box(&tensor_b));
                gpu_sync();
                result
            })
        });

        // CPU multiplication
        group.bench_with_input(BenchmarkId::new("cpu_mul", size), size_ref, |b, s| {
            let a = random_tensor(*s);
            let tensor_b = random_tensor(*s);
            b.iter(|| black_box(&a).elem_mul(black_box(&tensor_b)))
        });

        // GPU multiplication
        group.bench_with_input(BenchmarkId::new("gpu_mul", size), size_ref, |b, s| {
            let a = random_tensor(*s).to_device(Device::gpu().unwrap());
            let tensor_b = random_tensor(*s).to_device(Device::gpu().unwrap());
            b.iter(|| {
                let result = black_box(&a).elem_mul(black_box(&tensor_b));
                gpu_sync();
                result
            })
        });
    }

    group.finish();
}

// ===== UNARY OPERATIONS COMPARISON =====

fn bench_unary_ops_cpu_vs_gpu(c: &mut Criterion) {
    if !is_gpu_enabled() {
        return;
    }

    let mut group = c.benchmark_group("unary_ops_cpu_vs_gpu");

    for size in [4096, 16384, 65536] {
        let size_ref = &size;
        // CPU exp
        group.bench_with_input(BenchmarkId::new("cpu_exp", size), size_ref, |b, s| {
            let a = random_tensor(*s);
            b.iter(|| black_box(&a).exp())
        });

        // GPU exp
        group.bench_with_input(BenchmarkId::new("gpu_exp", size), size_ref, |b, s| {
            let a = random_tensor(*s).to_device(Device::gpu().unwrap());
            b.iter(|| {
                let result = black_box(&a).exp();
                gpu_sync();
                result
            })
        });

        // CPU relu
        group.bench_with_input(BenchmarkId::new("cpu_relu", size), size_ref, |b, s| {
            let a = random_tensor(*s);
            b.iter(|| black_box(&a).relu())
        });

        // GPU relu
        group.bench_with_input(BenchmarkId::new("gpu_relu", size), size_ref, |b, s| {
            let a = random_tensor(*s).to_device(Device::gpu().unwrap());
            b.iter(|| {
                let result = black_box(&a).relu();
                gpu_sync();
                result
            })
        });
    }

    group.finish();
}

// ===== REDUCTION OPERATIONS COMPARISON =====

fn bench_reduce_ops_cpu_vs_gpu(c: &mut Criterion) {
    if !is_gpu_enabled() {
        return;
    }

    let mut group = c.benchmark_group("reduce_ops_cpu_vs_gpu");

    for size in [4096, 16384, 65536] {
        let size_ref = &size;
        // CPU sum
        group.bench_with_input(BenchmarkId::new("cpu_sum", size), size_ref, |b, s| {
            let a = random_tensor(*s);
            b.iter(|| black_box(&a).sum())
        });

        // GPU sum
        group.bench_with_input(BenchmarkId::new("gpu_sum", size), size_ref, |b, s| {
            let a = random_tensor(*s).to_device(Device::gpu().unwrap());
            b.iter(|| {
                let result = black_box(&a).sum();
                gpu_sync();
                result
            })
        });

        // CPU mean
        group.bench_with_input(BenchmarkId::new("cpu_mean", size), size_ref, |b, s| {
            let a = random_tensor(*s);
            b.iter(|| black_box(&a).mean())
        });

        // GPU mean
        group.bench_with_input(BenchmarkId::new("gpu_mean", size), size_ref, |b, s| {
            let a = random_tensor(*s).to_device(Device::gpu().unwrap());
            b.iter(|| {
                let result = black_box(&a).mean();
                gpu_sync();
                result
            })
        });
    }

    group.finish();
}

// ===== MEMORY TRANSFER OVERHEAD =====

fn bench_memory_transfer(c: &mut Criterion) {
    if !is_gpu_enabled() {
        return;
    }

    let mut group = c.benchmark_group("memory_transfer");

    for size in [1024, 4096, 16384, 65536, 262144] {
        let size_ref = &size;
        group.throughput(Throughput::Bytes(size as u64 * 4));

        group.bench_with_input(BenchmarkId::new("cpu_to_gpu", size), size_ref, |b, s| {
            let a = random_tensor(*s);
            b.iter(|| {
                let result = black_box(&a).to_device(Device::gpu().unwrap());
                gpu_sync(); // Ensure transfer completes for accurate timing
                result
            })
        });
    }

    group.finish();
}

// ===== GPU BATCH PROCESSING =====

fn bench_gpu_batch_processing(c: &mut Criterion) {
    if !is_gpu_enabled() {
        return;
    }

    let mut group = c.benchmark_group("gpu_batch_processing");
    group.sample_size(10); // Reduce from 100 to 10 samples to prevent memory exhaustion

    // Compare processing many small operations vs one large operation
    // This demonstrates GPU kernel dispatch overhead: launching many small
    // kernels is much slower than one large kernel with the same total work.
    // Reduced to 20 tensors to prevent memory exhaustion during benchmarking.
    group.bench_function("many_small_ops", |b| {
        let tensors: Vec<_> =
            (0..20) // Reduced from 100 to 20 tensors
                .map(|_| random_tensor(256).to_device(Device::gpu().unwrap()))
                .collect();

        b.iter(|| {
            for t in &tensors {
                black_box(&t).relu();
            }
            // Sync to ensure GPU work completes and timing is accurate
            gpu_sync();
        })
    });

    group.bench_function("single_large_op", |b| {
        let large_tensor = random_tensor(5120).to_device(Device::gpu().unwrap()); // 20 × 256
        b.iter(|| {
            black_box(&large_tensor).relu();
            // Sync to ensure GPU work completes and timing is accurate
            gpu_sync();
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_matmul_cpu_vs_gpu,
    bench_binary_ops_cpu_vs_gpu,
    bench_unary_ops_cpu_vs_gpu,
    bench_reduce_ops_cpu_vs_gpu,
    bench_memory_transfer,
    bench_gpu_batch_processing,
);

criterion_main!(benches);

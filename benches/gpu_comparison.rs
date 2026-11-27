//! GPU vs CPU performance comparison benchmarks
//!
//! These benchmarks are only available when the `gpu` feature is enabled.
//! They compare CPU and GPU performance for key operations.
use criterion::{BenchmarkId, Criterion, Throughput, criterion_main};
use std::hint::black_box;
use volta::{Device, RawTensor, Tensor};

#[cfg(feature = "gpu")]
use volta::gpu::invalidate_all_tensor_caches;
#[cfg(feature = "gpu")]
use volta::gpu::{
    gpu_compact, gpu_pending_count, gpu_pool_stats, gpu_sync, gpu_sync_threshold, is_gpu_available,
};

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

/// Whether a GPU context could be created, so the benches can skip themselves when it could not.
#[cfg(feature = "gpu")]
fn is_gpu_enabled() -> bool {
    is_gpu_available()
}

/// Panics if the GPU's queued-submission backlog has outgrown the back-pressure threshold, or if
/// draining it times out, so a benchmark never starts or ends on a runaway queue. `when`
/// names the point in the benchmark for the message.
#[cfg(feature = "gpu")]
fn assert_gpu_drained(when: &str) {
    let pending = gpu_pending_count();
    let threshold = gpu_sync_threshold();
    assert!(
        pending <= threshold,
        "{pending} GPU submissions pending {when}, above the back-pressure threshold {threshold}"
    );
    assert!(gpu_sync(), "GPU sync timed out {when}");
}

/// Always `false`: without the `gpu` feature there is no GPU context to benchmark.
#[cfg(not(feature = "gpu"))]
fn is_gpu_enabled() -> bool {
    false
}

// ===== MATMUL COMPARISON =====

/// Square `n x n` matmul on CPU vs GPU for `n` in 256..=2048. Skipped when no GPU is available.
fn bench_matmul_cpu_vs_gpu(c: &mut Criterion) {
    if !is_gpu_enabled() {
        return; // Skip if GPU not available
    }

    assert_gpu_drained("before the benchmark");

    let mut group = c.benchmark_group("matmul_cpu_vs_gpu");

    // Track setup tensors for cache cleanup
    let mut setup_tensors: Vec<Tensor> = Vec::new();

    // Large matrices where GPU should shine
    for n in [256, 512, 1024, 2048] {
        // Fewer samples as the work grows (2048^3 is ~512x the flops of 256^3), so the heaviest
        // case cannot hold the machine for minutes. Criterion's minimum is 10.
        group.sample_size(if n >= 1024 { 10 } else { 30 });
        let n_ref = &n;
        // CPU version
        group.bench_with_input(BenchmarkId::new("cpu", n), n_ref, |b, s| {
            let a = random_tensor_2d(*s, *s);
            let tensor_b = random_tensor_2d(*s, *s);
            b.iter(|| black_box(&a).matmul(black_box(&tensor_b)));
        });

        // GPU version
        group.bench_with_input(BenchmarkId::new("gpu", n), n_ref, |b, s| {
            let a = random_tensor_2d(*s, *s).to_device(Device::gpu().unwrap());
            let tensor_b = random_tensor_2d(*s, *s).to_device(Device::gpu().unwrap());
            // Capture setup tensors for cleanup
            setup_tensors.push(a.clone());
            setup_tensors.push(tensor_b.clone());
            b.iter(|| {
                let result = black_box(&a).matmul(black_box(&tensor_b));
                assert!(gpu_sync(), "GPU sync timed out inside a timed iteration");
                result
            });
        });
    }

    group.finish();

    // Clear CPU caches on setup tensors to free memory
    if !setup_tensors.is_empty() {
        let tensor_refs: Vec<&Tensor> = setup_tensors.iter().collect();
        invalidate_all_tensor_caches(&tensor_refs);
        println!(
            "[Cleanup] Cleared CPU caches for {} setup tensors",
            tensor_refs.len()
        );
    }
    drop(setup_tensors); // Release tensor references

    // Explicit GPU cleanup with memory release
    println!("[Cleanup] Syncing GPU after matmul_cpu_vs_gpu...");
    if !gpu_sync() {
        eprintln!("WARNING: GPU sync timeout after matmul_cpu_vs_gpu");
    }

    // Clear buffer pools to release GPU memory
    if let Some((buffers, staging)) = gpu_pool_stats() {
        println!("[Cleanup] Clearing pools (buffers: {buffers}, staging: {staging})");
    }
    let _ = gpu_compact();

    // Cooldown period to allow GPU to fully drain command queues
    // This prevents accumulated stress from affecting subsequent benchmarks
    println!("[Cleanup] Cooldown period (500ms) to allow GPU recovery...");
    std::thread::sleep(std::time::Duration::from_millis(500));

    assert_gpu_drained("after cleanup");
}

// ===== BINARY OPERATIONS COMPARISON =====

/// Elementwise `add` and `mul` on CPU vs GPU over 4096, 16384, and 65536 elements. Skipped when no GPU is available.
fn bench_binary_ops_cpu_vs_gpu(c: &mut Criterion) {
    if !is_gpu_enabled() {
        return;
    }

    assert_gpu_drained("before the benchmark");

    let mut group = c.benchmark_group("binary_ops_cpu_vs_gpu");

    // Track setup tensors for cache cleanup
    let mut setup_tensors: Vec<Tensor> = Vec::new();

    for size in [4096, 16384, 65536] {
        let size_ref = &size;
        // CPU addition
        group.bench_with_input(BenchmarkId::new("cpu_add", size), size_ref, |b, s| {
            let a = random_tensor(*s);
            let tensor_b = random_tensor(*s);
            b.iter(|| black_box(&a).add(black_box(&tensor_b)));
        });

        // GPU addition
        group.bench_with_input(BenchmarkId::new("gpu_add", size), size_ref, |b, s| {
            let a = random_tensor(*s).to_device(Device::gpu().unwrap());
            let tensor_b = random_tensor(*s).to_device(Device::gpu().unwrap());
            // Capture setup tensors for cleanup
            setup_tensors.push(a.clone());
            setup_tensors.push(tensor_b.clone());
            b.iter(|| {
                let result = black_box(&a).add(black_box(&tensor_b));
                assert!(gpu_sync(), "GPU sync timed out inside a timed iteration");
                result
            });
        });

        // CPU multiplication
        group.bench_with_input(BenchmarkId::new("cpu_mul", size), size_ref, |b, s| {
            let a = random_tensor(*s);
            let tensor_b = random_tensor(*s);
            b.iter(|| black_box(&a).elem_mul(black_box(&tensor_b)));
        });

        // GPU multiplication
        group.bench_with_input(BenchmarkId::new("gpu_mul", size), size_ref, |b, s| {
            let a = random_tensor(*s).to_device(Device::gpu().unwrap());
            let tensor_b = random_tensor(*s).to_device(Device::gpu().unwrap());
            // Capture setup tensors for cleanup
            setup_tensors.push(a.clone());
            setup_tensors.push(tensor_b.clone());
            b.iter(|| {
                let result = black_box(&a).elem_mul(black_box(&tensor_b));
                assert!(gpu_sync(), "GPU sync timed out inside a timed iteration");
                result
            });
        });
    }

    group.finish();

    // Clear CPU caches on setup tensors to free memory
    if !setup_tensors.is_empty() {
        let tensor_refs: Vec<&Tensor> = setup_tensors.iter().collect();
        invalidate_all_tensor_caches(&tensor_refs);
        println!(
            "[Cleanup] Cleared CPU caches for {} setup tensors",
            tensor_refs.len()
        );
    }
    drop(setup_tensors); // Release tensor references

    // Explicit GPU cleanup with memory release
    println!("[Cleanup] Syncing GPU after binary_ops_cpu_vs_gpu...");
    if !gpu_sync() {
        eprintln!("WARNING: GPU sync timeout after binary_ops_cpu_vs_gpu");
    }

    // Clear buffer pools to release GPU memory
    if let Some((buffers, staging)) = gpu_pool_stats() {
        println!("[Cleanup] Clearing pools (buffers: {buffers}, staging: {staging})");
    }
    let _ = gpu_compact();

    // Brief cooldown to ensure clean state for next benchmark
    std::thread::sleep(std::time::Duration::from_millis(200));

    assert_gpu_drained("after cleanup");
}

// ===== UNARY OPERATIONS COMPARISON =====

/// Elementwise `exp` and `relu` on CPU vs GPU over 4096, 16384, and 65536 elements. Skipped when no GPU is available.
fn bench_unary_ops_cpu_vs_gpu(c: &mut Criterion) {
    if !is_gpu_enabled() {
        return;
    }

    assert_gpu_drained("before the benchmark");

    let mut group = c.benchmark_group("unary_ops_cpu_vs_gpu");

    // Track setup tensors for cache cleanup
    let mut setup_tensors: Vec<Tensor> = Vec::new();

    for size in [4096, 16384, 65536] {
        let size_ref = &size;
        // CPU exp
        group.bench_with_input(BenchmarkId::new("cpu_exp", size), size_ref, |b, s| {
            let a = random_tensor(*s);
            b.iter(|| black_box(&a).exp());
        });

        // GPU exp
        group.bench_with_input(BenchmarkId::new("gpu_exp", size), size_ref, |b, s| {
            let a = random_tensor(*s).to_device(Device::gpu().unwrap());
            // Capture setup tensor for cleanup
            setup_tensors.push(a.clone());
            b.iter(|| {
                let result = black_box(&a).exp();
                assert!(gpu_sync(), "GPU sync timed out inside a timed iteration");
                result
            });
        });

        // CPU relu
        group.bench_with_input(BenchmarkId::new("cpu_relu", size), size_ref, |b, s| {
            let a = random_tensor(*s);
            b.iter(|| black_box(&a).relu());
        });

        // GPU relu
        group.bench_with_input(BenchmarkId::new("gpu_relu", size), size_ref, |b, s| {
            let a = random_tensor(*s).to_device(Device::gpu().unwrap());
            // Capture setup tensor for cleanup
            setup_tensors.push(a.clone());
            b.iter(|| {
                let result = black_box(&a).relu();
                assert!(gpu_sync(), "GPU sync timed out inside a timed iteration");
                result
            });
        });
    }

    group.finish();

    // Clear CPU caches on setup tensors to free memory
    if !setup_tensors.is_empty() {
        let tensor_refs: Vec<&Tensor> = setup_tensors.iter().collect();
        invalidate_all_tensor_caches(&tensor_refs);
        println!(
            "[Cleanup] Cleared CPU caches for {} setup tensors",
            tensor_refs.len()
        );
    }
    drop(setup_tensors); // Release tensor references

    // Explicit GPU cleanup with memory release
    println!("[Cleanup] Syncing GPU after unary_ops_cpu_vs_gpu...");
    if !gpu_sync() {
        eprintln!("WARNING: GPU sync timeout after unary_ops_cpu_vs_gpu");
    }

    // Clear buffer pools to release GPU memory
    if let Some((buffers, staging)) = gpu_pool_stats() {
        println!("[Cleanup] Clearing pools (buffers: {buffers}, staging: {staging})");
    }
    let _ = gpu_compact();

    // Brief cooldown to ensure clean state for next benchmark
    std::thread::sleep(std::time::Duration::from_millis(200));

    assert_gpu_drained("after cleanup");
}

// ===== REDUCTION OPERATIONS COMPARISON =====

/// `sum` and `mean` reductions on CPU vs GPU over 4096, 16384, and 65536 elements. Skipped when no GPU is available.
fn bench_reduce_ops_cpu_vs_gpu(c: &mut Criterion) {
    if !is_gpu_enabled() {
        return;
    }

    assert_gpu_drained("before the benchmark");

    let mut group = c.benchmark_group("reduce_ops_cpu_vs_gpu");

    // Track setup tensors for cache cleanup
    let mut setup_tensors: Vec<Tensor> = Vec::new();

    for size in [4096, 16384, 65536] {
        let size_ref = &size;
        // CPU sum
        group.bench_with_input(BenchmarkId::new("cpu_sum", size), size_ref, |b, s| {
            let a = random_tensor(*s);
            b.iter(|| black_box(&a).sum());
        });

        // GPU sum
        group.bench_with_input(BenchmarkId::new("gpu_sum", size), size_ref, |b, s| {
            let a = random_tensor(*s).to_device(Device::gpu().unwrap());
            // Capture setup tensor for cleanup
            setup_tensors.push(a.clone());
            b.iter(|| {
                let result = black_box(&a).sum();
                assert!(gpu_sync(), "GPU sync timed out inside a timed iteration");
                result
            });
        });

        // CPU mean
        group.bench_with_input(BenchmarkId::new("cpu_mean", size), size_ref, |b, s| {
            let a = random_tensor(*s);
            b.iter(|| black_box(&a).mean());
        });

        // GPU mean
        group.bench_with_input(BenchmarkId::new("gpu_mean", size), size_ref, |b, s| {
            let a = random_tensor(*s).to_device(Device::gpu().unwrap());
            // Capture setup tensor for cleanup
            setup_tensors.push(a.clone());
            b.iter(|| {
                let result = black_box(&a).mean();
                assert!(gpu_sync(), "GPU sync timed out inside a timed iteration");
                result
            });
        });
    }

    group.finish();

    // Clear CPU caches on setup tensors to free memory
    if !setup_tensors.is_empty() {
        let tensor_refs: Vec<&Tensor> = setup_tensors.iter().collect();
        invalidate_all_tensor_caches(&tensor_refs);
        println!(
            "[Cleanup] Cleared CPU caches for {} setup tensors",
            tensor_refs.len()
        );
    }
    drop(setup_tensors); // Release tensor references

    // Explicit GPU cleanup with memory release
    println!("[Cleanup] Syncing GPU after reduce_ops_cpu_vs_gpu...");
    if !gpu_sync() {
        eprintln!("WARNING: GPU sync timeout after reduce_ops_cpu_vs_gpu");
    }

    // Clear buffer pools to release GPU memory
    if let Some((buffers, staging)) = gpu_pool_stats() {
        println!("[Cleanup] Clearing pools (buffers: {buffers}, staging: {staging})");
    }
    let _ = gpu_compact();

    // Brief cooldown to ensure clean state for next benchmark
    std::thread::sleep(std::time::Duration::from_millis(200));

    assert_gpu_drained("after cleanup");
}

// ===== MEMORY TRANSFER OVERHEAD =====

/// Host-to-device copy cost (`to_device` plus a sync) for 1024 to 262144 `f32` elements, reported as bytes per second. Skipped when no GPU is available.
fn bench_memory_transfer(c: &mut Criterion) {
    if !is_gpu_enabled() {
        return;
    }

    assert_gpu_drained("before the benchmark");

    let mut group = c.benchmark_group("memory_transfer");

    for size in [1024, 4096, 16384, 65536, 262_144] {
        let size_ref = &size;
        group.throughput(Throughput::Bytes(size as u64 * 4));

        group.bench_with_input(BenchmarkId::new("cpu_to_gpu", size), size_ref, |b, s| {
            let a = random_tensor(*s);
            b.iter(|| {
                let result = black_box(&a).to_device(Device::gpu().unwrap());
                assert!(gpu_sync(), "GPU sync timed out inside a timed iteration"); // Ensure transfer completes for accurate timing
                result
            });
        });
    }

    group.finish();

    // Explicit GPU cleanup with memory release
    println!("[Cleanup] Syncing GPU after memory_transfer...");
    if !gpu_sync() {
        eprintln!("WARNING: GPU sync timeout after memory_transfer");
    }

    // Clear buffer pools to release GPU memory
    if let Some((buffers, staging)) = gpu_pool_stats() {
        println!("[Cleanup] Clearing pools (buffers: {buffers}, staging: {staging})");
    }
    let _ = gpu_compact();

    // Brief cooldown to ensure clean state for next benchmark
    std::thread::sleep(std::time::Duration::from_millis(200));

    assert_gpu_drained("after cleanup");
}

// ===== GPU BATCH PROCESSING =====

/// Kernel-dispatch overhead on the GPU: `many_small_ops` runs an op over 20 tensors of 256 elements, `single_large_op` runs it once over one 5120-element tensor (the same total work). Uses 10 samples to avoid exhausting memory. Skipped when no GPU is available.
#[cfg(feature = "gpu")]
fn bench_gpu_batch_processing(c: &mut Criterion) {
    if !is_gpu_enabled() {
        return;
    }

    assert_gpu_drained("before the benchmark");

    let mut group = c.benchmark_group("gpu_batch_processing");
    group.sample_size(10); // Reduce from 100 to 10 samples to prevent memory exhaustion

    // Track setup tensors for cache cleanup
    let mut setup_tensors: Vec<Tensor> = Vec::new();

    // Compare processing many small operations vs one large operation
    // This demonstrates GPU kernel dispatch overhead: launching many small
    // kernels is much slower than one large kernel with the same total work.
    // Reduced to 20 tensors to prevent memory exhaustion during benchmarking.
    group.bench_function("many_small_ops", |b| {
        let tensors: Vec<_> =
            (0..20) // Reduced from 100 to 20 tensors
                .map(|_| random_tensor(256).to_device(Device::gpu().unwrap()))
                .collect();

        // Capture setup tensors for cleanup
        for t in &tensors {
            setup_tensors.push(t.clone());
        }

        // Sync after tensor setup to start with clean GPU state
        if !gpu_sync() {
            eprintln!("Warning: GPU sync timed out after tensor setup");
        }

        b.iter(|| {
            for t in &tensors {
                black_box(black_box(&t).relu());
            }

            // Sync to ensure GPU work completes and timing is accurate
            assert!(gpu_sync(), "GPU sync timed out inside a timed iteration");
        });
    });

    group.bench_function("single_large_op", |b| {
        let large_tensor = random_tensor(5120).to_device(Device::gpu().unwrap()); // 20 × 256

        // Capture setup tensor for cleanup
        setup_tensors.push(large_tensor.clone());

        // Sync after tensor setup to start with clean GPU state
        if !gpu_sync() {
            eprintln!("Warning: GPU sync timed out after tensor setup");
        }

        b.iter(|| {
            black_box(black_box(&large_tensor).relu());

            // Sync to ensure GPU work completes and timing is accurate
            assert!(gpu_sync(), "GPU sync timed out inside a timed iteration");
        });
    });

    group.finish();

    // Clear CPU caches on setup tensors to free memory
    if !setup_tensors.is_empty() {
        let tensor_refs: Vec<&Tensor> = setup_tensors.iter().collect();
        invalidate_all_tensor_caches(&tensor_refs);
        println!(
            "[Cleanup] Cleared CPU caches for {} setup tensors",
            tensor_refs.len()
        );
    }
    drop(setup_tensors); // Release tensor references

    // Explicit GPU cleanup with memory release
    println!("[Cleanup] Syncing GPU after gpu_batch_processing...");
    if !gpu_sync() {
        eprintln!("WARNING: GPU sync timeout after gpu_batch_processing");
    }

    // Clear buffer pools to release GPU memory
    if let Some((buffers, staging)) = gpu_pool_stats() {
        println!("[Cleanup] Clearing pools (buffers: {buffers}, staging: {staging})");
    }
    let _ = gpu_compact();

    // Brief cooldown to ensure clean state for next benchmark
    std::thread::sleep(std::time::Duration::from_millis(200));

    assert_gpu_drained("after cleanup");
}

/// No-op: without the `gpu` feature there is nothing to benchmark.
#[cfg(not(feature = "gpu"))]
fn bench_gpu_batch_processing(_c: &mut Criterion) {
    // No-op when GPU feature is disabled
}

// Runs the CPU-vs-GPU benchmarks, lightest first and the 2048x2048 matmul last. The GPU groups skip themselves when no GPU is available.
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
        super::bench_gpu_batch_processing, // Lightest: 20 tensors × 256 elements
        super::bench_binary_ops_cpu_vs_gpu,
        super::bench_unary_ops_cpu_vs_gpu,
        super::bench_reduce_ops_cpu_vs_gpu,
        super::bench_memory_transfer,
        super::bench_matmul_cpu_vs_gpu, // Heaviest: 2048×2048 matrices, run last
    );
}
use group::benches;

criterion_main!(benches);

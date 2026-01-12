//! GPU vs CPU performance comparison benchmarks
//!
//! These benchmarks are only available when the `gpu` feature is enabled.
//! They compare CPU and GPU performance for key operations.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use volta::{Device, RawTensor, Tensor, TensorOps};

#[cfg(feature = "gpu")]
use volta::{get_gpu_context, gpu_compact, gpu_pool_stats, gpu_sync};

#[cfg(feature = "gpu")]
use volta::gpu::monitor::{ResourceStatus, check_system_resources, get_process_memory_mb};

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

    // Reset timeout counter to isolate this benchmark from previous state
    if let Some(ctx) = get_gpu_context() {
        ctx.reset_timeout_counter();
    }

    // Pre-flight resource check
    match check_system_resources() {
        ResourceStatus::Critical(msg) => {
            eprintln!("CRITICAL: Cannot run matmul benchmarks - {}", msg);
            eprintln!("Skipping matmul_cpu_vs_gpu to avoid system freeze");
            return;
        }
        ResourceStatus::Warning(msg) => {
            eprintln!(
                "WARNING: Starting matmul benchmarks with elevated resources: {}",
                msg
            );
        }
        ResourceStatus::Healthy => {
            println!(
                "Resources healthy. Starting matmul_cpu_vs_gpu (Memory: {}MB)",
                get_process_memory_mb()
            );
        }
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

    // Post-flight check
    let final_memory = get_process_memory_mb();
    match check_system_resources() {
        ResourceStatus::Critical(msg) => {
            eprintln!(
                "CRITICAL after matmul_cpu_vs_gpu: {} (Memory: {}MB)",
                msg, final_memory
            );
        }
        ResourceStatus::Warning(msg) => {
            eprintln!(
                "Warning after matmul_cpu_vs_gpu: {} (Memory: {}MB)",
                msg, final_memory
            );
        }
        ResourceStatus::Healthy => {
            println!(
                "matmul_cpu_vs_gpu complete. Final memory: {}MB",
                final_memory
            );
        }
    }

    // Explicit GPU cleanup with memory release
    println!("[Cleanup] Syncing GPU after matmul_cpu_vs_gpu...");
    if !gpu_sync() {
        eprintln!("WARNING: GPU sync timeout after matmul_cpu_vs_gpu");
    }

    // Clear buffer pools to release GPU memory
    if let Some((buffers, staging)) = gpu_pool_stats() {
        println!(
            "[Cleanup] Clearing pools (buffers: {}, staging: {})",
            buffers, staging
        );
    }
    gpu_compact();

    // Cooldown period to allow GPU to fully drain command queues
    // This prevents accumulated stress from affecting subsequent benchmarks
    println!("[Cleanup] Cooldown period (500ms) to allow GPU recovery...");
    std::thread::sleep(std::time::Duration::from_millis(500));

    // Log memory after cleanup
    println!(
        "[Cleanup] Post-cleanup memory: {}MB",
        get_process_memory_mb()
    );

    // Verify cleanup succeeded
    if let ResourceStatus::Critical(msg) = check_system_resources() {
        panic!("CRITICAL after cleanup: {}", msg);
    }
}

// ===== BINARY OPERATIONS COMPARISON =====

fn bench_binary_ops_cpu_vs_gpu(c: &mut Criterion) {
    if !is_gpu_enabled() {
        return;
    }

    // Reset timeout counter to isolate this benchmark from previous state
    if let Some(ctx) = get_gpu_context() {
        ctx.reset_timeout_counter();
    }

    // Pre-flight resource check
    match check_system_resources() {
        ResourceStatus::Critical(msg) => {
            eprintln!("CRITICAL: Cannot run binary ops benchmarks - {}", msg);
            eprintln!("Skipping binary_ops_cpu_vs_gpu to avoid system freeze");
            return;
        }
        ResourceStatus::Warning(msg) => {
            eprintln!(
                "WARNING: Starting binary ops benchmarks with elevated resources: {}",
                msg
            );
        }
        ResourceStatus::Healthy => {
            println!(
                "Resources healthy. Starting binary_ops_cpu_vs_gpu (Memory: {}MB)",
                get_process_memory_mb()
            );
        }
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

    // Post-flight check
    let final_memory = get_process_memory_mb();
    match check_system_resources() {
        ResourceStatus::Critical(msg) => {
            eprintln!(
                "CRITICAL after binary_ops_cpu_vs_gpu: {} (Memory: {}MB)",
                msg, final_memory
            );
        }
        ResourceStatus::Warning(msg) => {
            eprintln!(
                "Warning after binary_ops_cpu_vs_gpu: {} (Memory: {}MB)",
                msg, final_memory
            );
        }
        ResourceStatus::Healthy => {
            println!(
                "binary_ops_cpu_vs_gpu complete. Final memory: {}MB",
                final_memory
            );
        }
    }

    // Explicit GPU cleanup with memory release
    println!("[Cleanup] Syncing GPU after binary_ops_cpu_vs_gpu...");
    if !gpu_sync() {
        eprintln!("WARNING: GPU sync timeout after binary_ops_cpu_vs_gpu");
    }

    // Clear buffer pools to release GPU memory
    if let Some((buffers, staging)) = gpu_pool_stats() {
        println!(
            "[Cleanup] Clearing pools (buffers: {}, staging: {})",
            buffers, staging
        );
    }
    gpu_compact();

    // Brief cooldown to ensure clean state for next benchmark
    std::thread::sleep(std::time::Duration::from_millis(200));

    // Log memory after cleanup
    println!(
        "[Cleanup] Post-cleanup memory: {}MB",
        get_process_memory_mb()
    );

    // Verify cleanup succeeded
    if let ResourceStatus::Critical(msg) = check_system_resources() {
        panic!("CRITICAL after cleanup: {}", msg);
    }
}

// ===== UNARY OPERATIONS COMPARISON =====

fn bench_unary_ops_cpu_vs_gpu(c: &mut Criterion) {
    if !is_gpu_enabled() {
        return;
    }

    // Reset timeout counter to isolate this benchmark from previous state
    if let Some(ctx) = get_gpu_context() {
        ctx.reset_timeout_counter();
    }

    // Pre-flight resource check
    match check_system_resources() {
        ResourceStatus::Critical(msg) => {
            eprintln!("CRITICAL: Cannot run unary ops benchmarks - {}", msg);
            eprintln!("Skipping unary_ops_cpu_vs_gpu to avoid system freeze");
            return;
        }
        ResourceStatus::Warning(msg) => {
            eprintln!(
                "WARNING: Starting unary ops benchmarks with elevated resources: {}",
                msg
            );
        }
        ResourceStatus::Healthy => {
            println!(
                "Resources healthy. Starting unary_ops_cpu_vs_gpu (Memory: {}MB)",
                get_process_memory_mb()
            );
        }
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

    // Post-flight check
    let final_memory = get_process_memory_mb();
    match check_system_resources() {
        ResourceStatus::Critical(msg) => {
            eprintln!(
                "CRITICAL after unary_ops_cpu_vs_gpu: {} (Memory: {}MB)",
                msg, final_memory
            );
        }
        ResourceStatus::Warning(msg) => {
            eprintln!(
                "Warning after unary_ops_cpu_vs_gpu: {} (Memory: {}MB)",
                msg, final_memory
            );
        }
        ResourceStatus::Healthy => {
            println!(
                "unary_ops_cpu_vs_gpu complete. Final memory: {}MB",
                final_memory
            );
        }
    }

    // Explicit GPU cleanup with memory release
    println!("[Cleanup] Syncing GPU after unary_ops_cpu_vs_gpu...");
    if !gpu_sync() {
        eprintln!("WARNING: GPU sync timeout after unary_ops_cpu_vs_gpu");
    }

    // Clear buffer pools to release GPU memory
    if let Some((buffers, staging)) = gpu_pool_stats() {
        println!(
            "[Cleanup] Clearing pools (buffers: {}, staging: {})",
            buffers, staging
        );
    }
    gpu_compact();

    // Brief cooldown to ensure clean state for next benchmark
    std::thread::sleep(std::time::Duration::from_millis(200));

    // Log memory after cleanup
    println!(
        "[Cleanup] Post-cleanup memory: {}MB",
        get_process_memory_mb()
    );

    // Verify cleanup succeeded
    if let ResourceStatus::Critical(msg) = check_system_resources() {
        panic!("CRITICAL after cleanup: {}", msg);
    }
}

// ===== REDUCTION OPERATIONS COMPARISON =====

fn bench_reduce_ops_cpu_vs_gpu(c: &mut Criterion) {
    if !is_gpu_enabled() {
        return;
    }

    // Reset timeout counter to isolate this benchmark from previous state
    if let Some(ctx) = get_gpu_context() {
        ctx.reset_timeout_counter();
    }

    // Pre-flight resource check
    match check_system_resources() {
        ResourceStatus::Critical(msg) => {
            eprintln!("CRITICAL: Cannot run reduce ops benchmarks - {}", msg);
            eprintln!("Skipping reduce_ops_cpu_vs_gpu to avoid system freeze");
            return;
        }
        ResourceStatus::Warning(msg) => {
            eprintln!(
                "WARNING: Starting reduce ops benchmarks with elevated resources: {}",
                msg
            );
        }
        ResourceStatus::Healthy => {
            println!(
                "Resources healthy. Starting reduce_ops_cpu_vs_gpu (Memory: {}MB)",
                get_process_memory_mb()
            );
        }
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

    // Post-flight check
    let final_memory = get_process_memory_mb();
    match check_system_resources() {
        ResourceStatus::Critical(msg) => {
            eprintln!(
                "CRITICAL after reduce_ops_cpu_vs_gpu: {} (Memory: {}MB)",
                msg, final_memory
            );
        }
        ResourceStatus::Warning(msg) => {
            eprintln!(
                "Warning after reduce_ops_cpu_vs_gpu: {} (Memory: {}MB)",
                msg, final_memory
            );
        }
        ResourceStatus::Healthy => {
            println!(
                "reduce_ops_cpu_vs_gpu complete. Final memory: {}MB",
                final_memory
            );
        }
    }

    // Explicit GPU cleanup with memory release
    println!("[Cleanup] Syncing GPU after reduce_ops_cpu_vs_gpu...");
    if !gpu_sync() {
        eprintln!("WARNING: GPU sync timeout after reduce_ops_cpu_vs_gpu");
    }

    // Clear buffer pools to release GPU memory
    if let Some((buffers, staging)) = gpu_pool_stats() {
        println!(
            "[Cleanup] Clearing pools (buffers: {}, staging: {})",
            buffers, staging
        );
    }
    gpu_compact();

    // Brief cooldown to ensure clean state for next benchmark
    std::thread::sleep(std::time::Duration::from_millis(200));

    // Log memory after cleanup
    println!(
        "[Cleanup] Post-cleanup memory: {}MB",
        get_process_memory_mb()
    );

    // Verify cleanup succeeded
    if let ResourceStatus::Critical(msg) = check_system_resources() {
        panic!("CRITICAL after cleanup: {}", msg);
    }
}

// ===== MEMORY TRANSFER OVERHEAD =====

fn bench_memory_transfer(c: &mut Criterion) {
    if !is_gpu_enabled() {
        return;
    }

    // Reset timeout counter to isolate this benchmark from previous state
    if let Some(ctx) = get_gpu_context() {
        ctx.reset_timeout_counter();
    }

    // Pre-flight resource check
    match check_system_resources() {
        ResourceStatus::Critical(msg) => {
            eprintln!("CRITICAL: Cannot run memory transfer benchmarks - {}", msg);
            eprintln!("Skipping memory_transfer to avoid system freeze");
            return;
        }
        ResourceStatus::Warning(msg) => {
            eprintln!(
                "WARNING: Starting memory transfer benchmarks with elevated resources: {}",
                msg
            );
        }
        ResourceStatus::Healthy => {
            println!(
                "Resources healthy. Starting memory_transfer (Memory: {}MB)",
                get_process_memory_mb()
            );
        }
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

    // Post-flight check
    let final_memory = get_process_memory_mb();
    match check_system_resources() {
        ResourceStatus::Critical(msg) => {
            eprintln!(
                "CRITICAL after memory_transfer: {} (Memory: {}MB)",
                msg, final_memory
            );
        }
        ResourceStatus::Warning(msg) => {
            eprintln!(
                "Warning after memory_transfer: {} (Memory: {}MB)",
                msg, final_memory
            );
        }
        ResourceStatus::Healthy => {
            println!("memory_transfer complete. Final memory: {}MB", final_memory);
        }
    }

    // Explicit GPU cleanup with memory release
    println!("[Cleanup] Syncing GPU after memory_transfer...");
    if !gpu_sync() {
        eprintln!("WARNING: GPU sync timeout after memory_transfer");
    }

    // Clear buffer pools to release GPU memory
    if let Some((buffers, staging)) = gpu_pool_stats() {
        println!(
            "[Cleanup] Clearing pools (buffers: {}, staging: {})",
            buffers, staging
        );
    }
    gpu_compact();

    // Brief cooldown to ensure clean state for next benchmark
    std::thread::sleep(std::time::Duration::from_millis(200));

    // Log memory after cleanup
    println!(
        "[Cleanup] Post-cleanup memory: {}MB",
        get_process_memory_mb()
    );

    // Verify cleanup succeeded
    if let ResourceStatus::Critical(msg) = check_system_resources() {
        panic!("CRITICAL after cleanup: {}", msg);
    }
}

// ===== GPU BATCH PROCESSING =====

#[cfg(feature = "gpu")]
fn bench_gpu_batch_processing(c: &mut Criterion) {
    if !is_gpu_enabled() {
        return;
    }

    // Reset timeout counter to isolate this benchmark from previous state
    if let Some(ctx) = get_gpu_context() {
        ctx.reset_timeout_counter();
    }

    // Pre-flight resource check
    match check_system_resources() {
        ResourceStatus::Critical(msg) => {
            eprintln!("CRITICAL: Cannot run GPU benchmarks - {}", msg);
            eprintln!("Skipping gpu_batch_processing to avoid system freeze");
            return;
        }
        ResourceStatus::Warning(msg) => {
            eprintln!(
                "WARNING: Starting benchmarks with elevated resources: {}",
                msg
            );
        }
        ResourceStatus::Healthy => {
            println!(
                "Resources healthy. Starting GPU benchmarks (Memory: {}MB)",
                get_process_memory_mb()
            );
        }
    }

    let mut group = c.benchmark_group("gpu_batch_processing");
    group.sample_size(10); // Reduce from 100 to 10 samples to prevent memory exhaustion

    // Compare processing many small operations vs one large operation
    // This demonstrates GPU kernel dispatch overhead: launching many small
    // kernels is much slower than one large kernel with the same total work.
    // Reduced to 20 tensors to prevent memory exhaustion during benchmarking.
    group.bench_function("many_small_ops", |b| {
        // Check resources before setup
        if let ResourceStatus::Critical(msg) = check_system_resources() {
            panic!("Benchmark aborted before setup - {}", msg);
        }

        let tensors: Vec<_> =
            (0..20) // Reduced from 100 to 20 tensors
                .map(|_| random_tensor(256).to_device(Device::gpu().unwrap()))
                .collect();

        // Sync after tensor setup to start with clean GPU state
        if !gpu_sync() {
            eprintln!("Warning: GPU sync timed out after tensor setup");
        }

        // Check resources after setup
        match check_system_resources() {
            ResourceStatus::Critical(msg) => {
                panic!("Benchmark aborted after setup - {}", msg);
            }
            ResourceStatus::Warning(msg) => {
                eprintln!("Warning after setup: {}", msg);
            }
            _ => {}
        }

        b.iter(|| {
            for t in &tensors {
                black_box(&t).relu();
            }

            // Sync to ensure GPU work completes and timing is accurate
            if !gpu_sync() {
                eprintln!("Warning: GPU sync timed out during benchmark");
            }

            // Check resources after each iteration
            if let ResourceStatus::Critical(msg) = check_system_resources() {
                panic!("Benchmark aborted during iteration - {}", msg);
            }
        })
    });

    group.bench_function("single_large_op", |b| {
        // Check resources before setup
        if let ResourceStatus::Critical(msg) = check_system_resources() {
            panic!("Benchmark aborted before setup - {}", msg);
        }

        let large_tensor = random_tensor(5120).to_device(Device::gpu().unwrap()); // 20 × 256

        // Sync after tensor setup to start with clean GPU state
        if !gpu_sync() {
            eprintln!("Warning: GPU sync timed out after tensor setup");
        }

        // Check resources after setup
        match check_system_resources() {
            ResourceStatus::Critical(msg) => {
                panic!("Benchmark aborted after setup - {}", msg);
            }
            ResourceStatus::Warning(msg) => {
                eprintln!("Warning after setup: {}", msg);
            }
            _ => {}
        }

        b.iter(|| {
            black_box(&large_tensor).relu();

            // Sync to ensure GPU work completes and timing is accurate
            if !gpu_sync() {
                eprintln!("Warning: GPU sync timed out during benchmark");
            }

            // Check resources after each iteration
            if let ResourceStatus::Critical(msg) = check_system_resources() {
                panic!("Benchmark aborted during iteration - {}", msg);
            }
        })
    });

    group.finish();

    // Final resource check
    let final_memory = get_process_memory_mb();
    match check_system_resources() {
        ResourceStatus::Critical(msg) => {
            eprintln!(
                "CRITICAL after benchmarks: {} (Memory: {}MB)",
                msg, final_memory
            );
        }
        ResourceStatus::Warning(msg) => {
            eprintln!(
                "Warning after benchmarks: {} (Memory: {}MB)",
                msg, final_memory
            );
        }
        ResourceStatus::Healthy => {
            println!("Benchmarks complete. Final memory: {}MB", final_memory);
        }
    }

    // Explicit GPU cleanup with memory release
    println!("[Cleanup] Syncing GPU after gpu_batch_processing...");
    if !gpu_sync() {
        eprintln!("WARNING: GPU sync timeout after gpu_batch_processing");
    }

    // Clear buffer pools to release GPU memory
    if let Some((buffers, staging)) = gpu_pool_stats() {
        println!(
            "[Cleanup] Clearing pools (buffers: {}, staging: {})",
            buffers, staging
        );
    }
    gpu_compact();

    // Brief cooldown to ensure clean state for next benchmark
    std::thread::sleep(std::time::Duration::from_millis(200));

    // Log memory after cleanup
    println!(
        "[Cleanup] Post-cleanup memory: {}MB",
        get_process_memory_mb()
    );

    // Verify cleanup succeeded
    if let ResourceStatus::Critical(msg) = check_system_resources() {
        panic!("CRITICAL after cleanup: {}", msg);
    }
}

#[cfg(not(feature = "gpu"))]
fn bench_gpu_batch_processing(_c: &mut Criterion) {
    // No-op when GPU feature is disabled
}

criterion_group!(
    benches,
    bench_gpu_batch_processing, // Lightest: 20 tensors × 256 elements
    bench_binary_ops_cpu_vs_gpu,
    bench_unary_ops_cpu_vs_gpu,
    bench_reduce_ops_cpu_vs_gpu,
    bench_memory_transfer,
    bench_matmul_cpu_vs_gpu, // Heaviest: 2048×2048 matrices, run last
);

criterion_main!(benches);

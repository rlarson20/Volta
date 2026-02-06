# GPU Batch Processing Benchmark Analysis

## Executive Summary

The `gpu_batch_processing` benchmark works correctly when run in isolation but throws `sync()` errors when run as part of the full `gpu_comparison` benchmark suite. This document analyzes the root causes and identifies the architectural issues involved.

---

## Benchmark Overview

### Location
- **File**: `benches/gpu_comparison.rs:535-686`
- **Feature Gate**: `#[cfg(feature = "gpu")]`

### Structure
The benchmark contains two sub-benchmarks:
1. **`many_small_ops`**: 20 tensors of size 256, each run through `relu()`
2. **`single_large_op`**: 1 tensor of size 5120 (20 × 256), run through `relu()`

### Purpose
Demonstrates GPU kernel dispatch overhead: launching many small kernels is slower than one large kernel with equivalent total work.

---

## The `sync()` Method

### Location
`src/gpu/context.rs:323-363`

### Implementation
```rust
pub fn sync(&self) -> bool {
    let pending = self.pending_submissions.load(Ordering::Relaxed);

    if pending == 0 {
        return true;  // Nothing to sync
    }

    let result = self.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(Duration::from_secs(get_sync_timeout_secs())),  // Default: 2 seconds
    });

    self.pending_submissions.store(0, Ordering::Relaxed);  // Reset regardless of outcome

    match result {
        Ok(_) => true,
        Err(e) => {
            eprintln!("[GPU] Sync timeout warning: {:?}", e);
            false
        }
    }
}
```

### Key Constants
| Constant | Value | Purpose |
|----------|-------|---------|
| `DEFAULT_SYNC_THRESHOLD` | 16 | Auto-sync after N pending submissions |
| `HARD_SYNC_TIMEOUT_SECS` | 2 | Maximum wait time for sync |
| `MAX_CONSECUTIVE_TIMEOUTS` | 3 | Strike count before `GpuLost` error |

### Timeout Configuration
Configurable via environment variable:
```bash
VOLTA_GPU_SYNC_TIMEOUT=5 cargo bench --bench gpu_comparison
```

---

## Why It Works When Run Alone

When you run `gpu_batch_processing` by itself:

```bash
cargo bench --bench gpu_comparison -- gpu_batch_processing
```

1. **Fresh GPU State**: The GPU context initializes without prior work
2. **Minimal Workload**: Only 20 tensors × 256 elements = 5,120 floats total
3. **Sufficient Timeout**: 2 seconds is plenty for this small workload
4. **No State Pollution**: No accumulated command queue from other benchmarks

---

## Why It Fails When Combined

### Execution Order

The `criterion_group!` macro defines execution order (lines 693-701):

```rust
criterion_group!(
    benches,
    bench_matmul_cpu_vs_gpu,      // 1st - Heavy: 2048×2048 matrices
    bench_binary_ops_cpu_vs_gpu,  // 2nd
    bench_unary_ops_cpu_vs_gpu,   // 3rd
    bench_reduce_ops_cpu_vs_gpu,  // 4th
    bench_memory_transfer,        // 5th
    bench_gpu_batch_processing,   // 6th - LAST
);
```

### Root Cause Analysis

#### 1. Shared Global GPU Context

The GPU context is a global singleton (`src/gpu/mod.rs:29-47`):

```rust
static GPU_CONTEXT: OnceLock<Option<GpuContext>> = OnceLock::new();

pub fn get_gpu_context() -> Option<&'static GpuContext> {
    GPU_CONTEXT.get_or_init(|| GpuContext::new().ok())
}
```

**Impact**: All benchmarks share the same device, queue, and internal state.

#### 2. Consecutive Timeout Counter Accumulation

The `GpuContext` tracks consecutive timeouts (`src/gpu/context.rs:74-75`):

```rust
consecutive_timeouts: AtomicU32::new(0),
```

This counter:
- Increments on each sync timeout
- Only resets on successful sync
- **Persists across all benchmark groups**

If earlier benchmarks cause timeouts, the counter accumulates, making `sync_checked()` eventually return `GpuLost`.

#### 3. Heavy Workload from Earlier Benchmarks

The `matmul_cpu_vs_gpu` benchmark (runs first) processes:
- Matrix sizes: 256, 512, 1024, **2048**
- 2048 × 2048 = **4,194,304 float operations per multiplication**

This can leave the GPU overwhelmed, especially on M2 Mac with:
- Unified memory architecture
- Thermal throttling under sustained load
- Shared GPU/CPU memory bandwidth

#### 4. Insufficient Cleanup Between Groups

Each benchmark group ends with cleanup:

```rust
println!("[Cleanup] Syncing GPU after matmul_cpu_vs_gpu...");
if !gpu_sync() {
    eprintln!("WARNING: GPU sync timeout after matmul_cpu_vs_gpu");
}
```

**Problem**: The cleanup only logs a warning if sync times out—it doesn't:
- Wait longer for completion
- Reset internal state
- Provide cooldown time for the GPU

#### 5. Aggressive Timeout for M2 Mac

The 2-second timeout (`HARD_SYNC_TIMEOUT_SECS = 2`) was reduced from 10 seconds to "detect problems faster" (comment at line 18).

**Problem**: M2 Mac's unified memory architecture can be slower to complete GPU work when:
- Memory pressure is high
- Multiple benchmark groups have run
- The GPU is thermally throttled

---

## Resource Monitoring Gaps

### Current Thresholds (`src/gpu/monitor.rs:56-85`)

| Status | Memory Threshold | Pending Ops Threshold |
|--------|-----------------|----------------------|
| Critical | >90% of 24GB | >100 |
| Warning | >70% of 24GB | >50 |
| Healthy | ≤70% | ≤50 |

### Limitations

1. **Memory Detection Assumes 24GB**: Hardcoded for M2 Mac (`src/gpu/monitor.rs:130`)
2. **No GPU Workload Measurement**: Only counts pending submissions, not actual GPU utilization
3. **No Thermal Monitoring**: Can't detect if GPU is throttled
4. **Thresholds Too High**: By the time pending ops hit 100, damage is done

---

## Data Flow During Failure

```
Previous Benchmarks Complete
         ↓
gpu_batch_processing starts
         ↓
Creates 20 GPU tensors
         ↓
gpu_sync() called after setup
         ↓
GPU still processing previous work → Timeout (2s)
         ↓
"Warning: GPU sync timed out after tensor setup"
         ↓
Benchmark iteration runs
         ↓
gpu_sync() called in iteration
         ↓
Timeout again (consecutive_timeouts = 1 or 2)
         ↓
Eventually: "Warning: GPU sync timed out during benchmark"
         ↓
If using sync_checked(): Err(GpuSyncError::Timeout) or Err(GpuSyncError::GpuLost)
```

---

## Key Files and Line References

| File | Lines | Purpose |
|------|-------|---------|
| `benches/gpu_comparison.rs` | 535-686 | `bench_gpu_batch_processing` function |
| `benches/gpu_comparison.rs` | 693-701 | `criterion_group!` ordering |
| `src/gpu/context.rs` | 14-22 | Timeout constants |
| `src/gpu/context.rs` | 323-363 | `sync()` implementation |
| `src/gpu/context.rs` | 427-487 | `sync_checked()` with 3-strike rule |
| `src/gpu/mod.rs` | 29-47 | Global `GPU_CONTEXT` singleton |
| `src/gpu/mod.rs` | 82-84 | `gpu_sync()` wrapper function |
| `src/gpu/monitor.rs` | 56-85 | Resource checking thresholds |

---

## Potential Solutions (Not Implemented)

The following are identified approaches that could address the issues:

### 1. Increase Timeout for M2 Mac
- Increase `HARD_SYNC_TIMEOUT_SECS` from 2 to 5-10 seconds
- Or dynamically adjust based on prior benchmark complexity

### 2. Add Cooldown Between Benchmark Groups
- Insert `std::thread::sleep()` after each group's cleanup
- Allow GPU to fully drain command queues

### 3. Reset Consecutive Timeout Counter
- Call `ctx.reset_timeout_counter()` at the start of each benchmark group
- Prevents state pollution across groups

### 4. Run `gpu_batch_processing` First
- Reorder `criterion_group!` to put lighter benchmarks first
- `gpu_batch_processing` is the lightest GPU benchmark

### 5. Separate Benchmark Binary
- Create a separate `gpu_batch_only.rs` benchmark file
- Ensures complete isolation from other GPU workloads

### 6. Lower Resource Thresholds
- Reduce "Warning" threshold for pending ops from 50 to 20
- Skip benchmarks earlier when GPU is stressed

---

## Environment Variables for Debugging

```bash
# Enable GPU debug logging
VOLTA_GPU_DEBUG=1 cargo bench --bench gpu_comparison

# Increase sync timeout to 10 seconds
VOLTA_GPU_SYNC_TIMEOUT=10 cargo bench --bench gpu_comparison

# Run only gpu_batch_processing
cargo bench --bench gpu_comparison -- gpu_batch_processing
```

---

## Summary

The `gpu_batch_processing` benchmark fails when run with other benchmarks due to:

1. **Shared mutable state** in the global `GpuContext` singleton
2. **Cumulative GPU stress** from earlier heavy benchmarks (especially matmul)
3. **Aggressive 2-second timeout** that's insufficient after prior GPU work
4. **No isolation mechanism** between benchmark groups
5. **Resource monitoring thresholds** that are too permissive

The benchmark itself is correctly implemented—the issue is in the benchmark suite architecture and GPU context management.

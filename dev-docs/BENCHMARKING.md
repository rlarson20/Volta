# Benchmarking Guide

This document provides detailed information about benchmarking in the Volta deep learning framework.

## Safety

`gpu_comparison` used to crash the machine under resource exhaustion. After the buffer-pool and cleanup work it completed in one full run (`just bench-gpu` at `8008088`, M-series Mac, peak process memory 146 MB at the 2048 matmul), so it is no longer marked unsafe. That is a single run, not a guarantee: watch memory the first time on a new machine. The run takes several minutes, so start it in the background.

`conv_algorithms` is designed to stay under a 16GB memory limit.

## GPU results

`just bench-gpu`, commit `8008088`, Criterion median, M-series Mac. Every GPU time includes a sync, so it is latency, not kernel throughput.

| Group | Size | CPU | GPU |
| --- | --- | --- | --- |
| add / mul | 4096 | 5.9 µs | 1.50 ms |
| add / mul | 65536 | 87 µs | 1.6 ms |
| exp | 4096 | 13.6 µs | 1.50 ms |
| exp | 65536 | 217 µs | 1.69 ms |
| sum / mean | 4096 | 6.5 µs | 4.3 ms |
| sum / mean | 65536 | 101 µs | 11 ms |
| matmul | 256 | 70 µs | 3.0 ms |
| matmul | 1024 | 3.0 ms | 37 ms |
| matmul | 2048 | 22 ms | 310 ms |
| `cpu_to_gpu` | 1024 floats | | 1.42 ms (2.7 MiB/s) |
| `cpu_to_gpu` | 262144 floats | | 1.8 ms (549 MiB/s) |

What the numbers say:

- The GPU loses to the CPU at every size measured. Elementwise GPU time is flat at about 1.5 ms from 4096 to 16384 elements and grows slowly after, so the cost is per-op overhead, not the kernel.
- `memory_transfer` is flat near 1.4 ms for the smaller sizes, so the MiB/s figures there are a latency floor divided into the payload, not bandwidth. Only the largest size starts to reflect transfer cost.
- GPU matmul at 2048 is about 55 GFLOP/s, so matmul is slow beyond the overhead floor. This has not been investigated.
- Reductions cost 3x to 7x an elementwise op and grow with size. Not investigated.
- Criterion's "regressed" and "improved" flags compare against whatever baseline was on disk, so they are not a verdict on a code change. The largest flags in this run were `gpu_sum` and `gpu_mean` at 4096 (+43% and +52%) and `many_small_ops` (+16%).

Open follow-ups: [#79](https://github.com/rlarson20/Volta/issues/79) (per-op floor, matmul/sum profiling, transfer sizing) and [#80](https://github.com/rlarson20/Volta/issues/80) (reductions).

## Benchmark Commands

```bash
# Run all benchmarks (except gpu_comparison; use `just bench-gpu`)
just bench

# Run convolution algorithm comparison (SAFE)
just bench-conv

# Quick matmul timing smoke check: N release runs of one test, averaged
just bench-matmul 10                          # extra cargo flags: just bench-matmul 10 --no-default-features

# Run specific benchmark
just bench-name tensor_ops
just bench-name neural_networks
just bench-name gpu_comparison

# CPU-only benchmarks (no acceleration)
just bench-cpu

# With BLAS acceleration (macOS)
just bench-accel

# GPU comparison benchmarks
just bench-gpu

# Save baseline for comparison
just bench-save

# Compare against saved baseline
just bench-compare

# Open HTML report
just bench-report
```

## Convolution Algorithm Benchmarks

The `conv_algorithms` benchmark compares Direct vs im2col+GEMM convolution algorithms.

### Benchmark Groups

1. **`conv_algorithm_comparison`**: Compares algorithms across various scenarios
   - Tiny/small/medium/large inputs
   - Different kernel sizes (1x1, 3x3, 5x5, 7x7)
   - ResNet-style layers

2. **`conv_kernel_comparison`**: Varies kernel size (1-7)
   - Fixed input: 8x64x56x56
   - Shows how kernel size affects performance

3. **`conv_batch_comparison`**: Varies batch size (1-32)
   - Fixed input: 64x56x56
   - Shows scaling with batch size

4. **`conv_spatial_comparison`**: Different spatial dimensions
   - MNIST (28x28), CIFAR (32x32)
   - ImageNet variants (56x56, 112x112, 224x224)

5. **`conv_auto_mode`**: Tests automatic algorithm selection
   - Verifies Auto mode chooses correctly

6. **`conv_memory_scaling`**: Shows memory usage patterns
   - Scales channels, spatial dimensions, and batch size
   - All benchmarks stay under 16GB limit

### Memory Safety

- All scenarios calculate im2col memory usage beforehand
- Maximum memory used: ~1GB (very_large scenario)
- Prints memory usage for each benchmark
- Asserts <16GB limit before running

### Example Output

```
tiny: B=1, C=3→16, H=W=16, K=3x3, im2col memory: 0.02 MB
small: B=4, C=3→16, H=W=32, K=3x3, im2col memory: 0.41 MB
medium: B=8, C=64→128, H=W=28, K=3x3, im2col memory: 67.5 MB
```

## Benchmarking Guidelines

For benchmark changes:

- Always implement resource cleanup (buffer pools, cache invalidation) after benchmark groups
- Use size-bucketed buffer pools to prevent exhaustion
- Add hard sync timeouts for GPU operations

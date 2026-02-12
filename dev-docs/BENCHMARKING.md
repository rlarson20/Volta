# Benchmarking Guide

This document provides detailed information about benchmarking in the Volta deep learning framework.

## Safety Warning

**⚠️ WARNING:** The `gpu_comparison` benchmark is NOT SAFE and will crash your computer.
DO NOT run `just bench-gpu` or `cargo bench --bench gpu_comparison`.

The new `conv_algorithms` benchmark is SAFE and designed to stay under 16GB memory limit.

## Benchmark Commands

```bash
# Run all benchmarks (except gpu_comparison)
just bench

# Run convolution algorithm comparison (SAFE)
just bench-conv

# Run specific benchmark
just bench-name tensor_ops
just bench-name neural_networks
just bench-name gpu_comparison # NOT FOR USE

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

# Conv2D Implementation Details

This document provides detailed information about the 2D convolution implementation in Volta.

## Algorithm Overview

The Conv2d implementation uses algorithm selection in `src/nn/layers/conv.rs` with three available algorithms.

### 1. Direct Convolution

- Computes each output pixel by iterating over kernel directly
- **Memory efficient**: No intermediate allocations
- **Best for**: Small inputs/kernels, memory-constrained scenarios
- **Trade-off**: Slower than im2col/iGEMM for larger inputs
- **✅ Full gradient support**: Input and weight gradients computed directly

### 2. im2col + GEMM

- Materializes intermediate matrix (B*H_out*W_out, C*K*K) then uses GEMM
- **Fast**: Leverages optimized matrix multiplication
- **Best for**: Large kernels, general training
- **Trade-off**: High memory usage (can OOM on large inputs)
- **✅ Full gradient support**: Gradients via col2im and GEMM

### 3. Implicit GEMM (iGEMM)

- Performs GEMM without materializing the im2col matrix
- **Balanced**: Good trade-off between Direct and im2col
- **Best for**: Medium-to-large inputs where memory is a concern
- **Approach**: Tiled computation for cache efficiency
- **✅ Full gradient support**: Tiled gradient computation matches forward pass
- **🔧 Extensible**: Architecture supports future variants (Winograd, FFT, Direct-to-GEMM)

## Algorithm Selection

```rust
// Auto-select based on input characteristics
let conv = Conv2d::new(3, 16, 3, 1, 1, true);
conv.set_algo(ConvAlgo::Auto); // Default behavior

// Force specific algorithm
conv.set_algo(ConvAlgo::Direct);  // Memory efficient
conv.set_algo(ConvAlgo::Im2col);  // Faster but uses more memory
conv.set_algo(ConvAlgo::IGEMM);   // Balanced approach
```

### Auto-selection Logic

- **GPU**: Always uses im2col (GPU-accelerated)
- **CPU small inputs**: Uses direct for small kernels (≤3x3) and small batches (≤4)
- **CPU large inputs (>5M im2col elements)**: Uses direct to save memory
- **CPU medium inputs**: Uses iGEMM for balanced performance/memory
- **All algorithms support training** with full gradient computation

## iGEMM Architecture

The iGEMM implementation is designed with extensibility for future optimizations.

### Available Variants

- **Tiled (current)**: Cache-friendly blocking with configurable tile sizes
- **Winograd (future)**: Transform-domain convolution for 3x3 kernels
- **FFT (future)**: Frequency-domain convolution for large kernels (audio)
- **Direct-to-GEMM (future)**: Hand-tuned micro-kernels for maximum performance

### Adding a New iGEMM Variant

To add a new iGEMM variant:

1. Add variant to `IGEMMVariant` enum in `src/nn/layers/conv.rs`
2. Implement forward/backward in `Conv2d::igemm_forward`
3. Update `Conv2d::igemm_backward` to handle the variant
4. Add tests and benchmarks

## Recent Improvements

- ✅ Direct convolution now supports full gradient computation
- ✅ iGEMM implementation with tiled forward and backward passes
- ✅ Extensible architecture for future convolution algorithms
- ✅ Memory-efficient training enabled for large inputs

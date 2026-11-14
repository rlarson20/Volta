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

## Algorithm Selection

```rust
let conv = Conv2d::new(3, 16, 3, 1, 1, true);

// Auto is the default. To force one algorithm, set it (these are alternatives):
conv.set_algo(ConvAlgo::Auto);
conv.set_algo(ConvAlgo::Direct);  // least memory
conv.set_algo(ConvAlgo::Im2col);  // materializes the im2col matrix
conv.set_algo(ConvAlgo::IGEMM);   // GEMM without the im2col matrix
```

### Auto-selection Logic

`ConvAlgo::auto_select` (`src/nn/layers/conv.rs`) is the source; its rustdoc has the same
table, and `auto_select_follows_documented_rule` checks both sides of each threshold. It
estimates the im2col matrix as

```text
im2col_elements = batch * (H - k + 1) * (W - k + 1) * channels * k * k
```

with `H` and `W` the padded input extents and `k = kernel_size`. The first rule that
matches wins:

| Device | Condition | Result |
| --- | --- | --- |
| GPU | `im2col_elements > 1_000_000` | `IGEMM` |
| GPU | otherwise | `Im2col` |
| CPU | `im2col_elements > 5_000_000` | `Direct` (saves memory) |
| CPU | `kernel_size <= 3 && batch <= 4` | `Direct` |
| CPU | `im2col_elements > 1_000_000` | `IGEMM` |
| CPU | otherwise | `Im2col` |

Two things to know about the heuristic:

- The small-kernel rule looks only at the kernel and the batch, not the input size. A
  `4 x 64 x 32 x 32` input with a 3x3 kernel (about 2.07M elements) runs `Direct`; the same
  input with batch 5 runs `IGEMM`.
- It assumes stride 1 and a square kernel (`Conv2d::forward` passes `kernel_h` only), so a
  strided or non-square convolution is sized as if it were stride 1 and square. A stride-2
  convolution produces about a quarter of the estimated matrix, so the 1M and 5M
  thresholds trip earlier than the real memory use warrants.

All three algorithms support training with full gradients.

## iGEMM

`IGEMMVariant` (private) has one variant, `Tiled`, and the grad fn stores it in an unused
field (`_variant`, #118). The backward is `igemm_backward_input` and
`igemm_backward_weight` on the grad fn. There is no extension point beyond that: a second
variant would need a dispatch added to the forward and both backward functions.

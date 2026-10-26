---
paths:
  - "src/nn/**"
---

# Neural network layers

- Implement `Module`: `forward`, `parameters`, `state_dict`, `load_state_dict`. `parameters()` must return every learnable tensor.
- Build outputs from graph ops (`shrink`, broadcast add, `matmul`, …) or from `edge()`. Never copy values into a fresh tensor, which severs the graph. LSTMCell's gate slice (B2) and PositionalEncoding (B3) both still do this.
- **Every parameterized layer gets an `assert_trainable!` test with random input** (`randn` plus `set_requires_grad(true)`). All-ones input makes gradients mathematically zero for LayerNorm, BatchNorm, MultiHeadAttention, and TransformerBlock, so those tests fail even though the layers are fine (verified 2026-10-26).
- Exception: `Embedding::forward` takes `&[usize]`, so it can't go through `Module`. Test its per-index gradients directly.
- `MultiHeadAttention` and `TransformerBlock` have an inherent `forward(x, mask, …)` alongside `Module::forward`. In generic code, call the trait version through UFCS: `<_ as Module>::forward(&layer, &x)`.
- `load_state_dict` must validate numel and preserve the destination's device (H7). Never replace `data` with `Storage::cpu(..)`.
- Layout is NCHW, following PyTorch conventions.
- **Known open defects** (don't treat passing tests as proof they work):
  - LSTMCell severs the graph at the gate slice (B2).
  - PositionalEncoding rebuilds its output from raw data (B3). It has no `assert_trainable!` test yet.
  - TransformerBlock uses a parameterless `SimpleAttention` placeholder (H6), so its trainability test covers only the norms and the FFN.
- Conv: there are three algorithms (direct, im2col+GEMM, iGEMM) plus auto-selection. Read `dev-docs/CONV2D_IMPLEMENTATION.md` before changing selection or any conv backward.

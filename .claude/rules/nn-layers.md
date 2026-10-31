---
paths:
  - "src/nn/**"
---

# Neural network layers

- A tensor-to-tensor layer implements `Module` (`parameters`, `state_dict`, `load_state_dict`) and `Layer` (`forward`). `parameters()` must return every learnable tensor. Index input is an `IndexedLayer`, carried state is a `RecurrentLayer`; both are sealed, so a new layer of those kinds goes in the crate.
- Build outputs from graph ops (`shrink`, broadcast add, `matmul`, …) or from `edge()`. Never copy values into a fresh tensor, which severs the graph; that is how LSTMCell's gate slice (B2) and PositionalEncoding (B3) shipped, and both are a `shrink` now. A layer that holds a non-parameter tensor (a fixed buffer such as PositionalEncoding's sinusoid table) overrides `Module::to_device` to move it too.
- **Every parameterized layer gets an `assert_trainable!` test**, with its input from `crate::macros::trainable_input(&shape)`. All-ones input makes gradients mathematically zero for normalization and attention layers.
- Exception: `Embedding` is an `IndexedLayer` (`forward_indices(&[usize])`), so it can't go through `Layer` or `assert_trainable!`. Test its per-index gradients directly. `LSTMCell` is a `RecurrentLayer` (`initial_state`, `step`); its `assert_trainable!` test goes through a test-only two-step `Layer` wrapper (one step from zero state gives `weight_hh` an exactly zero gradient).
- `MultiHeadAttention` and `TransformerBlock` have an inherent `forward(x, mask, …)` alongside `Layer::forward`. In generic code, call the trait version through UFCS: `<_ as Layer>::forward(&layer, &x)`.
- `load_state_dict` must validate numel and preserve the destination's device (H7). Never replace `data` with `Storage::cpu(..)`.
- Layout is NCHW, following PyTorch conventions.
- Kernel and backward indexing goes through `crate::index::{fetch, slot, span}` (panic on out of bounds). A flat output index is `b * stride + <within-sample offset>`: don't add `b` to a decoded `s_idx` that already contains it (the iGEMM batch bug). Test kernels with batch > 1 and a random weighted loss.
- Conv-family geometry (`Conv2d`, `ConvTranspose2d`, `MaxPool2d`) comes from the spec in `src/spec.rs`, built once in `forward` and stored in the `GradFn`. Backward and GPU wrappers read the spec; never re-derive dims with `shape.get(i).unwrap_or(1)`.
- Conv: there are three algorithms (direct, im2col+GEMM, iGEMM) plus auto-selection. Read `dev-docs/CONV2D_IMPLEMENTATION.md` before changing selection or any conv backward.

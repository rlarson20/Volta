# Backlog

Feature ideas after Phase 3, checked against the code on 2026-11-02. Don't start these unprompted (CLAUDE.md, Scope); `notes/TODO.md` is the live list of defects and open work. The earlier ten-item version (2026-05-08, with the full gradient-clipping design) is in git history: `git show ddd4a04:docs/backlog.md`.

Every item must meet the repo invariants: new code writes `grad` only through `set_grad`/`accumulate_grad` (3), reads and writes numeric data through `crate::index` (5), gets a gradient check or exact-value test (7), and ships a GPU path where the CPU one exists rather than a "CPU only" note (`.claude/rules/gpu.md`). User-chosen arguments return `VoltaError`; contract violations panic (8).

## Do next, in this order

### 1. Gradient clipping (`src/nn/optim/clip.rs`)

`clip_grad_norm(params, max_norm, norm_type) -> f32` (returns the norm before clipping) and `clip_grad_value(params, clip)`. Nothing named `clip_grad*` exists, and `char_language_model` and `mnist_rnn` train LSTMs without it.

- Read each gradient with `Tensor::grad_storage()` and write the rescaled one back with `set_grad`; never `borrow_mut` (invariants 2 and 3). The total norm spans all parameters, so it is one reduction over per-parameter norms.
- Support L1, L2, and L∞. `max_norm <= 0` or `norm_type <= 0` panics (the program chose them); empty input or no gradients returns 0.0; NaN and Inf propagate.
- GPU gradients get a GPU path (compose existing reduce and scale kernels, or write one), not a host round trip.
- Tests: norm 10 clipped to 1.0 ends at exactly 1.0; under the threshold is untouched; each norm type against a hand value; clipping then `step` on a toy problem. Then add it to the two LSTM examples between `backward()` and `step()`.

### 2. Learning-rate schedulers (`src/nn/optim/scheduler.rs`)

`StepLR`, `ExponentialLR`, `CosineAnnealingLR`, and linear warmup; `ReduceLROnPlateau` is optional. The `lr` field is private on `Adam`, `SGD`, and `Muon`, and there is no shared optimizer trait, so first add `lr()` / `set_lr()` to each (or a small `Optimizer` trait with both and `step`/`zero_grad`). A scheduler then owns its schedule and the user calls `opt.set_lr(sched.step())`; don't hold `&mut f32`.

- Tests: the exact LR sequence over epochs for each, against closed-form values.
- Update one example (`mnist_cnn`) to show the LR falling.

### 3. Loss reductions and missing losses (`src/nn/loss.rs`)

The losses are free functions at the bottom of `src/tensor.rs` and all hardcode mean. Move them into `nn/loss.rs`, add `Reduction::{Mean, Sum, None}` as an argument, and add `l1_loss`, `huber_loss`/`smooth_l1_loss`, and optionally `ignore_index` for `cross_entropy_loss`. Keep the old names working. Skip the `MSELoss::new(..)` struct layer.

- Tests: each reduction's output shape, a gradient check per loss with a weighted loss, and hand-computed values (`pos_weight` and label smoothing are easy to get subtly wrong; if added, check them against the PyTorch formula).

### 4. GRU and RNN cells, plus a sequence wrapper (`src/nn/layers/`)

`GRUCell` and `RNNCell` as `RecurrentLayer`s (`initial_state`, `step`), next to `LSTMCell`; and an unroll wrapper over `(B, T, F)` that removes the manual timestep loop in `char_language_model`. Single layer and single direction first; bidirectional and multi-layer later.

- Tests: `assert_trainable!` through the two-step wrapper (see DECISIONS.md, "`assert_trainable!`", for why one step is not enough), values against PyTorch's reference on fixed weights, and the wrapper against a manual loop of cell calls.

### 5. Training checkpoints (`src/io/checkpoint.rs`)

Model weights plus optimizer state (Adam's `m`, `v`, `t`; SGD and Muon velocity), epoch and step, scheduler state, and the RNG seed. Needs `state_dict`/`load_state_dict` on each optimizer, so do it after items 1–2.

- Use safetensors for tensors, not bincode (`deny.toml` already ignores a bincode advisory; a new file format on it adds to that).
- Do it with the open "atomic `load_state_dict`" item in `notes/TODO.md`, since both define what a failed load leaves behind.
- Test: save mid-run, reload, and continue; the final loss matches an uninterrupted run, and Adam's `m`/`v` are bit-identical after the round trip.

## Mostly done

- **`Dataset` / `DataLoader` trait.** `DataLoader` exists (`src/tensor.rs`): shuffle, `manual_seed` ordering, a smaller last batch, and device prefetch. What's missing is the `Dataset` trait, `TensorDataset`, `random_split`, and `drop_last`. That's generalization for datasets that don't exist yet, so wait for an example that needs it. Single-threaded only; no `num_workers`.

## Someday, with a precondition

- **Mixed precision / autocast.** Every compute path is f32 and GPU storage is f32-only (`notes/TODO.md`, the dtype item). Real bf16 kernels come first; autocast on top of f32-only paths would only be a cast.
- **Chunked im2col.** `ConvAlgo::Auto` and iGEMM already avoid the large intermediate; only an explicit `ConvAlgo::Im2col` on a large input hits it. Worth doing only if someone needs that path, with a spatial-chunk size and an equivalence test at chunk sizes down to 1.
- **`.pth` loader.** A hand-written pickle parser is the riskiest thing here, and converting to safetensors first (`scripts/export_pytorch_mnist.py` shows the route) costs the user one script. If it's ever built, whitelist opcodes and reject `GLOBAL`/`REDUCE`/`BUILD`; never add one more opcode to load one more checkpoint.
- **CI GPU lane.** GPU tests skip silently without an adapter. What would help is a lane that fails when no adapter is found (a software adapter such as lavapipe on Linux, or a self-hosted runner). Check the `ci.yml` macOS job first to see how much already exists. Coverage upload needs a Codecov token set by the user; `just coverage` covers it locally.
- **Load a real HuggingFace model** (a tiny BERT), as the test of the loader and `StateDictMapper`.
- **Remaining PyTorch examples:** multi-layer LSTM (after item 4), seq2seq with attention, and the ones that need pretrained weights (siamese, style transfer).

## Out of scope

ONNX, tokenizers, a model hub, and architecture inference from weights. This is an educational autograd engine, not a model runtime.

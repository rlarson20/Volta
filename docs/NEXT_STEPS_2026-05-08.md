# Next Steps — 2026-05-08

Ten focused feature ideas, each scoped for a single productive agent session. Each targets a real gap identified in `CLAUDE.md`, `notes/TODO.md`, `notes/REFACTOR_SUGGESTIONS.md`, or `notes/REPO_HEALTH.md`.

Every entry includes: **gap**, **proposed API**, **files touched**, **acceptance criteria**, and **risk notes** so an agent can pick it up cold.

---

## 1. Learning Rate Schedulers (`src/nn/optim/scheduler.rs`)

**Gap.** `CLAUDE.md` lists "no learning rate schedulers" as a known limitation. Examples currently train at a fixed LR for all epochs, which caps how good demonstrated results can get without manual tuning.

**Why it's a clean session.** `Adam.lr` and `SGD.lr` are already public, mutable `f32` fields. A scheduler just needs a reference to the optimizer (or to the `lr` field directly) and a `step()` method called once per epoch. No invasive refactor required.

**Proposed API.**

```rust
pub trait LRScheduler {
    fn step(&mut self);                    // advance one epoch
    fn get_last_lr(&self) -> f32;
}

pub struct StepLR<'a> { lr: &'a mut f32, base_lr: f32, step_size: usize, gamma: f32, epoch: usize }
pub struct MultiStepLR<'a> { ... milestones: Vec<usize>, ... }
pub struct ExponentialLR<'a> { ... gamma: f32, ... }
pub struct CosineAnnealingLR<'a> { ... t_max: usize, eta_min: f32, ... }
pub struct ReduceLROnPlateau<'a> { ... factor: f32, patience: usize, mode: Mode, ... fn step_with(&mut self, metric: f32) }
```

**Files touched.** New: `src/nn/optim/scheduler.rs`. Edit: `src/nn/optim/mod.rs` (re-export), `src/lib.rs` (re-export at crate root). One example updated to demonstrate (e.g., `mnist_cnn.rs` with `StepLR`).

**Acceptance criteria.**

- Five scheduler types with unit tests for the LR sequence each produces.
- `ReduceLROnPlateau` covers both `min` and `max` metric modes plus the `cooldown` window.
- One updated example shows the LR decreasing visibly across epochs.
- All existing tests still pass; clippy clean.

**Risks.** Borrow checker friction if the scheduler holds `&mut f32` while the optimizer is also borrowed elsewhere. Fallback: schedulers store the new LR and the user calls `optimizer.set_lr(scheduler.get_last_lr())` each epoch. Pick whichever feels less awkward in `mnist_cnn.rs`.

---

## 2. Gradient Clipping (`src/nn/optim/clip.rs`)

**Gap.** `grep` confirms no `clip_grad_norm_` or `clip_grad_value_` anywhere in the source. Yet `examples/char_language_model.rs` and `examples/mnist_rnn.rs` train LSTMs without any clipping — which is exactly the setup that famously explodes. The Transformer block in `src/nn/transformer/` has the same issue waiting to bite.

**Proposed API.**

```rust
// Returns the total norm BEFORE clipping (useful for logging).
pub fn clip_grad_norm(params: &[Tensor], max_norm: f32, norm_type: f32) -> f32;
pub fn clip_grad_value(params: &[Tensor], clip_value: f32);
```

Each function iterates `params`, reads `.grad` from the underlying `RefCell`, computes the norm (default `norm_type = 2.0`), and rescales in-place if the total norm exceeds `max_norm`.

**Files touched.** New: `src/nn/optim/clip.rs`. Edit: `src/nn/optim/mod.rs`, `src/lib.rs`. Update `char_language_model.rs` and `mnist_rnn.rs` to call `clip_grad_norm(&params, 1.0, 2.0)` between `loss.backward()` and `optimizer.step()`.

**Acceptance criteria.**

- Unit tests: gradients with norm 10 clipped to `max_norm=1.0` end up with norm exactly 1.0; gradients already under the threshold are untouched.
- Numerical gradient check still passes on the clipped path (clipping happens after `.backward()`, so it shouldn't affect grad correctness — but verify).
- L1, L2, and L∞ norm types all exercised.
- Updated examples train without NaN/Inf at higher LRs than before.

**Risks.** Need to mutate `.grad` through `Rc<RefCell<...>>` — make sure no other handle is borrowed when clipping runs. The autograd graph is fully built and consumed by `backward()` first, so this should be safe.

---

## 3. `nn::loss` Module with Reduction Modes (`src/nn/loss.rs`)

**Gap.** Loss functions already exist (`mse_loss`, `cross_entropy_loss`, `nll_loss`, `bce_loss`, `bce_with_logits_loss` — see `src/tensor.rs:364-465` and `:1374-1394`), but:

- They're split awkwardly between `RawTensor::*` methods and free functions at the bottom of `tensor.rs`.
- **All hardcode `mean` reduction** — no `sum` or `none` option.
- Several common losses are missing: `L1Loss`, `HuberLoss` / `SmoothL1Loss`, `KLDivLoss`, `FocalLoss`.
- No `Module`-style struct API (`MSELoss::new(Reduction::Mean).forward(&pred, &target)`), which is what most PyTorch users reach for.

**Proposed API.**

```rust
pub enum Reduction { Mean, Sum, None }

pub struct MSELoss      { reduction: Reduction }
pub struct L1Loss       { reduction: Reduction }
pub struct CrossEntropyLoss { reduction: Reduction, ignore_index: Option<i64>, label_smoothing: f32 }
pub struct BCEWithLogitsLoss { reduction: Reduction, pos_weight: Option<Tensor> }
pub struct HuberLoss    { reduction: Reduction, delta: f32 }
pub struct NLLLoss      { reduction: Reduction, ignore_index: Option<i64> }
pub struct KLDivLoss    { reduction: Reduction, log_target: bool }
```

Each implements `forward(&self, input: &Tensor, target: &Tensor) -> Tensor`. Free-function shims in `src/tensor.rs` continue to work and now accept an optional `Reduction`.

**Files touched.** New: `src/nn/loss.rs`. Edit: `src/nn/mod.rs`, `src/lib.rs`. Optionally migrate the `tensor.rs` free functions to delegate to the new module (preserve back-compat).

**Acceptance criteria.**

- All seven loss types with grad checks (`check_gradients_simple()`).
- Each `Reduction` mode tested for shape (`None` returns per-element, `Sum` returns scalar, `Mean` returns scalar).
- `CrossEntropyLoss` `ignore_index` correctness test (masked positions don't contribute to gradient).
- Existing examples still produce identical loss values when switched to the new API (`Reduction::Mean` is default).

**Risks.** `pos_weight` for `BCEWithLogitsLoss` and `label_smoothing` for `CrossEntropyLoss` are easy to get subtly wrong. Cross-check against the PyTorch source/formula in tests with hardcoded expected values.

---

## 4. PyTorch `.pth` State Loader (`src/io/pytorch.rs`)

**Gap.** `notes/TODO.md` flags "figure out how to load torch models" as the next step toward HuggingFace compatibility. SafeTensors works (`src/io.rs:load_safetensors`), but most pretrained checkpoints in the wild ship as `.pth` — a ZIP archive containing a pickled `OrderedDict` of tensor metadata plus raw storage blobs.

**Why it's tractable in one session.** We don't need a general-purpose pickle VM. PyTorch state dicts use a tiny, well-defined opcode subset (mostly `EMPTY_DICT`, `MARK`, `BINPUT`, `SETITEMS`, `SHORT_BINUNICODE`, `BINPERSID`, plus the storage typing tuples). A purpose-built reader is ~500 LOC.

**Proposed API.**

```rust
pub fn load_pytorch(path: impl AsRef<Path>) -> Result<StateDict, VoltaError>;
pub fn load_pytorch_with_mapping(path: ..., mapper: &StateDictMapper) -> Result<StateDict, VoltaError>;
```

**Security stance (non-negotiable).** The reader is a hand-written pickle parser that **whitelists** opcodes. Any opcode that could trigger code execution — `GLOBAL`, `STACK_GLOBAL`, `REDUCE`, `BUILD`, `INST`, `OBJ`, `NEWOBJ`, `NEWOBJ_EX` — returns `VoltaError::UnsafePickle`. The only "callable" path we accept is `torch._utils._rebuild_tensor_v2`, which we recognize by name and resolve internally to a pure data unpacking step. No user-supplied code is ever executed.

**Files touched.** New: `src/io/pytorch.rs`, `src/io/pickle.rs` (the parser). Edit: `src/io.rs` or `src/io/mod.rs` to re-export. Add fixture `.pth` file under `tests/fixtures/` (small, generated once via PyTorch and committed).

**Acceptance criteria.**

- Round-trip test: save weights from PyTorch (offline, fixture committed), load via Volta, compare against expected values.
- Adversarial test: a `.pth` containing a `GLOBAL os.system` opcode is **rejected** with `VoltaError::UnsafePickle`, not executed.
- Works on CPU `f32`, `f16`, `bf16` storage variants.
- Documented in a doc comment that this loader is intentionally restricted and will not load arbitrary objects.

**Risks.** This is the highest-risk item on the list. Stick to the whitelist; do not "just add" `GLOBAL` support to load one more checkpoint type. If a checkpoint requires arbitrary code, document it as "convert to safetensors first" rather than expanding the parser.

---

## 5. General `Dataset` / `DataLoader` Trait (`src/data/loader.rs`)

**Gap.** `src/data/` currently contains only `mnist.rs` (raw bytes loader) and `transforms.rs` (`normalize`, `to_one_hot`). Every example reimplements its own batching loop with manual indexing — see the inner `for batch_idx in 0..num_batches` in `mnist_cnn.rs`. There is no abstraction to share across datasets.

**Proposed API.**

```rust
pub trait Dataset {
    type Item;
    fn len(&self) -> usize;
    fn get(&self, idx: usize) -> Self::Item;
}

pub struct TensorDataset { tensors: Vec<Tensor> }   // zips tensors along dim 0

pub struct DataLoader<D: Dataset> {
    dataset: D,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    rng: StdRng,
}

impl<D: Dataset> DataLoader<D> {
    pub fn new(dataset: D, batch_size: usize) -> Self;
    pub fn with_shuffle(self, seed: u64) -> Self;
    pub fn with_drop_last(self, drop: bool) -> Self;
    pub fn iter(&mut self) -> impl Iterator<Item = Vec<D::Item>>;
}

pub fn random_split<D: Dataset>(d: D, sizes: &[usize], seed: u64) -> Vec<Subset<D>>;
```

**Files touched.** New: `src/data/loader.rs`, `src/data/dataset.rs`. Edit: `src/data/mod.rs`, `src/lib.rs`. Refactor `mnist_cnn.rs` to use it, as a smoke test that the API is ergonomic.

**Acceptance criteria.**

- `TensorDataset` over two tensors of differing shape errors out at construction.
- Shuffle with the same seed produces identical batch order across runs.
- `drop_last = true` skips the trailing partial batch; `drop_last = false` includes it at smaller size.
- `random_split([0.7, 0.15, 0.15])` produces non-overlapping subsets that cover the full dataset.
- `mnist_cnn.rs` produces the same final loss as before, within numerical noise.

**Risks.** Single-threaded only (the project explicitly uses `Rc<RefCell>`). Don't try to add `num_workers` parallelism — it'll fight the existing design. Document that limitation.

---

## 6. Streaming/Chunked im2col (`src/nn/layers/conv.rs`)

**Gap.** `notes/REFACTOR_SUGGESTIONS.md` calls this the "im2col memory crisis" — a 32×3×224×224 conv with 3×3 kernels materializes a ~17 GB intermediate matrix. Direct convolution and iGEMM mitigate this for the auto-selected path, but anyone explicitly choosing `ConvAlgo::Im2col` still hits the wall.

**Proposed API.**

```rust
pub struct Im2colConfig { pub spatial_chunk: usize }   // tile along H_out × W_out
// ConvAlgo::Im2col { config: Im2colConfig }   (or extend existing variant)
```

The new tiled algorithm:

1. Split the output spatial grid into chunks of `spatial_chunk` rows.
2. For each chunk: build the partial im2col matrix (only rows for that chunk), GEMM against weights, write directly into the output region.
3. Backward pass: accumulate `dW` chunk-by-chunk; `dx` via tiled `col2im`.

**Files touched.** Edit: `src/nn/layers/conv.rs` (add tiled forward + backward), `src/gpu/shaders/im2col.wgsl` (probably small change to take `chunk_offset`). Possibly `src/gpu/kernels.rs` to plumb the offset through.

**Acceptance criteria.**

- Numerical equivalence with the existing `Im2col` algorithm for forward and backward (within `1e-5`).
- Peak memory measurement (via `process_memory_mb`) shows >5x reduction at `spatial_chunk = 64` on a synthetic 224×224 case.
- Default `spatial_chunk` chosen so that small inputs aren't slowed down measurably (benchmark in `benches/conv_algorithms.rs`).

**Risks.** The backward pass `col2im` accumulation is where bugs hide — race-free chunked accumulation requires the chunks to write to **disjoint** output regions. Verify by running the existing conv backward grad checks at multiple `spatial_chunk` sizes including `1`.

---

## 7. GRU & Vanilla `RNNCell` (`src/nn/layers/rnn.rs`)

**Gap.** `LSTMCell` is the only recurrent unit (`src/nn/layers/lstm.rs`). `CLAUDE.md` notes "no full RNN/Transformer". Most academic NLP baselines and the time-series example in `examples/time_sequence.rs` would benefit from drop-in `GRUCell` / `RNNCell`. A small wrapper that unrolls a cell over a sequence dimension also removes the manual timestep loop in `char_language_model.rs:121-138`.

**Proposed API.**

```rust
pub struct RNNCell { ... nonlinearity: Nonlin /* Tanh | ReLU */ }
pub struct GRUCell { ... }

// Sequence wrappers — accept any cell, unroll along time dim.
pub struct RNN { cell: RNNCell, num_layers: usize, bidirectional: bool }
pub struct GRU { cell: GRUCell, ... }
pub struct LSTM { cell: LSTMCell, ... }
// All implement Module; forward takes (B, T, F) → (B, T, H)
```

**Files touched.** New: `src/nn/layers/rnn.rs` (or split into `rnn_cell.rs` + `rnn.rs`). Edit: `src/nn/layers/mod.rs`, `src/lib.rs`. Update `char_language_model.rs` to use the new `LSTM` wrapper as a smoke test.

**Acceptance criteria.**

- `GRUCell` matches PyTorch's reference output on hardcoded weight + input tensors.
- Numerical gradient check passes for both cells.
- The unrolled `LSTM` wrapper produces identical outputs to a manual loop of `LSTMCell` calls.
- Bidirectional flag verified on a tiny synthetic test.

**Risks.** Bidirectional + multi-layer interactions are easy to get wrong. Implement single-layer single-direction first, get tests passing, then add layers and direction.

---

## 8. Mixed-Precision Training (autocast scope)

**Gap.** Phase-1 dtype work is done — `src/dtype.rs` defines `f16`, `bf16`, `f32`, `f64`, `i32`, `i64`, `u8`, `bool` with promotion rules, and storage is byte-buffer-backed. None of this is wired into training. Modern GPUs (and Apple Silicon's neural engine) get 2-3x speedups from bf16 matmul/conv.

**Proposed API (PyTorch-style scope guard).**

```rust
pub struct Autocast { dtype: DType, prev: Option<DType> }
impl Autocast {
    pub fn new(dtype: DType) -> Self;   // installs a thread-local
}
impl Drop for Autocast { ... }          // restores previous

// Inside matmul / conv2d forward:
//   if let Some(cast) = autocast::current() { downcast inputs to cast.dtype }
// Backward keeps grads in master fp32.
```

**Scope.** Single session means **only matmul and conv2d** are autocast-aware. Element-wise ops, normalizations, and reductions stay in their original dtype (matches PyTorch's autocast op list approximately). Master weights stay in fp32 in the optimizer.

**Files touched.** New: `src/autocast.rs`. Edit: `src/ops/matmul.rs`, `src/nn/layers/conv.rs` (insert dtype-cast at op boundary), `src/lib.rs`.

**Acceptance criteria.**

- `Autocast::new(DType::BF16)` scope reduces matmul peak memory in a benchmark.
- Output of an autocast'd matmul matches fp32 within bf16 tolerance (~`1e-2`).
- Optimizer step still mutates fp32 master weights (verified by checking weight dtype after step).
- One example (e.g., `gpu_training.rs`) demonstrates an autocast scope and trains to comparable accuracy.

**Risks.** This is the largest item. If it's blowing up scope mid-session, fall back to "matmul-only autocast" and ship that. Keep the conv2d cast for a follow-up.

---

## 9. Training Checkpoint API (`src/io/checkpoint.rs`)

**Gap.** `save_state_dict` (`src/io.rs`) saves model weights only. Resuming a half-finished training run requires also persisting: optimizer state (Adam's `m`/`v` buffers!), current epoch, scheduler state, RNG seed, and ideally the loss curve so far. None of that has a canonical home.

**Proposed API.**

```rust
pub struct Checkpoint {
    pub model: StateDict,
    pub optimizer: OptimizerState,    // enum over Adam/SGD/Muon
    pub epoch: usize,
    pub global_step: usize,
    pub scheduler: Option<SchedulerState>,
    pub rng_seed: u64,
    pub metadata: BTreeMap<String, String>,   // free-form for loss curves, etc.
}

pub fn save_checkpoint(ckpt: &Checkpoint, path: impl AsRef<Path>) -> Result<...>;
pub fn load_checkpoint(path: impl AsRef<Path>) -> Result<Checkpoint, VoltaError>;
```

`OptimizerState` requires each optimizer to expose `state_dict() -> OptimizerState` and `load_state_dict(&OptimizerState)`. For Adam this is `m`, `v`, `t`. For SGD with momentum it's the velocity buffer.

**Files touched.** New: `src/io/checkpoint.rs`. Edit: `src/nn/optim/adam.rs`, `src/nn/optim/sgd.rs`, `src/nn/optim/muon.rs` (add `state_dict()` / `load_state_dict()`). Couples to #1 if schedulers ship first.

**Acceptance criteria.**

- Save mid-training, reload, continue 10 more epochs — final loss matches an uninterrupted 10+10 run within numerical noise.
- Adam's `m`/`v` are bit-identical pre-save and post-load.
- Backward-compat: a state-dict-only file (no checkpoint wrapper) still loads via the existing `load_state_dict`.

**Risks.** RNG state isn't easily serializable for `StdRng` from `rand 0.8` without `serde1` feature. If that turns out to be a snag, store only the seed + step count and reconstruct on load.

---

## 10. CI Coverage + GPU Smoke Job (`.github/workflows/ci.yml`)

**Gap.** `notes/REPO_HEALTH.md` flags "no code coverage tracking" and "GPU feature only `cargo check`'d, never tested". The current CI (just confirmed via the workflow file) is `cargo fmt --check`, `cargo clippy`, `cargo test`, `cargo test --doc`, `cargo doc`. The commented-out GPU line shows the intent was there but blocked by GitHub-hosted runners lacking GPUs.

**Proposed changes.**

**A. Coverage (Ubuntu only, runs on every PR).**

```yaml
- uses: taiki-e/install-action@v2
  with: { tool: cargo-llvm-cov }
- run: cargo llvm-cov --lcov --output-path lcov.info
- uses: codecov/codecov-action@v4
  with: { files: lcov.info, fail_ci_if_error: false }
```

**B. GPU smoke job (separate workflow, gated).**

- Runs `cargo test --features gpu --test gpu_smoke_test` and `tests/gpu_test.rs`.
- Gating options (pick whichever matches the user's setup):
  - `runs-on: self-hosted` if a runner with a GPU exists.
  - Or: `if: github.event_name == 'workflow_dispatch'` so it's manual-trigger only.
  - Or: keep the existing `cargo check --features gpu` line and call out in CI docs that GPU tests run locally pre-PR.

**Files touched.** Edit: `.github/workflows/ci.yml`. New: `.github/workflows/gpu.yml` (if going the separate-workflow route). Potentially new: `codecov.yml` for coverage thresholds.

**Acceptance criteria.**

- Coverage badge renders in `README.md`.
- A first coverage run completes successfully and shows a sensible number (~70-80% expected given the test density).
- GPU job either runs on a real GPU (if a self-hosted runner is configured) or is documented as manual-dispatch.
- No regression in CI runtime budget on PRs (coverage adds maybe 2-3 min on Ubuntu).

**Risks.** Codecov requires a token set up in repo secrets — flag this for the user to do, the agent can't. Self-hosted runners require the user to provision hardware. If both are blockers, ship just the coverage piece.

---

## Recommended order of attack

1. **#2 Gradient clipping** — tiniest, removes a real footgun in shipped examples. Could literally be a 90-minute session.
2. **#3 Loss module** — high impact (refactor + new losses), low risk, no architectural changes.
3. **#1 LR schedulers** — uses #2/#3 as nothing-changes-in-them prerequisites; immediately makes example results better.
4. **#5 DataLoader** — unlocks every subsequent example, cleans up boilerplate everywhere.
5. **#9 Checkpoints** — pairs naturally with #1 (scheduler state) and #5 (resumable training).
6. Pick from #4 / #6 / #7 / #8 / #10 based on what's blocking the actual user goal that month.

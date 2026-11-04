# TODO — live checklist

The plan is `docs/plans/ADDENDUM.md` § Sequencing. Defect evidence is in `docs/audits/OPUS_5_EVAL_AUDIT.md` (OPUS §n), with sources in `AUDIT_CITATION.md`. Phases 0–3 are done, and so are B1–B5 and H1–H8.
This file holds only open work. The done items, with their evidence (tests, mutations, measurements), are in git history: `git show ddd4a04:notes/TODO.md`. The older long-form notes are in `git show d771013:notes/TODO.md`.

**Status key:** `[ ]` = open. `[~]` = partly fixed. Last cleaned: 2026-11-04 (post-Phase-5 audit); sweep items added 2026-11-02.

## Correctness

- [ ] **`examples/dcgan.rs` is fixed but unverified: nobody has seen it learn.** The double normalization is gone (it asserts the data spans [-1, 1]) and it uses Adam (lr 2e-4, betas 0.5/0.999), but a CPU release run (`--no-default-features`) had not finished one 937-batch epoch after 13 minutes, so the D/G losses and the before/after generator stats it prints (it exits 1 on a non-finite loss or a collapsed output) have never been seen. Find out where the time goes (the two `ConvTranspose2d` layers have no GPU path and the example runs 4 forward/backward passes per batch) before judging the training.
- [ ] **GPU broadcast backward is broken, but unreachable.** Verified on `fff5397` with a 3x1 + 1x4 add: the race-free two-pass path (`binary_backward_broadcast_safe`) panics with a wgpu validation error (pass-1 bind group has 6 bindings; the auto layout keeps only the bindings the entry point uses: 3 for Add, 5 for Mul). Nothing reaches it: `gpu_binary_backward` treats differing shapes as a device-policy gap and takes the CPU backward. Fixing it means an explicit bind group layout (or matching bindings per entry point) plus a GPU forward that broadcasts. Repro: `cargo test --lib gpu_binary_backward_broadcast_matches_cpu -- --ignored`.

## Correctness: found by the 2026-11 sweep

Evidence, measured numbers, and the refuted claims are in `docs/audits/2026-11-sweep.md`. Each item below has an `#[ignore = "S<n>: ..."]` test that asserts PyTorch's answer; un-ignore it with the fix. Run them with `cargo test --no-default-features --test torch_ref_ops --test torch_ref_layers --test torch_ref_optim --test torch_ref_reductions --test sweep_repros -- --ignored`, and the GPU ones with `cargo test --test gpu_sweep -- --ignored --nocapture` (check for `GPU-RAN` in the output).

- [ ] **S10, S11: `TransformerBlock` adds the attention residual twice, and `MultiHeadAttention::new_with_norm` adds it to the normalized input.** The block computes `x + LN(x) + Attn(LN(x))`. Decide whether `MultiHeadAttention` owns a residual at all (PyTorch's does not); `examples/char_transformer_model.rs` works around it by hand, which `.claude/rules/tests.md` forbids.
- [ ] **S1: a second `backward()` through an intermediate node re-sends its stored gradient** (`sum(2a)` twice gives 6, torch 4). Intermediate nodes should not keep `grad` after their backward step, or `backward` should only propagate the new contribution. This also affects `examples/time_sequence.rs`, which calls `backward()` per timestep. `backward_twice_accumulates` (`autograd.rs`) asserts only `>`; make it exact.
- [ ] **S2, S3, S4: one rule for max, ties, and NaN.** `max_dim` and `MaxPool2d` fall back to flat index 0 of the whole tensor for a window with no winner, `max_reduce` panics on NaN and picks the last tie, `max_elem` gives the whole gradient to `a` on a tie, and `relu(NaN)` is 0. PyTorch propagates NaN and splits tied gradients evenly for `max()` and `maximum`. A fully masked row in `softmax` writes a NaN gradient into row 0 through S2.
- [ ] **S5: `modulo` is truncated on the CPU and floored on the GPU,** and is non-differentiable. Match `torch.remainder` (floored) on both, with gradients `1` and `-floor(a/b)`.
- [ ] **S6: sums are sequential f32** on the CPU (`reduce.rs`, `sum_dim`) and in the one-thread GPU kernel. A sum of 2^25 ones is 16,777,216 (torch 33,554,432). Use pairwise or Kahan summation; the GPU reduction item under GPU fixes it too.
- [ ] **S7, S8: `bce_loss` has no log clamp** (NaN where torch gives 40.24), and **`cross_entropy_loss` is NaN when any off-target logit is `-inf`** (0 times `-inf`).
- [ ] **S9: an optimizer step changes a parameter that a pending backward pass saved,** and backward uses the new value without an error. Needs a version counter like torch's, or a documented rule.
- [ ] **S12: `BatchNorm` with one sample per channel makes `running_var` NaN** (0/0 in the Bessel correction). Reject the batch, as torch does.
- [ ] **S14, S15, S16: transformer pieces.** The `dropout` argument to `FeedForward` and `MultiHeadAttention` is stored and never applied; `FeedForward::intermediate_dim()` returns the factor; `ScaledDotProductAttention` requires key length to equal query length, so there is no cross-attention.
- [ ] **S17: optimizers.** Adam keeps one global step count (torch counts per parameter), and optimizer state is allocated on the parameter's device in `new()`, so `model.to_device(gpu)` after building the optimizer makes `step()` panic. Allocate state lazily on the first step. Not yet run: SGD folding `lr` into its velocity (matters once `set_lr` lands).
- [ ] **S19, S20: boundary checks.** `StateDictMapper::add_prefix`/`rename`/`strip_prefix` drop a tensor when a new name collides with an old key; `DataLoader::new` accepts fewer targets than samples. Not yet run: the MNIST header product overflow (`mnist.rs:32`).
- [ ] **S21: GPU `max_reduce`** starts at -1e9 and finds its index with `abs(v - max) < 1e-6` and `map_or(0)`, which is a silent fallback (invariant 5) and picks the wrong element for a near-tie. **S22:** GPU `erf` has a relative error of about 8e-5 near 0.
- [ ] **S24: GPU `tanh` returns NaN for x >= 45** (measured on an Apple M2). `Tanh`, `LSTMCell`, and `tanh_backward` use it. Clamp the input or use a formulation that does not overflow.
- [ ] **S23: README and examples.** Three listed example commands do not exist (`char_lm`, `lstm_time_series`, `polynomial_regression`) and a CIFAR example is claimed. Not yet checked: the `.borrow()` snippets (not public), the GPU snippet panicking with `NoGradRequired`, the zero-copy safetensors claim, `layernorm_demo` checking zero gradients, `char_language_model`'s ignored `temperature`, and `dcgan` not detaching the fake images.
- [ ] **Not yet run:** `LSTMCell` and `MultiHeadAttention` against torch with loaded weights; the effect of S1 on `time_sequence.rs`.
- [ ] **Tests: extend the torch reference suite** (`scripts/torch_ref/gen_fixtures.py`) to `LSTMCell`, attention, `bce_with_logits_loss`, `kl_divergence_gaussian`, unary ops over wide ranges, and `stride_op` (`Embedding` and `expand`/`pad`/`shrink` are covered by the dtype fixtures since Phase 5). The CPU/GPU parity tests still only compare Volta with itself.

- [ ] **`RawTensor::contiguous` detaches.** It copies the data into a new leaf that takes `requires_grad` from its input, so a gradient through it never reaches the input (invariant 1: outputs come from `edge()`). No caller in the repo uses it. Make it a differentiable identity (a `Movement` reshape to the same shape) or remove it. Found 2026-11-03 while making it dtype-generic (Phase 5 P3).

- [ ] **`gpu_stress_tests::test_auto_sync_threshold` races other tests** on the process-wide pending-submission counter: it syncs, then asserts the count is 0 while tests in the same binary submit GPU work in parallel. It passes alone and with `--test-threads=1`; it failed once in a full parallel run on lavapipe (found in Phase 5 P8). Serialize it (a lock shared with the other stress tests) or count only its own submissions.
- [ ] **Training a model converted to half (`Module::to_dtype(BF16)`) has no f32 master weights.** Autocast (P7) keeps parameters f32, so it needs none; a half-converted model steps its parameters in half and loses small updates (the MNIST MLP trained this way ends 0.39 points below f32). Add f32 master copies of half parameters (and f32 optimizer state) as an optimizer option if half-stored training is wanted.
- [ ] **Autocast casts a weight on every use.** Torch caches the low-precision copy of each f32 leaf weight for the length of an autocast scope; Volta casts per op, which costs time and memory but not accuracy. Add a per-scope cache if a model reuses weights (RNN steps, tied embeddings).
- [ ] **Norm layers under autocast return f32 where torch returns the input dtype.** `BatchNorm`/`LayerNorm` with half input and f32 parameters promote to f32 (they are built from elementwise ops); torch's CPU kernels accept the mix, compute in f32 and return the half input's dtype. The next low-precision op casts back, so values agree; only the intermediate dtype (and its memory) differs.
- [ ] **Nightly CI lane for `just mnist-autocast`** (P7's done-criterion: 5-seed bf16 autocast within 0.2 points of f32). It needs the 55 MB download (cache it) and about 3 minutes; keep it off the per-PR jobs.
- [ ] **Scalars promote like `[1]` tensors (P6 decision (a)).** Volta has no zero-dim tensors or scalar ops, so `x_f16.elem_mul(&RawTensor::new(vec![2.0], &[1], false))` returns f32, as torch does for a `[1]` tensor, and silently widens a half model. Torch keeps `f16 * 2.0` in f16 (a scalar can't widen a tensor of its own category). Add scalar ops (`mul_scalar(f64)`, ...) with that rule if a user or P7 needs them; the library's own constants come from `constant_like` in the input's dtype, so only user code can hit this.
- [ ] **Integer arithmetic.** I64/I32/U8 tensors move, compare, select and index, but `add`, `sum`, `max` and the other arithmetic ops panic on them (ADDENDUM §11 P5 kept them data-only). Add integer kernels (wrapping, PyTorch semantics) when a use needs them; P6's promotion covers int + float by casting first. Since P6, `i64 / i64` works (true division in f32, as in torch) while `i64 + i64` panics: an asymmetry kept deliberately until integer kernels land.
- [ ] **A non-f32 model cannot be saved (audit A8).** `state_dict()` panics in `TensorData::from_tensor` for an f64, f16 or bf16 parameter, because `TensorData` holds f32, and there is no public way to build the `TypedTensorData` that `save_safetensors_typed` takes from a `Tensor`. Give `TensorData` a dtype (a bincode format change; safetensors already carries one), or add `TypedTensorData::from_tensor` and a typed `state_dict`. Loading already casts. Found in Phase 5 P3b.
- [ ] **Optimizer state is allocated at construction** in each parameter's dtype and device, so converting a model (`to_dtype`, `to_device`) after building its optimizer panics on the first step. S17 is the device half; allocate state lazily on the first step for both.

## Found by the post-Phase-5 audit (2026-11-04)

Evidence in `docs/audits/2026-11-post-phase5.md`; repros (`#[ignore = "A<n>: ..."]`) in `tests/audit_repros.rs`, run with `cargo test --test audit_repros -- --ignored --nocapture`.

- [ ] **A5: `DTypePolicy::Strict` + autocast panics in any norm layer** (bf16 activations meet f32 parameters: an implicit promotion). Same root as the norm item below; fixing norms to return the input's dtype (torch) fixes both.
- [ ] **A6: `check_gradients` on f16/bf16 reports a correct gradient as failing,** silently (torch's `gradcheck` warns that the input is not double). Warn, refuse, or document.
- [ ] **A7: class-index `cross_entropy_loss` builds a dense `[N, C]` one-hot** (exact, but O(N·C) memory). Gather the target log-probabilities instead once a gather op exists.
- [ ] **A9: 43 bare `.unwrap()`s in library code** without messages (A3's GPU crash was one). Enable `clippy::unwrap_used` for the library and give each an `expect` that names the broken contract.
- [ ] **Invariant debts.** Invariant 2: 17 constructor sites in `src/nn/layers/` set `requires_grad` through `borrow_mut()` instead of `set_requires_grad`. Invariant 3: `zero_grad` writes `grad = None` directly (`nn/mod.rs`, `Adam`, `SGD`, `RawTensor`); route it through one method or name it in the invariant. `src/utils/progress.rs` opts out of the print lint module-wide without a `reason`.
- [ ] **CI: run the default-feature suite on Linux with lavapipe** (`apt-get install mesa-vulkan-drivers`). The shader validation and the GPU half tests are `gpu`-gated, so today only the macOS lane runs them.
- [ ] **Shader validation covers a hand-kept list** (`shader_template::tests::SHADERS`); compare it with `src/gpu/shaders/` in the test, so a new shader can't skip validation and the direct-access check.
- [ ] **About 15 doc examples are ` ```ignore ` but pass with an adapter** (`src/gpu/*`, `storage.rs`, `tensor.rs`, `define_ops!`), so CI never compiles them. Make them `no_run` or guard them on `is_gpu_available()`.
- [ ] **Dead code:** the `cmplt` WGSL entry and pipeline (unused since P5, now compiled once per element type).
- [ ] **S24 is Metal-specific:** its repro passes on lavapipe. Keep the ignore; confirm on the Mac.

## GPU

- [ ] **Kernel gaps that `Strict` rejects.** Each one panics under `Strict` and runs on the CPU under `PreferGpu`: binary ops with differing (broadcast) shapes, forward and backward (`Linear`'s bias add, `mean_dim`'s divide); batched and vector (1-D) `matmul`, forward and backward; `transpose`; `sum_dim`; `max_dim` (and so `softmax`/`log_softmax`); `mul_acc` backward; movement ops above rank 4. With no GPU path at all: `Embedding`, `MaxPool2d`, `ConvTranspose2d`. `Where` and the comparisons have kernels for equal shapes since P8; broadcasting them is a gap. A kernel for any of these drops its name out of an `expect_gaps` assertion.
- [ ] GPU reductions are single-threaded (`workgroup_size(1)`); `gpu_max_reduce` reads back to CPU.
- [ ] **GPU Muon's Frobenius norm is a single-workgroup reduction** (`sum_squares` in `muon.wgsl`, one 256-thread workgroup with a strided loop). Correct (about 1e-6 against the CPU on 64x96), but it will be the slow step in `muon_newton_schulz` for a very large parameter. The fix is a two-pass reduction (per-workgroup partial sums, then one workgroup over the partials). Time `muon_newton_schulz` on a 4096x4096 matrix before changing it.
- [ ] GPU layer returns `Option` for everything; move to `Result<_, GpuError>` with a single dispatch choke point.
- [ ] `get_process_memory_ratio` (`src/gpu/monitor.rs`) divides by a hard-coded 24 GB (an M2 Mac) and returns 0.0 off macOS. Detect real system memory (the `sysinfo` crate, or drop the ratio).
- [~] Stop exporting diagnostic tooling as public API. `GpuKernels`, `OptimizerStepParams`, `BufferPool`, `BufferPoolConfig`, and `StagingBufferPool` are crate-private. Still public: the `monitor`, `system_monitor`, and `early_warning` modules (`tests/gpu_stress.rs` demonstrates them) and `GpuContext`.

## Performance

- [~] Every op deep-copied storage. CPU snapshots are shared through copy-on-write `Arc<[u8]>` now (DECISIONS.md, "CPU storage: `Arc<[u8]>` copy-on-write"). Op outputs still allocate, and broadcasts are still materialized (no stride-0 views). Phase 4 territory.
- [ ] Broadcast backward pass 2 is O(targets × out_size).
- [ ] Profile the 400+ `.clone()` sites in `src/` and use references where the hot ones allow. Find the hotspots before editing; the examples first noticed may not be the worst.
- [ ] Movement ops (`ops/movement.rs`) use deep recursive helpers (high cyclomatic complexity, many parameters, duplicated code between the forward and backward index transforms, no inlining, hard to test singly). Proposal: an iterator- or cursor-based index transform shared by both directions (look at `ndarray`'s strides pattern).

## API and boundary

- [ ] **Make `load_state_dict` atomic before any browser tool uses it.** It checks each entry as it reaches it, so a bad entry in layer 3 of 5 returns `Err` after layers 1 and 2 were overwritten, and a UI that shows "load failed" and keeps the model is running mixed weights. Do it as **plan, then apply**: each layer returns the `(tensor, entry)` pairs it would load (`Rc` handles and references, no weight copies), and a provided trait method checks every pair's shape and length before writing any. Nothing is written until the whole load is known to succeed, and that is the only option with no memory cost. Do not use snapshot-and-rollback or a fresh model plus a swap: both keep the old values alive, so peak memory is about twice the weights, which can crash a browser tab on the path meant to be safe. Validating up front against `self.state_dict()` is not airtight either (`Sequential` accepts a layer name or an index as prefix, and mapped keys need not match what `state_dict()` emits) and it clones every weight. Plan-then-apply rewrites the roughly ten layers that load weights, makes the apply step infallible (a GPU upload failure there would have to panic or be handled separately), and lets containers prefix keys while collecting, which would retire `VoltaError::with_key_prefix`. It changes the `Module` trait a second time, so do it together with the browser tool, not before.
- [ ] Ops that take a user-chosen `Axis`, `permute` order, or padding/shrink ranges (`max_dim`, `mean_dim`, `softmax`, `log_softmax`, `permute`) panic on a bad value and have no `try_` form (only `sum_dim`, `pad_with`, `shrink` do). A WASM binding has to range-check against `shape().rank()` itself. Add a `try_` form, or an `Axis::new(dim, rank)`, when the binding exists.
- [ ] `cargo check --lib --no-default-features --target wasm32-unknown-unknown` fails: `getrandom 0.3` (through `rand`) needs its `wasm_js` backend and `--cfg getrandom_backend="wasm_js"`. Also unverified for wasm32: `std::fs`/`std::time` use in `io` and `utils`, and the `accelerate` feature (macOS-only). The WASM binding (`pkg/`, `minigrad-react/src/wasm/`) is an untracked build of code that is not in this repo, so nothing here is tested against it. Add a wasm32 `cargo check` to CI once the target builds. The README copies in those two untracked directories still show `Module::forward` and `TensorOps`.
- [ ] `Tensor::from_raw` (`src/tensor.rs`) carried a note "investigate why I actually need this". Check its call sites; if each one could use an existing constructor or `edge()`, delete it (invariant 1 and 2 are why: a `RawTensor` wrapped by hand skips the graph).

## Feature proposals (not started; details in `docs/backlog.md`)

- [ ] Gradient clipping (`clip_grad_norm`, `clip_grad_value`), then add it to the two LSTM examples.
- [ ] `lr()` / `set_lr()` on the optimizers, then LR schedulers.
- [ ] `Reduction` argument, `l1_loss`, and `huber_loss`; move the losses out of `tensor.rs`.
- [ ] `GRUCell`, `RNNCell`, and a `(B, T, F)` unroll wrapper.
- [ ] Training checkpoints (optimizer state plus model), with the atomic `load_state_dict` item.

## Tests and tooling

- [ ] `tests/improvements_test.rs` is a vague name; rename it for what it covers. `tests/coverage_fill.rs` needs a `//!` line saying why it exists.
- [ ] `GpuBuffer::zeros` builds a host `vec![0.0; len]` and uploads it, even for a fresh buffer. Use `encoder.clear_buffer` (or wgpu's zero-initialised fresh buffers) so a matmul output does not pay for a host allocation and a copy.
- [ ] A synced GPU elementwise op costs about 1.5 ms at 4096 elements and `matmul` 256 about 3.0 ms (`just bench-gpu` at `8008088`; table in `dev-docs/BENCHMARKING.md`), so the floor is per-op overhead and not the kernel. Find where it goes (params buffer and bind group created per call, `GpuBuffer::zeros` upload, the sync poll) before quoting any CPU-vs-GPU number.
- [ ] GPU `matmul` 2048 runs at about 55 GFLOP/s (310 ms) and `sum`/`mean` cost 3x to 7x an elementwise op. Profile both once the per-op floor is understood.
- [ ] `memory_transfer` is flat near 1.4 ms up to 16384 floats, so its MiB/s figures are a latency floor, not bandwidth. Add a size large enough to saturate (several MiB) before quoting a transfer rate.
- [ ] `bench.sh` loops one test and averages; decide whether it belongs in the `justfile` (`dev-docs/BENCHMARKING.md` is the doc).

## Phase 4 — endgame (optional, last)

- [ ] Arena graph (§6), with the `TensorIndex`/`NodeIndex` newtypes (§1.3) it makes meaningful.
- [ ] Sealed `Backend` trait (§3.5).
- [ ] proptest DAG checks (§10.2).
  - Include `matmul` and `permute` with square and non-square operands, so cancelling transpose bugs get caught (GEMM report §4). `cargo-mutants` (§10.4) should flag blind tests.

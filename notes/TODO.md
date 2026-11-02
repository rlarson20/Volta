# TODO — live checklist

The plan is `docs/plans/ADDENDUM.md` § Sequencing. Defect evidence is in `docs/audits/OPUS_5_EVAL_AUDIT.md` (OPUS §n), with sources in `AUDIT_CITATION.md`. Phases 0–3 are done, and so are B1–B5 and H1–H8.
This file holds only open work. The done items, with their evidence (tests, mutations, measurements), are in git history: `git show ddd4a04:notes/TODO.md`. The older long-form notes are in `git show d771013:notes/TODO.md`.

**Status key:** `[ ]` = open. `[~]` = partly fixed. Last cleaned: 2026-11-02.

## Correctness

- [ ] **`examples/dcgan.rs` normalizes already-normalized data** (found 2026-11-01 while fixing the VAE). It calls `normalize(&mut train_data, 127.5, 127.5)` on pixels `load_mnist_images` already scaled to [0, 1], so every real image is about -1 (range [-1, -0.992]) instead of [-1, 1] and the discriminator sees almost constant 'real' data. Drop the call and map with `(x - 0.5) / 0.5`.
- [ ] The dtype system is decorative: every compute path is f32. Either make it real or stop advertising it.
- [ ] **GPU broadcast backward is broken, but unreachable.** Verified on `fff5397` with a 3x1 + 1x4 add: the race-free two-pass path (`binary_backward_broadcast_safe`) panics with a wgpu validation error (pass-1 bind group has 6 bindings; the auto layout keeps only the bindings the entry point uses: 3 for Add, 5 for Mul). Nothing reaches it: `gpu_binary_backward` treats differing shapes as a device-policy gap and takes the CPU backward. Fixing it means an explicit bind group layout (or matching bindings per entry point) plus a GPU forward that broadcasts. Repro: `cargo test --lib gpu_binary_backward_broadcast_matches_cpu -- --ignored`.

## GPU

- [ ] **Kernel gaps that `Strict` rejects.** Each one panics under `Strict` and runs on the CPU under `PreferGpu`: binary ops with differing (broadcast) shapes, forward and backward (`Linear`'s bias add, `mean_dim`'s divide); batched and vector (1-D) `matmul`, forward and backward; `transpose`; `sum_dim`; `max_dim` (and so `softmax`/`log_softmax`); `mul_acc` backward; movement ops above rank 4. With no GPU path at all: `Embedding`, `MaxPool2d`, `ConvTranspose2d`, `Where`. A kernel for any of these drops its name out of an `expect_gaps` assertion.
- [ ] GPU reductions are single-threaded (`workgroup_size(1)`); `gpu_max_reduce` reads back to CPU.
- [ ] **GPU Muon's Frobenius norm is a single-workgroup reduction** (`sum_squares` in `muon.wgsl`, one 256-thread workgroup with a strided loop). Correct (about 1e-6 against the CPU on 64x96), but it will be the slow step in `muon_newton_schulz` for a very large parameter. The fix is a two-pass reduction (per-workgroup partial sums, then one workgroup over the partials). Time `muon_newton_schulz` on a 4096x4096 matrix before changing it. Also: `workgroups_for` panics over 65535 workgroups (about 16.7M elements) for the elementwise entries.
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
- [ ] `bench.sh` loops one test and averages; decide whether it belongs in the `justfile` (`dev-docs/BENCHMARKING.md` is the doc).

## Phase 4 — endgame (optional, last)

- [ ] Arena graph (§6), with the `TensorIndex`/`NodeIndex` newtypes (§1.3) it makes meaningful.
- [ ] Sealed `Backend` trait (§3.5).
- [ ] proptest DAG checks (§10.2).
  - Include `matmul` and `permute` with square and non-square operands, so cancelling transpose bugs get caught (GEMM report §4). `cargo-mutants` (§10.4) should flag blind tests.

# TODO — live checklist

The plan is `docs/plans/ADDENDUM.md` § Sequencing. Defect evidence is in `docs/audits/OPUS_5_EVAL_AUDIT.md` (OPUS §n), with sources in `AUDIT_CITATION.md`.
The full long-form notes (audit transcription plus the preventive-architecture sketches) are in git history: `git show d771013:notes/TODO.md`.

**Status key:** `[x]` = verified fixed (evidence given). `[~]` = partly fixed. `[ ]` = open per the audit (2026-09-08) and not re-verified unless a note says so.
Last verified: 2026-10-29.

## Blocking (OPUS §1)

- [x] **B1** ConvTranspose2d weights get no gradient. Fixed by `edge()`; `test_conv_transpose2d_assert_trainable` passes.
- [ ] **B2** LSTMCell `slice_gate` severs the graph (`lstm.rs` gate slice → `RawTensor::new`). The test is `#[ignore]`d under B2; fix with `shrink` or a slice GradFn, then un-ignore.
- [ ] **B3** `PositionalEncoding::forward` rebuilds its output from raw data in both modes (`positional_encoding.rs:163,196`). It has no trainability test. The fix composes it from graph ops and adds `assert_trainable!` for learned mode. `examples/char_transformer_model.rs` hand-rolls positional encoding around this bug.
- [x] **B4** Unequal-rank batched matmul backward. Fixed by `MatmulSpec`: matrix operands of unequal rank are rejected in forward (the audit's `[2,3,4] @ [2,1,4,6]` and 3D@2D both panic with "equal rank"), so the bad backward is unreachable. Evidence: `ops::matmul::tests::test_matmul_rejects_unequal_rank_b4_repro`, plus finite-difference checks of both operands for 2D, equal-batch, and broadcast-batch shapes. NumPy-style unequal-rank broadcasting is not supported (DECISIONS.md, "MatmulSpec").
- [x] **B5** The CPU-only build failed to compile. Fixed in `c8a5006`; `--no-default-features` builds and tests (2026-10-26).

## High (OPUS §2)

- [x] **H1** MaxPool2d padded with `0.0` instead of `-inf`. Fixed by `PadValue`: `pool_forward` calls `pad_with(.., PadValue::NEG_INF)`, and padding above half the kernel is rejected. Evidence: `maxpool::tests::test_maxpool2d_padding_never_beats_negative_input_h1_repro` (all-`-5` input, was `0.0`), `..._padding_routes_one_unit_per_output_to_real_input`, `..._padding_over_half_kernel_panics`; GPU parity in `tests/movement_backward_test.rs` (ran on a real adapter).
- [x] **H2** `bce_with_logits_loss` now uses `max(x,0) - x*y + ln(1+exp(-|x|))`; `tests/bce_with_logits.rs` checks loss and gradient at ±100 and at x=0.
- [~] **H3** Unsafe GPU cache accessors. `invalidate_cpu_cache` now takes `&mut self` (`991a8b7`). The `unsafe` slice accessors and the f64/f16/bf16 GPU reinterpretation remain. Fixed by `OnceCell` storage (Phase 3).
- [ ] **H4** The library panics on ordinary data. Panic-vs-`Result` policy is undecided (see Rust/errors).
- [ ] **H5** GPU broadcast backward silently truncates rank > 4. The param packing in `binary_backward_broadcast_safe` now panics on rank > 4 instead; the other GPU wrappers still truncate by design with `.take(4)`.
- [ ] **H6** TransformerBlock is decorative: parameterless `SimpleAttention`. Trait split §3.2 (Phase 3).
- [x] **H7** `load_state_dict` demoted GPU tensors to CPU. Every layer now goes through `Tensor::load_weight`, which checks shape (and numel) and keeps the destination's device. Evidence: `tests/load_state_dict_device.rs`, plus `tensor::load_weight_tests` on GPU. `load_state_dict_checked` now reports shape mismatches and skips them instead of loading them (2026-10-26).
- [ ] **H8** Muon is a toy and panics on GPU params.

## Medium (OPUS §3)

- [ ] GPU `pad` packs dims 2 and 3 into 16 bits each (`GpuKernels::pad`), so a pad over 65535 on those dims truncates silently (invariant 5). It is also limited to rank <= 4.
- [x] `MaxPool2d::pool_forward` and `MaxPool2dGradFn::backward` skipped out-of-range reads and writes silently. They use `fetch`/`slot` now (2026-10-29).

- [x] The vector matmul arms ((2,1), (1,2), (1,1)) in `ops/matmul.rs` wrote through `if let Some` and `unwrap_or(0.0)`. They use `fetch`/`slot` now, so an index bug panics. They are still not covered by `MatmulSpec` (2026-10-29).
- [x] **CPU iGEMM was wrong for batch > 1** (found by the §8 conversion). `igemm_tiled`, `igemm_backward_weight`, and `igemm_backward_input` added the batch offset twice (`s_idx` already contains `b * spatial`), so for batch > 1 the forward wrote into the wrong place or past the end, and the backward weight/input read wrong or out-of-range gradients. The old `if let Some` / `unwrap_or(&0.0)` hid it, and every iGEMM correctness test used batch 1 while the batch > 1 tests checked only shapes. Fixed by using the within-sample offset. Regression: `conv2d_tests::test_conv2d_igemm_batch_gt_one_matches_direct` (seeded, weighted loss, matches Direct for output, input grad, and weight grad; each of the three fixes is individually guarded). Direct and im2col were unaffected (2026-10-29).
- [ ] Integer shape/stride defaults (`shape.get(i).copied().unwrap_or(1)`, about 70 sites) are still silent defaults. Out of scope for the §8 pass (f32 data only), and they overlap the `Shape` field flip. `just check-fallbacks` also does not catch the multi-line `if let Some(&v) = x.get(i) { v } else { 0.0 }` form; none remain.
- [ ] Every op deep-copies storage; broadcasts are materialized (no stride-0 views). Phase 3/4.
- [ ] im2col copies the whole tensor per kernel element (`to_vec` inside the `kw_idx` loop).
- [ ] Conv input gradient scans all output positions per input pixel. It should iterate the ≤K² contributing positions.
- [ ] GPU reductions are single-threaded (`workgroup_size(1)`); `gpu_max_reduce` reads back to CPU.
- [ ] Broadcast backward pass 2 is O(targets × out_size).
- [ ] `check_gradients` doesn't zero pre-existing grads, so reusing a tensor gives false failures.
- [ ] Conv finite-difference tolerances are loosened (5e-2), and GPU consistency tests use all-ones data.
- [x] `test_conv_transpose2d_gradient_5x5` was flaky (1 in 80 runs). Seeded, with `eps=0.5` (same cause and fix as `..._multiple_channels`, `b6602af`). 0/400 seeds fail; still fails with the analytic gradient scaled by 1.1 (2026-10-28).
- [x] `test_direct_conv_gradient_multiple_configs` was flaky (4 in 80 runs). Seeded, with `eps=0.5`: 9/400 seeds failed at `5e-2` and 0/400 at `0.5`; still fails with the analytic gradient scaled by 1.1 (2026-10-28).
- [ ] `linear_tests::backward_gradcheck_input` (`src/lib.rs`) is flaky: 7 failures in 300 runs on `f1c07fe`, 2 in 300 after the Phase 2 close-out (2026-10-28). It checks `x @ w + b` with unseeded `Linear::new` weights through `check_gradients_simple`. Like the conv flakes the function is linear in `x`, but the cause is not investigated. Seed it, then see whether the step is the issue.
- [ ] More unseeded conv gradient tests are flaky, beyond the three fixed above. Measured 2026-10-29 by running `cargo test --no-default-features --lib gradient` 60 times: 7 runs failed on a clean `f1c07fe`-descendant `HEAD` (`test_conv2d_gradient`, `_stride2`, `_5x5`, `test_conv2d_igemm_gradient` (13/1000 on `HEAD` at 2cb1f9a, 3/1000 after the `TensorOps` removal, run as the test binary with `--exact`), `test_direct_conv_gradient_no_padding`, `test_conv_transpose2d_gradient`) and 4 on the §8 branch (`test_conv2d_gradient_5x5`, `test_conv2d_igemm_gradient_no_bias`, `test_direct_conv_gradient_no_padding`). `test_conv_transpose2d_gradient_1x1` (unseeded `randn`, default `eps`) also failed once inside a full `cargo test` run on 2026-10-29 and 0 of 60 times in isolation, so the indexing pass did not cause them. Same fix as the others: seed and use `eps=0.5` (conv is linear in `x`, so the step only affects f32 roundoff), then confirm 0/400 seeds fail and that a 1.1x-scaled analytic gradient still fails.
- [ ] CPU-only clippy (`cargo clippy --lib --no-default-features -- -D warnings`) fails with 13 distinct errors on `f1c07fe` and after Phase 2 (2026-10-28): unused GPU-only variables (`conv.rs`, `ops/unary.rs`, `ops/ternary.rs`), `step_gpu_param_simple` never used (`optim/sgd.rs`), and `must_use` / doc-backtick lints in `storage.rs` and `device.rs`. Most are the same missing not-gpu stub pattern fixed for movement. Only the default-feature clippy is a usable gate until then.
- [x] **`RawTensor::matmul_raw` was unsound** (`ops/matmul.rs`); it is `pub(crate)` and asserts both operand lengths now (see Phase 3, matmul layout contract). The entry below is the original report. It is a safe `pub fn` that passes `a.as_ptr()`/`b.as_ptr()` and caller-supplied `m, k, n` to `cblas_sgemm` / `matrixmultiply::sgemm` without checking `a.len() >= m*k` or `b.len() >= k*n`. Repro (`--no-default-features`): `matmul_raw(&[1.0; 2], &[1.0; 2], 4, 4, 4)` returns without a panic and reads adjacent heap memory (a `1.14e-43` denormal in the output). The macOS path also truncates dims with `as i32`. Internal callers (Muon, backward helpers) all pass consistent lengths, so it is latent. Fix: make it `pub(crate)`, or assert both lengths and use `i32::try_from`; add a `#[should_panic]` test for the repro above.
- [ ] Square weights load silently transposed (GEMM report §2). `load_weight` checks shapes exactly, which is blind when the matrix is square. `MultiHeadAttention`'s q/k/v projections are all square `Linear(embed_dim, embed_dim)`. `transpose_pattern("weight")` (`io/mapping.rs`, used by `load_pytorch_linear_weights` and `load_huggingface_weights`) also transposes every 2-D `*weight*` key, including `Embedding.weight` and `LSTMCell.weight_ih/hh`, which already use PyTorch's layout. Loud for non-square shapes, silent for square ones. Fix: narrow the pattern (explicit `Linear` keys, or skip layers already in PyTorch layout), document the per-layer layout table next to `StateDictMapper` (`Linear` is `[in, out]`; `LSTMCell` is `[4H, in]`; `Embedding` is `[vocab, dim]`; PyTorch `nn.Linear` is `[out, in]`), and add a loader test on a square, asymmetric `Linear` that checks forward values against a PyTorch-computed output.
- [ ] Tests that cannot see a transpose (GEMM report §4). `test_batched_matmul` (`lib.rs:665`) uses all-ones inputs. `test_chained_transformations` (`io/mapping.rs:490`) transposes `[1,2,3,4]` as 2x2 and asserts only the shape, and its comment wrongly calls the matrix symmetric. `assert_matmul_parity` (`tests/common/mod.rs`) backprops a plain `.sum()` (so `dC` is all ones and an indexing bug in the GPU backward reads the same value everywhere), and its shapes (`3x4 @ 4x5`, `8x8`) all fit inside one 16x16 tile, so the multi-tile `k` loop and partial-tile edges never run against the CPU. Fix: assert data in the first, asymmetric data in the second, a randomly weighted loss in the third (reuse the `assert_matmul_gradients` pattern), and add a parity shape past one tile and not a multiple of 16 (for example `37x19 @ 19x23`).
- [ ] No reference matmul exists (GEMM report §5). The CPU oracle is Accelerate or `matrixmultiply` depending on platform, so GPU parity compares against an optimized third-party kernel. Add a `#[cfg(test)]` triple-loop GEMM with explicit `(rs, cs)` per operand, and compare CPU and GPU matmul against it over random non-square shapes. Conv already has this pattern (direct vs im2col/iGEMM).
- [ ] The `BackwardParams` comments in `gpu/shaders/matmul_backward.wgsl` (lines ~12-16) describe `m` and `k` wrongly for `backward_b` (`grad` is `[M, N]`). The code is correct; fix the comments.
- [x] Doctests in `embedding.rs` and `pixelshuffle.rs` called the now-private `borrow()`. They use `shape()` now; `cargo test --doc --no-default-features` passes (2026-10-28).
- [ ] `Sequential::load_state_dict` prefix match is greedy (`"10.weight"` matches `"1."`).
- [ ] RNG reproducibility holes: the dataloader shuffle is unseeded, and `with_rng` falls back silently under a reentrant borrow.
- [x] `src/io/config.rs` was dead pseudocode that no module declared. Deleted (2026-10-28).
- [ ] The dtype system is decorative: every compute path is f32.
- [ ] `Embedding` implements `Module::forward` as a panic stub (trait split, Phase 3).
- [ ] Mixed-device binary ops silently produce CPU output (`DevicePolicy`, Phase 3).
- [ ] **GPU broadcast backward is broken, but unreachable** (moved out of Phase 2; it is a feature, not type work). Verified on `fff5397` (before `define_ops!`) with a 3x1 + 1x4 add: the race-free two-pass path (`binary_backward_broadcast_safe`) panics with a wgpu validation error (pass-1 bind group has 6 bindings; the auto layout keeps only the bindings the entry point uses: 3 for Add, 5 for Mul), and the deleted non-safe path returned wrong gradients silently (`[1,0,0]`/zeros, expected `[4,4,4]`/`[3,3,3,3]`). Nothing reaches it today because GPU forward never broadcasts, so a mismatched-shape op yields a CPU tensor and takes the CPU backward. Fixing it means an explicit bind group layout (or matching bindings per entry point) plus a GPU forward that broadcasts. Repro: `cargo test --lib gpu_binary_backward_broadcast_matches_cpu -- --ignored`. The orphaned non-safe pipelines are gone, so the two-pass path is the only one left to fix.

## Tests (OPUS §5)

- [x] `assert_trainable!` exists for every parameterized layer, with a weighted loss and seeded random inputs (2026-10-26).
- [ ] Add a regression test for multi-consumer (diamond) gradient accumulation.
- [ ] Add a gradient test for `PositionalEncoding` (B3).
- [ ] Remove example workarounds for broken components once the components are fixed.
  - `examples/char_transformer_model.rs` hand-rolls positional encoding (B3).
  - `examples/load_external_mnist.rs::run_validation` does not compare against PyTorch outputs; it skips the JSON parse and runs the demo inference.
  - `examples/vae.rs` falls back to constant synthetic data (0.5) when MNIST is missing.

## Rust / API (OPUS §7–14)

- [x] Opaque `Tensor` newtype (Phase 1).
- [x] One op-construction helper: `edge()` (Phase 1).
- [x] Library no longer prints; uses the `log` facade (Phase 0).
- [x] CI "no features" step actually runs `--no-default-features`, plus cargo-hack (Phase 0).
- [x] `approx` moved to a dev-dependency; `statrs` dropped (Phase 0).
- [ ] Pick one error regime (panic on programmer error, `Result` on runtime failure), document it in `dev-docs/DEVELOPMENT.md`, and prune redundant `try_*` variants. The §8 indexing pass added two panics that this decision should revisit (DECISIONS.md, "Indexing policy"): `data::transforms::to_one_hot` asserts `label < num_classes` (the label can be runtime data, so it may become a `Result`), and `binary_backward_broadcast_safe` panics on rank > 4 (H5; unreachable today).
- [ ] GPU layer returns `Option` for everything; move to `Result<_, GpuError>` with a single dispatch choke point.
- [ ] Rewrite the `storage.rs` unsafe block as owned data or a `OnceCell` cache (H3).
- [x] 16 positional `usize` arguments on GPU kernel wrappers. The eight conv wrappers take a conv spec and the three matmul wrappers take a `MatmulSpec` (Phase 2). No wrapper in `kernels.rs` takes three or more bare `usize`s now.
- [ ] Move about 5,000 lines of tests out of `lib.rs`.
- [ ] Stop exporting diagnostic tooling as public API.
- [ ] Merge gate: remove `#![allow(missing_docs, clippy::missing_docs_in_private_items)]` in `src/lib.rs` before merging into `main`. About 688 library items are undocumented (2026-10-28 count: `gpu/kernels.rs` 140, `tensor.rs` 91, `gpu/context.rs` 83, `spec.rs` 56, `ops/movement.rs` 26, the rest spread over 40 files). `examples/`, `benches/`, and `tests/` are done.

- [ ] Pedantic lint backlog (2026-10-29). `bacon`'s `pedantic` job no longer allows `cast_possible_wrap`, `cast_sign_loss`, `cast_possible_truncation`, `cast_precision_loss`, `needless_pass_by_value`, or `redundant_closure_for_method_calls`; it used to hide them behind "TEMP IMP TO FIX". The crate-wide casts were never enabled under `cargo clippy`, so the old `#[allow]`s for them were removed as no-ops. Decide per lint whether to enable it in `Cargo.toml` and fix the sites, or to leave it off for good.
- [ ] Root `CONTRIBUTING.md` and `DEVELOPMENT.md` are unchecked-box stubs, and `DEVELOPMENT.md` shares its name with `dev-docs/DEVELOPMENT.md`. Fill them in or delete the stubs.
- [ ] Consider `cargo-deny` (`deny.toml`) in place of the weekly `cargo audit` workflow: it adds license and duplicate-version checks.

## Phase 2 — types for dims and ops (done)

- [x] `Shape` / `Axis` / `Strides` defined (`f1cc5ea`).
- [x] `edge()` takes `Shape`; tensor constructors, `expand`, `pad`, `broadcast_to`, and `try_broadcast_shape` validate through `Shape::from_dims`; `Axis` on `sum_dim`/`max_dim`/`mean_dim`/`softmax`/`log_softmax`/`permute` (§1.2; `31e2edd`, `b12efe5`, `567946f`).
- [x] `Strides::from_shape` replaced `RawTensor::compute_strides` at all 21 call sites (`ops/movement.rs`, `ops/binary.rs`, `tensor.rs`); the free fn is gone. Call sites demote with `truncate_rank()`. The new body is a running product, with no `if let Some` skip.
- [x] Conv specs: `Conv2dSpec`, `ConvTranspose2dSpec`, `Window2dSpec` in `src/spec.rs` replace positional args in Conv2d, ConvTranspose2d, MaxPool2d, and the conv GPU wrappers (§1.3; DECISIONS.md, "Conv-family specs").
- [x] `MatmulSpec` in `src/spec.rs` (§1.2/§1.3). Retires B4. Covers rank ≥ 2 operands; the vector forms are untouched.
- [x] Closed `GradFn` enum instead of `Box<dyn GradFn>` (§2.5; DECISIONS.md, "Closed `GradFn` enum").
- [x] `define_ops!` macro (§5.1) for `UnaryOp` and `BinaryOp` (DECISIONS.md, "`define_ops!` for unary and binary ops"). Adds GPU `erf`/`erf_backward` kernels, which were named but never existed.
- [x] Movement ops share dispatch (§5.1 tail), without a generated table: `expand`, `pad`, `shrink`, and `stride_op` each pick GPU-or-CPU storage and end in one `RawTensor::movement_edge` call, so the recorded `GradFn` can't differ between the paths. A second macro grammar was declined (DECISIONS.md). The five forward `gpu_*` wrappers now have not-gpu stubs, so `movement.rs` has no forward `cfg`; that also cleared 5 CPU-only clippy errors (2026-10-28).
- [x] Flaky test: `conv_transpose::tests::test_conv_transpose2d_gradient_multiple_channels` failed about 3% of runs. Fixed in `b6602af`: seeded, and `eps=0.5` (the layer is linear in its input, so the step only affects f32 roundoff; 13/400 seeds failed at 5e-2, 0/400 at 0.5). Tolerance unchanged.
- [x] Removed the orphaned non-safe broadcast-backward GPU pipelines: `ComputePipelines::{add,sub,mul,div,max}_broadcast`, the matching entry points in `binary_backward.wgsl`, and the helpers and `params` binding only they used. `just validate-shaders` passes; `gpu_binary_backward_same_shape_matches_cpu` and `tests/movement_backward_test.rs` pass on a real adapter (2026-10-28).
- [x] `PadValue` (§2.4) in `src/spec.rs`, re-exported from the crate root. `pad` is gone; `pad_with`/`try_pad_with` take a required value. Retires H1.
- [x] `into_loss` / `ScalarLoss` (§2.2). `TensorOps::backward` is gone; `Tensor::into_loss()?.backward()` is the only entry point, and `tests/compile_fail/backward_on_tensor.rs` pins that. `assert_trainable!` and `check_gradients` use it.
- [x] Flaky test: `conv::conv2d_tests::test_direct_conv_gradient_multiple_configs`. Duplicate of the Medium entry above; fixed there.
- [x] `tests/gpu_test.rs` had an unconditional `use volta::gpu;`, so `cargo test --no-default-features --tests` failed to compile it. Gated on `feature = "gpu"`; both feature sets compile (2026-10-28).

## Phase 3 — traits and storage

- [ ] Sealed and split traits with trybuild tests (§3). Retires H6 and the Embedding panic.
- [ ] `DevicePolicy` (§2.1). Retires the silent GPU→CPU fallback.
- [ ] Storage: `Arc` / `OnceCell` / no `Deref` (§7). Retires H3 and the clone economy. Matmul operand reads must materialize contiguous (DECISIONS.md, "Matmul layout: contiguous-only").
- [ ] Flip `RawTensor.shape` to `Shape` and make `Tensor::shape()` return it. Moved here from Phase 2: it touches about 440 kernel reads of the field, the same sites the storage rewrite rewrites (DECISIONS.md, "Shape boundary").
- [x] Indexing policy (§8): `crate::index::{fetch, slot, span}` panic on out of bounds in every profile (not NaN: DECISIONS.md, "Indexing policy"). All `unwrap_or(0.0)` and the positional `if let Some` reads and writes of f32 data are converted (library via the helpers, tests, examples, and benches via `.expect`), including `broadcast_to` / `sum_over_broadcast_dims`. `just check-fallbacks` (in `just ci` and pre-commit) keeps it that way. It also made `to_one_hot` and the GPU broadcast-backward param packing reject out-of-range input loudly. The integer shape defaults are a follow-up (Medium).
- [x] Delete `TensorOps` (§3.4). The methods are inherent on `Tensor` (DECISIONS.md, "Delete `TensorOps`"). `pkg/README.md` and `minigrad-react/src/wasm/README.md` are separate copies of the old README and still mention it.
- [x] Matmul layout contract (ADDENDUM §7, "Matmul under views"): contiguous-only (DECISIONS.md, "Matmul layout: contiguous-only"). `matmul_raw` is `pub(crate)` and asserts operand lengths; fence tests pass today and guard it once views land.

## Phase 4 — endgame (optional, last)

- [ ] Arena graph (§6), with the `TensorIndex`/`NodeIndex` newtypes (§1.3) it makes meaningful.
- [ ] Sealed `Backend` trait (§3.5).
- [ ] proptest DAG checks (§10.2).
  - Include `matmul` and `permute` with square and non-square operands, so cancelling transpose bugs get caught (GEMM report §4). `cargo-mutants` (§10.4) should flag the blind tests listed under Medium.

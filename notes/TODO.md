# TODO — live checklist

The plan is `docs/plans/ADDENDUM.md` § Sequencing. Defect evidence is in `docs/audits/OPUS_5_EVAL_AUDIT.md` (OPUS §n), with sources in `AUDIT_CITATION.md`.
The full long-form notes (audit transcription plus the preventive-architecture sketches) are in git history: `git show d771013:notes/TODO.md`.

**Status key:** `[x]` = verified fixed (evidence given). `[~]` = partly fixed. `[ ]` = open per the audit (2026-09-08) and not re-verified unless a note says so.
Last verified: 2026-10-26.

## Blocking (OPUS §1)

- [x] **B1** ConvTranspose2d weights get no gradient. Fixed by `edge()`; `test_conv_transpose2d_assert_trainable` passes.
- [ ] **B2** LSTMCell `slice_gate` severs the graph (`lstm.rs` gate slice → `RawTensor::new`). The test is `#[ignore]`d under B2; fix with `shrink` or a slice GradFn, then un-ignore.
- [ ] **B3** `PositionalEncoding::forward` rebuilds its output from raw data in both modes (`positional_encoding.rs:163,196`). It has no trainability test. The fix composes it from graph ops and adds `assert_trainable!` for learned mode. `examples/char_transformer_model.rs` hand-rolls positional encoding around this bug.
- [~] **B4** Unequal-rank batched matmul backward. 3D@2D now *panics in forward* ("Batched matmul requires rank >= 3"), so it is loud instead of wrong. Unequal ranks ≥3 are not re-verified. The structural fix is a `MatmulSpec` (Phase 2); then add a test.
- [x] **B5** The CPU-only build failed to compile. Fixed in `c8a5006`; `--no-default-features` builds and tests (2026-10-26).

## High (OPUS §2)

- [ ] **H1** MaxPool2d pads with `0.0` instead of `-inf`. Re-verified 2026-10-26: all-`-5` input with pad=1 outputs `0.0`. Fixed by `PadValue` (Phase 2).
- [x] **H2** `bce_with_logits_loss` now uses `max(x,0) - x*y + ln(1+exp(-|x|))`; `tests/bce_with_logits.rs` checks loss and gradient at ±100 and at x=0.
- [~] **H3** Unsafe GPU cache accessors. `invalidate_cpu_cache` now takes `&mut self` (`991a8b7`). The `unsafe` slice accessors and the f64/f16/bf16 GPU reinterpretation remain. Fixed by `OnceCell` storage (Phase 3).
- [ ] **H4** The library panics on ordinary data. Panic-vs-`Result` policy is undecided (see Rust/errors).
- [ ] **H5** GPU broadcast backward silently truncates rank > 4.
- [ ] **H6** TransformerBlock is decorative: parameterless `SimpleAttention`. Trait split §3.2 (Phase 3).
- [x] **H7** `load_state_dict` demoted GPU tensors to CPU. Every layer now goes through `Tensor::load_weight`, which checks shape (and numel) and keeps the destination's device. Evidence: `tests/load_state_dict_device.rs`, plus `tensor::load_weight_tests` on GPU. `load_state_dict_checked` now reports shape mismatches and skips them instead of loading them (2026-10-26).
- [ ] **H8** Muon is a toy and panics on GPU params.

## Medium (OPUS §3)

- [ ] Every op deep-copies storage; broadcasts are materialized (no stride-0 views). Phase 3/4.
- [ ] im2col copies the whole tensor per kernel element (`to_vec` inside the `kw_idx` loop).
- [ ] Conv input gradient scans all output positions per input pixel. It should iterate the ≤K² contributing positions.
- [ ] GPU reductions are single-threaded (`workgroup_size(1)`); `gpu_max_reduce` reads back to CPU.
- [ ] Broadcast backward pass 2 is O(targets × out_size).
- [ ] `check_gradients` doesn't zero pre-existing grads, so reusing a tensor gives false failures.
- [ ] Conv finite-difference tolerances are loosened (5e-2), and GPU consistency tests use all-ones data.
- [ ] `test_conv_transpose2d_gradient_multiple_channels` is flaky (about 1 in 15 runs): unseeded random init against a finite-difference tolerance.
- [ ] `test_direct_conv_gradient_multiple_configs` is flaky too: 4 failures in 80 runs on a clean `HEAD`, 3 in 80 after the spec change (2026-10-26). Same cause: unseeded `randn` against a finite-difference tolerance.
- [ ] Doctests in `embedding.rs` and `pixelshuffle.rs` call the now-private `borrow()` and fail under `cargo test --doc`.
- [ ] `Sequential::load_state_dict` prefix match is greedy (`"10.weight"` matches `"1."`).
- [ ] RNG reproducibility holes: the dataloader shuffle is unseeded, and `with_rng` falls back silently under a reentrant borrow.
- [ ] `src/io/config.rs` is dead pseudocode that no module declares. Delete it.
- [ ] The dtype system is decorative: every compute path is f32.
- [ ] `Embedding` implements `Module::forward` as a panic stub (trait split, Phase 3).
- [ ] Mixed-device binary ops silently produce CPU output (`DevicePolicy`, Phase 3).

## Tests (OPUS §5)

- [x] `assert_trainable!` exists for every parameterized layer, with a weighted loss and seeded random inputs (2026-10-26).
- [ ] Add a regression test for multi-consumer (diamond) gradient accumulation.
- [ ] Add a gradient test for `PositionalEncoding` (B3) and one for unequal-rank matmul (B4).
- [ ] Remove example workarounds for broken components once the components are fixed.

## Rust / API (OPUS §7–14)

- [x] Opaque `Tensor` newtype (Phase 1).
- [x] One op-construction helper: `edge()` (Phase 1).
- [x] Library no longer prints; uses the `log` facade (Phase 0).
- [x] CI "no features" step actually runs `--no-default-features`, plus cargo-hack (Phase 0).
- [x] `approx` moved to a dev-dependency; `statrs` dropped (Phase 0).
- [ ] Pick one error regime (panic on programmer error, `Result` on runtime failure), document it in `dev-docs/DEVELOPMENT.md`, and prune redundant `try_*` variants.
- [ ] GPU layer returns `Option` for everything; move to `Result<_, GpuError>` with a single dispatch choke point.
- [ ] Rewrite the `storage.rs` unsafe block as owned data or a `OnceCell` cache (H3).
- [~] 16 positional `usize` arguments on GPU kernel wrappers. The eight conv wrappers now take a spec (Phase 2). The matmul wrappers' `m, k, n` wait for `MatmulSpec`.
- [ ] Move about 5,000 lines of tests out of `lib.rs`.
- [ ] Stop exporting diagnostic tooling as public API.
- [ ] Merge gate: remove `#![allow(missing_docs)]` in `src/lib.rs` before merging into `main`.

## Phase 2 — types for dims and ops (in progress)

- [x] `Shape` / `Axis` / `Strides` defined (`f1cc5ea`).
- [x] `edge()` takes `Shape`; tensor constructors, `expand`, `pad`, `broadcast_to`, and `try_broadcast_shape` validate through `Shape::from_dims`; `Axis` on `sum_dim`/`max_dim`/`mean_dim`/`softmax`/`log_softmax`/`permute` (§1.2; `31e2edd`, `b12efe5`, `567946f`).
- [ ] Flip `RawTensor.shape` to `Shape` and make `Tensor::shape()` return it. Deferred with the storage work (§7): about 440 kernel reads of the field (DECISIONS.md, "Shape boundary").
- [ ] `Strides` is defined but unused; `RawTensor::compute_strides` is still a free fn returning `Vec<usize>`.
- [x] Conv specs: `Conv2dSpec`, `ConvTranspose2dSpec`, `Window2dSpec` in `src/spec.rs` replace positional args in Conv2d, ConvTranspose2d, MaxPool2d, and the conv GPU wrappers (§1.3; DECISIONS.md, "Conv-family specs").
- [ ] `MatmulSpec` (§1.2/§1.3). Retires B4. Goes in `src/spec.rs`.
- [ ] Closed `GradFn` enum instead of `Box<dyn GradFn>` (§2.5).
- [ ] `define_ops!` macro (§5.1).
- [ ] `PadValue` (§2.4). Retires H1.
- [ ] `into_loss` / `ScalarLoss` (§2.2); switch `assert_trainable!` to it.

## Phase 3 — traits and storage

- [ ] Sealed and split traits with trybuild tests (§3). Retires H6 and the Embedding panic.
- [ ] `DevicePolicy` (§2.1). Retires the silent GPU→CPU fallback.
- [ ] Storage: `Arc` / `OnceCell` / no `Deref` (§7). Retires H3 and the clone economy.
- [ ] Indexing policy plus `fetch` (NaN, not zero) (§8). Retires the silent-zero class.
- [ ] Delete `TensorOps` (§3).

## Phase 4 — endgame (optional, last)

- [ ] Arena graph (§6), with the `TensorIndex`/`NodeIndex` newtypes (§1.3) it makes meaningful.
- [ ] Sealed `Backend` trait (§3.5).
- [ ] proptest DAG checks (§10.2).

# Volta Improvement Plan

**Date:** 2026-07-09
**Method:** Full-source review of the tensor/autograd core, ops, nn layers, optimizers, storage, and GPU paths, plus a clean run of the test suite (519 unit + ~130 integration tests, all passing) and `cargo clippy --all-targets` (clean under default features).

This document is grounded in specific file/line evidence rather than general impressions. It supersedes the older audit documents in `notes/` (`SENIOR_REPORT.md`, `REPO_HEALTH.md`) where they conflict.

---

## 1. How well has the project achieved its goals?

Volta's stated goals (README/CLAUDE.md): educational clarity, correctness-first, PyTorch-like API, verified gradients, experimental GPU acceleration.

**Where it genuinely succeeds:**

- **The autograd engine is correct and well-designed.** Iterative post-order DFS with proper topological ordering (`src/autograd.rs`) handles diamond graphs correctly, avoids stack overflow on deep graphs, and accumulates gradients properly. This is the heart of the project and it's solid.
- **Gradient verification culture is real.** Central-difference gradient checking with a sensible small-magnitude fallback (`src/tensor.rs:889-971`), used pervasively across ~650 passing tests including CPU/GPU parity suites.
- **API breadth is impressive for a solo educational project:** broadcasting, 16 layers including a working transformer stack, three conv algorithms with auto-selection, SafeTensors interop with a genuinely useful `StateDictMapper`, named-layer serialization, 16 runnable examples.
- **Tooling hygiene:** CI on two OSes, pre-commit hooks, an unusually thorough clippy config, weekly `cargo audit`.

**Where the goals are undermined:**

- **The correctness claim has real holes** — a handful of concrete bugs (§2) including one instance of undefined behavior, silent gradient-dropping in GPU backward paths, and a loss function whose docstring promises numerical stability it doesn't have.
- **The "defensive programming" style backfires** (§3.1). Denying `indexing_slicing` pushed hot numeric loops into `if let Some(...)` chains that _silently skip work_ when an index is wrong. For a library whose selling point is verified correctness, converting loud panics into silent zeros is the wrong trade — a bug in an index computation now produces a subtly wrong gradient instead of a crash the test suite would catch.
- **The GPU backend costs more than it returns right now** (§3.2). It roughly doubled the codebase's complexity (kernels.rs alone is 3,000 lines), introduced the only `unsafe` soundness bug, and is effectively untested in CI because every GPU test silently skips when no adapter exists (all GPU test binaries complete in 0.00s).
- **Documentation has drifted.** README says transformers/attention are missing (they exist and work); CLAUDE.md says "No RNN/Transformer layers"; Cargo.toml makes `gpu` + `accelerate` **default features** while README documents them as opt-in; CI steps labeled "no features" actually run with defaults (i.e. with GPU code compiled in).

**Verdict:** As an educational autograd engine, the project has substantially achieved its goal — the core is correct, tested, and readable in structure. As a _codebase_, it's carrying accumulating drift: stale docs, duplicated audit reports, silent-failure patterns, and a GPU layer growing faster than its verification. The highest-value work is consolidation and correctness hardening, not new features.

---

## 2. Concrete bugs

### P1-1: `xavier_uniform` panics for any non-2D shape — `src/tensor.rs:254-265`

It samples `fan_in * fan_out` values (from `shape[0]`, `shape[1]`) but constructs the tensor with the _full_ shape — `RawTensor::new` asserts data length == shape product, so any rank-3+ shape panics, and the fan computation ignores receptive fields entirely. It only works today because `Linear` is its sole caller. Fix by sizing from `shape.iter().product()` and computing fans the way `he_initialization` does (which handles conv shapes correctly).

### P1-2: `load_state_dict` silently moves GPU models to CPU and desyncs the `device` field — `src/nn/layers/linear.rs:37-48`, same pattern in batchnorm, conv, etc.

Every layer's `load_state_dict` does `t.data = Storage::cpu(...)` without touching `t.device`. Loading a checkpoint into a `new_on_device(GPU)` model leaves tensors whose `device` says GPU but whose storage is CPU. Downstream, `common_gpu_device()` reports GPU and kernels receive CPU storage — behavior then depends on each kernel's failure handling (see P0-2 for how that ends). **Fix:** in `load_state_dict`, restore data onto the tensor's existing device (`Storage::cpu(...).to_device(&t.device)`), or set `t.device = Device::CPU` and document that callers must re-`to_device`. The former matches PyTorch semantics.

### P1-3: `backward()` on non-scalar outputs silently seeds all-ones — `src/autograd.rs:50-61`

PyTorch errors on `backward()` from a non-scalar without an explicit gradient; Volta fills the gradient with ones. The special case for shape `[1]` computes the same `grad_size` as the general path, so the branch is dead code. Users who call `.backward()` on a non-reduced tensor get mathematically valid-looking but almost-certainly-unintended gradients. **Fix:** assert the tensor is scalar (or accept an explicit seed gradient parameter). This is a one-line guard that prevents a whole class of silent user error.

### P1-4: `Sequential::load_state_dict` prefix collision — `src/nn/layers/sequential.rs:51-83`

Layer prefix matching uses `key.starts_with("{i}.")`. For an unnamed model with ≥ 10 parameterized layers, layer `1`'s prefix `"1."` also matches keys `"10.weight"`, `"11.bias"`, … The sub-key extraction then produces keys like `"0.weight"` which won't match — but wrong keys can shadow correct lookups and the failure is silent (loading skips unmatched keys without error). **Fix:** split on the first `.` and compare the segment exactly. Also consider making `load_state_dict` return a `Result` reporting missing/unexpected keys — today a typo'd mapping loads nothing and reports nothing (the README's own external-loading example would fail silently if a key were misspelled).

### P2 (smaller, real):

- **`reshape` claims to be a view but deep-copies on CPU** (`src/ops/movement.rs:618-647`): the comment says "data stays in place," but `Storage::clone` on CPU copies the whole `Vec<u8>`. Misleading, and it matters because reshape is on every hot path (BatchNorm reshapes twice per forward).
- **Mixed-device binary ops silently produce CPU results** (`src/ops/binary.rs:607-688`): GPU⊕CPU inputs fall to the CPU path and the output stays on CPU, whereas `sum_dim`/`max_dim` move results back to the input device. Pick one policy (error on mixed devices, like PyTorch, is the honest one) and apply it everywhere.
- **`try_new` hard-caps tensors at 100M elements and forbids zero-element tensors** (`src/tensor.rs:120-129`): arbitrary, undocumented in the API surface, and `broadcast_to` has its own separate 100M assert. A 100M-element tensor is 400 MB of f32 — legitimate for embeddings.
- **`Adam::step_cpu_param` panics if a CPU parameter has a GPU gradient** (`expect("State should be CPU storage")`) — reachable through the P1-2 desync above.
- **Muon** (`src/nn/optim/muon.rs`) uses the cubic Newton–Schulz iteration and a magic `* 0.1` stability factor; standard Muon uses a tuned quintic `(3.4445, −4.7750, 2.0315)` and normalizes per the smaller matrix dimension. Fine as "experimental," but label it in code, not just README.

---

## 3. Poor decisions worth reversing (judgment calls, not bugs)

### 3.1 `indexing_slicing = deny` produces silent-skip numerics — the single most damaging pattern in the codebase

The intent (no panicking indexing) is reasonable for a parser; for a numerics library it has produced hundreds of loops shaped like:

```rust
if let Some(val) = grad_data.get(grad_idx)
    && let Some(slot) = result.get_mut(i)
{
    *slot = *val;
}
// else: silently do nothing — the gradient element stays 0.0
```

If an index computation is ever wrong (the exact bug class this codebase is most at risk of, given all the manual stride arithmetic), the result is a **silently wrong gradient**, not a panic. Numerical gradient checks catch this only for the shapes the tests happen to exercise. It also buries the actual math under bounds-checking noise, hurting the educational goal, and adds branch overhead in hot loops.

**Recommendation:** keep the lint, but change the idiom in numeric kernels:

- Prefer iterator/`chunks_exact`/`zip` formulations that make bounds violations impossible by construction (many loops here can be rewritten this way and get _faster_).
- Where direct indexing is genuinely needed, use `#[allow]` on the small kernel function with a debug assertion of the invariant, or `.expect("index invariant: <why>")` — a loud failure is the correct behavior for an internal invariant violation in a correctness-first library.
- Never write an `if let Some` whose `else` silently skips a write in a gradient computation.

### 3.2 The GPU backend's cost/benefit is currently inverted

Facts on the ground: `gpu` is a default feature (contradicting docs); GPU tests silently no-op without an adapter, so CI has never executed a GPU kernel; the GPU code contains the only UB (P0-1) and the gradient-dropping fallbacks (P0-2); the benchmark suite crashes under full load (README admits this); every GPU op falls back to CPU by returning `Option`, with `eprintln!` warnings as the only signal; and `Storage`'s `Deref<Target=[f32]>` means any innocent `data.get(i)` triggers a full GPU→CPU readback invisibly.

**Recommendations, in order of leverage:**

1. **Make `gpu` opt-in again** (`default = []`, or at most `["accelerate"]` guarded to macOS docs). This matches the README, shrinks default builds enormously (wgpu is a heavy dependency tree), and stops shipping the riskiest code to every consumer.
2. **Get an actual GPU into CI** via a software adapter (Mesa `lavapipe`/`llvmpipe` works with wgpu on Linux runners) so the parity tests execute rather than skip. Until then, the honest status for GPU is "unverified in CI."
3. **Replace silent `Option`-fallback with a single choke point**: one `dispatch(op) -> Result<Storage, GpuError>` that either logs-and-falls-back _consistently_ (including moving results back to the right device) or errors. Today each op hand-rolls its own fallback and three of them got it wrong (§2).
4. **Remove `Deref`/`Index` from `Storage`.** They panic on non-f32 (contradicting the defensive philosophy), hide device transfers, and encourage exactly the CPU-touching patterns that make the GPU path pointless. An explicit `.as_f32_slice()` at least shows up in code review; `&data[i]` doesn't.

### 3.3 Documentation and repo hygiene drift

- **Stale claims:** README "What's Missing: full attention mechanisms" vs. a working `MultiHeadAttention`/`TransformerBlock`; CLAUDE.md "No RNN/Transformer layers"; README example list references `lstm_time_series`/`polynomial_regression` while Cargo.toml defines `time_sequence`/`regression`; README installation implies CPU-only default while Cargo.toml defaults to `gpu + accelerate`.
- **`notes/` is a graveyard of five overlapping LLM-generated audits** (SENIOR_REPORT, REPO_HEALTH, REFACTOR_SUGGESTIONS, ADAPTATION_ACCURACY, analysis/) with contradictory dates and version references (one says v0.1.0). Anyone (human or agent) reading the repo gets conflicting guidance. Consolidate to one living document; archive or delete the rest. Consider whether `sys-prompts/` and `responses/` belong in the published crate/repo at all (`cargo package` will include them unless excluded).
- **`src/lib.rs` is 5,051 lines, ~4,900 of which are tests.** Move test modules to `tests/` or per-module `#[cfg(test)]` files; lib.rs should be the crate's front door.
- **CI is mislabeled and incomplete:** the "no features" steps run default features (which now include `gpu`). Add an explicit matrix: `--no-default-features`, `--features gpu`, `--all-features`; this also catches the `#[cfg]` breakages that feature-gated code is prone to.

### 3.4 Dual panicking/`try_` API without a policy

Roughly half the operations have `try_*` variants; the `TensorOps` trait exposes only panicking versions; some ops assert, some return `Result`, some do both in different code paths. Pick a rule — e.g. _"user-facing `TensorOps` panics on programmer error (shape mismatch); everything that can fail at runtime (IO, device) returns `Result`"_ — write it in DEVELOPMENT.md, and delete the `try_` variants that exist only to satisfy the lint config. Every dual entry point is ~30 lines of maintenance and doc burden.

---

## 4. Performance (deliberately last)

The README is honest that performance isn't the goal, so these are noted for when it becomes one. The dominant costs, in order:

1. **Every op deep-copies its inputs.** The `let (data, shape, ...) = { s.borrow(); (s.data.clone(), ...) }` prologue clones full buffers (CPU `Storage::clone` copies the `Vec`) before any math happens. Holding the borrow through the computation (or wrapping CPU storage in `Rc<[f32]>`) removes an O(n) copy from literally every operation — likely the single biggest CPU win available, with no architectural change.
2. **Broadcasting materializes both operands** (`broadcast_to` allocates the full output-shaped copy of each input) and computes index mappings with a division per dimension per element. Index-arithmetic-in-the-loop (or the classic stride-0 trick) avoids both allocations.
3. **Axis reductions and softmax always run on CPU**, then `to_device` back — a GPU forward pass round-trips per `sum_dim`. Also `softmax`/`log_softmax` chain 5–6 full-tensor primitive ops; they're worth fusing eventually.
4. `matmul` is already routed through `matrixmultiply`/Accelerate — good; batched matmul re-broadcasts full tensors (see #2).

Non-goal: don't migrate `Rc<RefCell>` → `Arc<Mutex>` speculatively. Single-threaded is a legitimate, documented scope decision for an educational engine; the migration would touch everything and buy nothing until there's a concrete parallelism plan.

---

## 5. Recommended sequence

**Phase 0 — correctness (small diffs, high value, do first):**

1. Fix P0-1 (`invalidate_cpu_cache` soundness) — `&mut self` is a one-line signature change plus fallout.
2. Fix P0-2 (gradient-dropping fallbacks) with a shared "all required grads computed" helper + regression tests.
3. Fix P0-3 (`bce_with_logits_loss`) + gradient test at extreme logits.
4. Fix P1-1 through P1-4 (each is < 30 lines).

**Phase 1 — stop the bleeding on the GPU/feature front:** 5. `default = []`; fix README/CI labels; add the feature matrix to CI. 6. Software-rasterizer GPU in CI so parity tests actually run. 7. Centralize GPU dispatch/fallback; remove `Deref`/`Index` from `Storage`.

**Phase 2 — hygiene that compounds:** 8. Move tests out of lib.rs; consolidate `notes/` into one living doc; fix stale README/CLAUDE.md claims; exclude non-crate dirs in `Cargo.toml` (`exclude = ["notes", "responses", "sys-prompts", "docs/plans"]`). 9. Establish the panics-vs-Result policy; prune redundant `try_` duplicates. 10. Replace silent-skip `if let Some` numerics with iterator forms or invariant `expect`s, module by module (start with `autograd.rs`, `binary.rs`, `movement.rs` — the gradient-critical paths).

**Phase 3 — when performance becomes a goal:** 11. Eliminate per-op input clones (borrow or `Rc` storage). 12. Virtual broadcasting (stride-0), then fused softmax/layernorm. 13. GPU-resident reductions; only then revisit GEMM tiling.

The through-line: **Volta's differentiator is verified correctness. Every hour spent on Phase 0/1 defends that claim; features added before then inherit the silent-failure patterns.**

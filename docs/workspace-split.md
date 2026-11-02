# Should Volta become a workspace of crates?

tldr: don't split for now, something to handle much later

Evaluation, 2026-11-02, against `improvements` at `039c4a2`. This is a recommendation, not a roadmap item: `docs/plans/ADDENDUM.md` § Sequencing stays the only plan, and nothing here is scheduled until it is added there.

## TL;DR

- **Don't split now.** The library is one strongly connected component: `tensor` ↔ `autograd` ↔ `ops` ↔ `nn` ↔ `io`, with `storage` → `gpu` → `ops`/`tensor` closing a second loop. Any cut along today's module lines fails to compile, or compiles only after `edge()`, `GradFn`, and `RawTensor`'s fields become `pub`.
- **Making them `pub` breaks invariants 1–3.** Those hold today because `edge()`, `grad_fn`, and `RawTensor`'s fields are `pub(crate)` (231 `pub(crate)` items in all). A second crate can only see them if they are public, and then any user can attach a `grad_fn`. That is how B2/B3 shipped.
- **The payoff is small.** About 29.5k lines under `src/` (inline test modules included) plus 5.2k in `src/lib_tests/` and 4.8k in `tests/`. An incremental `build --tests` after a real edit takes about 2.5 s, and building the crate from scratch takes 4.2 s (see [Measurements](#measurements)). Neither justifies the migration cost: splitting won't make wgpu compile any faster, and the `gpu` feature already keeps wgpu out of CPU-only and WASM builds. The open wasm32 failure (`notes/TODO.md`) is `getrandom`'s backend, which a split would not fix.
- **The right seam is Phase 4's sealed `Backend` trait (ADDENDUM §3.5).** Once it exists, a `volta-wgpu` backend crate falls out naturally. Cutting before then would draw the lines in the wrong places, and they would have to be cut again.
- **Do the in-crate layering work now.** Steps 0a–0d below cost little, are worth doing even if the crate never splits, and make a later split mechanical. Each step is one PR, and `just ci` checks it.

## What I looked at

The dependency graph comes from `crate::<module>` references in each module, with `lib_tests/` excluded. Counts are reference sites.

```
Cycles (each arrow is a `use crate::...`; every cycle must stay inside one crate):

  tensor ──► ops ──► autograd ──► tensor          (ops are `impl Tensor`; edge() builds Tensors)
  autograd ──► nn::layers::{conv, maxpool, embedding} ──► autograd      (closed GradFn enum)
  io ──► nn (Module) ──► io (StateDict, TensorData);   tensor ──► io (TensorData)
  storage ──► gpu ──► ops (BinaryOp, UnaryOp), spec ──► ... ──► storage
  device ──► gpu;   nn::optim ──► gpu

Leaves: shape, dtype, error, index, data, utils   (spec mentions Tensor only in a doc link)
```

| Edge                                                                         | Why it exists                                                                                                                                  | Blocks a split because                                                                                                                               |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `autograd → nn::layers::{conv, conv_transpose, maxpool, embedding}`          | The closed `GradFn` enum has variants for these ops' backward structs (DECISIONS "Closed `GradFn` enum" already names this as a layering cost) | `GradFn` must be in the same crate as every variant's struct, so those kernels can't live in an `nn` crate                                           |
| `ops/* → impl Tensor` (9 files)                                              | Ops are inherent methods on `Tensor`                                                                                                           | Inherent impls must be in `Tensor`'s crate. Moving ops out would bring back an extension trait, which "Delete `TensorOps`" (§3.4) removed on purpose |
| `io ↔ nn`                                                                    | `io` uses `Module`; layers use `StateDict`/`TensorData`; `tensor::load_weight` uses `TensorData`                                               | A cycle at the crate level                                                                                                                           |
| `storage → gpu`, `gpu::kernels → ops::{BinaryOp, UnaryOp}`, `spec`, `tensor` | GPU storage and the op enums the kernels dispatch on                                                                                           | `gpu` can't sit below core or above it; it sits _inside_ it                                                                                          |
| `nn → crate internals`                                                       | 13 `edge()` calls, 209 `.borrow()`, 83 `.borrow_mut()`, 115 `index::{fetch, slot, span, ...}` calls                                            | `nn` is not a client of the public API. It is part of the core                                                                                       |

Only `data` (190 lines: MNIST plus transforms, depending on `error`, `index`, `shape`) and `utils::progress` (79 lines) are true leaves.

### A side finding

Invariant 3 ("only `accumulate_grad` and `set_grad` write `grad`") is broken today: `zero_grad` writes `grad = None` directly in `src/nn/optim/sgd.rs:65`, `src/nn/optim/adam.rs:75`, and `src/nn/mod.rs:65`. So does `RawTensor::zero_grad` (`src/tensor.rs:1375`), which already exists and which the `nn` sites could call. Strictly, it is a third writer too: either the invariant should name it or it should go through `set_grad`. The 83 `borrow_mut()` calls in `nn` likewise go against invariant 2. A crate boundary would turn both into compile errors, which is the one real benefit of splitting. Step 0c gets that benefit without the split.

## Costs and benefits

|                                                                           | Split now                                                                                                                                                                                                                                                                                | Stay single crate (+ step 0)           |
| ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| Encapsulation of `edge`/`GradFn`/`RawTensor`                              | Weakened (`pub` or `#[doc(hidden)] pub`; trybuild can't pin "hidden")                                                                                                                                                                                                                    | Kept (`pub(crate)`)                    |
| Layering enforced by the compiler                                         | Only if internals stay private, which needs step 0 first; with `pub` internals, no                                                                                                                                                                                                       | Only by review, until 0d adds a check  |
| Incremental build                                                         | 0.7 s lib / 2.5 s tests today. Gains only when the edit is in a leaf crate, and most edits are in core                                                                                                                                                                                   | Unchanged                              |
| Tooling to port                                                           | `clippy.toml`, `check-fallbacks.sh` (`dirs=(src tests examples benches)` hard-coded, so it would **pass vacuously** after files move to `crates/`), `validate-shaders` path, 6 trybuild `.stderr` files (crate paths in diagnostics), CI lanes, pre-commit, `build.rs` (Accelerate link) | None                                   |
| Unit tests (`src/lib_tests/`, 5.2k lines, 13 of 18 files touch internals) | Must be split per crate and keep their internal access                                                                                                                                                                                                                                   | None                                   |
| Publishing                                                                | N crates to version and publish in order; every `pub` internal becomes semver surface                                                                                                                                                                                                    | One crate                              |
| Roadmap fit                                                               | Cuts before the Backend seam exists, so it will be cut again                                                                                                                                                                                                                             | Leaves Phase 4 free to choose the seam |

## Recommendation

1. **Now:** do step 0 (in-crate layering) as ordinary TODO items. Do not create a workspace.
2. **Revisit when any one of these is true:**
   - The Phase 4 `Backend` trait lands. This is the expected trigger.
   - A second in-repo consumer appears, for example the WASM binding moving into this repo as its own crate (`notes/TODO.md` wasm32 item).
   - An incremental `cargo build --tests` after a one-line edit in `nn` regularly takes over about 30 s on the Mac. Measure it; don't go by feel.
3. **When triggered,** follow the plan below. The plan is written so that step 0 is useful even if no trigger ever fires.

Rejected alternatives:

- **A single-member workspace now** (`[workspace]` + `[workspace.lints]`, one member). It gets nothing while there is one crate and moves every path today.
- **Extracting only `volta-data`.** 190 lines buys nothing in build time, and it adds a crate to publish. The only reason to do it is as the first step of a real split, to prove the tooling works (step 2 below).
- **`#[doc(hidden)] pub mod __internal`** to reach internals across crates. It is the usual escape hatch, but it hands every user `edge()` and `grad_fn`, and it is exactly the kind of "pub but don't touch" that invariant 1 exists to prevent.

## Plan (conditional)

### Target layout

```
Cargo.toml                    # [workspace], [workspace.package], [workspace.dependencies], [workspace.lints]
crates/
  volta-core/                 # shape, dtype, error, index, spec, storage (CPU), device, device_policy,
                              # autograd (+ GradFn), tensor, ops — including the conv/pool/embedding
                              # kernels and their GradFns; io::TensorData + safetensors bytes
  volta-wgpu/                 # wgpu runtime, kernels, shaders; implements core's sealed Backend trait
  volta-nn/                   # Module/Layer traits, layers (parameter holders composing core ops),
                              # optimizers, transformer, state-dict mapping/loading
  volta-data/                 # mnist, transforms, ProgressBar
  volta/                      # facade: re-exports today's `volta::` paths unchanged;
                              # examples, benches, integration tests, trybuild live here
```

The dependency direction is `volta` → {`nn`, `data`, `wgpu`} → `core`. `volta-wgpu` depends on `core` and not the other way round, which only works once `Storage` dispatches through the `Backend` trait instead of naming `GpuBuffer`. That is why the split waits for Phase 4.

### Step 0: in-crate layering (do now; one PR each, `just ci` green)

- **0a. Move the custom-backward kernels out of `nn`.** Move the conv, conv-transpose, maxpool, and embedding forward/backward kernels and `*GradFn` structs from `nn/layers/` into `ops/` (`ops/conv.rs`, `ops/pool.rs`, `ops/embedding.rs`), exposed as `Tensor` methods. The layers keep their parameters and call those methods. Afterwards `autograd` imports only from `ops` and `tensor`. The move is mechanical: same code, new path.
- **0b. Break `io ↔ nn`.** Keep `TensorData` and the safetensors byte read/write in `io` (core). Move the functions that take a `Module` (`load_state_dict_*`, `mapping`, `save_safetensors(model)`) under `nn` (`nn::state`), and keep the existing `volta::` re-exports so the public API doesn't change.
- **0c. Give `nn` the methods it reaches around.** Route the three `zero_grad`s through the existing `RawTensor::zero_grad` and name it in invariant 3 (or make it go through `set_grad`). Add `Tensor` methods for the optimizer's in-place parameter update, `requires_grad` on construction, and BatchNorm's running stats. Then remove `borrow_mut()` from `nn` (fixes invariant 2). Target: no `.borrow`, `edge(`, or `crate::index` in non-test `nn` code. Several PRs, one per layer family.
- **0d. Lock it in.** Add a script check in `just ci` that fails on `crate::autograd`, `edge(`, or `borrow_mut()` in non-test `src/nn/**`, the same mechanism as `check-fallbacks`. Once this passes, `nn` uses only the public API and the eventual move is a file move.

Each item needs a line in `notes/TODO.md`. 0c's invariant-3 fix is worth doing whatever happens to the split.

### Step 1: convert to a workspace (after the trigger)

1. Create the root `[workspace]` with `members = ["crates/*"]` and `resolver = "3"`. Hoist `[lints.*]` to `[workspace.lints]` (each member gets `lints.workspace = true`), dependencies to `[workspace.dependencies]`, and `edition`/`rust-version`/`license`/`version` to `[workspace.package]`.
2. `git mv` everything into `crates/volta/` as a single member. No code changes.
3. Port the tooling **before** doing anything else, and make each port fail loudly:
   - `check-fallbacks.sh`: glob the members' `src`/`tests`/`examples`/`benches`, and **assert the file list is non-empty** so a moved path can't make the check pass with nothing to scan.
   - `clippy.toml`: one at the root applies to every member. Confirm with a seeded violation.
   - `justfile` `validate-shaders`, CI `paths`, pre-commit entries, `build.rs` (moves with whichever crate calls `cblas_sgemm`, i.e. `volta-core`).
   - Re-bless the trybuild `.stderr` files and review the diff by hand. Only paths should change.
4. Gate: `just ci` + `cargo test --no-default-features`. The test count must match before and after. Diff `cargo test -- --list`.

### Step 2: extract `volta-data` (proves the pipeline)

It is a leaf with no internal access beyond `index`. Either give `volta-data` its own copy of `fetch`/`span` (small, `#[track_caller]`) or have `volta-core` export `index` publicly. Exporting is fine: these helpers carry no graph invariants. The facade re-exports, and the public paths stay the same.

### Step 3: extract `volta-nn`

This needs step 0 to be complete. If 0d passes, the move is `git mv` plus import rewrites. Unit tests in `nn` move with it. Any test that needs core internals stays in `volta-core` or becomes an integration test.

### Step 4: extract `volta-wgpu`

This needs the Phase 4 `Backend` trait. `volta-core` defines the sealed trait and the `Cpu` backend. `volta-wgpu` implements `Gpu`. The `gpu` feature on the facade becomes an optional dependency on `volta-wgpu`. GPU parity tests run on the Mac lane only, as now.

### Step 5: publishing

The members share one `workspace.package.version`. Publish in dependency order: core, then data/nn/wgpu, then the facade. Before step 1, check that the `volta-*` names are free on crates.io (not checked here).

### Questions to settle at trigger time

1. Should `index` be public API in `volta-core`, or copied into each crate? I recommend public: there is no invariant to protect.
2. Should the internal crates be published, or vendored into the facade with `publish = false`? Crates.io doesn't allow `publish = false` dependencies of a published crate, so publishing the facade means publishing all of them.
3. Should `GradFn` stay closed? A public custom-op API (`GradFn::Custom`) is the only thing that would let `nn` define new backward passes outside core. DECISIONS says to add it only if that API exists.

## Measurements

Linux cloud VM, CPU-only (`--no-default-features`), shared and noisy (`dev-docs/CLOUD.md`), with dependencies already built. These are single runs, so treat them as rough sizes, not benchmarks. Re-measure on the Mac before using a number as a trigger.

| Scenario (`--no-default-features`)                                                        | Wall time     |
| ----------------------------------------------------------------------------------------- | ------------- |
| Real edit in `nn/layers/conv.rs` (a new fn appended), `cargo check --lib`                 | 0.6 s         |
| Same, `cargo build --lib`                                                                 | 0.7–0.8 s     |
| Same, `cargo build --tests` (lib unit tests + 22 integration test binaries)               | 2.3–2.5 s     |
| Real edit in `shape.rs` (a leaf every module uses), `cargo check --lib` / `build --tests` | 0.5 s / 2.5 s |
| `volta` from scratch (deps cached), `build --lib`                                         | 4.2 s         |
| Then every test target from scratch                                                       | 10.0 s        |

These numbers come from a 4-core VM. An edit inside a hot function body invalidates more than an appended fn does, so expect somewhat higher times in practice, but not by enough to matter. At these sizes, a crate split has almost no build time left to save. The roughly 2 s of a test build goes mostly to linking about 23 test binaries, and a split would add more crates to link, not fewer.

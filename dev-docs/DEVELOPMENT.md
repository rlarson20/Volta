# Development Guide

How to add an op or a layer. The rules behind each step are in `.claude/rules/` (`autograd.md`, `gpu.md`, `nn-layers.md`, `tests.md`) and load when you edit matching files; the repo-wide invariants are in `CLAUDE.md`. Where the two disagree with this file, they win.

## Adding a new operation

Every op ends in `edge()`, the only function that attaches `parents` and `grad_fn` (invariant 1). Never copy values into `RawTensor::new` to build an op's output.

**A unary or binary elementwise op** is one row in the `define_ops!` table, in `src/ops/unary.rs` or `src/ops/binary.rs`:

1. Add the row: the CPU forward closure, the CPU backward closure, and a `gpu(forward, backward)` clause. The row generates the enum variant, the dispatch, and the GPU kernel name, so there is no `match` arm to forget.
2. For a GPU kernel, write the WGSL entry point in `src/gpu/shaders/`, add a field to `ComputePipelines` (`src/gpu/context.rs`), and name it in the row. With no kernel, a unary row says `cpu_only`; a binary op that cannot be differentiated goes in the `non_differentiable` section. Run `just validate-shaders` after editing a shader.
3. Add the inherent method on `Tensor` (`src/tensor.rs`): a documented one-line forward with `#[must_use]`. The logic stays in `src/ops/`.

**Any other op** (reductions, matmul, a new movement op):

1. Compute the output storage in `src/ops/<family>.rs`. For the GPU, call the `gpu_*` wrapper; for a case with no kernel, compute on the CPU and pass `Device::CPU` to `edge()` so `DevicePolicy` decides (`.claude/rules/gpu.md`). A kernel that covers the case and returns `None` calls `kernel_failed(op)`.
2. Call `edge(inputs, data, Shape::derived(out_dims), device, GradFn::MyOp(MyGradFn { .. }))`. A movement op ends in one `RawTensor::movement_edge(..)` call with a `MovementOp` variant instead.
3. `GradFn` is a closed enum in `src/autograd.rs`, not a trait. Add a variant, and an arm in both `GradFn::backward` and `GradFn::is_trivial` (the compiler rejects a missing one). The struct derives `Clone` and has a `pub(crate) fn backward` returning `Vec<Option<Storage>>`, one slot per parent: bare storage (`Storage::from_compute` for a CPU kernel result, the kernel's buffer on the GPU, `Tensor::into_grad_storage` for a gradient built from tensor ops), checked against the parent's element count in `run_backward`. Compute every required gradient or panic; never return `None` for a parent that needs one.
4. Validate user-supplied dims once in forward, into a spec if the geometry is non-trivial (`MatmulSpec`, the conv specs in `src/spec.rs`), and store it in the `GradFn`. Backward takes its dims from the tensor whose gradient it computes.
5. Add the inherent method on `Tensor`, as above.

**Tests for an op:**

- A gradient check, `check_gradients`, with seeded random inputs and a randomly weighted loss (a plain `.sum()` hides wrong pairings and zeroes some gradients). Cover unequal-rank and broadcast shapes; B4 shipped because only equal-rank shapes were tested.
- Where the op has a GPU path, a CPU/GPU parity test in `tests/parity_*.rs`, and run `just test-gpu` on real hardware. GPU tests skip silently without an adapter, so "passes" does not mean GPU-verified.
- Put the test next to the op, or in its family's one file, `src/lib_tests/<family>_tests.rs` (`unary`, `binary`, `reduce`, `ternary`, `movement`, ...; declared in `lib.rs` with `#[path]`), where its name is `<family>_tests::<test>`.

## Adding a new neural network layer

1. Create `src/nn/layers/my_layer.rs` and declare it `pub(crate) mod my_layer;` in `src/nn/layers/mod.rs` and add it to the `pub use` lists there and in `src/nn/mod.rs`, so its one public path is `volta::nn::MyLayer` (no root re-export; DECISIONS.md, "One path per item; the root re-exports the core").
2. Implement `Module` (`parameters`, `state_dict`, `plan_load`; the provided `load_state_dict` and `load_state_dict_partial` build a `LoadPlan` from `plan_load`, check every entry, then write them all) and `Layer` (`forward`). `parameters()` returns every learnable tensor. Index input is an I64/I32 tensor through `Layer` (`Embedding`); carried state is a `RecurrentLayer` (`LSTMCell`), which is sealed.
3. Build the output from graph ops or `edge()`. A layer that holds a fixed non-parameter tensor overrides `Module::to_device` to move it, and builds constants with `x.constant_like(..)` so they land on the input's device.
4. `plan_load` names each tensor with `plan.param("weight", &self.weight)` under its `state_dict` key, and each sub-module with `plan.child("ffn.", &self.ffn)`. `LoadPlan::apply` checks each entry with `Tensor::check_load` and writes through `Tensor::load_weight`, which keeps the destination's device.
5. Add an `assert_trainable!` test with its input from `crate::macros::trainable_input(&shape)`, plus value tests against a hand-computed or reference result.
6. Add an example in `examples/` if the layer is complex. Examples must not route around a broken component; file the defect as a GitHub issue instead.

## Errors: panic or `Result`

Decide by where an argument comes from (DECISIONS.md, "Panic inside the program, `Result` where data comes from outside").

- **Panic** when the calling program's own code decides every argument: a shape mismatch between two tensors it built, a wrong rank passed to a layer. Document it under `# Panics`, and put `#[track_caller]` on the public panicking function so the message points at the caller. The attribute only helps if every frame between the user and the `panic!` carries it, and it does not reach into a closure: panic in the function body (`match`, `let .. else`, `assert!`, `.expect(..)`), never inside `unwrap_or_else(|e| panic!(..))`.
- **`Result<_, VoltaError>`** when an argument can come from a file, a checkpoint, a dataset, JSON, or a UI: shapes and data for a new tensor, a state dict, labels, bytes, axes and ranges a user picks. Document it under `# Errors`.
- **Both forms** only for those boundary arguments. Write the `try_` form, and make the panicking form `#[track_caller]` plus `try_x(..).or_panic("x")` (`crate::error::OrPanic`). Do not add a `try_` twin to an op just for symmetry.
- **Validate before you allocate.** Run `Shape::from_dims` (or `RawTensor::checked_numel`) on a caller's shape before `product()`, `vec![..; n]`, or any buffer: an overflowing product or an oversized allocation is not an error you can catch afterwards.
- A new error that names a state-dict entry carries its `key`, the full key in the loaded state dict (`LoadPlan` builds it with every container's prefix).
- A panic for a caller's mistake reads `<public op or layer>: <what was wrong> (got <value>)` (#142, `tests/panic_messages.rs`). Panic from a `VoltaError` with `.or_panic("<op>")`, never `.expect(..)` (which prints the `Debug` dump). Use `assert!(cond, "..")` with the values in the message, not `assert_eq!`. A layer checks its input's rank and width itself, under its own name, before calling an op, and an op that calls another (`softmax` calling `max_dim`) checks its axis first so the message names the op the user called. A rank mistake must never reach `crate::index`, whose message says the bug is inside Volta.

## Testing workflow

- Before opening a PR, run clippy and the tests your change touches; GitHub Actions runs the full gate on the PR (CLAUDE.md, "Commands"). `just ci` runs the same checks locally (fmt, clippy, test and `-D warnings` doc, each with no features and with `gpu` + `accelerate`, then examples and the cargo-hack feature powerset when installed) when you want them before pushing. The fast loop is `cargo test --lib <name-filter>` (CPU-only, the default); the filter matches test names, and there are no category modules.
- `check_gradients_simple` is the quick check; `check_gradients` is the full one. Avoid ops with non-deterministic gradients on ties (argmax, argmin), and make sure the gradient shape matches the input's.
- Never write a test that asserts a known bug. Use `#[ignore = "<defect ID>: <why>"]`.
- Where the tests live: unit tests beside the code or in `src/lib_tests/`; integration, parity, and GPU tests in `tests/` (`tests/gpu_*.rs`, `tests/parity_*.rs`); compile-fail tests in `tests/compile_fail/`.

## Examples smoke run

`just examples-smoke` (`scripts/examples-smoke.sh`, in `just ci` and the `examples-smoke` CI job) builds the examples below in release with no default features and runs each in an empty directory under a 120 s limit. It fails on a non-zero exit, a timeout, or a printed `NaN`/`inf`. Every example in the set that trains asserts at the end that its final loss is finite and below its first (`readme2` takes one step on random targets, so it asserts only that the loss is finite). Timings are from a 4-core Linux VM.

| In the smoke set | Time |
| --- | --- |
| `readme1`, `readme2`, `showcase`, `regression`, `layernorm_demo`, `gpu_training` (its CPU path), `load_external_mnist` (its demo path, #143) | < 1 s each |
| `char_transformer_model`, `char_language_model` | 1 s, 3 s |
| `super_resolution` | 9 s |
| `time_sequence` | 30 s |

| Left out | Why |
| --- | --- |
| `mnist_cnn`, `mnist_rnn`, `mnist_autocast`, `vae`, `dcgan` | Need MNIST in `data/mnist`, and run for minutes (#59 is the MNIST lane). One `dcgan` epoch takes hours on the CPU, almost all of it in convolution backward passes (#230, #231). |
| `gpu` | Does nothing without the `gpu` feature. |

A new example joins the set when it needs no downloaded data and runs in seconds; add it to the `smoke` list in the script and to this table, with its loss assertion.

## Serving the built docs

`just serve-docs` (`scripts/serve_docs.py`, standard library only, needs `uv`) runs `cargo doc --no-deps`, copies `target/doc` to a snapshot under `target/doc-serve/`, and serves it on every interface at port 7447, logging the link to open (`http://<hostname>:7447/volta/`, which other devices on the tailnet or LAN can reach). Pass flags through the recipe: `just serve-docs --port 9000 --features gpu,accelerate`. `--rebuild-every <seconds>` rebuilds in the background (with `--pull`, after `git pull --ff-only`) and swaps to the new snapshot only when the build succeeds, so a failed build keeps the last good docs up. `--no-build` serves the latest snapshot without building. Arguments after `--` go to `cargo doc`.

It has no auth or TLS, so the host firewall decides who can reach it. On NixOS, open the port only on Tailscale: `networking.firewall.interfaces.tailscale0.allowedTCPPorts = [ 7447 ];`. `--host 127.0.0.1` limits it to the machine itself.

## Nix dev shell

`flake.nix` gives a dev shell with everything the justfile calls, for a NixOS machine or any host with Nix. Enter it with `nix develop`, or `direnv allow` once (the committed `.envrc` is `use flake`). It is a convenience, not part of CI: GitHub Actions installs its own toolchains, and `flake.lock` pins this shell's separately (`nix flake update` moves it).

| Shell | What it has | For |
| --- | --- | --- |
| `default` | Stable Rust from fenix (clippy, rustfmt, rust-analyzer, llvm-tools, the `wasm32-unknown-unknown` and `aarch64-apple-darwin` std), sccache as `RUSTC_WRAPPER`, `just`, `uv` with Nix's Python 3.12, `pre-commit`, `gh`, `git-cliff`, `bacon`, the cargo tools (`hack`, `llvm-cov`, `mutants`, `deny`, `machete`, `expand`, `bloat`), and, on Linux, unwrapped clang | Everything on the CPU: `just ci`'s checks, `just serve-docs`, `just torch-fixtures`, `just wasm-check`, `just macos-cross` |
| `.#gpu` (Linux) | `default` plus the Vulkan loader and Mesa's lavapipe (software Vulkan), added beside any driver NixOS provides | `VOLTA_REQUIRE_GPU=1 just test-gpu` without a GPU, as the CI lavapipe lane runs it |
| `.#msrv` | Rust 1.89.0 (Cargo.toml's `rust-version`) and `just` | Reproducing a failure in CI's `1.89` lane: `nix develop .#msrv -c cargo test` |

What it does not cover:

- **rustup.** The toolchain comes from Nix, so `cargo +<toolchain>` does not work. `just miri` and `just asan` need `cargo +nightly-…` and still need rustup. The `*-if-available` recipes in `just ci` detect a target with `rustup target list`, so inside this shell they print "skipped" even though the targets are installed; run `just wasm-check` and `just macos-cross` directly.
- **macOS.** The flake evaluates for `aarch64-darwin`, but the shell has not been built on a Mac; Metal and Accelerate there are untested through Nix.
- **NixOS firewall.** `just serve-docs` still needs port 7447 open on `tailscale0` ("Serving the built docs").

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
3. `GradFn` is a closed enum in `src/autograd.rs`, not a trait. Add a variant, and an arm in both `GradFn::backward` and `GradFn::is_trivial` (the compiler rejects a missing one). The struct derives `Clone` and has a `pub(crate) fn backward`. Compute every required gradient or panic; never return `None` for a parent that needs one.
4. Validate user-supplied dims once in forward, into a spec if the geometry is non-trivial (`MatmulSpec`, the conv specs in `src/spec.rs`), and store it in the `GradFn`. Backward takes its dims from the tensor whose gradient it computes.
5. Add the inherent method on `Tensor`, as above.

**Tests for an op:**

- A gradient check, `check_gradients`, with seeded random inputs and a randomly weighted loss (a plain `.sum()` hides wrong pairings and zeroes some gradients). Cover unequal-rank and broadcast shapes; B4 shipped because only equal-rank shapes were tested.
- Where the op has a GPU path, a CPU/GPU parity test in `tests/parity_*.rs`, and run `just test-gpu` on real hardware. GPU tests skip silently without an adapter, so "passes" does not mean GPU-verified.
- Put the test next to the op, or in `src/lib_tests/<family>_tests.rs` (declared in `lib.rs` with `#[path]`).

## Adding a new neural network layer

1. Create `src/nn/layers/my_layer.rs` and export it in `src/nn/layers/mod.rs`.
2. Implement `Module` (`parameters`, `state_dict`, `load_state_dict`) and `Layer` (`forward`). `parameters()` returns every learnable tensor. Index input is an I64/I32 tensor through `Layer` (`Embedding`); carried state is a `RecurrentLayer` (`LSTMCell`), which is sealed.
3. Build the output from graph ops or `edge()`. A layer that holds a fixed non-parameter tensor overrides `Module::to_device` to move it, and builds constants with `x.constant_like(..)` so they land on the input's device.
4. `load_state_dict` goes through `Tensor::load_weight`, which checks the shape and keeps the destination's device.
5. Add an `assert_trainable!` test with its input from `crate::macros::trainable_input(&shape)`, plus value tests against a hand-computed or reference result.
6. Add an example in `examples/` if the layer is complex. Examples must not route around a broken component; file the defect in `notes/TODO.md` instead.

## Errors: panic or `Result`

Decide by where an argument comes from (DECISIONS.md, "Panic inside the program, `Result` where data comes from outside").

- **Panic** when the calling program's own code decides every argument: a shape mismatch between two tensors it built, a wrong rank passed to a layer. Document it under `# Panics`, and put `#[track_caller]` on the public panicking function so the message points at the caller.
- **`Result<_, VoltaError>`** when an argument can come from a file, a checkpoint, a dataset, JSON, or a UI: shapes and data for a new tensor, a state dict, labels, bytes, axes and ranges a user picks. Document it under `# Errors`.
- **Both forms** only for those boundary arguments. Write the `try_` form, and make the panicking form `try_x(..).unwrap_or_else(|e| panic!("x: {e}"))`. Do not add a `try_` twin to an op just for symmetry.
- **Validate before you allocate.** Run `Shape::from_dims` (or `RawTensor::checked_numel`) on a caller's shape before `product()`, `vec![..; n]`, or any buffer: an overflowing product or an oversized allocation is not an error you can catch afterwards.
- A new error that names a state-dict entry carries its `key`; a container puts its prefix back with `VoltaError::with_key_prefix`.

## Testing workflow

- Run `just ci` before committing (fmt, clippy with and without default features, test, doc, examples). The fast loop is `cargo test --no-default-features --lib <name-filter>`; the filter matches test names, and there are no category modules.
- `check_gradients_simple` is the quick check; `check_gradients` is the full one. Avoid ops with non-deterministic gradients on ties (argmax, argmin), and make sure the gradient shape matches the input's.
- Never write a test that asserts a known bug. Use `#[ignore = "<defect ID>: <why>"]`.
- Where the tests live: unit tests beside the code or in `src/lib_tests/`; integration, parity, and GPU tests in `tests/` (`tests/gpu_*.rs`, `tests/parity_*.rs`); compile-fail tests in `tests/compile_fail/`.

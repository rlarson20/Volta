# Development Guide

This document provides detailed guidance for adding new features to the Volta deep learning framework.

TODO: UPDATE WITH PHASE 1 CHANGES

## Adding New Operations

To add a new tensor operation:

1. **Implement the operation logic** in the appropriate file under `src/ops/`:
   - `binary.rs` for element-wise binary operations (add, sub, mul, etc.)
   - `unary.rs` for element-wise unary operations (sigmoid, relu, sqrt, etc.)
   - `reduce.rs` for reduction operations (sum, mean, max)
   - `movement.rs` for shape manipulation (reshape, permute, transpose)
   - `matmul.rs` for matrix multiplication
   - `ternary.rs` for three-operand operations

2. **Add the enum variant** to the corresponding operation enum (e.g., `BinaryOp::NewOp`)

3. **Implement forward pass** in the operation's helper function

4. **Implement backward pass** by creating a GradFn struct that implements the `GradFn` trait

5. **Add an inherent method** on `Tensor` in `src/tensor.rs` that forwards to the helper function, with a doc comment and `#[must_use]`

6. **Add gradient check test** in `src/lib.rs` to verify correctness

7. **Update documentation** in CLAUDE.md if the operation is significant

## Adding New Neural Network Layers

To add a new neural network layer:

1. **Create new file** in `src/nn/layers/` (e.g., `my_layer.rs`)

2. **Implement the Module trait**:

   ```rust
   impl Module for MyLayer {
       fn forward(&self, x: &Tensor) -> Tensor { /* ... */ }
       fn parameters(&self) -> Vec<Tensor> { /* ... */ }
       fn state_dict(&self) -> StateDict { /* ... */ }
       fn load_state_dict(&mut self, state: &StateDict) { /* ... */ }
   }
   ```

3. **Export the layer** in `src/nn/layers/mod.rs`

4. **Add tests** in `src/lib.rs` for forward/backward passes

5. **Add example** in `examples/` if the layer is complex

## Errors: panic or `Result`

Decide by where an argument comes from (DECISIONS.md, "Panic inside the program, `Result` where data comes from outside").

- **Panic** when the calling program's own code decides every argument: a shape mismatch between two tensors it built, a wrong rank passed to a layer. Document it under `# Panics`, and put `#[track_caller]` on the public panicking function so the message points at the caller.
- **`Result<_, VoltaError>`** when an argument can come from a file, a checkpoint, a dataset, JSON, or a UI: shapes and data for a new tensor, a state dict, labels, bytes, axes and ranges a user picks. Document it under `# Errors`.
- **Both forms** only for those boundary arguments. Write the `try_` form, and make the panicking form `try_x(..).unwrap_or_else(|e| panic!("x: {e}"))`. Do not add a `try_` twin to an op just for symmetry.
- **Validate before you allocate.** Run `Shape::from_dims` (or `RawTensor::checked_numel`) on a caller's shape before `product()`, `vec![..; n]`, or any buffer: an overflowing product or an oversized allocation is not an error you can catch afterwards.
- A new error that names a state-dict entry carries its `key`; a container puts its prefix back with `VoltaError::with_key_prefix`.

## Testing Workflow Conventions

- Always run `cargo test` before committing changes
- If tests fail, fix them before proceeding with commit
- Use `check_gradients_simple()` for basic gradient validation
- Use full `check_gradients()` for comprehensive validation

### Gradient Checks

When implementing gradients, follow these guidelines:

- Verify gradient accumulation works correctly
- Avoid operations that produce non-deterministic results (e.g., argmax/argmin on ties)
- Ensure gradient shapes match input tensor shapes

## Test Coverage

The test suite (tests in `src/lib.rs`) validates:

**Test Organization:**

Tests are organized into modules by category:

- **core**: Basic tensor operations, chain rule, shape manipulation
- **grad_check**: Numerical gradient validation for all operations
- **broadcasting**: NumPy-style broadcasting rules
- **neural**: Neural network layer tests (Linear, Conv2d, MaxPool2d, etc.)
- **optimizer**: Optimizer convergence tests (Adam, SGD)
- **axis_reduce_tests**: Dimension reduction operations (sum_dim, max_dim, softmax)
- **shape_tests**: Shape manipulation operations (reshape, permute, transpose)

**Test Coverage:**

- **Core operations**: Basic ops, chain rule, tensor shapes
- **Gradient correctness**: All operations verified with numerical gradient checking
- **Broadcasting**: NumPy-style broadcasting rules
- **Neural networks**: Linear, Sequential, Conv2d, ConvTranspose2d, MaxPool2d layers
- **Optimizers**: Adam, SGD convergence tests
- **Edge cases**: Numerical edge cases, empty tensors, scalar tensors, shape operations

**Recent testing improvements:**

- Comprehensive Conv2d layer test coverage
- Comprehensive MaxPool2d layer test coverage
- Scalar tensor and empty tensor edge case tests
- Numerical edge case tests with improved error handling
- Gradient checks for missing unary operations

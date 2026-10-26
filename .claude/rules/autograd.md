---
paths:
  - "src/autograd.rs"
  - "src/tensor.rs"
  - "src/shape.rs"
  - "src/ops/**"
---

# Autograd core

- `Tensor` is an opaque newtype over `Rc<RefCell<RawTensor>>`. `borrow()`, `borrow_mut()`, and `as_ptr()` are `pub(crate)`; never widen them. Code outside the crate uses `shape()`, `data()`, `device()`, `grad_storage()`, `detach()`, `set_requires_grad()`, and `set_grad()`.
- **New op:** compute the output storage, then call `edge(inputs, data, shape, device, Box::new(MyGradFn { .. }))`. `edge` derives `requires_grad` from the inputs and `is_grad_enabled()`. Never compute or gate on `requires_grad` yourself (that was B1).
- `edge` currently takes `Vec<usize>` and `Box<dyn GradFn>`. Phase 2 moves it to `Shape` and a closed `GradFn` enum (ADDENDUM §1.2, §2.5). In new code, prefer `Shape`/`Axis` wherever the callee accepts them, and don't add new loose `Vec<usize>` shape parameters.
- `NO_GRAD` is a thread-local, reentrant depth counter (`Cell<u32>`), so `no_grad()` scopes nest. Don't simplify it to a bool.
- `TensorOps` is slated for deletion (ADDENDUM §3). Keep any method you add to it as a one-line forward; the logic belongs in `src/ops/`.
- **Every new op needs a numerical gradient check** (`check_gradients`) with random inputs, covering unequal-rank and broadcast cases. B4 (wrong unequal-rank matmul backward) shipped because only equal-rank shapes were tested.
- When a backward pass derives dims, take them from the tensor whose gradient you are computing. Better still, compute a validated spec once in forward (ADDENDUM §1.2, `MatmulSpec`).

---
paths:
  - "src/autograd.rs"
  - "src/tensor.rs"
  - "src/shape.rs"
  - "src/ops/**"
---

# Autograd core

- `Tensor` is an opaque newtype over `Rc<RefCell<RawTensor>>`. `borrow()`, `borrow_mut()`, and `as_ptr()` are `pub(crate)`; never widen them. Code outside the crate uses `shape()`, `data()`, `device()`, `grad_storage()`, `detach()`, `set_requires_grad()`, and `set_grad()`.
- **New op:** compute the output storage, then call `edge(inputs, data, Shape::derived(out_dims), device, Box::new(MyGradFn { .. }))`. `edge` derives `requires_grad` from the inputs and `is_grad_enabled()`. Never compute or gate on `requires_grad` yourself (that was B1).
- **Shapes:** `edge` takes a `Shape`. Use `Shape::derived` for dims an op computed from existing tensors (it panics, because an invalid one is an op bug) and `Shape::from_dims` for user-supplied dims (it returns `Err` on a zero dim or more than `MAX_NUMEL` elements). Don't re-implement the zero/size-cap checks. `RawTensor.shape` is still a `Vec<usize>` field (DECISIONS.md, "Shape boundary"); demote with `truncate_rank()` only at the existing boundary sites.
- **Dims:** reductions, `softmax`, `log_softmax`, and `permute` take `Axis`. Get an axis's size with `Shape::dim`, never `shape.get(i).unwrap_or(&1)` (invariant 5).
- `edge` still takes `Box<dyn GradFn>`. Phase 2 moves it to a closed enum (§2.5). Don't add new loose `Vec<usize>` shape or `usize` dim parameters to public functions.
- `NO_GRAD` is a thread-local, reentrant depth counter (`Cell<u32>`), so `no_grad()` scopes nest. Don't simplify it to a bool.
- `TensorOps` is slated for deletion (ADDENDUM §3). Keep any method you add to it as a one-line forward; the logic belongs in `src/ops/`.
- **Every new op needs a numerical gradient check** (`check_gradients`) with random inputs, covering unequal-rank and broadcast cases. B4 (wrong unequal-rank matmul backward) shipped because only equal-rank shapes were tested.
- When a backward pass derives dims, take them from the tensor whose gradient you are computing. Better still, compute a validated spec once in forward (ADDENDUM §1.2, `MatmulSpec`).

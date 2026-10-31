---
paths:
  - "src/autograd.rs"
  - "src/tensor.rs"
  - "src/shape.rs"
  - "src/ops/**"
---

# Autograd core

- `Tensor` is an opaque newtype over `Rc<RefCell<RawTensor>>`. `borrow()`, `borrow_mut()`, and `as_ptr()` are `pub(crate)`; never widen them. Code outside the crate uses `shape()`, `data()`, `device()`, `grad_storage()`, `detach()`, `set_requires_grad()`, and `set_grad()`.
- **New op:** compute the output storage, then call `edge(inputs, data, Shape::derived(out_dims), device, GradFn::MyOp(MyGradFn { .. }))`. `GradFn` is a closed enum in `src/autograd.rs`: add a variant and an arm in both `GradFn::backward` and `GradFn::is_trivial`, and give the struct `#[derive(Clone)]` and a `pub(crate) fn backward`. `edge` derives `requires_grad` from the inputs and `is_grad_enabled()`. Never compute or gate on `requires_grad` yourself (that was B1).
- **New unary or binary op:** add one row to the `define_ops!` table in `src/ops/unary.rs` or `binary.rs`; don't hand-write a `match` arm, a kernel-name string, or a `GradFn` variant. A GPU kernel also needs a WGSL entry point and a `ComputePipelines` field (`src/gpu/context.rs`), named in the row's `gpu(..)` clause. No kernel means `cpu_only` (unary); say so rather than naming one that doesn't exist.
- **GPU paths follow the device policy** (`src/device_policy.rs`): a kernel that covers the case and returns `None` calls `kernel_failed(op)`; a case with no kernel computes on the CPU and passes `Device::CPU` to `edge()`, which panics under `Strict` or moves the result under `PreferGpu`. Never fall back silently, and never `to_device` an op's own result. Constants an op or layer combines with its input come from `x.constant_like(..)`.
- **New movement op:** compute the output storage (GPU wrapper, else CPU), then end in one `RawTensor::movement_edge(..)` call with a `MovementOp` variant. Don't build `GradFn::Movement` per path. Movement ops are not in `define_ops!`.
- **Shapes:** `edge` takes a `Shape`. Use `Shape::derived` for dims an op computed from existing tensors (it panics, because an invalid one is an op bug) and `Shape::from_dims` for user-supplied dims (it returns `Err` on a zero dim or more than `MAX_NUMEL` elements). Don't re-implement the zero/size-cap checks. `RawTensor.shape` is a `Shape`: read it with its accessors (`rank`, `numel`, `dim_at`, `fixed::<N>`, `numel_before`/`numel_from`, `with_dim`/`squeeze`/`unsqueeze`), build backward outputs with `RawTensor::from_shape(data, x.shape.clone(), ..)`, and demote with `truncate_rank()` only at a raw index-math, public `&[usize]`, or serialization boundary (DECISIONS.md, "`RawTensor.shape` is a `Shape`").
- **Dims:** reductions, `softmax`, `log_softmax`, and `permute` take `Axis`. Get an axis's size with `Shape::dim`, never `shape.get(i).unwrap_or(&1)` (invariant 5). On a raw `&[usize]` whose rank is already checked, destructure with `crate::shape::fixed_dims::<N>`; in a rank-bounded loop, use `fetch`.
- **Kernel indexing:** read with `crate::index::fetch`, write with `slot`, slice with `span`. They panic on an out-of-bounds index, so a mapping bug fails at the site. Never `.get(i).unwrap_or(0.0)` or `if let Some(slot) = v.get_mut(i)`; `just check-fallbacks` rejects both. If a zero is the meaning (padding), check the coordinate explicitly and comment it.
- Don't add new loose `Vec<usize>` shape or `usize` dim parameters to public functions.
- **Backward** runs only through `Tensor::into_loss()?.backward()` (`ScalarLoss`). Don't add another public entry point or a seeded-gradient variant; reduce with `.sum()` first.
- `NO_GRAD` is a thread-local, reentrant depth counter (`Cell<u32>`), so `no_grad()` scopes nest. Don't simplify it to a bool.
- User-facing ops are inherent methods on `Tensor` (`impl Tensor` in `src/tensor.rs`; the `TensorOps` trait is gone). Keep each a documented one-line forward with `#[must_use]`; the logic belongs in `src/ops/`.
- **Every new op needs a numerical gradient check** (`check_gradients`) with random inputs, covering unequal-rank and broadcast cases. B4 (wrong unequal-rank matmul backward) shipped because only equal-rank shapes were tested.
- When a backward pass derives dims, take them from the tensor whose gradient you are computing. Better still, compute a validated spec once in forward and store it in the `GradFn` (`MatmulSpec` and the conv specs in `src/spec.rs`).

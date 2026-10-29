# DECISIONS.md — Volta Autograd Engine Audit

Design decisions, implementation rationale, and known gaps.
Updated: 2026-10-26

---

## Autograd Engine

### Topological Sort

**Decision:** Iterative DFS with a `HashSet` for visited tracking.

Recursive DFS risks stack overflow on deep computation graphs. Iterative DFS with an explicit stack avoids this. The `HashSet` ensures each node is processed once — critical for diamond-shaped graphs where multiple paths converge on a shared ancestor.

**Diamond graphs:** When two downstream nodes share an upstream node, that node's `.grad` receives contributions from both paths. The engine handles this by _accumulating_ (summing) gradients at each node rather than overwriting. This is mathematically correct: partial derivatives from all downstream paths sum to give the total gradient.

> ⚠ **Gap:** Could not derive the multivariate chain rule justification for gradient summation unprompted. Understand the conclusion, not the derivation.

> ⚠ **Gap:** Confused root-level `assert` (sanity check on the output node) with per-node behavior during traversal. These are distinct: the assert fires once, traversal logic is per-node.

---

### `Rc` and Node Liveness

**Decision:** Parents are stored as `Rc<RefCell<Node>>` in each node's `grad_fn`.

`Rc` keeps the reference count > 0 as long as any child holds a reference to a parent. This means the full computation graph stays alive for the duration of the backward pass — no manual lifetime annotation needed.

> ⚠ **Gap:** Could not trace the `Rc` → liveness guarantee unprompted. Understand it when shown; not yet automatic.

**`RefCell` borrow discipline:** `RefCell` provides interior mutability but panics on simultaneous mutable + immutable borrows. The engine must drop the `RefCell` borrow on a node _before_ calling `backward` on its parents, since backward may need to re-borrow the same node. Cloning the node's `GradFn` out of the borrow is what makes this safe.

---

### Closed `GradFn` enum

**Decision:** `GradFn` is a `pub(crate)` enum with one variant per op struct (`Unary(UnaryGradFn)`, `Binary(BinaryGradFn)`, ...), replacing `Box<dyn GradFn>` and `clone_box` (ADDENDUM §2.5). `GradFn::backward` and `GradFn::is_trivial` are exhaustive matches, so `wildcard_enum_match_arm` (denied in `Cargo.toml`) makes a new variant a compile error until it has both arms. `is_trivial` lists every variant rather than using a default, so a new op has to decide.

**Why closed:** an open trait let an op ship with no backward pass wired in, and cost every implementor a hand-written `clone_box`. `Clone` is derived now.

**Still true from the trait version:** `backward` clones the node's `GradFn` out of the `RefCell` borrow before recursing into parents. That drops the borrow, so a backward pass can re-borrow the same node.

**Deviations from the sketch:**
- No `Custom(CustomGrad)` variant. `edge()` is `pub(crate)`, so no downstream crate could construct one. Add it with a public custom-op API, if that ever exists.
- Variants wrap the existing structs instead of carrying inline fields (`SumDim { shape, axis, keepdim }`). The change is mechanical and behavior-identical. Flattening belongs with `define_ops!` (§5.1) and the `Shape` field flip.
- `RawTensor.grad_fn` is now `pub(crate)`, and the `*GradFn` structs and the `GradFn` re-export left the public API. With the trait gone they had no public methods or constructors.
- The variant is `Igemm`, not `IGEMM`, for `clippy::upper_case_acronyms`. The struct keeps its name.
- `autograd.rs` imports the conv, pool, and embedding structs from `nn::layers`, a crate-internal layering cost of a closed enum.

---

### `requires_grad` Short-Circuiting

**Decision:** If `requires_grad = false`, no `grad_fn` is attached.

During the backward pass, nodes without a `grad_fn` are leaves — no further traversal. This prunes the graph cheaply: frozen parameters (e.g. a pretrained backbone) don't accumulate gradients and don't participate in the backward DFS.

---

### Composition Pattern

**Decision:** High-level layers are composed from primitive ops; they inherit `grad_fn` for free.

Example: Dropout is implemented as elementwise multiply by a binary mask (`elem_mul`). Because `elem_mul` already has a registered `grad_fn`, Dropout gets backward support without writing any backward code. This is the general pattern: build new ops from primitives already in the graph; the chain rule handles the rest automatically.

> ⚠ **Gap:** This composition pattern was not internalized — had to be shown explicitly.

---

## Convolution — im2col / GEMM

### im2col

**Decision:** Unfold input patches into columns, then dispatch to GEMM.

im2col reshapes each receptive field into a column vector. The full unfolded matrix has shape `(B·H_out·W_out, C·K·K)`. A single GEMM then computes all convolution outputs simultaneously.

**Memory cost:** `O(B · H_out · W_out · C · K²)` — each input element is duplicated into every receptive field it participates in. For large kernels this is significant.

### Alternatives considered

| Method                | When it wins                                                                          |
| --------------------- | ------------------------------------------------------------------------------------- |
| im2col + GEMM         | Mid-to-large kernels; GEMM is highly optimized                                        |
| iGEMM (implicit GEMM) | Avoids materializing the im2col matrix; wins on memory-bound workloads                |
| Direct convolution    | Small inputs / small kernels where im2col setup overhead dominates the actual compute |

---

## Broadcasting

### Algorithm

**Decision:** Right-align shapes, then stretch size-1 dimensions to match.

1. Right-align the two shapes (pad shorter shape with 1s on the left).
2. For each dimension: if sizes match, keep. If one is 1, stretch it. If both differ and neither is 1, error.

> ⚠ **Gap:** Algorithm not fully internalized — understood the output rule but couldn't reconstruct the step-by-step procedure unprompted.

### Backward

During backward, gradients must be summed over axes that were broadcast. Two cases:

- **Stretched axis (size 1 in original):** sum gradient over that axis, keep dim.
- **Nonexistent axis (shape was shorter):** sum gradient over that axis, drop dim.

This restores the gradient to the original tensor's shape.

---

## GPU Dispatch

**Decision:** `cfg` feature flag + runtime device check + `Option` fallback.

Compile-time `cfg` gates GPU code so CPU-only builds don't pull in CUDA dependencies. At runtime, a device check selects the kernel; if GPU is unavailable, falls back to `None` → CPU path.

### Broadcast Backward on GPU — Race Condition

Naïve parallel reduction over broadcast axes causes write races when multiple threads accumulate into the same output element.

**Fix:** Two-pass scatter/reduce.

1. **Scatter:** each thread writes its partial gradient to a private slot.
2. **Reduce:** a second pass atomically reduces private slots into the output.

This serializes the final accumulation, eliminating the race.

---

## Optimizers

### SGD

Standard: `θ ← θ - α·g`. No state.

### Adam

**State per parameter:** `m` (1st moment), `v` (2nd moment), `t` (step count).

**Recurrences:**

```
m_t = β₁·m_{t-1} + (1 - β₁)·g_t
v_t = β₂·v_{t-1} + (1 - β₂)·g_t²
```

**Bias correction:**

```
m̂_t = m_t / (1 - β₁ᵗ)
v̂_t = v_t / (1 - β₂ᵗ)
```

Required because `m` and `v` are initialized at 0. Early in training, `(1 - βᵗ)` is small, so the correction inflates the estimate back to the true EMA value. As `t → ∞`, the correction → 1 and vanishes.

**Update:**

```
θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)
```

**Hyperparameter defaults and roles:**

| Param | Default | Role                                                    |
| ----- | ------- | ------------------------------------------------------- |
| β₁    | 0.9     | Decay rate for gradient EMA (~10-step memory)           |
| β₂    | 0.999   | Decay rate for squared gradient EMA (~1000-step memory) |
| ε     | 1e-8    | Prevents div-by-zero in denominator                     |
| α     | 1e-3    | Global learning rate                                    |

**Intuition:** `√v̂` estimates the RMS of past gradients. Dividing by it gives each parameter an adaptive learning rate — parameters with large historical gradients take smaller steps.

> ⚠ **Gap:** Adam update rule required reference during audit. Understood bias correction purpose; recurrences and exact formula not memorized.

---

## Invariant Wall (ADDENDUM Phase 1)

### Opaque `Tensor`

**Decision:** `Tensor` is `struct Tensor { inner: Rc<RefCell<RawTensor>> }` instead of a bare type alias. `borrow()`/`borrow_mut()`/`as_ptr()` are `pub(crate)` forwarding methods.

With the alias, `x.borrow_mut().requires_grad = true` was the documented idiom, so every invariant was only a convention. Keeping the forwarding methods under their original names meant every internal `t.borrow()` call site compiled unchanged, because method resolution is by name. Meanwhile the fields became unreachable from outside the crate.

**Consequence:** `tests/` and `examples/` see only the public API, so `Tensor` gained `shape()`, `data()`, `device()`, `grad_storage()`, `detach()`, a public `set_requires_grad()`, and `set_grad()`. `set_grad` is the second sanctioned `grad` writer after `accumulate_grad`. It exists so optimizer tests can seed a synthetic gradient without running `backward()`.

### `edge()` — single graph-edge constructor

**Decision:** `autograd::edge()` is the only code that attaches `parents`/`grad_fn`. It derives `requires_grad` as "grad enabled AND any input requires grad".

Before this, about 25 hand-written copies of the attach ritual existed, and each was a chance to gate on the wrong flag (B1: ConvTranspose gated on the input only, which froze the weights) or to drop parents (B2/B3). Migrating to `edge()` also exposed the same latent gating bug in `conv.rs`'s `direct_conv`/`igemm_forward`.

**Deviation from the ADDENDUM sketch:** `edge()` took `Vec<usize>` and `Box<dyn GradFn>` because the `Shape` and `GradFn` enum were Phase 2 work. (Since Phase 2, `edge()` takes `Shape` and the closed `GradFn` enum.) It also takes an explicit `device`, since `Storage` doesn't uniquely determine a `Device` and every call site already tracked one.

### `NO_GRAD` as a reentrant depth counter

**Decision:** `thread_local! { static NO_GRAD: Cell<u32> }`, not `Cell<bool>`.

A bool breaks as soon as one `no_grad()` scope nests inside another, for example a layer's own `no_grad()` running inside `check_gradients`. The inner guard's `Drop` would re-enable tracking while the outer scope was still active. It is `thread_local!` because a bare `static Cell<_>` doesn't compile (`Cell` isn't `Sync`).

### `assert_trainable!`

**Decision:** every parameterized layer has a test asserting that each parameter gets a nonzero gradient after forward and backward. This is the check that would have caught B1–B3.

**Revision (2026-10-26):** the macro first backpropped a plain `.sum()` of the output, and the tests fed all-ones inputs. That produced five false failures. Under a plain sum, BatchNorm's gamma gradient is exactly zero for any input, because the normalized activations sum to zero. Constant inputs also make LayerNorm's normalized values and attention's scores constant, which zeroes their gradients mathematically. The macro now backprops `(output ⊙ randn_like(output)).sum()`, and the tests take seeded random inputs from `macros::trainable_input`. After the fix, only LSTMCell (B2) fails; it is `#[ignore]`d with the defect ID until fixed.

## Types for Dims and Ops (ADDENDUM Phase 2)

### Shape boundary: signatures, not the field

**Decision:** `Shape` is the type at function boundaries (`edge()`, tensor construction and validation), and `Axis` is the type of dim arguments. `RawTensor.shape` stays a `Vec<usize>`, and `Tensor::shape()` still returns one.

Roughly 440 places in `src/` read the `.shape` field (about 365 outside `lib.rs`): kernels, `GradFn` structs, and conv/matmul index math. `Shape` deliberately has no `Deref`; stepping outside it must be spelled `truncate_rank()` so the demotion stays greppable. Storing a `Shape` in the field would turn those reads into hundreds of `truncate_rank()` calls, and a grep that matches everything finds nothing. The field flips naturally with the storage work (ADDENDUM §7), when kernels stop reading it directly.

**Rejected:** adding `Deref<Target = [usize]>` so the field could change without touching readers. It compiles with a small diff, but it removes the explicit demotion that §1.2 is built around, and any `&[usize]`-taking function would then silently accept a shape.

**Consequence:** `truncate_rank()` has six calls outside `shape.rs`, in four places: `edge()`, the two `RawTensor` constructors (two calls each), and `try_broadcast_shape`. Each is where a validated `Shape` is written into the field or handed back as a `Vec`. Add a new one only at a boundary like those.

### One validation point, and `Shape::derived`

**Decision:** `Shape::from_dims` is the only place that rejects a zero dimension or more than `MAX_NUMEL` (100M) elements. `expand`, `pad`, `broadcast_to`, and the tensor constructors call it instead of keeping private `MAX_ALLOC` checks. Op code that builds an output shape from existing tensors uses `Shape::derived`, which panics on failure.

Before this, the zero/cap checks lived in `Tensor::try_new` only. `try_new_with_storage`, which `edge()` reaches, checked neither, and `expand`, `pad`, and `broadcast_to` each carried a copy of the cap. `pad`'s message even said "Expand would create tensor", copied from `expand`. An op whose output was zero-sized could therefore build a degenerate tensor that no check ever saw.

**Consequence:** an op that produces a zero-sized or over-cap output now panics at `edge()` instead. No existing test hit this. `reshape` needed no change: its size-equality assert already guarantees the target is a valid `Shape`, and `edge()` validates it.

### `Axis` and the silent default

**Decision:** an axis's size comes from `Shape::dim(axis)`, which returns `DimensionOutOfBounds`. The `shape.get(dim).unwrap_or(&1)` pattern in `sum_dim`, `max_dim`, and `mean_dim` is gone.

A size silently defaulting to 1 is the same class as `.get(i).unwrap_or(0.0)` (invariant 5): after the up-front bounds check it could not fire, but it would have hidden a future mismatch between the check and the lookup. `max_dim`, `mean_dim`, and `softmax` panic with the `DimensionOutOfBounds` text; `try_sum_dim` returns it.

### Conv-family specs (ADDENDUM §1.3)

**Decision:** `src/spec.rs` holds `Conv2dSpec`, `ConvTranspose2dSpec`, and `Window2dSpec`. Each is built once in `forward` from the input and weight `Shape`s, validated (4D, channels agree, stride non-zero, kernel not larger than the padded input, transposed output positive), and then stored in the `GradFn`. Backward and the GPU wrappers read the spec instead of re-deriving dims from tensors. The eight conv `GpuKernels` wrappers and their `RawTensor::gpu_*` forwarders take `&Conv2dSpec` (or `&Window2dSpec` for `im2col`/`col2im`) in place of 11 to 15 positional `usize`s. The `#[repr(C)]` uniform structs are unchanged, so the WGSL shaders did not change.

`Window2dSpec` exists because `im2col`/`col2im` run on the already-padded tensor and have no weight. It serves `MaxPool2d` too, and `Conv2dSpec::padded_window()` builds it from a conv spec, so the padded and unpadded extents cannot drift apart.

**Rejected:**
- Spec accessors returning `Shape` (`input_shape()`). The old code passes `&[usize]` to `RawTensor::new`, so each call site would need `truncate_rank()`, against the "Shape boundary" rule above. They return `[usize; 4]` (`input_dims()`, `weight_dims()`, `output_dims()`), which coerces to `&[usize]` and goes through `Shape::derived` for `edge()`.
- Making the specs `pub` so the `pub` `GpuKernels` wrappers could keep their visibility. That widens the public API for a type with no external use. The eight wrappers are now `pub(crate)`; only `gpu_ops.rs` calls them.
- `Channel(usize)` and `TensorIndex`/`NodeIndex` newtypes from §1.3. Named spec fields already remove the `batch_size`/`channels` transposition risk `Channel` targeted, and the index newtypes only mean something once the arena exists (§6, Phase 4).

**Consequence:** invalid geometry now fails in `forward` with a message from the spec. Before, a kernel larger than the padded input was an arithmetic-underflow panic (debug) or a wrap (release). A zero stride was a divide-by-zero in `Conv2d` and `MaxPool2d` and, in `ConvTranspose2d`, silently produced a wrong-sized output.

**Found on the way:** `ConvAlgo::auto_select` computed `height - kernel_size` from the *unpadded* height, so a valid conv whose kernel fits only after padding (5x5 kernel, pad 1, 3x3 input) panicked under `Auto`. `forward` now passes the padded extents, which makes the heuristic's "stride 1, pad 0" estimate exactly the stride-1 output size. The selection heuristic for padded convs now sees larger (correct) output sizes.

### MatmulSpec (ADDENDUM §1.2)

**Decision:** `MatmulSpec` in `src/spec.rs` is built once in `matmul` from the two operand `Shape`s and stored in `MatMulGradFn` (`Option<MatmulSpec>`; `None` for the vector forms). It validates rank >= 2 on both sides, **equal rank**, matching inner dims, and that the batch dims broadcast (equal or 1 per axis). Forward, backward, and the three GPU matmul wrappers read `m`, `k`, `n`, and the broadcast shapes from it. The batched forward and backward loops use `chunks_exact` plus length asserts, so a mismatch fails instead of skipping.

**Why equal rank is rejected:** B4 came from the backward indexing one operand's shape with the other's rank. With a spec that never happens, but supporting `[2,3,4] @ [2,1,4,6]` or 3D@2D would still need a correct unequal-rank forward (the audit found today's `broadcast_to` path wrong for these shapes) and a matching backward. That is a new feature, not a defect fix, so the spec rejects it with an `equal rank` message, as ADDENDUM §1.2 and the audit propose.

**Rejected:**
- NumPy/PyTorch right-aligned batch broadcasting for unequal ranks. Deferred as a feature; it can be added by teaching `MatmulSpec::new` to left-pad the shorter batch.
- Folding the vector forms into the spec. They have different semantics (the rank-1 operand is not a matrix), and `MatmulSpec` is "a matrix product" by definition. They keep their own forward and backward arms, so the `None` case is the vector forms.
- Letting a batched spec reach the GPU kernels. The kernels are 2D only; `GpuKernels::matmul*` assert `!spec.is_batched()` instead of silently using only `m, k, n`.

**Also changed:**
- `GpuKernels::matmul`, `matmul_backward_a`, and `matmul_backward_b` are now `pub(crate)`, for the same reason as the conv wrappers. `test_gpu_matmul` was the only external caller; it moved from `tests/gpu_test.rs` into `src/gpu/kernels.rs` as a unit test (with a non-square case, so swapped `m/k/n` shows up).
- The GPU matmul backward returned early even when a gradient kernel failed (a `FIXME`), which silently froze a parameter. It now returns early only if every required gradient was computed, and otherwise falls through to the CPU path with a `warn`. No test forces a kernel failure, so the fall-through path itself is untested.

### PadValue (ADDENDUM §2.4)

**Decision:** `PadValue` (`src/spec.rs`, re-exported at the crate root) is the required fill argument of `pad_with`/`try_pad_with`. It has `ZERO` and `NEG_INF` and nothing else. Convolution pads with `ZERO`; `MaxPool2d` pads with `NEG_INF`. H1 existed because max pooling reused a `pad` whose zero fill was implicit.

**Why `pad` was removed, not kept as a zero shorthand:** a shorthand leaves the implicit default in reach, and the next op with a neutral element that isn't zero would reuse it the same way. Every caller (conv, tests, benches) now names its value.

**Why the half-kernel guard lives in `MaxPool2d`:** `-inf` padding makes a window that is all padding output `-inf`, and `max_idx` stays 0, so backward adds that window's gradient to element 0. PyTorch forbids padding above half the kernel for this reason. `Window2dSpec` is shared with conv and im2col, where large padding is legal, so the check is in `pool_forward`, after the spec is built. Four existing tests used larger padding. Three now use a legal geometry and `test_maxpool2d_large_padding` is gone; the guard has its own test.

**Rejected:**
- A public `PadValue::constant(f32)`. No caller needs it, and an arbitrary value is how a bare `0.0` crept back in. Add it with the first constant-pad caller.
- Reducing the pooled maximum over only the real elements, which would allow any padding. That is a different kernel and would change the backward routing; out of scope for a fill-value fix.

**GPU:** the value rides in `MovementParams.padding[1]` as `f32::to_bits`, which was an unused uniform slot, so the `#[repr(C)]` layout and the other movement shaders are unchanged. `pad` in `movement.wgsl` reads it with `bitcast<f32>`. Backward needs nothing: it shrinks the gradient, which never touches the border.

### `define_ops!` for unary and binary ops (ADDENDUM §5.1)

**Decision:** `UnaryOp` and `BinaryOp` are each one table in `src/ops/unary.rs` / `binary.rs`, expanded by `define_ops!` (`src/macros/define_ops.rs`, plain `macro_rules!`). A row generates the enum, `ALL`, the CPU `forward`, the CPU `backward`, and the GPU pipeline lookup. Adding an op is one row, plus a WGSL entry point and a `ComputePipelines` field if it has a kernel.

**Rows name `ComputePipelines` fields, not strings.** The sketch had `("neg", "neg_backward")`. Strings still left `_ => None` / `_ => panic!` arms in `GpuKernels`, which is how Erf named two kernels that didn't exist (forward silently ran on CPU, backward panicked). With field idents, `gpu(neg, neg_backward)` against a missing field is a compile error. `GpuKernels::unary_op`, `unary_backward`, `binary_op`, `binary_backward_a`/`_b`, and `binary_backward_broadcast_safe` take the op enum, a public API change.

**`cpu_only` is explicit.** An op with no kernel says so in its row, and `gpu_pipelines` returns `None`, so the caller takes the CPU path instead of the GPU function panicking. No current op uses it (Erf got real kernels); the binary table has no such clause because every binary op has a forward kernel.

**Binary rows are two sections** (`differentiable { .. }`, `non_differentiable { .. }`) because one repetition can't alternate between two row shapes. Mod and Cmplt having no derivative is now part of the grammar: `is_trivial` is generated from the section, and `BinaryOp::backward` on one panics.

**One generic binary CPU backward.** The row gives the per-element `(∂L/∂a, ∂L/∂b)` at the broadcast shape; `BinaryGradFn` broadcasts both operands, calls the row, and runs `sum_over_broadcast_dims` once per operand. This replaced about 170 lines of per-op copies of that reduction. Cost: Add and Sub now materialise `broadcast_to` copies of their operands, which they used to skip. Accepted ("correctness over speed"). Max ties go to the first operand, pinned by a test.

**GPU backward never returns a partial result.** The old code returned `vec![None, gy]` when only one gradient came back (a FIXME), which freezes a parameter. `gpu_binary_backward` (in `gpu_ops.rs`, with a not-gpu stub, so `binary.rs` has no `cfg(feature = "gpu")`) returns `None` unless every required gradient was computed, logs at `warn`, and the CPU path runs.

**Not covered:**
- Movement ops keep their per-op structs and hand-written dispatch. They carry real state and different kernels and would need a second grammar. They share the one thing that could drift instead: each of `expand`, `pad`, `shrink`, and `stride_op` computes GPU-or-CPU storage and ends in a single `RawTensor::movement_edge` call, where it used to build an identical `GradFn::Movement` literal once per path. That needed not-gpu stubs for the five forward `gpu_*` wrappers (`gpu_ops.rs`), so `movement.rs` carries no forward `cfg`.
- The non-safe `GpuKernels::binary_backward_broadcast` (atomic-add scatter, dead code behind `#[allow(dead_code)]`) was deleted rather than given a row slot. It returned wrong gradients silently when checked on the commit before this change, so nothing working was lost. The race-free two-pass path remains but also fails (wgpu bind-group validation error); both are unreachable, since GPU forward never broadcasts (`notes/TODO.md`). The non-safe `*_broadcast` pipelines and shader entry points were deleted afterwards, along with the helpers and `params` binding only they used; `binary_backward.wgsl` now holds only the same-shape `*_backward_a/b` entries.

### `into_loss` / `ScalarLoss` (ADDENDUM §2.2)

**Decision:** `Tensor::into_loss(self) -> Result<ScalarLoss>` is the only way to start backward. `ScalarLoss::backward(&self)` is infallible, with no `Result`, because `into_loss` has already checked both conditions: one element (`VoltaError::NotScalar { shape }`) and `requires_grad` (`VoltaError::NoGradRequired`). The engine is `RawTensor::run_backward`, `pub(crate)`.

**`TensorOps::backward` was removed, not deprecated or kept as a forwarding shim.** A forwarder would leave the guarantee at runtime only, so there would be nothing for a trybuild test to pin. With `-D warnings`, a deprecation would have forced every call site to change anyway. `tests/compile_fail/backward_on_tensor.rs` is the first trybuild case and pins the E0599.

**Non-scalar roots migrate with an explicit `.sum()`, not a seeded-backward API.** The old code seeded a non-scalar root with all ones, which is an implicit `.sum()`. `sum`'s backward broadcasts ones, so gradients are identical and no second public entry point is needed. The `[1]` special case in the old seed code computed the same size as the general path, so it was dead and was deleted.

**`into_loss` takes `self`, as the ADDENDUM sketch does.** A call site that uses the tensor afterwards writes `loss.clone().into_loss()`; `Tensor` is an `Rc`, so that is a refcount bump.

**`backward` can be called twice.** It takes `&self` and accumulates, as before. `set_requires_grad(false)` on another handle after `into_loss` is the only way to break the check, and `run_backward` has a `debug_assert!` for it.

## Traits and Storage (ADDENDUM Phase 3)

### Matmul layout: contiguous-only

**Decision:** matmul only ever sees dense row-major operands. `matmul_raw` is `pub(crate)`, documents that contract, and asserts `a.len() == m*k` and `b.len() == k*n`. The same assumption holds in the vector forms, the batched `chunks_exact` slicing, the CPU backward, and the GPU forward and backward kernels. When storage gains strides, the public `matmul` makes a view contiguous before computing; it does not panic on one, because `x.t().matmul(y)` is ordinary usage.

Choke points for whoever lands views: the operand reads in `RawTensor::matmul` (`s.data.clone()`, `o.data.clone()`) and in `MatMulGradFn::backward` (`x.data`, `y.data`, `out_grad.data`) must go through a contiguous materialization.

**Rejected:** a strided `MatmulSpec` carrying per-operand `(rs, cs)`. It cannot be tested until views exist, so the bounds check and the strides would be written against nothing real. On the default macOS build `cblas_sgemm` takes only `lda`, so it would also mean two backends with different copy behaviour behind one API. Contiguous-only is the restrictive option, so widening it later is additive; the reverse is not.

**Consequence:** the `transpose_2d` copies in backward and the materialized batch broadcasts stay. That costs speed, not correctness. The contract is vacuously true today, so `matmul_of_transposed_operand_matches_reference`, `matmul_of_permuted_batched_operand_matches_reference`, and `test_matmul_through_transpose_gradients` pass now and are the fence that starts guarding it when `transpose`/`permute` return views.

**Likely next step, only if profiling shows the backward copies matter:** a per-operand transpose flag (`CblasTrans` on cblas, swapped strides on `matrixmultiply`). Not checked against the call sites.

### Indexing policy: panic, not NaN (ADDENDUM §8)

**Decision:** kernel code reads and writes through `crate::index::{fetch, slot, span}`, which panic on an out-of-bounds index in every build profile, with `#[track_caller]` naming the kernel site. `.get(i).unwrap_or(0.0)`, `if let Some(slot) = v.get_mut(i)`, and the `if let Some(&v) = x.get(i) { v } else { 0.0 }` form are banned for numeric data; `just check-fallbacks` enforces it. `clippy::indexing_slicing` stays denied, since the helpers are built on `get`.

**Why not §8's NaN:** §8 sketched `fetch` as `debug_assert!` plus a NaN fallback in release. A NaN can make a bad *read* loud, but a skipped *write* has no value to poison: the slot just keeps its old contents, so the write half of the policy would still be silent in release. Splitting the rule (reads NaN, writes panic) was the alternative. One rule for both is easier to review, and a panic costs the same as `get` plus `unwrap_or`.

**Zero that means something:** conv and im2col padding reads as zero because the coordinate is outside the image. That check is spelled out (`ih < height && iw < width`) and commented at the site, and the in-range read then goes through `fetch`. A slice bound never implements padding.

**Two panics that wait on the error-handling policy:** the §8 pass added two panics that go past "kernel mapping bug" and touch the undecided panic-vs-`Result` question (`notes/TODO.md`, H4 and Rust/errors). `to_one_hot` asserts `label < num_classes`, and that label can come from real data, so it may belong on the `Result` side. `GpuKernels::binary_backward_broadcast_safe` panics on rank > 4 instead of truncating the shape into a 4-wide array (H5); the path is unreachable today. Both were kept loud because the alternative was silent corruption. Revisit them when the error regime lands.

**Out of scope:** `HashMap` lookups (`state.get("weight")`, where a missing key is legitimate) and the integer `unwrap_or(1)` shape/stride defaults. The latter are tracked in `notes/TODO.md` and overlap the `Shape` field flip.

### Delete `TensorOps` (ADDENDUM §3.4)

**Decision:** the 47-method `TensorOps` trait is gone. Its methods are inherent methods on `Tensor` (`impl Tensor` in `src/tensor.rs`), each a one-line forward to the `RawTensor` function in `src/ops/`, with a doc comment and `#[must_use]`. Nothing replaces the `use volta::TensorOps` import, so callers just drop it.

**Why no re-export-only extension trait:** §3.4 allowed one "for discoverability". It would keep an import that does nothing, and the crate root already re-exports `Tensor`. It would also be one more public trait to seal under §3.1.

**Why now, before the storage and `Shape` work:** the trait duplicated every op signature, so the `RawTensor.shape` flip and the storage rewrite would have changed each one twice.

**What changed at call sites:** `#[must_use]` is new. It flagged 15 places that discarded a result, all `#[should_panic]` tests and bench loops, now `let _ =` or `black_box(..)`. `use_self` also applies, so signatures say `Self`. The trybuild case `backward_on_tensor` still pins E0599; it used to import `TensorOps` to prove the trait method was gone, and an inherent `backward` would now be found without an import, so the import is no longer needed.

### Split layer traits; `Module` and `Layer` stay open (ADDENDUM §3.1, §3.2)

**Decision:** `Module` is the base trait (parameters, state dict, `zero_grad`, `to_device`, `train`/`eval`) and has no `forward`. `Layer: Module` adds `forward(&Tensor) -> Tensor`, and `Sequential` holds `Box<dyn Layer>`. `IndexedLayer: Module` (`Embedding::forward_indices(&[usize])`) and `RecurrentLayer: Module` (`LSTMCell`: `initial_state`, `step(x, &mut state) -> h`) are separate. Putting an `Embedding` in a `Sequential` is now a compile error (`tests/compile_fail/embedding_in_sequential.rs`), not a runtime panic.

**Departure from §3.1:** only `IndexedLayer` and `RecurrentLayer` are sealed. `Module` and `Layer` are open. There is no `GradFn::Custom` and `edge()` is `pub(crate)`, so a user-written `impl Layer` that composes built-in layers is the one extension path the library has, and it is how PyTorch users build models. Sealing it would also have broken `tests/improvements_test.rs` (`CountingModule`), which is outside code and now doubles as the proof that the open traits stay open. The cost is that we can't add a required method to `Module` or `Layer` without a breaking change; provided methods are still free. The two sealed traits have no outside use case, and sealing them lets `State` and the step signature change within 0.x. Trybuild cases `seal_indexed_layer` and `seal_recurrent_layer` pin the seal. The marker is named `SealedLayer`, not `Sealed`: with the `gpu` feature a dependency (`unicode_width`) has its own private `Sealed`, and rustc adds a "similarly named trait" note that made the pinned `.stderr` differ by feature set.

**LSTM:** `forward_step(x, Option<(&h, &c)>) -> (h, c)` was deleted rather than kept beside `step`. `State = (Tensor, Tensor)` carries `c`, which `Module::forward` used to drop. `step` still goes through `slice_gate`, so B2 (the severed graph) is untouched and its `assert_trainable!` test stays `#[ignore]`d, driven through a test-only one-step `Layer` wrapper.

**H6:** `TransformerBlock` holds a real `MultiHeadAttention`; `SimpleAttention` and the `Rc<dyn Module>` are gone. `causal` reaches the attention, and `load_state_dict` now loads the `attention.*` keys (it silently skipped them before). Tests: `test_block_causal_changes_output`, `test_block_attention_state_dict_roundtrip`, and the existing `assert_trainable!` now covers the attention projections.

**Not done:** `load_state_dict` still returns `()`, and the `*Layer` traits don't yet cover attention (`forward(x, causal, mask)` stays inherent on `MultiHeadAttention`).

### CPU storage: `Arc<[u8]>` copy-on-write (ADDENDUM §7)

**Decision:** `Storage::Cpu` holds `Arc<[u8]>`. `clone` is a refcount bump, and the one write path, `as_f32_slice_mut` (behind `DerefMut`, `IndexMut`, `iter_mut`, and `as_mut_slice`), calls `Arc::make_mut`, which copies only when another storage shares the buffer. Value semantics are unchanged: a write to one tensor is never visible through another. Before this, every `data.clone()` snapshot in an op prologue or a `GradFn` was a full memcpy, including `reshape`, whose comment claimed it was a view.

**Why `Arc`, not `Rc`:** §7 specifies it, the GPU arm already uses `Arc<GpuBuffer>`, and `Rc` would build a `!Send` type into storage, which the arena route to threading (§6, "Stay synchronous") would then have to undo. Whether `Storage` is `Send` today with the `gpu` feature on was not checked. The atomic refcount costs nothing measurable next to the copies it removes. No speed numbers were taken; this was done in a cloud session where benches aren't valid.

**Why `Arc<[u8]>`, not `Arc<Vec<u8>>`:** one indirection instead of two, and every typed constructor (`cpu`, `cpu_f64`, ...) already copies once from the typed `Vec` into bytes, so building the `Arc` straight from that slice costs no more than before. The exception is `try_from_bytes(Vec<u8>)` (and `to_dtype`'s `U8`/`Bool` arms), which used to take the `Vec` without copying and now copy it once into the `Arc`.

**Alignment (unchanged):** `cast_slice::<u8, f32>` needs 4-byte alignment. The old `Vec<u8>` relied on the allocator handing out aligned blocks. `Arc<[u8]>` puts the bytes after its two-`usize` header, so they are 8-aligned in practice. Neither is a documented guarantee. bytemuck panics on a misaligned cast, so a failure would be loud. A typed backing store would fix it properly; that waits for the `Deref` removal.

**Not changed:** op outputs still allocate, and broadcasts still materialize. The GPU host cache (`OnceCell`, H3) and the `Deref` removal are separate PRs.

**Tests:** `storage::tests::{cpu_clone_shares_buffer, cpu_write_after_clone_copies, cpu_unique_write_is_in_place}` pin sharing, copy-on-write, and the no-copy path for a unique owner (the optimizer case). `autograd::tests::optimizer_write_does_not_leak_into_shared_reshape` checks that an SGD step on `x` leaves a reshape of `x`, which shares its buffer, unchanged.

## Concurrency and Async

### Stay synchronous

**Decision:** no async in the core or the GPU API. Revisit only for isolated data loading or serving.

Autograd is a sequential graph walk over `Rc<RefCell<_>>`, and nothing in it is `Send`. The GPU path blocks on submission, which is coherent for a single-threaded engine. The prerequisites are explicit transfers (ADDENDUM §7) and the arena graph (§6), which is the actual route to threading. Premature `Arc<Mutex>` is how `BufferPool` ended up guarding state only one thread can reach. The full analysis lives in git history (`docs/plans/2026-02-23-async-*.md`, removed 2026-10-25).

## Known Gaps Summary

| Area                   | Gap                                                                    |
| ---------------------- | ---------------------------------------------------------------------- |
| Chain rule             | Cannot derive multivariate chain rule justification for grad summation |
| Traversal              | Conflated root assert with per-node behavior                           |
| `Rc` liveness          | Cannot trace guarantee unprompted                                      |
| Composition pattern    | Not internalized without prompting                                     |
| Broadcasting algorithm | Step-by-step procedure not reproducible unprompted                     |
| Adam                   | Recurrences and update formula required reference                      |

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

**`RefCell` borrow discipline:** `RefCell` provides interior mutability but panics on simultaneous mutable + immutable borrows. The engine must drop the `RefCell` borrow on a node _before_ calling `backward` on its parents, since backward may need to re-borrow the same node. `clone_box` is the mechanism that makes this safe.

---

### `clone_box` and `dyn GradFn`

**Decision:** `clone_box` works around the `Clone: Sized` bound that prevents `Clone` on trait objects.

`Box<dyn GradFn>` can't implement `Clone` directly because `Clone` requires `Sized`. `clone_box` is a method on the trait that each implementor defines, returning a `Box<dyn GradFn>`. This lets the engine clone grad functions without knowing their concrete type.

**Secondary role:** Calling `clone_box` to get an owned copy of the grad function lets the engine drop the `RefCell` borrow on the current node before recursing into parents — preventing a borrow panic.

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

**Deviation from the ADDENDUM sketch:** `edge()` took `Vec<usize>` and `Box<dyn GradFn>` because the `Shape` and `GradFn` enum were Phase 2 work. (Since Phase 2, `edge()` takes `Shape`; `GradFn` is still boxed.) It also takes an explicit `device`, since `Storage` doesn't uniquely determine a `Device` and every call site already tracked one.

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

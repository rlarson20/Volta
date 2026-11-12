# DECISIONS.md — Design decisions

Why the code is the way it is, and what was tried and rejected. What was done is in the commit log; what is open is in the GitHub issues.
Updated: 2026-11-12

---

## Autograd Engine

### Topological Sort

**Decision:** Iterative DFS with a `HashSet` for visited tracking.

Recursive DFS risks stack overflow on deep computation graphs. Iterative DFS with an explicit stack avoids this. The `HashSet` ensures each node is processed once — critical for diamond-shaped graphs where multiple paths converge on a shared ancestor.

**Diamond graphs:** When two downstream nodes share an upstream node, that node receives contributions from both paths. The engine handles this by _accumulating_ (summing) them at each node rather than overwriting (in the pass's own table for an op output, in `grad` for a leaf; see "Only leaves keep a gradient"). This is mathematically correct: partial derivatives from all downstream paths sum to give the total gradient.

---

### Only leaves keep a gradient (S1)

**Decision:** `run_backward` keeps the gradient of an op output in a table that lives for one pass. Only a leaf has its `grad` written, by `Tensor::accumulate_grad`, so a second `backward()` adds only its own contribution. The root is seeded with 1 on every pass. This is PyTorch's rule: `.grad` of a non-leaf is `None` unless `retain_grad()` was called, and there is no `retain_grad` here.

**Before:** every node's incoming gradient was accumulated into its `grad`, and a node's whole `grad` was what it sent to its parents. Nothing cleared an intermediate, so a second pass re-sent the first pass's gradient: `sum(2a)` twice gave `a.grad = 6` (torch 4), and two losses through `b = 2a` gave 10 (torch 8). The root escaped only because it was seeded "if not already set" and never accumulated into. `examples/time_sequence.rs`, which carries LSTM state across timesteps and calls `backward()` each step, counted earlier steps again on every pass.

**Rejected: clear each intermediate's `grad` after its backward step.** It fixes the numbers, but an op output still has a `grad` for a moment, so it is observable mid-pass and stale if a pass panics. It also makes "who clears `grad`" two answers (invariant 3 says `zero_grad` only).

**Consequences:** a test that read an op output's gradient must read the leaf. The main case is a CPU leaf moved with `to_device`: the copy is an op output, so its gradient comes back on the source leaf (see "`to_device` is a differentiable op"). A GPU leaf is made by moving a tensor that doesn't require grad and then marking it.

**Tests:** `tests/sweep_repros.rs` (`second_backward_through_an_intermediate_adds_only_the_new_gradient`, `two_losses_sharing_an_intermediate_sum_their_gradients`) and `autograd::tests` (`backward_twice_accumulates` with exact values, `op_outputs_keep_no_gradient`, `leaf_root_accumulates_its_seed`, `shared_intermediate_sums_within_a_pass_and_adds_per_pass`).

---

### `Rc` and Node Liveness

**Decision:** Parents are stored as `Rc<RefCell<Node>>` in each node's `grad_fn`.

`Rc` keeps the reference count > 0 as long as any child holds a reference to a parent. This means the full computation graph stays alive for the duration of the backward pass — no manual lifetime annotation needed.

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

### In-place writes bump a version counter; backward checks the inputs it reads (#43, S9)

**Decision:** every `RawTensor` has a `version`, bumped by each in-place write to its `data` (an optimizer step, `load_weight`, `cast_in_place`, `Module::to_device`, BatchNorm's running statistics) through `RawTensor::bump_version`. `edge()` records the version of each parent whose values the op's backward reads (`GradFn::saves_input`), and `run_backward` panics before calling that op's backward if one has moved: `Binary(Mul): input 1 was modified by an inplace operation ...`. It is PyTorch's rule and message.

**Before:** nothing was saved at forward time. Backward read each parent's current `data`, so `y = w * x`, then a step on `w`, then `backward()` gave `x.grad` = the new `w`, silently. An optimizer step was one writer; loading a state dict or moving a model's dtype or device did the same.

**Choices:**
- *Only the inputs backward reads are checked.* `Add`, `Sub`, `Neg`, `Sum`, `Mean`, reshapes and casts have gradients that don't depend on an input's values, so a step in between is harmless and torch allows it. `Where`'s mask, `MaxPool2d`'s indices and `Embedding`'s indices live in the `GradFn`, not in a parent. `saves_input` spells out every variant, as `is_trivial` does, so a new op has to decide. A mask that errs toward `true` only costs a spurious panic; toward `false` it reopens S9.
- *A panic, not a `Result`.* Both sides are tensors the program built, so this is a contract violation (invariant 8), and backward has no error channel.
- *One bump per write path, no wrapper around `data`.* `data` is `pub`, but `borrow_mut()` is `pub(crate)` and invariant 2 keeps writers in `autograd.rs`, `ops/` and the few places listed above, so the call sites are countable. A new in-place writer calls `bump_version`.

**Rejected: save a copy-on-write snapshot of each parent's `Storage` in `edge()`.** Backward would use the forward-time values and never fail. But the usual `loss.backward(); step()` leaves the loss graph alive while the step writes the parameter, so every step would copy every parameter to keep the snapshot unshared. It also quietly differs from torch, where this is an error, and the issue asks for torch's behaviour.

**Rejected: a documented rule only.** The ignored repro could not pass, and the hole stayed open for every other in-place writer.

**Not covered:** a `detach()`ed tensor has its own counter (it copies on write, so it keeps the old values and is safe). The GPU optimizer paths bump the same field, since the counter is on the host-side `RawTensor`, not the buffer.

**Tests:** `tests/inplace_version.rs` (each writer, the ops that read and the ones that don't, the training-loop order) and `tests/sweep_repros.rs` (`backward_after_an_optimizer_step_refuses_the_stale_graph`).

---

### `requires_grad` Short-Circuiting

**Decision:** If `requires_grad = false`, no `grad_fn` is attached.

During the backward pass, nodes without a `grad_fn` are leaves — no further traversal. This prunes the graph cheaply: frozen parameters (e.g. a pretrained backbone) don't accumulate gradients and don't participate in the backward DFS.

---

### Composition Pattern

**Decision:** High-level layers are composed from primitive ops; they inherit `grad_fn` for free.

Example: Dropout is implemented as elementwise multiply by a binary mask (`elem_mul`). Because `elem_mul` already has a registered `grad_fn`, Dropout gets backward support without writing any backward code. This is the general pattern: build new ops from primitives already in the graph; the chain rule handles the rest automatically.

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

### Backward

During backward, gradients must be summed over axes that were broadcast. Two cases:

- **Stretched axis (size 1 in original):** sum gradient over that axis, keep dim.
- **Nonexistent axis (shape was shorter):** sum gradient over that axis, drop dim.

This restores the gradient to the original tensor's shape.

---

## GPU Dispatch

**Decision:** `cfg` feature flag at the definition site, a runtime device check, and the device policy for anything without a kernel.

Compile-time `cfg` gates the GPU code, with a not-gpu stub at each definition so call sites carry no `cfg` (ADDENDUM §9). A GPU input to an op with no kernel for the case is a gap: it panics under `DevicePolicy::Strict` and runs on the CPU under `PreferGpu`. A kernel that exists and returns `None` always panics. See "`DevicePolicy`: the device label is a promise" below. The early design (`None` falls back to the CPU path) is what that entry replaced.

### Broadcast Backward on GPU — Race Condition

Naïve parallel reduction over broadcast axes causes write races when multiple threads accumulate into the same output element.

**Fix:** Two-pass scatter/reduce.

1. **Scatter:** each thread writes its partial gradient to a private slot.
2. **Reduce:** a second pass atomically reduces private slots into the output.

This serializes the final accumulation, eliminating the race.

### Full-tensor GPU reductions: a multi-pass tree, f32 partials, nothing read back (#80)

**Decision:** `sum`, `mean`, `max_reduce` and the Muon Frobenius norm run `reduce.wgsl`: a 256-thread workgroup folds 1024 values (4 per thread, then a shared-memory tree) into one f32 partial, and the host chains passes in one command buffer until one value is left. The scalar stays in a one-element GPU buffer, rounded once to the input's dtype by the cast kernel. A max pass carries `(value, index)`, starts from a real element (not a sentinel), and leaves the index in a GPU buffer that `MaxReduceGradFn` holds (`MaxIndex::Device`). The backward kernels read the upstream scalar from `out_grad` on the device.

**Why:** the kernels were `workgroup_size(1)` loops with a host readback after the forward and another before each backward (and `max_reduce` downloaded the whole tensor to find its index). Measured on the Apple M2, release, median of 5 (ms, `before -> after`):

| n | sum | max | sum fwd+bwd |
| --- | --- | --- | --- |
| 65,536 | 15.4 -> 1.45 | 15.9 -> 1.43 | 16.8 -> 1.78 |
| 1,048,576 | 154.8 -> 1.42 | 144.4 -> 1.42 | 176.1 -> 3.35 |
| 8,388,608 | 1308 -> 1.42 | 1250 -> 2.75 | 1302 -> 18.1 |

A Muon step on 4096x4096 took 25.5 s before and after: it is the matmuls, not the norm, so the issue's worry about the single-workgroup norm did not hold at that size. The norm still moved onto the shared kernel, which is exact in a tree and needs no second code path. The 1.4 ms floor in the table is the synchronous `item()` readback, not the kernel.

**Choices:**
- *The tree is also the accuracy fix:* a chain of 2^25 f32 adds stalls at 2^24, a tree does not (S6, #41; the CPU half of S6 is separate).
- *The tie rule is one WGSL function, `better`,* currently the CPU's `max_by` (the highest index wins), so CPU and GPU agree. Torch splits the gradient evenly over ties (S4, #40); that change is `better` plus the backward kernel, and the max value buffer is already on the device for the `x == max` mask.
- *The index lives in a Bool-typed buffer* (one `u32` word per element, like `amp_unscale`'s `found` flag), never exposed as a tensor. `DeviceIndex` is defined in `gpu_ops.rs` with a not-gpu stub, so `reduce.rs` carries no `cfg`.

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

**Revision (2026-10-30):** B2 is fixed (`slice_gate` is a `shrink`), and the LSTMCell test passes un-ignored. It drives the cell through a test-only wrapper that takes two steps, not one. One step from the zero initial state gives `weight_hh` an exactly zero gradient (`dgatesᵀ · h` with `h = 0`), so a one-step wrapper fails the macro on a correct cell. The same holds for any layer whose initial state zeroes a parameter's gradient: drive it past that state, and don't loosen the macro.

**Calls go through UFCS** (`<_ as Module>::forward(..)`), because `MultiHeadAttention` and `TransformerBlock` have an inherent `forward` of a different arity next to `Module::forward`, and an inherent method wins `.forward()` resolution whatever its arity.

## Types for Dims and Ops (ADDENDUM Phase 2)

### Shape boundary: signatures, not the field

**Superseded** by "`RawTensor.shape` is a `Shape`" (Phase 3). Kept for the reasoning.

**Decision:** `Shape` is the type at function boundaries (`edge()`, tensor construction and validation), and `Axis` is the type of dim arguments. `RawTensor.shape` stays a `Vec<usize>`, and `Tensor::shape()` still returns one.

Roughly 440 places in `src/` read the `.shape` field (about 365 outside `lib.rs`): kernels, `GradFn` structs, and conv/matmul index math. `Shape` deliberately has no `Deref`; stepping outside it must be spelled `truncate_rank()` so the demotion stays greppable. Storing a `Shape` in the field would turn those reads into hundreds of `truncate_rank()` calls, and a grep that matches everything finds nothing. The field flips naturally with the storage work (ADDENDUM §7), when kernels stop reading it directly.

**Rejected:** adding `Deref<Target = [usize]>` so the field could change without touching readers. It compiles with a small diff, but it removes the explicit demotion that §1.2 is built around, and any `&[usize]`-taking function would then silently accept a shape.

**Consequence:** `truncate_rank()` has six calls outside `shape.rs`, in four places: `edge()`, the two `RawTensor` constructors (two calls each), and `try_broadcast_shape`. Each is where a validated `Shape` is written into the field or handed back as a `Vec`. Add a new one only at a boundary like those.

### One validation point, and `Shape::derived`

**Decision:** `Shape::from_dims` is the only place that rejects a zero dimension or more than `MAX_NUMEL` (100M) elements. `expand`, `pad`, `broadcast_to`, and the tensor constructors call it instead of keeping private `MAX_ALLOC` checks. Op code that builds an output shape from existing tensors uses `Shape::derived`, which panics on failure.

Before this, the zero/cap checks lived in `Tensor::try_new` only. `try_new_with_storage`, which `edge()` reaches, checked neither, and `expand`, `pad`, and `broadcast_to` each carried a copy of the cap. `pad`'s message even said "Expand would create tensor", copied from `expand`. An op whose output was zero-sized could therefore build a degenerate tensor that no check ever saw.

**Consequence:** an op that produces a zero-sized or over-cap output now panics at `edge()` instead. No existing test hit this. `reshape` needed no change: its size-equality assert already guarantees the target is a valid `Shape`, and `edge()` validates it.

**Public constructors keep `&[usize]`** (`new`, `zeros`, `randn`, `reshape`, `expand`), so literal shapes stay ergonomic; they still go through the one checked constructor. **`Strides::from_shape`** (`pub(crate)`) replaced `RawTensor::compute_strides`, which is deleted. It takes `&[usize]` because op code holds raw dims, and call sites demote with `truncate_rank()`. Its body is a running product with no `if let Some` skip.

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

**Fields beyond the sketch:** `a_batch`, `b_batch`, and `out_batch` carry the broadcast batch dims, so backward reduces over them without re-deriving them from the operands.

**Also changed:**
- `GpuKernels::matmul`, `matmul_backward_a`, and `matmul_backward_b` are now `pub(crate)`, for the same reason as the conv wrappers. `test_gpu_matmul` was the only external caller; it moved from `tests/gpu_test.rs` into `src/gpu/kernels.rs` as a unit test (with a non-square case, so swapped `m/k/n` shows up).
- The GPU matmul backward returned early even when a gradient kernel failed (a `FIXME`), which silently froze a parameter. Each required gradient now comes from its kernel or panics: a `None` calls `kernel_failed("MatMul")` (see "`DevicePolicy`"), and only a case with no kernel (batched, vector) is a gap. No test forces a kernel failure, so the panic branch is checked by review only.

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

**Binary rows are two sections** (`differentiable { .. }`, `non_differentiable { .. }`) because one repetition can't alternate between two row shapes. Mod and Cmplt having no derivative is now part of the grammar: `is_trivial` is generated from the section, and `BinaryOp::backward` on one panics. *(P5 removed `Cmplt`; comparisons are `CmpOp`, returning Bool. #45 (S5) made `Mod` floored, as `torch.remainder`, and differentiable (`∂a = g`, `∂b = -g·floor(a/b)`), which leaves `non_differentiable {}` empty but in the grammar; the macro emits one arm per variant so an empty section compiles.)*

**One generic binary CPU backward.** The row gives the per-element `(∂L/∂a, ∂L/∂b)` at the broadcast shape; `BinaryGradFn` broadcasts both operands, calls the row, and runs `sum_over_broadcast_dims` once per operand. This replaced about 170 lines of per-op copies of that reduction. Cost: Add and Sub now materialise `broadcast_to` copies of their operands, which they used to skip. Accepted ("correctness over speed"). Max ties go to the first operand, pinned by a test.

**GPU backward never returns a partial result.** The old code returned `vec![None, gy]` when only one gradient came back (a FIXME), which freezes a parameter. `gpu_binary_backward` (in `gpu_ops.rs`, with a not-gpu stub, so `binary.rs` has no `cfg(feature = "gpu")`) returns every required gradient or panics through `kernel_failed`. It returns `None` only for CPU inputs and for differing shapes, which are a gap, so the CPU path runs and the device policy applies when the gradient reaches its GPU parameter.

**Not covered:**
- Movement ops keep their per-op structs and hand-written dispatch. They carry real state and different kernels and would need a second grammar. They share the one thing that could drift instead: each of `expand`, `pad`, `shrink`, and `stride_op` computes GPU-or-CPU storage and ends in a single `RawTensor::movement_edge` call, where it used to build an identical `GradFn::Movement` literal once per path. That needed not-gpu stubs for the five forward `gpu_*` wrappers (`gpu_ops.rs`), so `movement.rs` carries no forward `cfg`.
- The non-safe `GpuKernels::binary_backward_broadcast` (atomic-add scatter, dead code behind `#[allow(dead_code)]`) was deleted rather than given a row slot. It returned wrong gradients silently when checked on the commit before this change, so nothing working was lost. The race-free two-pass path remains but also fails (wgpu bind-group validation error); both are unreachable, since GPU forward never broadcasts (#60). The non-safe `*_broadcast` pipelines and shader entry points were deleted afterwards, along with the helpers and `params` binding only they used; `binary_backward.wgsl` now holds only the same-shape `*_backward_a/b` entries.

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

**Two panics that wait on the error-handling policy:** the §8 pass added two panics that go past "kernel mapping bug" and touch the undecided panic-vs-`Result` question (then in `notes/TODO.md`, H4 and Rust/errors; that file was retired in #87, last version at `9d8aff5`; since settled, see "Errors" below). `to_one_hot` asserts `label < num_classes`, and that label can come from real data, so it may belong on the `Result` side. `GpuKernels::binary_backward_broadcast_safe` panics on rank > 4 instead of truncating the shape into a 4-wide array (H5); the path is unreachable today. Both were kept loud because the alternative was silent corruption. Revisit them when the error regime lands.

**Out of scope:** `HashMap` lookups (`state.get("weight")`, where a missing key is legitimate).

**Integer shape reads (follow-up, first PR of the `Shape` flip):** the same rule now covers shape and stride reads. The ~70 `shape.get(i).copied().unwrap_or(1)` defaults turned a rank bug into a size-1 axis, which in broadcasting code silently changes the answer. A read of a shape whose rank the caller has already checked uses `crate::shape::fixed_dims::<N>(&shape)`, which returns `[usize; N]` and panics, naming the caller, on any other rank. A positional read inside a rank-bounded loop uses `fetch`. Every converted site sat behind a rank guard (a `len()` match, an `assert_eq!` on rank, or a loop bounded by rank), so none of them could fire on a valid input; `just ci` passing unchanged is the evidence. Where a 1 really is the meaning, the code says so with an explicit `match` on the shape slice: `xavier_uniform` on `[n]` is a bias vector (`n × 1`), and Muon treats a vector as `(n, 1)`. `xavier_uniform` on rank ≥ 3 used to default and then fail as a data/shape mismatch; it now panics up front. `just check-fallbacks` rejects `unwrap_or(1)` and `unwrap_or(&1)` everywhere, examples included. Doing this before the field flip lets the flip map each loud read straight to a `Shape` accessor.

### Delete `TensorOps` (ADDENDUM §3.4)

**Decision:** the 47-method `TensorOps` trait is gone. Its methods are inherent methods on `Tensor` (`impl Tensor` in `src/tensor.rs`), each a one-line forward to the `RawTensor` function in `src/ops/`, with a doc comment and `#[must_use]`. Nothing replaces the `use volta::TensorOps` import, so callers just drop it.

**Why no re-export-only extension trait:** §3.4 allowed one "for discoverability". It would keep an import that does nothing, and the crate root already re-exports `Tensor`. It would also be one more public trait to seal under §3.1.

**Why now, before the storage and `Shape` work:** the trait duplicated every op signature, so the `RawTensor.shape` flip and the storage rewrite would have changed each one twice.

**What changed at call sites:** `#[must_use]` is new. It flagged 15 places that discarded a result, all `#[should_panic]` tests and bench loops, now `let _ =` or `black_box(..)`. `use_self` also applies, so signatures say `Self`. The trybuild case `backward_on_tensor` still pins E0599; it used to import `TensorOps` to prove the trait method was gone, and an inherent `backward` would now be found without an import, so the import is no longer needed.

### Split layer traits; `Module` and `Layer` stay open (ADDENDUM §3.1, §3.2)

**Decision:** `Module` is the base trait (parameters, state dict, `zero_grad`, `to_device`, `train`/`eval`) and has no `forward`. `Layer: Module` adds `forward(&Tensor) -> Tensor`, and `Sequential` holds `Box<dyn Layer>`. `IndexedLayer: Module` (`Embedding::forward_indices(&[usize])`) and `RecurrentLayer: Module` (`LSTMCell`: `initial_state`, `step(x, &mut state) -> h`) are separate. Putting an `Embedding` in a `Sequential` is now a compile error (`tests/compile_fail/embedding_in_sequential.rs`), not a runtime panic. *(Superseded by P5: `IndexedLayer` is gone; `Embedding` is a `Layer` over I64/I32 index tensors, as the alternative in ADDENDUM §3.2 proposed once dtypes were real. Only `RecurrentLayer` is sealed.)*

**Departure from §3.1:** only `IndexedLayer` and `RecurrentLayer` are sealed (`RecurrentLayer` alone since P5). `Module` and `Layer` are open. There is no `GradFn::Custom` and `edge()` is `pub(crate)`, so a user-written `impl Layer` that composes built-in layers is the one extension path the library has, and it is how PyTorch users build models. Sealing it would also have broken `tests/improvements_test.rs` (`CountingModule`; renamed to `tests/safety_and_state_dict.rs` for #74), which is outside code and now doubles as the proof that the open traits stay open. The cost is that we can't add a required method to `Module` or `Layer` without a breaking change; provided methods are still free. The two sealed traits have no outside use case, and sealing them lets `State` and the step signature change within 0.x. Trybuild cases `seal_indexed_layer` and `seal_recurrent_layer` pin the seal. The marker is named `SealedLayer`, not `Sealed`: with the `gpu` feature a dependency (`unicode_width`) has its own private `Sealed`, and rustc adds a "similarly named trait" note that made the pinned `.stderr` differ by feature set.

**LSTM:** `forward_step(x, Option<(&h, &c)>) -> (h, c)` was deleted rather than kept beside `step`. `State = (Tensor, Tensor)` carries `c`, which `Module::forward` used to drop. `step` still goes through `slice_gate`, so B2 (the severed graph) is untouched and its `assert_trainable!` test stays `#[ignore]`d, driven through a test-only one-step `Layer` wrapper.

**Update (2026-10-30):** B2 is fixed: `slice_gate` is a `shrink`, so the gates keep their gradient and their device. The test wrapper takes two steps (see the `assert_trainable!` revision), and `test_lstmcell_step_gradcheck` checks one step's gradients for `x`, `h`, and `c` against finite differences.

**H6:** `TransformerBlock` holds a real `MultiHeadAttention`; `SimpleAttention` and the `Rc<dyn Module>` are gone. `causal` reaches the attention, and `load_state_dict` now loads the `attention.*` keys (it silently skipped them before). Tests: `test_block_causal_changes_output`, `test_block_attention_state_dict_roundtrip`, and the existing `assert_trainable!` now covers the attention projections.

**Not done:** `load_state_dict` still returns `()`, and the `*Layer` traits don't yet cover attention (`forward(x, causal, mask)` stays inherent on `MultiHeadAttention`).

### CPU storage: `Arc<[u8]>` copy-on-write (ADDENDUM §7)

**Decision:** `Storage::Cpu` holds `Arc<[u8]>`. `clone` is a refcount bump, and the one write path (`as_f32_slice_mut` at the time, `as_host_f32_mut` since the `Deref` removal) calls `Arc::make_mut`, which copies only when another storage shares the buffer. Value semantics are unchanged: a write to one tensor is never visible through another. Before this, every `data.clone()` snapshot in an op prologue or a `GradFn` was a full memcpy, including `reshape`, whose comment claimed it was a view.

**Why `Arc`, not `Rc`:** §7 specifies it, the GPU arm already uses `Arc<GpuBuffer>`, and `Rc` would build a `!Send` type into storage, which the arena route to threading (§6, "Stay synchronous") would then have to undo. Whether `Storage` is `Send` today with the `gpu` feature on was not checked. The atomic refcount costs nothing measurable next to the copies it removes. No speed numbers were taken; this was done in a cloud session where benches aren't valid.

**Why `Arc<[u8]>`, not `Arc<Vec<u8>>`:** one indirection instead of two, and every typed constructor (`cpu`, `cpu_f64`, ...) already copies once from the typed `Vec` into bytes, so building the `Arc` straight from that slice costs no more than before. The exception is `try_from_bytes(Vec<u8>)` (and `to_dtype`'s `U8`/`Bool` arms), which used to take the `Vec` without copying and now copy it once into the `Arc`.

**Alignment (unchanged):** `cast_slice::<u8, f32>` needs 4-byte alignment. The old `Vec<u8>` relied on the allocator handing out aligned blocks. `Arc<[u8]>` puts the bytes after its two-`usize` header, so they are 8-aligned in practice. Neither is a documented guarantee. bytemuck panics on a misaligned cast, so a failure would be loud. A typed backing store would fix it properly.

**Not changed:** op outputs still allocate, and broadcasts still materialize. The GPU host cache (`OnceCell`, H3) is a separate PR; the `Deref` removal is the next entry.

**Tests:** `storage::tests::{cpu_clone_shares_buffer, cpu_write_after_clone_copies, cpu_unique_write_is_in_place}` pin sharing, copy-on-write, and the no-copy path for a unique owner (the optimizer case). `autograd::tests::optimizer_write_does_not_leak_into_shared_reshape` checks that an SGD step on `x` leaves a reshape of `x`, which shares its buffer, unchanged.

### No `Deref` on `Storage` (ADDENDUM §7)

**Decision:** `Storage` has no `Deref`/`DerefMut`, no `Index`/`IndexMut`, no `IntoIterator`, and no `iter`/`iter_mut`/`as_slice`/`as_f32_slice`/`as_f32_slice_mut`. Host access is three explicit methods:

- `as_host_f32() -> Result<&[f32], VoltaError>`: CPU F32 only, never transfers. GPU storage is `DeviceError`; another dtype is `DTypeMismatch`.
- `to_host_f32() -> Cow<'_, [f32]>`: the host transfer, spelled out. CPU F32 is borrowed; GPU F32 syncs on first read and borrows the cached copy; other dtypes convert into an owned `Vec`.
- `as_host_f32_mut() -> Result<&mut [f32], VoltaError>`: the one write path, copy-on-write. GPU storage and other dtypes are errors. Callers `.expect` with a message naming the site.

Before, `fetch(&s.data, i)`, `zip(&data)`, and `data.iter()` coerced silently, so a GPU→host readback or a GPU-storage panic (`DerefMut` on GPU, the H8 class) had no trace at the call site. The removal is pinned by `tests/compile_fail/storage_no_deref.rs`.

**Why kernels call `to_host_f32`, not `as_host_f32`:** GPU storage can reach CPU kernels today through gaps (`PreferGpu` runs them on the CPU), and which of the 144 `to_host_f32` call sites in `src/` can see it was not checked on a GPU. `to_host_f32` keeps each site's behavior on both devices; `as_host_f32().expect(..)` would have added panics that only a GPU run could exercise. `DevicePolicy` now decides what a GPU operand at a CPU kernel does; narrowing individual sites to `as_host_f32` is not part of this decision.

**Why the GPU arm borrows the existing cache:** an owned readback per call would re-copy from the GPU on every read where today's cache serves it once, a performance change that can't be measured off a GPU. The cache, and its `unsafe` block, stay until the `OnceCell` PR (H3), which changes only the internals of `to_host_f32`.

**Hoisting:** the conversion was compiler-driven, and its first pass put `to_host_f32()` inside element loops. Those 28 sites now take one host view before the loop. In GPU-capable functions the view is taken after the GPU dispatch (matmul backward, for example), so a successful GPU path never syncs to the host.

**Writes keep their failure mode:** every former `DerefMut`/`iter_mut` write (optimizers, BatchNorm running stats, `check_gradients`, Muon) already panicked on GPU storage. It now panics at a named `expect` instead of inside the trait impl. Muon's names H8.

**Public API:** this breaks `Storage` users that indexed, iterated, or coerced it (pre-1.0). In-repo, two integration tests changed (`coverage_fill`, `test_error_paths`).

### GPU storage: shared write-once host copy, copy-on-write writes (ADDENDUM §7, H3)

**Decision:** `Storage::Gpu(Arc<GpuStorage>)`, where `GpuStorage { buffer: GpuBuffer, host: OnceLock<Vec<f32>> }`. `clone` is a refcount bump and shares the buffer *and* its host copy. The one GPU write path is `Storage::gpu_buffer_mut(&mut self)`: if the `Arc` is shared it copies the buffer on the device (`GpuBuffer::copy_region`) into a fresh `GpuStorage`, then drops the now-unique host copy. This is the CPU `Arc::make_mut` rule applied to a device buffer. `gpu_optimizer_step` takes `&mut Storage` for the parameter and both state buffers and writes only through it. It was the only kernel that writes a buffer it was handed: every other `GpuKernels` entry point returns a new buffer (checked by listing the `Option<()>`-returning kernels, and gradient accumulation goes through `gpu_add`).

**What was wrong:** clones shared one `Arc<GpuBuffer>` but each had its own `RefCell<Option<Vec<u8>>>` host cache, handed out as `&[f32]` through `unsafe { *cache.as_ptr() }` under a "never cleared" SAFETY comment that `invalidate_cpu_cache` and both optimizers contradicted. The optimizer wrote through `&Storage` into the shared buffer and cleared only the parameter's own cache. A `detach()` snapshot that had not been read yet therefore showed the stepped values (`tests/gpu_storage.rs::detached_snapshot_keeps_its_value_across_an_optimizer_step`, red on the old code). One that had been read kept the old values only because its private cache was stale, not because anything protected it. The cache was also a `Vec<u8>` cast to `&[f32]`, whose alignment is not guaranteed.

**Why one `Arc` for buffer and host copy:** a host copy shared separately from the buffer is exactly how they diverged. In one `Arc` the copy is valid whenever the buffer is unchanged, and the buffer can only change while unshared, where `gpu_buffer_mut` clears the copy. `OnceLock`, not `OnceCell`: the old `Arc<GpuBuffer>` was `Send + Sync`, and an `Arc` around a `OnceCell` would not be (clippy's `arc_with_non_send_sync` caught it). `Tensor` is `Rc`-based, so nothing needs the `Send` today, but the storage should not be the thing that rules it out.

**Removed:** the f64/f16/bf16 accessors' GPU arms. They reinterpreted f32 bytes under another dtype, and GPU storage is f32-only, so `dtype()` returns `F32` for it and the field is gone. *(Superseded by P8: GPU buffers carry a dtype (F32, F16, BF16, Bool) and the host copy is a typed `HostBuf`.)* `invalidate_cpu_cache` now drops the host copy only if the storage is unshared; a shared copy is still correct and other storages may be reading it.

**Behavior change:** a failed GPU optimizer step used to be dropped (`.is_some()` into a flag), leaving the parameter silently untrained. It is now an `expect`: a CPU fallback cannot write GPU storage.

**Tests:** `ops::gpu_ops::tests::gpu_optimizer_step_{does_not_change_an_unread_snapshot, refreshes_the_params_own_host_read, does_not_change_a_read_snapshot}` and `tests/gpu_storage.rs` (SGD with momentum and Adam against CPU over three steps with a host read after each, the `detach` snapshot, and the bytes round trip). Of these only the unread-snapshot pair and the param-refresh test fail on the old code; the rest are regression guards for the new design. All ran on a real adapter (2026-10-30).

### `RawTensor.shape` is a `Shape` (supersedes "Shape boundary")

**Decision:** the field is a `Shape`. Op, layer, and backward code reads it through typed accessors, not by demoting it:

| Need | Accessor |
|---|---|
| rank, element count | `rank()`, `numel()` |
| a user-supplied axis | `dim(Axis) -> Result` |
| a position bounded by rank (loop, prior check) | `dim_at(i)`, panics naming the caller |
| all dims of a known-rank shape | `fixed::<N>() -> [usize; N]` |
| outer and inner sizes around an axis | `numel_before(i)`, `numel_from(i)` |
| keepdim and squeeze shapes | `with_dim(i, n)`, `squeeze(i)`, `unsqueeze(i)` |
| iterate the dims | `iter()` |
| strides | `strides()` |

A backward pass hands its parent's shape on by value: `RawTensor::from_shape(data, x.shape.clone(), false)` (new, `pub(crate)`), and `new_with_storage`/`try_new_with_storage` now take a `Shape`. `GradFn` fields that store a tensor's shape (`input_shape`, `original_shape`, `true_shape`/`false_shape`) are `Shape`s. User-supplied targets (`MovementOp::Reshape/Expand { new_shape }`, `LayerNorm::normalized_shape`) stay `Vec`, because they are inputs, not a tensor's shape.

**Where `truncate_rank()` remains:** 64 calls outside `shape.rs` (was 8; `Strides::truncate_rank` not counted). Each one sits at one of three kinds of boundary:
1. Raw index-math helpers that really are slice code: `broadcast_to`, `sum_over_broadcast_dims`, `transpose_2d`, `MovementContext`, and the GPU wrappers' parameter packing. Most of the 64 are here; `movement.rs` alone has 24.
2. Public `&[usize]` APIs called internally: `broadcast_shape`, `reshape`, `ones`/`zeros`/`randn`.
3. Serialization and error payloads: `TensorData.shape`, `ShapeDataMismatch`, and `load_weight`'s comparison.

The "Shape boundary" worry was that the flip would mean a `truncate_rank()` at every read. It didn't come true: about 440 implicit raw reads became about 64 named demotions, and everything else is an accessor. A new `truncate_rank()` belongs only at one of these three boundaries.

**Public accessor:** `Tensor::shape()` returns `Shape` (it returned `Vec<usize>`). Outside the crate, a dim is read with `shape.dim(Axis(i))` (a `Result`), and `truncate_rank()` is the explicit escape to a slice, for example to destructure `let &[b, s, f] = shape.truncate_rank() else { .. }`. `dim_at`, `fixed`, and the other new accessors stay `pub(crate)` until an outside use needs them. In-repo the change touched three examples and nothing in `tests/`, because the `PartialEq` impls below keep `assert_eq!(t.shape(), vec![..])` compiling. Inside the crate, `Shape::derived(x.shape())` round trips (conv, conv-transpose, maxpool specs) collapse to `x.shape()`.

**Comparison:** `Shape` implements `PartialEq` with `[usize]`, `[usize; N]`, and `Vec<usize>` (both directions for `Vec`). Equality hands no slice to a shape-blind API, so it isn't a demotion, and `assert_eq!(t.shape, vec![2, 3])` keeps reading naturally.

**Lost test:** `test_try_new_with_storage_rejects_zero_dim` is deleted. `try_new_with_storage` takes a `Shape` now, and a zero-dim `Shape` can't be constructed; `shape::tests::from_dims_zero_dim_errs` and `derived_panics_on_zero_dim` cover that.

**Fallout:** `axis_size` (a `&[usize]` → `Shape::from_dims` → `dim` round trip) is gone; reductions call `shape.dim(axis)`.

### `Sequential` layer names can't collide (ADDENDUM §10)

**Decision:** `SequentialBuilder::build()` panics on a layer name that is used twice, is a decimal number, or contains `.`. Each one made two layers write the same `state_dict` keys, so saving silently kept one layer's weights: two `"fc"` layers gave two keys for four parameters, a layer named `"1"` collided with the unnamed layer at index 1, and `"a.b"` collided with layer `"b"` nested in `"a"`. A clash is a programmer error, so it panics in `build()`, where it is local to the mistake and the signature stays the same; `build() -> Result` would break every builder caller and pre-empt the H4 error-regime decision. `build()` is the only constructor that takes names (`Sequential::new` takes none).

**Not a bug:** the audit's greedy prefix (`"10.weight"` matching `"1."`) does not reproduce. The prefix includes the dot. `test_state_dict_round_trip_past_ten_layers` pins it.

**Departure from §10:** no `insta`. The defects are key collisions, which `assert_eq!` on a short key list catches just as well (`test_state_dict_keys`), without a dev-dependency and the `cargo-insta` review step. Revisit if snapshot coverage grows to many layers' key sets.

### `DevicePolicy`: the device label is a promise (ADDENDUM §2.1)

**Decision:** an op's output is on its inputs' device, and nothing moves a tensor silently. Three cases break that, and each is handled in one place (`src/device_policy.rs`):

- **Mixed devices** (inputs on different devices) panic under every policy. `edge()` checks it for every op; `Where` checks its condition, which is not a graph input.
- **A gap** (a GPU input reaching an op, or a case of an op, with no GPU kernel) goes through the policy. `DevicePolicy::Strict`, the default, panics with the op's name and the opt-in. `DevicePolicy::PreferGpu` runs the op on the CPU, moves the result to the GPU, and records a `FallbackEvent`. `edge()` is the forward check (a CPU output for GPU inputs is a gap) and `Tensor::accumulate_grad` the backward one (a CPU gradient for a GPU tensor is a gap), so an op that is never edited is still covered.
- **A failure** (a kernel that covers the case returns `None`) panics under every policy via `kernel_failed`, at the op site. Only the op knows whether a kernel covers the case: `UnaryOp::has_gpu_kernel()` (from `define_ops!`), differing binary shapes, batched and vector matmul, movement ops above rank 4.

**Why `Strict` by default:** it is PyTorch's behavior (mixed devices raise; a missing kernel raises `NotImplementedError`; the only fallback, `PYTORCH_ENABLE_MPS_FALLBACK`, is opt-in), and invariant 5: a silent fallback is how a "GPU test" passed with nothing on the GPU. No one depends on the old behavior, so GPU models that hit gaps (`Linear`'s bias add, attention's batched matmuls) now opt in, and the panics are the list of missing kernels (#67).

**Why `PreferGpu` never covers failures:** a kernel that exists and returns nothing is a bug or a broken device. Retrying it on the CPU is the silent fallback this replaces. PyTorch does not retry a CUDA OOM on the CPU either.

**Device equality compares names.** `GPU("a") != GPU("b")`, so the check is already correct for more than one GPU. To make that safe with one context, `to_device(GPU(name))` panics unless `name` is the adapter's (`Device::gpu()`), as PyTorch rejects `cuda:7` on a one-GPU machine. Test helpers stopped inventing names (`"TestDevice"`, `"ParityTest"`).

**Device labels never lie:** `to_device`, `Storage::gpu`, and `Storage::new_zeros` panic without a GPU instead of returning CPU storage under a GPU label, and `new_with_storage` asserts the two agree. (`to_device` moved the gradient too until it became differentiable; see "`to_device` is a differentiable op".)

**Departures from the §2.1 sketch:**
- No `CpuOnly`: keeping tensors on the CPU already gives it.
- A scoped guard, `with_device_policy(p)`, instead of a bare `set_device_policy()`: it restores the previous policy on drop and on unwind, so scopes nest (the `no_grad()` shape).
- Fallbacks are recorded in a thread-local log drained by `take_fallback_events()`, not returned from each op: ops return `Tensor`. `PreferGpu` also logs a `warn` the first time each op falls back on a thread, not on every call.
- No CPU-scalar exception. PyTorch lets a 0-dim CPU tensor meet a CUDA tensor; Volta's scalars are shape `[1]`, and the exception would reopen the mixed-device hole. Library code builds constants on its input's device (`Tensor::constant_like`).
- Errors are panics, not `Result`. Ops return `Tensor`; the H4 error-policy decision revisits this.

**Also fixed on the way:** the GPU movement kernels pack shapes into `[u32; 4]`, so a rank-5 permute returned zeros (H5's truncation). Movement ops above rank 4, or with a pad width over 16 bits on dims 2–3, are now gaps. GPU binary backward with differing shapes is a gap too: its kernels are broken (#60), and the forward check made them reachable.

**Tests:** `tests/device_policy.rs` (forward and backward gaps under both policies, mixed devices, unknown GPU names, layers and losses keeping constants on the GPU, `no_grad`, event order, rank-5 movement), `device_policy::tests`, and `expect_gaps` in the parity and smoke tests, which asserts the exact set of ops each GPU test runs on the CPU. All ran on a real adapter (2026-10-30).

### Linear weights are transposed by name, not by key pattern

**Decision:** `load_pytorch_linear_weights(state, linear_layers)` and `load_huggingface_weights(state, prefix, linear_layers)` take the `Linear` layer names and transpose exactly `{name}.weight`, through `StateDictMapper::transpose_linear`, which panics if the key is missing or not 2-D. They used to call `transpose_pattern("weight")`, which also transposed `Embedding.weight` and PyTorch's LSTM `weight_ih_l0`/`weight_hh_l0`, whose layouts already match Volta's. On a non-square matrix that failed loudly at `load_weight`; on a square one it loaded silently wrong.

**Why names, not a better pattern:** PyTorch names a `Linear` weight and an `Embedding` weight the same way (`*.weight`, both 2-D), so no key pattern can separate them; skipping names containing `embed` would still be a guess. Shapes can't decide it either, for exactly the square case. The caller knows which layers are `Linear`.

**Breaking change:** both helpers gained a parameter. Nothing in the repo called them outside their own tests; `examples/load_external_mnist.rs` already used explicit `transpose(key)` calls. `transpose` and `transpose_pattern` keep their lenient behavior (missing or non-2-D keys are skipped) for general use, and their docs point to `transpose_linear` for PyTorch `Linear` weights.

### `to_device` is a differentiable op (PyTorch's `.to()`)

**Decision:** `to_device` builds its output through `edge()` with `GradFn::ToDevice { source }`, whose backward moves the gradient to the source device. The result is a new node whose parent is the input: it has no gradient of its own, and on a leaf it is not a leaf, so a CPU leaf moved to the GPU and used in a GPU loss gets its gradient back, on the CPU. A same-device move returns the tensor itself.

**Before:** `to_device` copied the tensor with its `grad`, `requires_grad`, `grad_fn`, and `parents`. On a leaf that made a new, independent leaf, so the original never got a gradient. On a non-leaf it made a second node sharing the first's parents, and the backward loop special-cased "a node on another device than its parents" to move its gradient back. That special case is gone: the transfer is an ordinary `GradFn`.

**`edge()` exemption:** a transfer is the one op whose output is meant to be on another device than its input, so `edge()` skips its input-to-output device check for `GradFn::ToDevice` (the check would call GPU-to-CPU a gap and panic on CPU-to-GPU). The mixed-input check still runs, and nothing else may use the exemption.

**Making a leaf on another device:** move a tensor that doesn't require grad, then mark it: `RawTensor::new(data, shape, false).to_device(d)` and `set_requires_grad(true)` (PyTorch's `x.to(d).requires_grad_()`). `set_requires_grad(true)` panics on a non-leaf, so a half-migrated call site fails loudly. `Module::to_device` still moves parameters in place, so a model moved to the GPU keeps leaf parameters. `Linear::new_on_device` and `Conv2d::new_on_device` used the old idiom (mark, then move), which would have made every GPU parameter a copy of a hidden CPU leaf; they move first now.

**Migration:** about 60 test sites built GPU leaves as `RawTensor::new(.., true).to_device(gpu)` or copied a `requires_grad` CPU input to the GPU for a parity check. The parity copies now go through `tests/common::gpu_leaf` (`t.detach().to_device(gpu)` plus `set_requires_grad`); left alone, the GPU loss would have flowed back into the CPU input and doubled its gradient, which is how they failed. One test overwrote each parameter in place with its own `to_device` copy, which made each parameter its own parent; it uses `Module::to_device` now.

**Tests:** `tests/device_policy.rs`: `to_device_routes_grad_to_source` (both directions, under `Strict`, no gaps), `to_device_copy_is_a_graph_node` (no inherited gradient, not a leaf, untracked under `no_grad`, same-device identity), `backward_through_to_device_is_not_a_gap`, and `new_on_device_parameters_are_gpu_leaves` (fails if `Linear::new_on_device` marks before moving). All ran on an Apple M2.

### `PositionalEncoding` is a `Layer`, and its rows are a `shrink` (B3)

**Decision:** `PositionalEncoding` implements `Module` and `Layer`, like every other tensor-to-tensor layer, and `forward` adds `table.shrink(rows 0..seq_len) → reshape [1, S, D] → expand [B, S, D]` to `x` with a same-shape `add`. `table` is the fixed sinusoid buffer or, in learned mode, the `Embedding`'s weight. Before, both modes copied `x` and the encoding to the host and built the sum with `RawTensor::new`, so no gradient reached `x` or the learned weight, and a GPU input came back on the CPU.

**Why a `shrink`, not `Embedding::forward_indices(0..seq_len)`:** the positions are always `0..seq_len`, so the lookup is a contiguous row slice. `shrink` has a GPU kernel and `Embedding` has none, and its backward writes only the used rows.

**Why expand, not a broadcast `add`:** GPU binary ops with differing shapes are a device-policy gap (no kernel), and `expand` plus a same-shape add are not. A GPU `PositionalEncoding` runs forward and backward with no gaps, so it works under `Strict` (`tests/device_policy.rs::gpu_positional_encoding_stays_on_device` asserts an empty `expect_gaps`).

**API change:** the inherent `forward`, `parameters`, `state_dict`, and `load_state_dict` are gone; they are the trait methods now, so callers import `Layer`/`Module`. `to_device` is overridden: the sinusoid table is not a parameter, so the default would leave it behind and the next forward would hit the mixed-device panic. `forward` now asserts the input's last dim equals `embed_dim`; the old code shrank the table to the input's width, so a narrower input was silently given a truncated encoding.

**Tests:** the output against the sin/cos formula and against the learned weight rows; the exact input gradient (`w` for the loss `sum(out ⊙ w)`, both modes); the exact learned-weight gradient (`Σ_b w[b, p, :]` for `p < seq_len`, zero past it); `assert_trainable!` in learned mode; and GPU parity of the output and both gradients on an Apple M2. The gradient tests fail when the output is replaced by a detached copy of the same values.

### Pedantic cast lints: three denied, two allowed (2026-11-01)

**Decision:** `cast_possible_truncation`, `cast_possible_wrap`, and `cast_sign_loss` are `deny`; `cast_precision_loss` and `needless_pass_by_value` are `allow` with the reason in `Cargo.toml`; `redundant_closure_for_method_calls` is `deny`.

**Why the first three:** they are the silent-wrong-number class that `fetch`/`slot` closed for indexing, one step earlier: an `as` that wraps a window coordinate or truncates a kernel parameter produces a plausible index or dimension instead of a failure. The iGEMM batch offset bug lived in exactly that arithmetic. `Shape` caps a tensor at 1e8 elements, below `u32::MAX`, so none of the converted sites could fire, and the checked helpers (`signed`, `unsigned`, `pack_u32`) change no behavior today; they make the cap load-bearing in one place instead of assumed in 250. Where narrowing is the point (f64 to f32, float to an integer dtype), the function says so with `#[expect(.., reason)]`, which fails to compile if the lint stops firing.

**Why the two allowed:** `usize as f32` for counts, means, and positions rounds above 2^24 (a relative error around 6e-8), which cannot change a result beyond f32's own rounding, and converting 106 sites buys no safety. `needless_pass_by_value` flags the `Storage` constructors and `set_grad`, which take ownership because the caller has just built the buffer; taking a slice would make every caller clone.

**Not done:** the lint stays an allow, not a convention, so an `as f32` that ever feeds an integer-exact comparison is not caught. Revisit if one appears.

### compile_fail requires `rust-src` (2026-11-06)

**Decision:** the pinned `tests/compile_fail/*.stderr` are blessed with the `rust-src` component installed, CI installs it, and `tests/compile_fail.rs` panics with an install hint when it is missing.

**Why:** rustc quotes libcore (`pub struct PhantomData<T: PointeeSized>;`) under the `$RUST/core/src/marker.rs` note in `guards_not_send` only when `rust-src` is present. The files were first blessed without it, so `just ci` failed on every machine that had it.

**Rejected:** changing the guards' non-`Send` marker so the diagnostic never reaches libcore. That edits library code to suit a test, and the output that avoids libcore can only be found by trial.

## Dtypes (ADDENDUM §11)

### Mixed precision: four decisions (2026-11-02)

**Context:** `DType` was decorative (`notes/TODO.md`, retired in #87; last version at `9d8aff5`): every `Tensor` is f32 and the tag only labels bytes in `Storage` and safetensors. The two honest options were to shrink it to an interchange tag or make it real. The maintainer chose full mixed precision (ADDENDUM §11), against the recommendation to shrink it first and treat mixed precision as backlog.

1. **Half types compute in f32 and round the output** (PyTorch CPU semantics), rather than rounding after every element operation. It reuses the f32 kernels, and torch fixtures can check it. Per-step rounding would show precision loss more vividly but disagree with torch, leaving nothing to check it against.
2. **Promotion is implicit and follows `torch.result_type`,** implemented by inserting differentiable `Cast` nodes so kernels only ever see one dtype. Rejected: strict `DTypeMismatch` on every mixed op. It is more explicit, but less like the PyTorch API the project teaches. Until P6 lands, mixed operands are a `DTypeMismatch`.
3. **The `improvements` → `main` gate waits for §11** (the maintainer's call; the recommendation was to let P0 clear the "decorative" item and merge meanwhile).
4. **GPU f16 is in scope** (P8, wgpu `SHADER_F16`). An unsupported dtype or adapter is an error at `to_device`, never the old silent upcast to f32. The cloud can write P8 but not verify it.

### Typed host buffer; kernels refuse other dtypes (P1, supersedes the byte layout in "CPU storage")

**Decision:** `Storage::Cpu` holds a `HostBuf`, one `Arc<[T]>` variant per dtype, instead of `Arc<[u8]>` plus a tag. A sealed `Element` trait (`f16`, `bf16`, `f32`, `f64`, `i32`, `i64`, `u8`, `bool`) gives `Storage::from_typed::<T>`, `as_host::<T>`, and `as_host_mut::<T>` (copy-on-write, as before), replacing `cpu_f64`/`cpu_f16`/`cpu_bf16`/`cpu_f32` and the `as_*_slice` accessors. Bytes are decoded by copying (`try_from_bytes(&[u8], DType)`), so they need no alignment, and a `Bool` byte other than 0 or 1 is an error.

**Why:** the old byte buffer relied on the allocator for alignment, which "CPU storage" called out as unguaranteed; f64 makes it load-bearing. A typed buffer makes the element type and alignment part of the type.

**Kernels refuse other dtypes:** `to_host_f32`, the read every kernel uses, panics with `DTypeMismatch` on a non-f32 storage instead of converting it, and now returns `&[f32]` (it was a `Cow` only to hold conversions). `Storage::to_vec` is its owned copy, also f32-only; `to_f32_vec` is the deliberate, lossy decode of any dtype, used by `Tensor::data()` and the safetensors loader. This turns the old silent path (an f64 operand read as f32) into a panic. On top of it, every op entry calls `tensor::expect_f32` before any work (the five dispatchers, movement ops, `sum_dim`/`max_dim`, `contiguous`, `Conv2d`, `ConvTranspose2d`, `MaxPool2d`, and the optimizers' `step`), so the panic names the op and the offending input (`Binary(Add): DType mismatch: expected F32, got F64 (input 1)`) instead of a kernel line. Composite ops (`softmax`, losses, `BatchNorm`) inherit it from their primitives. The kernel-read panic stays as the backstop. Movement ops are refused too, though they only move data: their kernels read through the f32 path until P3/P5. P3/P4 replace these panics with real kernels, and P6 adds promotion.

**Tensors:** `Tensor::dtype()`; `RawTensor::from_typed`/`try_from_typed` take any `Element`; `from_vec` stays f32-only, because an unsuffixed float literal is `f64` and a generic `from_vec(vec![1.0])` built an f64 tensor unasked (one existing test caught it). Integer and bool tensors cannot require gradients (`set_requires_grad(true)` panics with `NotDifferentiable`, a new `VoltaError` variant, as PyTorch raises). `set_grad` panics on a gradient of another dtype. `load_weight` returns `DTypeMismatch` for a non-f32 destination, because `TensorData` is f32. `Storage::to_device` panics instead of upcasting a non-f32 storage to the GPU.

**Removed:** `Storage::to_dtype` (no callers; P2 brings a differentiable cast with checked semantics) and `try_as_bytes` (`as_bytes` cannot fail). `TypedTensorData::to_storage`/`to_tensor_data` now return `Result`, since their fields are public bytes.

### `to_dtype` is a differentiable cast that matches torch bit for bit (P2)

**Decision:** `Tensor::to_dtype` builds a `GradFn::Cast` through `edge()`; backward casts the gradient back to the input's dtype, as PyTorch's `.to(dtype)` does, so rounding to f16 is the identity in backward. The values come from `Storage::cast`, which `tests/torch_ref_casts.rs` checks exactly against `x.to(dst)` for all 64 dtype pairs (fixture group `casts`, torch 2.15.0). Same dtype returns the tensor itself, as PyTorch does.

**Semantics, as measured on torch:** float to float rounds to nearest even, but f64 (and every integer) to f16/bf16 goes through f32, so two roundings: `1 + 2^-11 + 2^-30` becomes f16 `1.0`, not `1.0009765625`. Integer narrowing wraps. Anything to bool is `x != 0` (NaN is true, `-0.0` false).

**Where Volta differs:** float to integer for NaN, an infinity, or a value whose truncation is out of range panics, naming the element. Torch leaves these undefined (on x86 they return whatever the conversion instruction gives), and a saturated or wrapped value would be a plausible wrong number (invariant 5). The fixture leaves them out.

**Gradients and dtypes:** `edge()` never makes a non-float output require grad, whatever its inputs (PyTorch's rule), so a cast to an integer or bool is a dead end for autograd, like comparisons will be (P5). `accumulate_grad` checks that a gradient has its tensor's dtype, and adds in that dtype (half types add in f32 and round, decision 1 above). `Tensor::grad()` decodes to f32 like `data()`; `grad_storage()` keeps the dtype.

**GPU:** `to_dtype` on a GPU tensor to anything but F32 panics. That is not a gap with a CPU fallback: GPU storage cannot hold the result until P8. *(Since P8, GPU casts between F32, F16, BF16 and Bool run on the device; F64 and integer targets still panic.)*

**Tests:** besides the fixture, three mutants were run against it and all fail it (rounding instead of truncating float to int, saturating instead of wrapping, NaN to `false`). `tests/dtype_cast_grad.rs` covers the f64 round trip (exact values, unit gradient, gradcheck), the f16 round trip, gradient accumulation at a shared f64 node (fails if the f64 add drops a contribution), and integer and bool casts carrying no gradient.

### Core ops in f64: `Real`, two instances per `define_ops!` row (P3a)

**Decision:** CPU kernels are generic over a crate-private `Real` trait (f32, f64) and dispatch on the input dtype with `dispatch_real!`. `define_ops!` emits each row's body twice (`forward`/`backward` for f32, `forward_f64`/`backward_f64` for f64). A body names no float type: it uses literals and `fl::LN_2`, `fl::PI`, `fl::erf`, which a per-type `use crate::real::fl32/fl64 as fl` resolves. So the table stays one row per op, with no runtime trait object. P3 is split: P3a converts the core ops (elementwise, ternary, whole and axis reductions, movement, matmul and transpose, `contiguous`); P3b converts layers and optimizers, which keep the f32-only `expect_f32` check until then.

**Dtype check:** `real::compute_dtype` replaces `expect_f32` on the core ops. All inputs must share a dtype that is F32 or F64, and the panic names the op: a mixed pair is a `DTypeMismatch` (promotion is P6), half types say "not implemented until P4", and integer/bool say "no i64 kernel". *(Since P4 half types compute, and since P6 elementwise ops promote mixed operands; `compute_dtype` now guards ops that need one dtype.)* `Where`'s condition is checked separately and may be either float dtype; it is now a `Vec<bool>` mask in the grad fn, not an f32 copy.

**Matmul:** f64 goes through `matrixmultiply::dgemm` on every platform. An Accelerate `cblas_dgemm` path would be faster on macOS but unverifiable from the cloud, so f32 keeps Accelerate and f64 does not.

**Seeds and constants:** the backward seed is cast to the loss's dtype. `constant_like` builds its constant in the receiver's dtype (values given as f32 and widened exactly), which `mean_dim` needs.

**Evidence:** `src/lib_tests/f64_tests.rs` gradchecks every core op family in f64 at a relative tolerance of 1e-7 (the f32 checks use 1e-2): all unary ops, all differentiable binary ops at equal and broadcast shapes for both operands, `Sum`/`Mean`/`Max`, `sum_dim`/`max_dim`/`mean_dim` on every axis with and without `keepdim`, softmax and log-softmax, `mulacc` for each input, `where` for both branches with a broadcast condition, every movement op, every matmul form for both operands, and transpose. Value tests pin f64 rounding (ten 0.1s, 1 + 1e-12, `exp(1e-12)`, libm's f64 `erf`) and f64 matmul against a reference loop. A mutant computing f64 `erf` through f32 fails both a value test and the gradcheck. The f32 tests pass unchanged.

**Compile time (measured 2026-11-03):** two instances per kernel cost about 12% more wall time to compile the crate on the CPU (debug 1.88 to 2.11 s, release 4.95 to 5.50 s; user CPU +15-16%) and 5% with default features (2.83 to 2.98 s, where the GPU code dilutes it). `libvolta.rlib` grows 5% (debug) and 10% (release). Method: medians of three alternated runs per tree, `volta` only (dependencies prebuilt), `CARGO_INCREMENTAL=0`, on a shared 4-core Linux VM, against #21's tip. The instantiations are crate-private (`Real` is `pub(crate)`), so the cost stays in this crate and does not grow per downstream crate. P3b and P4 add the layers and the half types (which reuse the `f32` instance), so re-measure there; a third float instance would be the point to reconsider.

**Not done here:** torch fixtures in f64 (P3b): checking them from `tests/` needs a public way to read typed values back, and `data()` decodes to f32. The f64 gradcheck is test-only for the same reason.

### Layers and optimizers in f64; `Module::to_dtype` (P3b)

**Decision:** every layer and optimizer kernel dispatches on dtype like the core ops (P3a), and `Module::to_dtype` converts a model in place (PyTorch's `module.to(dtype)`). The default converts `parameters()`; `BatchNorm1d/2d` also convert their running statistics, `PositionalEncoding` its sinusoid table, and `Sequential` forwards to each child so buffers are reached. The transformer containers hold only parameter-only children, so the default covers them. `Tensor::to_vec_as::<T>()` is the public lossless readback (approved on #22); `data()` still decodes to f32.

**Choices made here:**
- **Loading casts.** `load_state_dict` casts the f32 `TensorData` to each parameter's dtype, exactly for f64, as PyTorch's `load_state_dict` (`copy_`) does. This replaces P1's refusal; an integer destination is still an error.
- **Saving stays f32-only.** `state_dict()` on a non-f32 model panics in `TensorData::from_tensor`: `TensorData` is f32, and narrowing silently would lose the point of f64. A typed `TensorData` changes the bincode format, so it is a TODO, not part of P3b.
- **Optimizers step each parameter in its own dtype.** A mixed f32/f64 list is fine (as in PyTorch); a parameter with no kernel (f16, until P4 added half kernels) was refused before any parameter was updated. Optimizer state (Adam's moments, SGD's and Muon's velocity) is allocated on a parameter's first step, in its current dtype and device (S17), so a model converted with `to_dtype` or `to_device` after its optimizer was built still steps. A conversion after the first step panics ("convert the model before the optimizer's first step"): reallocating would silently reset the momentum. Adam counts steps per parameter, incremented only when that parameter has a gradient, as torch does.
- **Hyperparameters stay f32.** `lr`, betas, eps, momentum, and weight decay are widened to the parameter's dtype, so `lr = 0.01` is `0.009999999776` in an f64 step. Training does not care; the torch f64 fixture uses powers of two so both sides agree exactly.
- **Helper tensors follow the input.** Dropout's mask, the causal masks, and LSTM's zero state come from `constant_like` (input's device and dtype). BatchNorm's eval path reshapes the running statistics instead of `constant_like(.., data())`, which would have rounded f64 statistics through f32.
- `check_gradients` perturbs in the input's dtype and computes errors in f64, so it checks f64 tensors through its existing signature. `expect_f32` is gone: nothing is f32-only on the CPU any more.

**Compile time after P3b (measured 2026-11-03):** medians of three rotated runs per tree, `volta` only, `CARGO_INCREMENTAL=0`, against #21's tip (P2), P3a, and P3b:

| Build | P2 | P3a | P3b |
| --- | --- | --- | --- |
| debug, CPU | 2.04 s | 2.26 s (+11%) | 2.40 s (+18%) |
| release, CPU | 5.25 s | 5.77 s (+10%) | 6.89 s (+31%; user CPU +37%) |
| debug, default features | 3.24 s | 3.42 s (+5%) | 3.47 s (+7%) |
| lib unit-test binary, CPU | 4.94 s | 5.31 s (+8%) | 5.82 s (+18%) |

`libvolta.rlib` (release) grows 3.64 to 4.00 to 4.21 MB. The release jump is LLVM optimizing a second copy of the large kernels (the three conv algorithms, conv-transpose, pooling): each `dispatch_real!` duplicates a whole loop nest. Accepted: it is about 1.6 s per release rebuild of this crate, paid once by a dependent, and P4's half types reuse the f32 instance. If it grows, the cheapest lever is a default-on `f64` cargo feature that compiles the f64 arms only when enabled.

**Evidence:** `src/lib_tests/f64_tests.rs` gradchecks every `Layer` (Linear, Conv2d with each of its three algorithms, ConvTranspose2d, MaxPool2d, BatchNorm1d/2d, LayerNorm, GELU, PixelShuffle, FeedForward, MultiHeadAttention, TransformerBlock, both PositionalEncodings), `Embedding`, and `LSTMCell` over two steps, in f64 at 1e-7 for the input and for every parameter (perturbed in place). A mutant that scales Conv2d's direct weight gradient by 1 + 1e-6 fails it, while all 98 f32 conv tests still pass. `tests/torch_ref_f64.rs` matches PyTorch's float64 results at 1e-10 relative for exp/tanh/sigmoid/erf, both softmaxes, broadcast batched matmul, Linear, Conv2d (all three algorithms against one fixture), ConvTranspose2d, MaxPool2d, BatchNorm2d (training), LayerNorm, Embedding, and three Adam steps with weight decay. Optimizer tests pin an f64 Adam step against the formula, a 1e-12 SGD update surviving, and mixed f32/f64 parameter lists.

### Half types run the f32 kernels and round each output (P4)

**Decision:** F16 and BF16 add no kernels. `dispatch_real!` binds `T = f32` for them; `Storage::host::<f32>()` returns a `Cow` that borrows f32 and f64 storage and decodes half storage (exactly) into an owned f32 copy; and every kernel output goes through `Storage::from_compute(dtype, Vec<T>)` (or `RawTensor::from_shape_compute` for a gradient), which rounds f32 to the half dtype, nearest-even. In-place updates (optimizer steps, BatchNorm's running statistics, gradcheck's perturbation) go through `Storage::host_mut::<T>()`, a guard that is the storage itself for `T`'s dtype and a decoded f32 copy, rounded back on drop, for a half one. This is PyTorch's CPU behavior: an op loads half values to f32, computes, and rounds its output, and reductions accumulate in f32.

**Guards against a missed site:** `edge()` asserts that an op's output has its inputs' dtype (except `Cast`), and `Tensor::accumulate_grad` already asserts that a gradient has its tensor's dtype. A kernel that forgets to round fails at the op, naming it, instead of leaking an f32 tensor into a half model. During P4 the gradient check caught the backward constructors the first conversion pass missed (matmul's gradients and the shrink/expand backward), on the first half fixture run.

**Choices made here:**
- **Optimizers round once per step.** A half parameter steps in f32 and its parameter and state are rounded once; PyTorch rounds after each in-place op of its update. The fixtures show no difference beyond 1 eps over three steps. State stays in the parameter's dtype; f32 master weights are P7.
- **Halves save storage, not time or peak memory.** Each op decodes its inputs to f32 and builds its output in f32 before rounding, so a half op allocates about three times its tensors' half bytes transiently, and movement ops decode and re-encode (exactly) rather than copy bits. Native half kernels would need half arithmetic or a per-dtype kernel copy; neither matches the semantics better, and the GPU path (P8) is where speed is expected.
- **No compile-time cost.** F16 and BF16 share the `f32` arm, so they add no monomorphized kernel copies.

**Rejected:** inserting `Cast` nodes around every op on half inputs. It computes the same values, but adds three graph nodes per op, saves f32 copies of every activation for backward (twice the memory of the half model), and touches every op entry. Also rejected: a `Real` instance for `f16`/`bf16`, which would round after every multiply-add inside a kernel (a matmul or a sum accumulating in half), not what PyTorch computes.

**Evidence:** `tests/torch_ref_half.rs` runs 12 families in f16 and in bf16 against torch 2.15 computing natively in each (fixture group `half`): all 13 unary rows, the differentiable binary rows with broadcast, `mulacc`/`where`, full and per-axis reductions, both softmaxes, movement (permute, reshape, shrink, expand, transpose), broadcast batched matmul, MSE and cross-entropy, Linear, Conv2d (all three algorithms), ConvTranspose2d, MaxPool2d, BatchNorm2d, LayerNorm, Embedding, and three SGD-momentum and Adam steps. Forward outputs of elementwise, movement and pooling ops are bit-exact; every other value is within 1.83 eps of the dtype (f16 cross-entropy's gradient) relative to `max(|torch|, rms)`, and the bounds are about twice what was measured. Mutants: rounding bf16 by truncation fails 8 of the 12 bf16 families; an `host_mut` guard that drops its write-back fails the f16 SGD test. Decoding bf16 through f16 survives, but it is an equivalent mutant on this data (every bf16 value in the fixture is exact in f16).

### I64 and Bool are data: movement, comparisons, `where`, indices, targets (P5)

**Decision:** integer and bool tensors get the data ops and no arithmetic. Movement ops (reshape, permute, expand, shrink, pad, stride, transpose, `contiguous`) work on every dtype; `lt`/`le`/`gt`/`ge`/`elem_eq`/`elem_ne` take two tensors of one dtype (any dtype) and return Bool; `where` takes a Bool condition and branches of one dtype (any dtype); `Embedding` is a `Layer` over an I64 or I32 index tensor of any shape (output `[..indices, dim]`); `cross_entropy_loss` and `nll_loss` take I64/I32 class indices `[N]` as well as float class probabilities `[N, C]`, choosing by the targets' dtype as PyTorch does. Integer arithmetic (`add`, `sum`, ...) still panics, naming the op. Decided with the maintainer, with the next three.

**Choices made here:**
- **Comparisons and `where` panic on the GPU until P8** (lifted in P8: Bool GPU buffers and kernels). GPU buffers are f32-only, so a Bool result or condition can't live there. The `cmplt` WGSL entry and pipeline stay, unused, for P8. Nothing in the library or the examples compared or selected on the GPU.
- **`where` takes Bool only** (PyTorch). A float condition panics, naming `lt`/`elem_eq` or `to_dtype(DType::Bool)`. `Cmplt` (f32 `0`/`1`) is gone from the binary table; `elem_eq`/`elem_ne` are named like `elem_mul` so they can't be mistaken for `PartialEq`.
- **`IndexedLayer` and `forward_indices` are removed.** An `Embedding` goes in a `Sequential` and trains there (`tests/dtype_tensors.rs`); the two compile-fail tests that pinned the old split are deleted. The examples build an I64 tensor from their token indices.
- **Movement copies bits.** A movement kernel never reads a value, so it runs on a `real::Bits` carrier of the element's width (`u8`, `u16`, `u32`, `u64`): all eight dtypes share four instances, an i64 beyond 2^53 or a bool pattern moves exactly, and f16/bf16 movement became a bit copy instead of P4's exact decode and re-encode. Pad's fill is converted to the dtype like `to_dtype`; `-inf` (max pooling's border) on an integer tensor panics instead of wrapping. Movement backward stays on `Real`, since only floats have gradients.
- **Class-index targets become a one-hot constant.** Multiplying `log_probs` by an exact 1 and summing exact zeros is PyTorch's gather, bit for bit, in every dtype, and reuses the existing loss and its gradient. An index outside `0..C` panics (there is no `ignore_index`).
- **A comparison's grad fn is never attached.** `edge()` takes a `GradFn::Compare` to name the op; a Bool output never gets one. `edge()`'s dtype assertion exempts comparisons and casts.

**Evidence:** `tests/torch_ref_intbool.rs` (fixture group `intbool`, torch 2.15): all six comparisons on f32 with ties and a NaN and on i64 beyond 2^40, broadcast; `where` with a broadcast bool condition (float output and both branch gradients; i64 branches exactly); cross-entropy and NLL with i64 targets (loss and gradient); `Embedding` with 2-D i64 indices and repeats (output and weight gradient); permute, shrink, expand, transpose and pad on i64 (beyond 2^53) and bool, exactly. Mutants: `Le` as `Lt`, an inverted `where` mask, and a mirrored class index each fail their test. `tests/dtype_tensors.rs` trains `Sequential[Embedding, Linear]` with i64 cross-entropy to an exact fit, and checks the refusals (float `where` condition, float embedding indices, negative or out-of-range index, out-of-range target, `-inf` pad on i32, mixed-dtype comparison).

### Implicit promotion with a `DTypePolicy` (P6)

**Decision:** elementwise ops promote mixed operands as PyTorch does, by inserting casts. `binary_op` (add, sub, mul, div, max, mod), the comparisons, `mulacc`, and `where`'s branches call `real::promote_operands`, which computes `DType::promote` of all operands (torch's `result_type` for dimensioned tensors, checked across all 64 pairs since P0) and casts each operand that differs with `to_dtype`. The cast is a graph node (`GradFn::Cast`), so each gradient comes back in its own operand's dtype and kernels stay single-dtype. `div` is true division: an integer or bool result dtype becomes f32 (`i64 / i64` is f32, as in torch). Matmul, the convolutions and the layers still need one dtype, as in torch (which errors there too); their panic now says to cast with `to_dtype` or `Module::to_dtype`. Losses built from elementwise ops (`mse_loss`, probability-target `cross_entropy_loss`) promote through them, which matches torch's result dtype. Decided with the maintainer, with the next three.

**Choices made here:**
- **`DTypePolicy`, `Promote` by default.** Promotion is the one implicit conversion in the library, against the fail-loudly rule (ADDENDUM §8), so `with_dtype_policy(DTypePolicy::Strict)` turns it off for a scope: mixed operands panic, naming the op and "under DTypePolicy::Strict". It is a thread-local with a `!Send` restoring guard, the shape of `DevicePolicy` and `no_grad`. Strict refuses only mixed operands; `i64 / i64` in f32 is `div`'s definition, not a mix.
- **No scalars, so `[1]` tensors promote fully** (decision (a)). Volta has no zero-dim tensors or scalar ops, so `f16[n] * f32[1]` is f32, exactly torch's rule for a `[1]` tensor. Torch keeps `f16 * 2.0` in f16 through its wrapped-number rule; scalar ops with that rule are a TODO. The library's constants come from `constant_like` in the input's dtype, so only user code can widen a half model this way.
- **Integer arithmetic stays out** (decision on P5, kept): `i64 + i64` promotes to i64 and panics "no i64 kernel", while `i64 / i64` works. The asymmetry is noted in TODO until integer kernels land.
- `where`'s condition stays Bool-only; only its branches promote.

**Evidence:** `tests/torch_ref_promotion.rs` (fixture group `promotion`, torch 2.15) runs add, mul, div, lt and where on all 64 ordered dtype pairs and matches torch's result dtype and values exactly; where torch's add/mul result is an integer or bool dtype, it checks the op panics naming the missing kernel. For the 16 float pairs, the gradients of `sum(a * b * w)` have each operand's dtype and torch's values, and the product has the promoted dtype. Mutants: promoting to the first operand's dtype, and dropping true division, each fail the 64-pair test. `tests/dtype_tensors.rs` checks an f32 × f64 product's value and per-operand gradient dtypes, the Strict refusal and the guard's restore, int true division, the `i64 + i64` panic, an i64/f32 comparison in f32 (2^24 + 1 equals 2^24), and the new layer message. `guards_not_send` pins the new guard as `!Send`.

### Autocast with f32 parameters, `GradScaler`, an `Optimizer` trait (P7)

**Decision:** mixed precision follows PyTorch's autocast, not "convert the model and keep master weights". `amp::autocast(DType::BF16 | F16)` is a thread-local, nesting, `!Send` guard. Inside it, `matmul`, `Linear`, `Conv2d` and `ConvTranspose2d` cast their float inputs (not f64) to the low dtype, and the losses (`mse_loss`, `cross_entropy_loss`, `nll_loss`, `bce_loss`, `bce_with_logits_loss`, `kl_divergence_gaussian`) cast theirs to f32; every other op runs in its inputs' dtype. The casts are `to_dtype` graph nodes, so parameters stay f32, receive f32 gradients, and the optimizers are unchanged: there are no master weights to keep. `GradScaler` (torch's defaults: scale 2^16, growth 2, backoff 0.5, interval 2000) scales the loss, unscales the gradients in their dtype, skips a step whose gradients hold an inf or NaN, and adapts the scale; it drives an optimizer through the new `Optimizer` trait (`step`, `zero_grad`, `parameters`), which SGD, Adam and Muon implement by forwarding to their inherent methods. Decided with the maintainer, with the next three.

**Choices made here:**
- **PyTorch's CPU op lists, not its CUDA ones.** Measured on torch 2.15: on the CPU, softmax, log_softmax, layer/batch norm, sum, mean, exp, relu and max pooling stay in the half dtype; only the losses go to f32. The CUDA lists also force softmax, norms and reductions to f32, which suits f16 better, but there is no CUDA here to test against; P8 can revisit them for the GPU.
- **Layers are cast as units.** `Linear` is a matmul plus a bias add, so casting only the matmul would promote the add back to f32 against the f32 bias; torch's fused `linear` returns the low dtype. `Linear` and the convolutions cast input, weight and bias together (`amp::lower_layer`). Composites built on `matmul` (attention, LSTM) follow it.
- **Autocast's casts are explicit,** so `DTypePolicy::Strict` does not refuse them; only implicit promotion is affected by that policy.
- **Master weights for half-converted models are a TODO,** as is torch's per-scope weight-cast cache. *(Norm layers under mixed inputs, a TODO here, follow torch since the audit fix for A5: a half input with f32 parameters normalizes in f32 and returns the input's dtype, through explicit casts, so `DTypePolicy::Strict` accepts it. `nn::norm_mixed`; checked against torch for `LayerNorm` after `Linear` and `BatchNorm2d` after `Conv2d` in the `autocast` fixture group.)*
- **`GradScaler` checks inf/NaN after rounding** to the gradient's dtype (a finite f32 product can overflow f16), refuses a second `unscale` or an `update` before `step` (as torch does), and is meant for f16: bf16 has f32's exponent range.

**The MNIST criterion and its margin** were fixed before autocast was written, from measurements on the same MLP (784 → 256 → ReLU → 10, Adam lr 1e-3, batch 64, 3 epochs, 5 seeds, one core, release):

| Run | Test accuracy, seeds 1-5 (%) | Mean |
| --- | --- | --- |
| torch 2.15, f32 | 97.18 97.39 97.46 97.39 97.44 | 97.37 |
| torch 2.15, bf16 autocast | 97.13 97.50 97.42 97.40 97.36 | 97.36 |
| Volta, f32 (harness) | 97.40 97.46 97.47 97.48 97.23 | 97.41 |
| Volta, model stored in bf16 (no f32 parameters) | 97.06 97.13 96.99 97.11 96.82 | 97.02 |

Seeds spread about 0.25 points, so a 5-seed mean has a standard error near 0.05; torch's own gap is 0.01; storing the model in bf16 loses 0.39. The criterion, `|mean(f32) - mean(bf16 autocast)| <= 0.2` points with the same seeds, initializations and data order, passes torch's behavior and rejects the bf16-stored one by about three standard errors. `examples/mnist_autocast.rs` (`just mnist-autocast`, after `just mnist-data` downloads and checks the files) measured, on 2026-11-03: f32 97.56 97.51 97.56 97.54 97.61 (mean 97.56), bf16 autocast 97.61 97.51 97.50 97.61 97.60 (mean 97.57), gap -0.01, in about 27 s per seed pair. (The harness and the example shuffle differently, hence the different f32 means.) It is not in CI; a nightly lane is a TODO.

**Evidence:** `tests/amp.rs` checks every output dtype in the measured torch table under bf16 and f16, f32 gradients for every parameter of an autocast MLP, nesting and f64 eligibility, and the scaler: a loss scaled by 2e-8 loses every f16 weight gradient to underflow and keeps them under a 2^16 scale (within f16 precision of the f32 run), an inf or NaN gradient skips the step and halves the scale, three clean steps grow a scale with interval 3 and a skip resets the count, and misuse panics. `tests/torch_ref_autocast.rs` (fixture group `autocast`) matches torch under bf16 and f16 for an MLP's loss and the f32 gradients of its input and parameters, Conv2d (all three algorithms) and ConvTranspose2d outputs and gradients, and broadcast batched matmul, within 2 eps of the low dtype except the f16 MLP's gradients (up to 4.55 eps, 0.44% relative; Volta's `Linear` rounds the product and the bias add apart where torch's `addmm` rounds once). Mutants: leaving `Linear`'s bias uncast and casting to bf16 whatever the scope's dtype each fail `tests/amp.rs`.

### GPU f16, bf16 and Bool: native where the hardware has it, emulated where it can store it (P8)

**Decision:** GPU buffers carry a dtype: F32, F16, BF16, or Bool (one `u32` per element). Every shader is one source expanded per element type by `gpu::shader_template`: data buffers are `array<ELEM>`, read with `ld_NAME(i)`, written with `st_NAME(i, v)` and bounded with `len_NAME()`, all in f32. F16 uses native `array<f16>` where the adapter has `SHADER_F16` (requested when present); bf16 always, and f16 without the feature, is stored as two values per `u32` word, low half first, and a store is an `atomicAnd` that clears its half and an `atomicOr` that sets it. That layout is the CPU's `HostBuf` layout (and `array<f16>`'s), so uploads and downloads are byte copies, and the native and emulated paths are interchangeable on one buffer. Kernels compute in f32 and round each stored element to nearest-even in integer arithmetic, so rounding, overflow to inf and NaN don't depend on the GPU's float conversion, which WGSL leaves partly implementation-defined. F64 and integer tensors are refused at `to_device`. Decided with the maintainer: bf16 by PyTorch's rule (option A), Bool as `u32` per element (1b).

**Why emulate bf16 (PyTorch's rule).** On CUDA devices without native bf16, `c10::BFloat16` is a `uint16_t`; every op widens it to f32 (`f32_from_bits`, a shift by 16), computes, and stores with `round_to_nearest_even` (NaN as `0x7FC0`), and `torch.cuda.is_bf16_supported(including_emulation=True)` reports such devices as supported. PyTorch's MPS backend supports f32, f16 and (macOS 14+) bf16 and refuses f64 (`supportedFloatingType`). WGSL has no 16-bit integer type, so the only spec-safe 16-bit storage without `SHADER_F16` is packing into `u32` words, and a half-word store must be atomic so it can't clobber a neighbor another thread writes. Emulated f16 reuses the same path, so f16 and bf16 work on every adapter; a crate-private switch (`set_f16_emulation`) forces f16 through it so it is tested on hardware that has native f16.

**Choices made here:**
- **Pipelines per element type,** compiled on first use (`GpuContext::pipelines_for(dtype)`); cast pipelines by (input, output) pair. Kernel wrappers choose by their first input's dtype, allocate outputs in it (`GpuBuffer::zeros_of`), and assert that every bound data buffer shares it (`assert_same_dtype`): during P8 that check caught SGD/Adam/Muon allocating GPU state as f32 next to an f16 parameter.
- **Reductions accumulate in f32** (the reduce shader's result is f32) and the host rounds the scalar to the input's dtype, as the CPU does.
- **GPU casts** (`Storage::cast`, so `to_dtype` and autocast on GPU tensors) run on the device between F32, F16, BF16 and Bool (`cast.wgsl`; float to Bool is "nonzero", NaN included, from the bits).
- **Bool on the GPU** lifts P5's panic: comparisons (`compare.wgsl`, NaN detected from the bits, since WGSL lets an implementation assume NaN never reaches a float comparison) and `where` (`where_select.wgsl`) run on equal-shape GPU operands; broadcasting is a device-policy gap, like the binary ops.
- **Muon on half parameters** casts parameter, gradient and velocity to f32 on the device, runs the f32 kernels and casts back (one rounding per step, as on the CPU), rather than rounding Newton-Schulz scratch to 16 bits. Its kernels assert f32 buffers.
- **`GradScaler::unscale` on GPU gradients** runs a fused kernel (`amp_unscale.wgsl`, PyTorch's `_amp_foreach_non_finite_check_and_unscale_`): `g * inv` stored in `g`'s dtype and an atomic flag for any non-finite rounded result, so one `u32` comes back instead of the gradients.
- **Validation without a device:** naga is a dev-dependency (the same 27.x wgpu uses) and `shader_template`'s test validates all 25 shaders under all 16 element-type pairs and checks that no shader indexes a data buffer directly. `just validate-shaders` runs it.

**Evidence:** `src/lib_tests/gpu_half_tests.rs` runs, for native f16, emulated f16 and bf16, every unary op with a GPU kernel, the binary ops (bit-exact except division: WGSL allows f32 division 2.5 ulp), matmul, the full reductions, movement (bit-exact forward), all three Conv2d algorithms, casts, the six comparisons and `where` (exact), three SGD-momentum/Adam/Muon steps, an autocast MLP's loss and f32 gradients, and `GradScaler` (an f16 underflow rescued on the device, an inf skip on a GPU parameter), each against the CPU in the same dtype, within 0 to 4 epsilons. On lavapipe (Mesa 25.2, LLVM 20) all three modes ran and every test passes, as does the whole default-feature suite (the f32 expansion of every shader behaves as before). Mutants: dropping the packed store's `atomicAnd` fails the optimizer test (only in-place updates overwrite non-zero words), and truncating instead of rounding in `f32_to_f16_bits` fails the binary, cast and optimizer tests. Not yet run on Metal: `just test-gpu` on the Mac is the remaining check.

## Errors

### Panic inside the program, `Result` where data comes from outside

**Decision:** a function takes the panicking form when every argument is decided by the calling program's own code, and returns `Result` when an argument can come from outside the program: a file, a checkpoint, a dataset, JSON, a UI, a config. Ops and layers whose arguments are tensors the caller already built (`add`, `matmul`, `forward`) panic on a contract violation, documented under `# Panics`. A shape, axis, range, label, byte buffer, or state dict that a user can type is validated and reported as a `VoltaError`. Where both forms exist, the panicking one is the `try_` one plus a panic (`try_x(..).unwrap_or_else(|e| panic!("x: {e}"))`, `#[track_caller]`), so the check and the message live once. A `try_` form exists exactly when an argument can come from outside; it is not generated for every op.

**Why a library cannot just panic:** the first consumer outside the repo is a browser tool built on WASM. On `wasm32` a panic traps the whole module instance, so one bad shape typed into a UI would kill the page, and `catch_unwind` is not a boundary there (`panic = "abort"`). A server or a notebook has the same problem. A library should not decide to take down its host over data it was handed.

**Why not `Result` everywhere (Candle's choice):** autograd code is chains of ops, and a `?` after every `matmul`, `add`, and `relu` buries the math. `edge()`, `GradFn::backward`, and `define_ops!` are written for infallible ops, so a `Result` would run through the closed `GradFn` enum. A shape mismatch between two tensors the program built is a bug with one fix (change the code); no caller handles it. A panic also gives a backtrace at the faulting op, which a `?` chain loses. The same reasoning declined a `try_` twin of every op (tch-rs's `f_` convention): it doubles the op surface for a library this size.

**Where the line was drawn:** `Tensor::new`/`from_vec` and the shape constructors (`zeros`, `ones`, `empty`, `constant`, `rand`, `randn`, `xavier_uniform`, `he_initialization`) each have a `try_` twin, and so do `reshape` and `expand`. `Module::load_state_dict` returns `Result<()>`, `load_state_dict_checked` returns `Result<StateDictDiff>`, and `to_one_hot` returns `Result<Vec<f32>>`. The existing `try_sum_dim`, `try_pad_with`, `try_shrink`, `try_broadcast_shape`, `try_from_bytes`, and `try_as_bytes` stay: `Axis(pub usize)` is not validated at construction, and padding, ranges, and bytes can come from outside. File I/O was already `Result`.

**Validate, then allocate:** `zeros`, `ones`, and the others computed `shape.iter().product()` and allocated the `Vec` before `Shape::from_dims` ran. A shape such as `[usize::MAX, 2]` overflowed that product, and `[MAX_NUMEL + 1]` allocated first and checked later. An allocation failure aborts rather than unwinds, so for a user-supplied shape the check has to come first. Every `try_` constructor calls `RawTensor::checked_numel` (which is `Shape::from_dims`) before it builds anything. `reshape` used `Shape::derived`, whose panic says "op computed an invalid output shape"; that is wrong when the caller typed the shape.

**State-dict errors name the entry:** `load_weight` returns `StateShapeMismatch { key, expected, actual }` or `StateDataLength { key, expected, len }`, and a container (`Sequential`, `TransformerBlock`, `MultiHeadAttention`, `FeedForward`, `PositionalEncoding`) puts its layer prefix back on the key (`VoltaError::with_key_prefix`), so a UI can say `2.weight` and not `weight`. The LSTM bias loads had been passing the name `"bias"` for the keys `bias_ih` and `bias_hh`; they pass the real keys now.

**Missing keys still load nothing:** `load_state_dict` skips keys the module lacks and leaves parameters with no entry untouched. That is the old behavior, kept; `load_state_dict_checked` reports them. Strictness about missing keys is a separate decision.

**Open: a failed load is not atomic.** `load_state_dict` checks each entry as it reaches it, so an `Err` can leave earlier layers overwritten. The trait doc says to discard the module. A browser tool must not rely on that. The fix has to avoid keeping a second copy of the weights, which rules out snapshot-and-rollback and a fresh model plus a swap (peak memory about twice the weights, gigabytes for a large model, in a browser tab). That leaves plan-then-apply: layers return the `(tensor, entry)` pairs they would load, one provided method validates all of them, and only then writes. Deferred until the browser tool exists, because it changes the `Module` trait again; the design and its costs are in #65.

**Not changed:** ops that take an `Axis` (`max_dim`, `mean_dim`, `softmax`, `permute`, ...) and the GPU layer's `Option` returns stay as they were. A binding range-checks an axis against `shape().rank()` itself until those get a `try_` form.

## Layers

### Attention owns no residual (#39, S10/S11)

**Decision:** `MultiHeadAttention::forward` returns `out_proj(Attn(x))`, as `torch.nn.MultiheadAttention` does. The residual and the layer norm belong to the block: `TransformerBlock` is Pre-LN, `x + Attn(LN1(x))` then `+ FFN(LN2(.))`, which is `torch.nn.TransformerEncoderLayer(norm_first=True)`. `new_with_norm` and the optional `norm` field are removed.

**Why:** attention added its own input, so the block computed `x + LN(x) + Attn(LN(x))` (S10), and `new_with_norm` returned `LN(x) + Attn(LN(x))` (S11). Without a residual inside attention, both layers can be checked against torch with loaded weights (`tests/torch_ref_transformer.rs`).

**Rejected:**
- *Keep the residual in attention and drop it from the block.* It works, but attention would not match any torch module, and a model ported from PyTorch adds the residual around attention itself, so it would add it twice.
- *Keep `new_with_norm` as `Attn(LN(x))`.* Nothing called it, and a norm without the residual is half a sublayer with no torch counterpart. A Pre-LN sublayer is `TransformerBlock`'s job.

## Optimizers

### Muon follows the reference iteration, on the CPU and the GPU

**Decision:** `Muon` orthogonalizes with the reference quintic Newton-Schulz (`a, b, c = 3.4445, -4.7750, 2.0315`), scales the step by `sqrt(max(1, rows / cols))`, and runs on a GPU parameter with its state on the device. It is exported from the crate root with `Adam` and `SGD`.

**Why the quintic:** the old code iterated `0.5·X·(3I - XᵀX)`, the classic cubic. It converges to exactly orthogonal on a well-conditioned matrix, so a Gram-matrix check cannot tell the two apart (measured: the cubic scores better on random 8x12 and 6x24 inputs). The quintic's point is the slope at zero, 3.44 against 1.5: a singular value around 5e-3 (a direction the gradient has nearly lost) reaches the band in five steps, where the cubic leaves it near 0.04. `muon::tests::newton_schulz_recovers_a_nearly_vanished_direction` pins that, and the cubic fails it. The quintic settles in about `[0.7, 1.2]` rather than at 1, which is what the reference accepts.

**Shape scale and orientation:** the reference multiplies by `sqrt(max(1, out / in))` for PyTorch's `[out, in]` weights. Volta treats a parameter as `[dim0, rest]`, which matches that for conv weights (`[out, in·kh·kw]`) and is the transpose for `Linear`, which stores `[in, out]`. The orthogonalization is transpose-equivariant (a test pins it), so only this factor differs: for a `Linear` it is `sqrt(max(1, in / out))`. Making Muon aware of each parameter's layout would need per-parameter metadata the optimizer API does not carry; documented on `Muon` instead. The old `* 0.1` "stability" constant is gone.

**Non-matrices:** a parameter of rank below 2, or with one row or column, gets plain momentum with no orthogonalization and no scale. The reference sends those to Adam; here the caller can put them in an `Adam`.

**GPU:** a GPU parameter keeps its velocity on the device, and `step` runs momentum, Newton-Schulz, and the update there with no host round-trip (`RawTensor::gpu_muon_step`). The kernels are `muon.wgsl` (momentum, sum of squares, normalize, `axpby`) composed with the existing `matmul` and `permute`; the update itself goes through the SGD arm of `optimizer_step`, with `lr · scale` as the rate. The Frobenius norm is computed by a one-workgroup reduction and read by the next kernel from a device buffer, so it costs no sync. The quintic coefficients and epsilon are passed in from `muon.rs`, so the CPU and GPU iterations cannot drift apart. Each kernel takes its element count from a uniform, not `arrayLength`, because pooled buffers are power-of-2 buckets. Evidence: `tests/gpu_muon.rs` (CPU/GPU parity over wide, tall, square, conv, vector, and column shapes, three steps, both momentum forms) and `gpu::kernels::muon_tests` (Newton-Schulz parity to 1e-5, measured worst 1.1e-6 on an Apple M2).

**History:** the first H8 fix made `Muon::new` panic on a GPU parameter instead of writing the kernel. That was wrong: a missing GPU path is a gap to close (`.claude/rules/gpu.md`), and refusing the input where the CPU path works only moves the failure. It was replaced by the kernels above.

## Concurrency and Async

### Stay synchronous

**Decision:** no async in the core or the GPU API. Revisit only for isolated data loading or serving.

Autograd is a sequential graph walk over `Rc<RefCell<_>>`, and nothing in it is `Send`. The GPU path blocks on submission, which is coherent for a single-threaded engine. The prerequisites are explicit transfers (ADDENDUM §7) and the arena graph (§6), which is the actual route to threading. Premature `Arc<Mutex>` is how `BufferPool` ended up guarding state only one thread can reach. The full analysis lives in git history (`docs/plans/2026-02-23-async-*.md`, removed 2026-10-25).

## Process

### Rulesets enforce the merge gate (#149)

**Decision:** `improvements` and `main` each have a ruleset, kept in `.github/rulesets/*.json` and applied with `just rulesets`. Both require a pull request and the four `ci.yml` checks (`cpu-only` on 1.89 and stable, `linux-lavapipe`, `macos` on stable), and block deletion and force pushes. `improvements` allows only squash merges; `main` allows only merge commits, so a phase landing keeps its history.

**Why:** until then the merge gate was convention. The only rule was "no force push", so a red CI run did not block a merge, and #148 reached `improvements` as a merge commit instead of a squash.

**Choices:**
- *0 required approvals.* There is one maintainer; the PR requirement exists to make CI run first, not to add a reviewer.
- *Repo admin bypasses through a PR only* (`bypass_mode: pull_request`): an emergency merge of a red PR is possible, a direct push is not.
- *Status checks are not strict* (a branch need not be up to date with the base), to avoid a rebase and a full CI re-run whenever another PR merges first.
- The checks are listed by name, so a renamed job or a change to the toolchain matrix in `ci.yml` (see #121) blocks every merge until the JSON is updated. Change both in the same PR.
- The JSON is the source of truth, but nothing applies it automatically: after editing a file, run `just rulesets`. The recipe creates or updates by `name`, so it is safe to repeat.
- *The `macos` job runs stable only.* GitHub's macOS runners are intermittently unavailable (a run failed because `macos (stable)` got no runner), and each extra macOS job is another chance to hit that. The MSRV is checked by `cpu-only (1.89)`. What this gives up: `accelerate` (and the macOS-only code paths) are no longer compiled with 1.89, so an MSRV break in that code is not caught by CI. Revisit if one slips through.

### `unused_self` and `needless_pass_by_value` are denied; exceptions are per-site `#[expect]` (#171)

**Decision:** both lints are `deny` in `[lints.clippy]`, and each exception is a `#[expect(.., reason = "..")]` at the item. The crate-wide `needless_pass_by_value = "allow"` is gone. Methods that never read `self` are associated functions, except the cases below.

**Why:** a crate-wide allow hides every new by-value argument, and it had a rationale that covered one site (`Tensor::set_grad`). `#[expect]` fails the build when the exception stops being needed, so the list cannot rot.

**Exceptions, and why they stay:**
- *`step_gpu_param` stubs in `adam.rs` and `sgd.rs`.* They keep the GPU variant's `&mut self` signature so the shared call site compiles with and without the `gpu` feature.
- *`backward` on the unit-struct grad fns `TransposeGradFn`, `MulAccGradFn`, `BceGradFn`.* They hold no state, so `self` is unused, but every other `GradFn` variant's `backward` takes `&self`, and `GradFn::backward` dispatches all of them as `g.backward(out_grad, parents)`. Keeping that uniform was chosen over three arms that call an associated fn.
- *`Tensor::set_grad` and `Storage::gpu` take `Storage` / `Vec<f32>` by value:* the caller hands over a buffer it just built, and a reference would make every caller clone.

**Revisit:** if more unit-struct grad fns appear, turning their `backward` into associated fns (and the dispatch arms into `Self::X(_) => XGradFn::backward(..)`) removes the `#[expect]`s. The cost is a dispatch that is no longer uniform.

---

### One definition of CI (#120, #121)

**Decision:** the justfile defines each check (`lint`, `lint-cpu`, `test-cpu`, `doc`, `doc-cpu`, `hack`), and `just ci`, the pre-commit clippy hook and `ci.yml` call those recipes. `just ci` now also runs `cargo test --no-default-features`, rustdoc with `-D warnings` for both feature sets, and the cargo-hack powerset (skipped with a notice when `cargo-hack` is not installed). Workflows run with `permissions: contents: read`, a per-job `timeout-minutes`, and actions pinned to commit SHAs that Dependabot updates. Job names are unchanged, so the rulesets are too.

**Why:** a change could pass `just ci` and fail GitHub (an unresolved intra-doc link, a CPU-only test), and a hung job could sit on a scarce macOS runner for six hours.

**Choices:**
- *`concurrency` cancels only pull-request runs.* A superseded PR push stops holding runners; a push to `main` or `improvements` always finishes, so the merged state is tested (`improvements` is now in the `push` branches).
- *`dtolnay/rust-toolchain` is pinned to a `master` SHA,* which Dependabot cannot follow. Bump it by hand.
- *`cargo deny` runs on PRs that touch `Cargo.toml`, `Cargo.lock` or `deny.toml`,* and weekly. It is not a required check: a path-filtered check that never reports would block every other PR.
- *`stable` keeps floating.* `tests/compile_fail.rs` pins rustc's wording, so a new stable can turn an untouched branch red. Pinning would rename the `(stable)` checks, edit the rulesets, and need a bump every six weeks. When it happens, re-bless on stable in its own PR.
- *Pre-commit does not run the CPU-only test, the strict docs or cargo-hack,* to keep commits fast. `just ci` does.

---

### Coverage and mutation testing are signals, not gates; dependency updates are grouped (#154)

**Decision:** `coverage.yml` runs `just coverage-cpu` (`cargo llvm-cov --no-default-features --summary-only`) on every PR and push to `improvements` and posts the autograd-core rows and the total to the job summary. `mutants.yml` runs `just mutants-core` (src/autograd.rs and src/ops/, CPU lane) weekly and by hand in 8 shards and lists surviving mutants in each shard's summary. Neither is a required check and neither fails on a number. Dependabot opens one weekly PR for all minor and patch crate bumps, one per major bump, and one for all action bumps.

**Why:** invariant 7 is a claim about test quality, and nothing measured it on a PR. A threshold needs a baseline first; at filing the CPU lane covered 91.8% of regions overall and 86.5% of `autograd.rs`.

**Choices:**
- *`src/ops/gpu_ops.rs` is excluded from mutation testing.* It is compiled out without `gpu`, so every mutant there would survive and drown the real ones. GPU-only paths elsewhere (`ToDeviceGradFn::backward`) still show as missed, which is accurate for the CPU lane.
- *Mutants run weekly, not per PR,* because ~400 mutants at ~20 s each (measured on 4 cores: 59 s baseline build, 38 s baseline test, 5 s build + 7 s test per mutant) is over two hours of runner time. `--in-diff` per PR is the next step if the weekly signal proves useful.
- *Missed mutants go to the job summary, not an uploaded `mutants.out`,* so the list is readable on the run page and the workflow needs no new action pin. The full `mutants.out` is a `just mutants-core` away locally.
- *A major crate bump gets its own PR* because it usually needs code changes (wgpu); the `cpu-only (1.89)` lane catches a bump that raises the MSRV.

**Revisit:** gate coverage (e.g. no drop on `autograd.rs` and `ops/`) once a few weeks of summaries give a baseline.

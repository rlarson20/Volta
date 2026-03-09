# DECISIONS.md — Volta Autograd Engine Audit

Design decisions, implementation rationale, and known gaps.
Updated: 2026-02-27

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

## Known Gaps Summary

| Area                   | Gap                                                                    |
| ---------------------- | ---------------------------------------------------------------------- |
| Chain rule             | Cannot derive multivariate chain rule justification for grad summation |
| Traversal              | Conflated root assert with per-node behavior                           |
| `Rc` liveness          | Cannot trace guarantee unprompted                                      |
| Composition pattern    | Not internalized without prompting                                     |
| Broadcasting algorithm | Step-by-step procedure not reproducible unprompted                     |
| Adam                   | Recurrences and update formula required reference                      |

# Interview prep

The first draft of `DECISIONS.md` was written to practise justifying this project in an interview. The parts that explain the textbook idea, or record what I could not yet reconstruct unprompted, live here so `DECISIONS.md` can stay a record of design choices.

## Gaps I flagged

- **Topological Sort:** Could not derive the multivariate chain rule justification for gradient summation unprompted. Understand the conclusion, not the derivation.
- **Topological Sort:** Confused root-level `assert` (sanity check on the output node) with per-node behavior during traversal. These are distinct: the assert fires once, traversal logic is per-node.
- **`Rc` and Node Liveness:** Could not trace the `Rc` → liveness guarantee unprompted. Understand it when shown; not yet automatic.
- **Composition Pattern:** This composition pattern was not internalized — had to be shown explicitly.
- **Broadcasting:** Algorithm not fully internalized — understood the output rule but couldn't reconstruct the step-by-step procedure unprompted.
- **Adam:** Adam update rule required reference during audit. Understood bias correction purpose; recurrences and exact formula not memorized.

## Optimizers (textbook)

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

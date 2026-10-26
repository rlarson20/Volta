# ADDENDUM — Preventive Architecture for Volta

_Companion to `glm-53-adversarial-audit.md`.
Parts I and II established what's broken and why.
This document is not about fixing those bugs:
it's about the structural changes that make each bug **class** impossible or compiler-enforced,
so the next hundred ops and layers can't reintroduce them._

_Every recommendation names the finding it prevents
(B1–B5, H1–H8, and the Part II Rust findings are referenced by their IDs),
shows the pattern concretely in Volta's terms, and states the cost.
Recommendations are honest about what NOT to do.
Sequencing plan at the end._

---

## Traceability: defect class → prevention

| Defect (from report)                | Class                                       | Prevented by                                   |
| ----------------------------------- | ------------------------------------------- | ---------------------------------------------- |
| B1 ConvTranspose grad gate          | graph attachment gated on the wrong input   | §1.4 single graph-edge constructor             |
| B2/B3 LSTM/PosEnc severed graph     | layers building raw `RawTensor::new` copies | §1.4 + §1.5 (`no_grad` + private constructors) |
| B4 unequal-rank matmul grad         | dims derived from the wrong tensor's rank   | §1.2 `Shape`/`Axis` newtypes                   |
| B5 no-features build broken         | cfg triplication                            | §9 single-source cfg dispatch                  |
| H1 MaxPool 0-padding                | semantic constants implicit                 | §2.4 `PadValue`                                |
| H3 unsafe cache invariant           | hand-rolled unsafe on a false invariant     | §7 `OnceCell` storage                          |
| H6 TransformerBlock decorative      | trait can't express attention               | §3.2 trait split                               |
| Silent zeros (77× `unwrap_or(0.0)`) | bounds failures become data                 | §8 loud indexing policy                        |
| Silent GPU→CPU fallback             | `Option` → warning → behavior change        | §2.1 `DevicePolicy`                            |
| Clone economy (424+ clones)         | deep-copy-per-op ownership design           | §6 arena + §7 `Arc` storage                    |
| GradFn boilerplate / `clone_box`    | 30 lines per op, vtable dispatch            | §4.1 closed enum + §5.1 `define_ops!`          |
| "Tests assert the bug" (LSTM)       | tests can encode the defect                 | §10 `assert_trainable!` + cargo-mutants        |
| Embedding panics in `Sequential`    | trait contract violated at runtime          | §3.2 sealed, split traits                      |
| CI step that doesn't run its name   | config drift                                | §9 feature matrix                              |

---

## 1. Make illegal states unrepresentable

This is the theme that retires the most findings per line changed. Three of the five Part I blockers (B1, B2, B3) are _writes to state that should have been impossible to write incorrectly_, and they were all one `borrow_mut()` deep.

### 1.1 Opaque `Tensor` newtype

**Prevents:** B1, H7, and the entire "invariants are conventions" class (Part II §1).

Today `pub type Tensor = Rc<RefCell<RawTensor>>` with seven public fields means `x.borrow_mut().requires_grad = true` is the _documented_ idiom and `load_state_dict` is un-checkable field surgery. Wrap it:

```rust
pub struct Tensor {
    inner: Rc<RefCell<RawTensor>>,  // fields now private
}

impl Tensor {
    // The only ways to mutate. Each is named, findable, greppable,
    // and can carry debug_assert! invariant checks in ONE place.
    pub fn requires_grad(&self) -> bool { self.inner.borrow().requires_grad }
    pub(crate) fn set_requires_grad(&self, on: bool) { /* + invariant asserts */ }
    pub fn detach(&self) -> Tensor { /* new leaf sharing data */ }
    pub(crate) fn accumulate_grad(&self, g: Storage) { /* the ONLY grad writer */ }
    pub(crate) fn snapshot(&self) -> TensorSnapshot<'static> { /* owned clone of fields */ }
}
```

`load_state_dict` becomes `layer.load_weight(key, &data)` — a method that validates numel and preserves device, instead of every layer hand-writing `t.data = Storage::cpu(...)` (twelve copies today). The ConvTranspose bug stops being _expressible_: the layer never touches `requires_grad`; it hands parents to the constructor in §1.4 and the flag is computed.

**Cost:** mechanical churn across ops (each `t.borrow().shape.clone()` becomes `t.shape()`), roughly a weekend; behavior identical if done before anything else. This is the prerequisite for nearly everything below — do it first or don't bother with the rest.

**Rule of thumb going forward:** every time you write `borrow_mut()` outside `src/autograd.rs` and `src/ops/`, you have found a missing method on `Tensor`.

### 1.2 `Shape` and `Axis` newtypes

**Prevents:** B4, every dim-mixup, and the `Vec<usize>` == `Vec<usize>` ambiguity that made the batched-matmul bug silent.

Shapes are `Vec<usize>` today, passed around loose, re-derived from the wrong tensor (`matmul.rs:506-559` reads `n` from `y.shape` at `x`'s rank). Wrap them where misuse becomes a type error or a single checked constructor error:

```rust
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Shape(Vec<usize>);

impl Shape {
    /// The ONLY constructor that reaches user code. Rejects zeros,
    /// enforces the 100M cap, and is the one place to debug_assert
    /// the data.len() == numel() invariant at attach time.
    pub fn from_dims(dims: impl Into<Vec<usize>>) -> Result<Self, VoltaError>;
    pub fn rank(&self) -> usize;
    pub fn numel(&self) -> usize;
    pub fn strides(&self) -> Strides;               // another newtype
    pub fn broadcast(&self, other: &Self) -> Result<Self, VoltaError>;  // the try_ twin, finally
    pub fn dim(&self, axis: Axis) -> Result<usize, VoltaError>;         // rank-checked access
    pub fn truncate_rank(&self) -> &[usize];        // explicit, named demotion
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Axis(pub usize);   // `sum_dim(Axis(1))` — an Axis can't be confused with a size
```

B4 dies structurally: `MatmulSpec::new(a: &Shape, b: &Shape) -> Result<MatmulSpec, VoltaError>` computes and validates `m, k, n, batch` once; the backward reads `spec`, not `y.shape.get(x.rank() - 1)`. A rank-mismatched product is an `Err` at forward time instead of a 29%-wrong gradient at backward time.

**Cost:** large but mechanical signature churn. Worth it — `Vec<usize>` shape bugs are this codebase's most productive bug family.

### 1.3 `#[repr(transparent)]` index newtypes everywhere indices meet

`TensorIndex(u32)`, `NodeIndex(u32)`, `Channel(usize)` where it disambiguates. Zero runtime cost, and `#[repr(transparent)]` documents that. The moment an API takes two same-primitive params that mean different things (`batch_size, channels` — 14 of them per GPU wrapper), that API is asking for transposition bugs; a newtype is cheaper than the outage.

### 1.4 One graph-edge constructor (the single highest-leverage change)

**Prevents:** B1, B2, B3, and every future op that attaches its grad fn under the wrong condition.

Every op today hand-writes the attach ritual, and each hand-writing is a chance to gate on the wrong flag (B1), forget parents (B2/B3), or set the flag without the fn. There are ~25 copies of this ritual. Replace them with exactly one:

```rust
// src/autograd.rs — the only code allowed to write grad_fn/parents.
pub(crate) fn edge(
    inputs: impl AsRef<[Tensor]>,
    data: Storage,
    shape: Shape,
    grad: GradFn,          // §4.1 enum
) -> Tensor {
    let requires_grad = inputs.as_ref().iter().any(|t| t.requires_grad())
        && !grad_is_trivial(&grad);        // Mod/Cmplt return false here, once
    // attach parents + grad_fn iff requires_grad, in this function only
}
```

Consequences:

- **B1 is dead.** `ConvTranspose2d::forward` calls `autograd::edge([x, w, b], ...)`; there is no `if x.requires_grad` to write wrongly — "any input requires grad" is the only semantics that exists.
- **B2/B3 are dead.** `LSTMCell::slice_gate` and `PositionalEncoding` needed raw `RawTensor::new` _because_ the graph path was harder to use than the raw path. With `RawTensor::new` demoted to `pub(crate)`, and `edge` as easy as the raw path, the easy way and the correct way become the same way. Slicing a gate becomes `gates.shrink(&ranges)` because that's now the _shortest_ code that compiles.
- The borrow-dance boilerplate (Part II §5) collapses into `edge` too.

This is the pattern PyTorch's `Node`/`Edge` construction centralizes, and its absence is the single reason three layers shipped gradient-dead.

### 1.5 RAII `no_grad` guard

**Prevents:** the `check_gradients` stale-grad contamination (Part I, probe E), BatchNorm running-stat pollution during gradient checking, and the future "inference path accidentally builds graph" class.

```rust
static NO_GRAD: Cell<bool> = Cell::new(false);

pub fn no_grad() -> NoGradGuard {
    NO_GRAD.set(true);
    NoGradGuard
}
pub struct NoGradGuard;
impl Drop for NoGradGuard { fn drop(&mut self) { NO_GRAD.set(false) } }

// edge() consults NO_GRAD; ops don't.
```

BatchNorm's running-stat update becomes `let _g = no_grad(); update_stats(...)` — the mutation is scoped and visible. `check_gradients` opens with `zero_grad` + `no_grad()` around the numerical evaluations so the loss_fn's side effects can't poison the comparison. Also the natural place to hang `torch.inference_mode`-style behavior later.

---

## 2. Typestate and phantom types — where they pay, and where they don't

Typestate is the reflexive answer to "prevent wrong-state bugs," but it's a spectrum. Volta should take the two cheap ends and decline the expensive middle.

### 2.1 Decline: full device typestate (`Tensor<Cpu>` / `Tensor<Gpu>`)

**Verdict: don't.** Full phantom-typed devices infect every signature (`fn f(t: &Tensor<Gpu>)`), force either duplication or generics-with-trait-bounds on every layer, and fight the current design where tensors legitimately move devices mid-graph (`to_device` preserving autograd metadata is a _feature_ here, `tensor.rs:711-736`). PyTorch, burn, and candle all keep device runtime-dynamic on the tensor for exactly this reason.

**Take instead — a typestate _policy_, not a typestate _tensor_:**

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DevicePolicy {
    /// Mixed devices or missing kernels are Err(DeviceMismatch). Default.
    Strict,
    /// Fall back to CPU on kernel miss; returns the policy decision.
    PreferGpu,
    CpuOnly,
}
```

This kills the silent-fallback class (Part II §3): `binary_op` takes the policy (thread-local with `set_device_policy()` for ergonomics), and `PreferGpu` fallbacks return a `FallbackEvent` your training loop can count and assert on, instead of an `eprintln!` that may be buried in logs. `Strict` — the default — converts the H5-class "silent wrongness" scenarios into errors. Same runtime cost, all of the prevention, none of the generic-signature plague.

### 2.2 Decline: `Variable`/`Tensor` typestate for `requires_grad`

PyTorch _had_ this (`Variable` pre-0.4) and merged it away because dynamic control flow makes the distinction leaky — every `if` on data values changes the type you wish you had. The compile-time guarantee is illusory the moment graphs are dynamic. Keep `requires_grad` a runtime flag, but take the tiny bounded version: `backward()` is only callable on a `Scalar` newtype:

```rust
impl Tensor {
    pub fn into_loss(self) -> Result<ScalarLoss, VoltaError>;  // Err if numel != 1
}
impl ScalarLoss {
    pub fn backward(&self);   // the assert!("doesn't require grad") panic becomes a Result
}
```

That converts `autograd.rs:43-46`'s panic into a type-checked flow at zero ergonomic cost — `loss.into_loss()?.backward()` reads fine.

### 2.3 Take: ZST markers where you already have implicit state

Zero-sized types give you named, checked state for free. You already have the good version of this (`pub struct ReLU;` — activation layers are ZSTs; `UnaryOp`/`BinaryOp` are fieldless enums). Extend the discipline:

```rust
pub struct Cpu;
pub struct Gpu;                 // phantom markers for a future Backend trait (§3.5)
pub type CpuTensor = Tensor;    // document intent at alias sites, no runtime change

pub struct PadValue(f32);        // §2.4
impl PadValue {
    pub const ZERO: Self = Self(0.0);
    pub const NEG_INF: Self = Self(f32::NEG_INFINITY);
}
```

`PhantomData<D>` on any future backend-generic struct costs nothing and documents the parameter. The rule: **if a bare primitive crosses an API boundary carrying semantics (0.0 "meaning" zero-pad vs neutral-pad), it wants to be a ZST-constructible newtype.**

### 2.4 Take: `PadValue` (prevents H1)

`MaxPool2d` pads with 0.0 because it reused the generic `pad` op whose semantic "pad value = 0" was implicit in the implementation. Make it a parameter of the op:

```rust
impl Tensor {
    pub fn pad_with(&self, pads: &[(usize, usize)], value: PadValue) -> Result<Tensor, VoltaError>;
}
// MaxPool2d::pool_forward calls pad_with(..., PadValue::NEG_INF)
```

H1 becomes unrepresentable: nobody can call the pooling path without choosing a pad value, and the correct one is a named constant at the call site instead of an implementation detail two layers down. Same treatment for any future op with hidden scalar semantics (e.g., `leaky_relu` slope, `elu` alpha — put them in the op spec from day one).

### 2.5 Take: closed `GradFn` enum instead of `Box<dyn GradFn>`

**Prevents:** the `clone_box` boilerplate tax (every GradFn carries one), vtable dispatch in the hot backward loop, and — most importantly — _unhandled new ops_.

```rust
#[derive(Clone)]
pub(crate) enum GradFn {
    Unary(UnaryOp),
    Binary(BinaryOp),
    Matmul,
    SumDim { shape: Shape, axis: Axis, keepdim: bool },
    Conv { spec: ConvSpec },       // §6: the struct that replaces 16 positional usizes
    Im2col { spec: ConvSpec },
    Movement(MovementOp),
    Custom(CustomGrad),            // escape hatch: keeps the trait for exotic cases
}
```

Backward becomes an exhaustive `match`. You already deny `wildcard_enum_match_arm` in `Cargo.toml:147` — which means **the compiler will force the next contributor to handle every existing variant when they touch this match**, and force them to add a variant's arm when they add an op. That lint, currently decorative, becomes load-bearing. `Clone` is derived; `clone_box` disappears; the enum fits `SmallVec`-style inline storage later if profiling ever cares.

Tradeoff: downstream crates can't add built-in-style ops. They use `GradFn::Custom` — same escape hatch PyTorch has (custom autograd functions). For a framework this is the right default; openness here has bought you nothing except 25 hand-written `clone_box`es.

---

## 3. Trait design

### 3.1 Seal every trait meant for internal extension

**Prevents:** downstream half-implementations of `Module`/`TensorOps`/`GradFn` breaking invariants your library assumes (e.g., "parameters() and state_dict() agree on keys" — today a convention, enforced by nobody).

```rust
pub mod sealed { pub trait Sealed {} }   // deliberately undocumented

pub trait Module: sealed::Sealed {
    fn forward(&self, x: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<Tensor>;
    // ...
}
impl sealed::Sealed for Linear {}
impl sealed::Sealed for Conv2d {}
// ...
```

Sealing buys you the right to add provided methods and change signatures within 0.x without semver-breaking well-meaning downstream impls. Pair it with a **trybuild test** (§10) that proves `impl Module for OutsiderType {}` fails to compile — that's how you _verify_ a seal instead of believing in it.

### 3.2 Split `Module` by input contract (prevents H6 and the Embedding panic)

The trait currently forces three unrelated contracts through one method, which is why `Embedding`'s impl panics by design (`embedding.rs:121-127`) and `LSTMCell`'s silently drops the cell state (`lstm.rs:149-153`):

```rust
pub trait Layer: sealed::Sealed {
    fn forward(&self, x: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<Tensor>;
    fn state_dict(&self) -> StateDict;
    fn load_state_dict(&mut self, &StateDict) -> Result<(), VoltaError>;
    fn train(&mut self, mode: bool) { }
}

/// Embedding lives here, NOT behind Box<dyn Layer>.
pub trait IndexedLayer: sealed::Sealed {
    fn forward_indices(&self, idx: &[u32]) -> Tensor;
}

/// LSTM lives here; the type system now carries (h, c) instead of discarding c.
pub trait RecurrentLayer: sealed::Sealed {
    type State;
    fn initial_state(&self, batch: usize) -> Self::State;
    fn step(&self, x: &Tensor, state: &mut Self::State) -> Tensor;
}
```

`Sequential` stores `Box<dyn Layer>` and now it is a **type error** to put an `Embedding` in one — the runtime panic is gone at compile time. (Alternative worth considering: keep Embedding a `Layer` by accepting an `I64`-dtype index tensor — that's the PyTorch shape of the API — but only if dtypes become more than decorative; otherwise the split is more honest.)

This also gives `TransformerBlock` a place to be honest (H6): either it holds a `Box<dyn Layer>` attention that _must_ be real (no `SimpleAttention` identity shim compiling under the same signature), or attention is generic `A: AttentionLayer` and the placeholder can't masquerade.

### 3.3 `#[non_exhaustive]` on everything public that can grow

`VoltaError`, `DType`, `ConvAlgo`, `Device`, the config structs. This is free at 0.3.0 and expensive to retrofit at 0.4. `VoltaError` especially: you will want new variants (`DeviceMismatch`, `KernelUnavailable`, `RankMismatch`) and every addition is a breaking change for downstream matches unless the enum is non-exhaustive _now_.

### 3.4 Delete `TensorOps` (Part II §2)

With the opaque `Tensor` of §1.1, the trait's reason-to-exist (methods on a type alias) evaporates: make them inherent. If you still want a named import surface for discoverability, keep a sealed, re-export-only extension trait — but the 140-line delegation blob (`tensor.rs:1009-1208`) should not survive the refactor.

### 3.5 Later: a sealed `Backend` trait (sketch only)

The endgame for the CPU/GPU split is burn's shape — a sealed trait with an associated tensor type, `Cpu`/`Gpu` ZST implementors from §2.3, and the current `Storage` becoming the `Cpu` backend's internals. Don't start here — start with `DevicePolicy` (§2.1) and let the seams form — but every new GPU wrapper should be written as if this trait exists (take a context/params struct, return `Result`), so the eventual migration is a move, not a rewrite.

---

## 4. Common trait impls — a specific list, not "implement more traits"

| Impl                                      | For                                                        | Why / guardrail                                                                                                                                                                                           |
| ----------------------------------------- | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Debug` (custom)                          | `Tensor`                                                   | Already half-exists (`tensor.rs:92-101`); keep shape/dtype/grad-flag, **not** full data (a 100M-param debug print is a DoS on your own logs). `Debug` on `Shape`, `Axis`, specs: derive.                  |
| `Display`                                 | `Shape`, `Device`, `DType`                                 | `"2×3"` / `"cpu"` for error messages people actually read.                                                                                                                                                |
| `TryFrom<(Vec<f32>, Shape-ish)>`          | `Tensor`                                                   | The constructor story: `try_new` exists; `TryFrom` is the trait people reach for, and it's what lets `?` work at call sites.                                                                              |
| `FromIterator<f32>`                       | `Tensor` builder                                           | Only with an explicit shape step — a bare `collect()` can't know the shape, so this wants to be a `TensorBuilder` instead (see below).                                                                    |
| `Index<[usize; N]>` / `Index<&[usize]>`   | `Tensor`                                                   | Checked coordinate indexing returning `f32` (copy). Kills the temptation to hand-roll linearization at use sites — the #1 source of index math bugs in layer code.                                        |
| **Do not** impl `PartialEq`               | `Tensor`                                                   | Float equality on tensors is a footgun factory. Provide `allclose(&self, other, rtol, atol) -> Result<(), Mismatch>` plus an `assert_close!` test macro (§5.4). `PartialEq` on `Shape`/`Axis`/specs: yes. |
| `Default`                                 | `TileConfig`, `BufferPoolConfig`, optimizers' hyper-params | `TileConfig::default()` already exists; `#[derive(Default)]`-able configs should derive, and bool-heavy constructors want the builder below.                                                              |
| `Error`                                   | `VoltaError`                                               | Already via thiserror — add `#[non_exhaustive]` and a `source()` chain for the GPU error types that currently `Display` without impl'ing `std::error::Error` (`gpu/context.rs:40-57`).                    |
| `#[track_caller]`                         | every panic-adjacent helper                                | When a bounds assert fires inside a kernel, the caller's line is the only useful one. Cheap attribute, transform the debugging experience.                                                                |
| `Serialize`/`Deserialize` (feature-gated) | `Shape`, `StateDict`                                       | Behind a `serde` feature; keeps the core compile lean and makes `StateDict` interop with the ecosystem instead of bincode-only.                                                                           |
| `#[repr(transparent)]` + `From<inner>`    | all §1.3 newtypes                                          | So they interop with `usize` boundaries without ceremony.                                                                                                                                                 |

**Bool-parameter constructors** (`Conv2d::new(in, out, k, s, p, use_bias)` — six positional params where two are indistinguishable scalars and one is a bool): replace with

```rust
Conv2d::builder(in_ch, out_ch, 3)     // required
    .stride(2).padding(1).bias()      // optional, named
    .build()?
```

The existing `SequentialBuilder` is the precedent — the codebase already knows this pattern and just doesn't apply it where the parameter lists are longest. Note your own clippy config denies `fn_params_excessive_bools`; the builder is how you comply without the `#[allow]`.

---

## 5. Macros — two that matter, one to skip

### 5.1 `define_ops!` — one table per op family, emitting forward, backward, and GPU kernel name

**Prevents:** forward/backward drift, CPU↔GPU kernel-name drift, and the boilerplate that makes people avoid writing ops correctly (the root cause of B2/B3).

Today the unary family lives in four hand-maintained places: the `UnaryOp` enum, the forward match (`unary.rs:258-275`), the backward match (`unary.rs:76-165`), and _two separate_ kernel-name tables (`unary.rs:175-211`). Five places to forget each other. One table:

```rust
define_ops! {
    // variant, forward expr, backward expr (g = out_grad, x = input), gpu kernels
    Neg   => |x| -x,                     |g, _x| -g,                        ("neg", "neg_backward");
    Exp   => |x| x.exp(),                |g, x|  g * x.exp(),              ("exp", "exp_backward");
    Sqrt  => |x| x.sqrt(),               |g, x|  g / (2.0 * x.sqrt()),     ("sqrt", "sqrt_backward");
    ReLU  => |x| x.max(0.0),             |g, x|  if x > 0.0 { g } else { 0.0 }, ("relu", "relu_backward");
    // ... 9 more rows. The whole unary family: 13 lines.
}
```

which expands to the enum, the forward arms, the backward arms, and — critically — **a single function mapping op → GPU kernel pair**, so `unary_kernel_name(op)` and the shader pipeline table are generated from the same row. Adding `SiLU` becomes one line and a WGSL entry; forgetting the backward arm or the kernel is a compile error, not a silently-slow CPU fallback.

The same shape works for binary ops (the `BinaryOp` family, `binary.rs:13-21` + the three kernel tables at `binary.rs:365-386`). Movement ops keep their per-op structs (they have real state) but should share the generated dispatch match.

**Cost:** declarative `macro_rules!` only — a day of work, no proc-macro. This is the highest ratio of prevented-drift to effort in this document.

### 5.2 `param_test!` — collapse the test matrices

`conv.rs`'s test module is ~1,900 lines of near-identical parameterized tests (kernel sizes, strides, paddings, channels — `conv.rs:1966-2906`). The repetitive ones become:

```rust
param_test! {
    conv_forward_shape: Conv2d forward shape in=(in_ch, out_ch, k, s, p, h, w) expect=(oh, ow) => [
        (3, 16, 3, 1, 1, 32, 32, 32, 32),
        (3,  8, 3, 2, 1, 16, 32,  8, 16),
        (3, 64, 5, 2, 2, 32, 32, 16, 16),
    ]
}
```

Same assertions, one failure message format with the full parameter tuple, and adding a configuration to the matrix is a row edit. This matters beyond tidiness: the reason the ConvTranspose tests never covered the weights-grad path is that writing the 30-line test body each time discourages matrix extension. Cheap tests are exhaustive tests.

### 5.3 `assert_trainable!` — the macro that would have caught B1, B2, and B3

This is the single most important test artifact in this document. One macro, applied to every layer, encodes the property whose absence shipped three blockers:

```rust
#[macro_export]
macro_rules! assert_trainable {
    ($layer:expr, $input:expr) => {{
        let output = $layer.forward(&$input);
        let loss = output.sum().into_loss().expect("layer output must be differentiable");
        loss.backward();
        for (i, p) in $layer.parameters().iter().enumerate() {
            let g = p.grad().unwrap_or_else(|| panic!(
                "parameter {} has no gradient — graph severed or gate bug", i));
            assert!(
                g.iter().any(|v| v.abs() > 0.0),
                "parameter {} gradient is identically zero", i
            );
        }
    }};
}

// In every layer's test module:
assert_trainable!(ConvTranspose2d::new(2, 4, 3, 1, 1, true), rand_input(&[1, 2, 6, 6]));
assert_trainable!(LayerNorm::new(vec![64]),               rand_input(&[2, 8, 64]));
// Embedding via its IndexedLayer test path, LSTM via a 2-step unroll.
```

Note what this encodes: **input does NOT require grad** — the real training configuration, the one every existing ConvTranspose test forgot to exercise. Zero gradients under a `sum()` loss means a severed graph or a wrong gate, always. A green run of this macro across all nineteen layers is a regression fence around the entire B1/B2/B3 class.

### 5.4 Small test ergonomics

`tensor![[1.0, 2.0], [3.0, 4.0]]` (shape-inferred literal), `assert_close!(a, b, rtol, atol)` (moves `approx` to dev-dependencies where it belongs, Part II §7), and a `gradcheck!` wrapper that zeroes grads first — fixing the `check_gradients` contamination footgun (Part I, probe E) at every call site.

### 5.5 Skip: proc-macros

A proc-macro deriving `GradFn` from struct definitions, a TVM-style op DSL, or derive-based `Module` registration would each add a compile-time dependency and a debugging tax that this codebase's scale does not justify. `macro_rules!` covers §5.1–5.4 at zero dependency cost. Revisit only if the op count triples.

---

## 6. Arena-based graph — the structural endgame

**Prevents, wholesale:** reference-cycle risk (currently avoided by discipline), the clone economy (Part II §5), `Rc` blocking `Send` forever, and double-borrow panics under re-entrant backward (Part II §1).

The `Rc<RefCell>` per-node design makes every tensor an independently-locked shared mutable node. The textbook alternative for dynamic graphs on a single thread:

```rust
pub struct Graph {
    nodes: Vec<Node>,               // Node { data, shape, grad, parents: SmallVec<[NodeId; 3]>, grad_fn: GradFn }
}
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct NodeId(u32);             // index, not pointer

pub struct Tensor {
    graph: Rc<Graph>,               // one Rc per tensor, not per node
    id: NodeId,
}
```

What this buys, concretely:

- **Cycles become unconstructible.** Parents are indices into a Vec owned by one graph; you cannot create a reference cycle in an index. The current design's cycle-safety is a property of every future op author's care.
- **Clones become copies of `(Rc, u32)`.** The 424+-clone economy mostly evaporates; `Storage` clones (§7) are the remaining deep copies and get their own fix.
- **Topological order is free.** `backward()` walks `id.0 ..= root` — nodes are created in topological order by construction. The DFS in `autograd.rs:69-95` and its diamond handling (correct today, but the most subtle code in the engine) stops existing.
- **Memory frees deterministically.** Drop the loss tensor, drop the graph. No strong-count archaeology (Part I probe 10).
- **`Send` becomes reachable.** `Rc<Graph>` → `Arc<Graph>` later is a one-type change instead of a redesign, because the interior `RefCell`s are gone (a graph under construction is owned by one thread).

**Costs, stated honestly:** tensors from two graphs can't interact without a check (an `assert_eq!(graph_ptr)` — same rule PyTorch's autograd engine has); in-place ops need generation counters or become forbidden (forbid them — §1.1's `accumulate_grad` is the sanctioned mutation); this is the largest change in this document, touching every op. Do it _after_ §1 and §4 land (opaque `Tensor` makes the migration invisible to users; the `GradFn` enum makes `Node` a plain struct). Precedent: this is the shape micrograd-family engines and burn's training graphs converge on.

---

## 7. Storage: cheap clones and a real cache invariant

**Prevents:** the deep-copy-per-op economics (Part II §5) and the unsafe block's false invariant (H3/Part II §4).

```rust
pub enum Storage {
    Host { bytes: Arc<[u8]>, dtype: DType },
    #[cfg(feature = "gpu")]
    Gpu  { buf: Arc<GpuBuffer>, dtype: DType, cache: OnceCell<CachedHost> },
}
```

- `Arc<[u8]>` turns every op's `data.clone()` snapshot from an O(n) memcpy into a refcount bump. Mutation goes copy-on-write: `fn make_mut(&mut self)` clones when `Arc::strong_count > 1` — the standard `Rc::make_mut` pattern, applied to storage. The optimizer's in-place param update is the only writer in the hot path and typically holds the only reference.
- `OnceCell` (or `OnceLock`) replaces the `RefCell<Option<Vec<u8>>>` + `unsafe as_ptr` block (`storage.rs:237-255`): a write-once cache with a _type-system-enforced_ write-once guarantee. `invalidate_cpu_cache` either disappears (honest name: the "cache" becomes a snapshot with a generation number) or the unsafe block is deleted in favor of `OnceCell::get` — either way the SAFETY comment stops contradicting reality (H3). **Delete the f64/f16/bf16-on-GPU accessors outright** rather than porting them — they are wrong today and unreachable (Part II §4).
- **Delete `Deref<Target = [f32]>`** and replace with explicit host access:

```rust
impl Storage {
    pub fn as_host_f32(&self) -> Result<&[f32], VoltaError>;   // Err for GPU / non-F32
    pub fn to_host_f32(&self) -> Cow<'_, [f32]>;               // explicit, greppable transfer
}
```

Every hidden GPU→CPU sync (today invisible behind `data.iter()`) and every hidden panic (`iter_mut` on GPU storage — the Muon bug, H8) becomes a visible `Result` or an explicit `Cow` at the call site. Grep-ability is the feature: `to_host_f32` in a hot loop is a reviewable smell; `Deref` coercion is not.

---

## 8. Loud-by-default kernels: the indexing policy

**Prevents:** the silent-zero class — the 77 `.unwrap_or(0.0)` sites that let B4 ship green (Part II §3).

The rule that made the lint table counterproductive was "never index, always `.get().unwrap_or(default)`." The correct policy for numerical kernels separates two cases that the lint conflates:

```rust
/// For indices that are IN CONSTRUCTION (im2col maps, backward scatters):
/// debug builds assert, release builds clamp-free fetch — but the default is
/// WRONG-BY-DEFAULT loudness, not zero.
#[inline]
#[track_caller]
fn fetch(slice: &[f32], i: usize) -> f32 {
    debug_assert!(i < slice.len(), "index {i} OOB len {} — mapping bug", slice.len());
    slice.get(i).copied().unwrap_or(f32::NAN)   // NaN propagates; zero hides
}
```

The choice of `NAN` over `0.0` as the fallback default is the whole point: an index-mapping bug in a gradient kernel now produces a NaN that fails the _next_ assertion in any test loudly, instead of a plausible-looking number that passes a 2e-2-tolerance gradient check. Debug builds stop at the exact site (`#[track_caller]` names the caller). Combined with the `define_ops!` single-sourcing (§5.1) — fewer hand-rolled index maps to get wrong — and `Shape`-checked specs (§1.2), the silent-wrong-number pipeline is closed at three independent valves.

Pair with: `log`/`tracing` crate behind a facade instead of 47 `eprintln!`/`println!` sites (Part II §3), so fallbacks and syncs are observable when asked for and invisible otherwise.

---

## 9. cfg hygiene and CI that runs what it names

**Prevents:** B5 and its recurrence.

The cfg triplication in `DirectConvGradFn::backward` (`conv.rs:262-337` — each CPU call written three times across cfg arms) exists because feature-gating happens at _call sites_. Gate at _definition_ sites, once:

```rust
// One file, one definition, two cfg'd bodies. No call site knows about features.
#[cfg(feature = "gpu")]
pub(crate) fn conv_backward_input(...) -> Option<Storage> { /* real impl */ }

#[cfg(not(feature = "gpu"))]
pub(crate) fn conv_backward_input(...) -> Option<Storage> { None }  // 3-line stub
```

Callers invoke `conv_backward_input(...)` unconditionally; the not-gpu build compiles the stub. `sgd.rs`'s inverted `#[cfg(feature = "gpu")]` on the CPU path (the direct cause of B5's fifth error) becomes un-writable: there is one `step_cpu_param`, ungated, and one GPU dispatch function, gated. Audit rule of thumb: **grep for `cfg(feature` outside `src/gpu/` — every hit is a B5 seed.** After this refactor the count should be zero.

CI changes (Part II §8):

- The step named "no features" gets `--no-default-features` on both commands — or rename it. The current mismatch is how B5 shipped under a green-checkmark named after the very configuration that was broken.
- Add `cargo hack --each-feature check` (or at minimum `check --no-default-features`, `check`, `check --all-features`). Feature-combination rot is exactly what cargo-hack exists to catch, and it caught nothing here because nothing ran it.
- Keep `-D warnings`, but pair it with the §5.1/§8 changes so the fix is cheaper than the `#[allow]`.

---

## 10. A test architecture that fences the bug classes

The Part I finding was that tests were _plentiful and well-placed but blind to configuration_ — they differentiated w.r.t. inputs, asserted known bugs as expected behavior, and never composed ops the way models do. Four additions fence each of those:

1. **`assert_trainable!` everywhere (§5.3).** The regression fence for B1/B2/B3. Every layer, input-without-grad, every parameter nonzero.

2. **Property-based gradient checks.** The existing FD checks are one-graph-each, hand-written, with hand-picked tolerances (some loosened to 2e-2 — Part I). Turn the _invariant_ into the test with `proptest`:

   ```rust
   proptest! {
       #[test]
       fn grads_are_correct_for_random_compositions(seed in any::<u64>(), depth in 2..5usize) {
           // generate a random DAG over {add, mul, matmul, relu, sum_dim, permute}
           // with broadcasting forced at some nodes; FD-check EVERY input of the DAG
       }
   }
   ```

   A random-graph FD check would have caught B4 (unequal-rank matmul composes trivially in a random DAG) and keeps catching whatever op ships next with a bad backward, at tolerances the generator controls rather than the author relaxes.

3. **trybuild UI tests for the type-system guarantees.** Every pattern in this document that promises "this misuse no longer compiles" — sealed traits (§3.1), `IndexedLayer`/`Sequential` (§3.2), `into_loss` (§2.2) — gets a `tests/compile_fail/*.rs` with an expected error. This is how you _prove_ a prevention instead of hoping: the pattern's value is a compile error, so the test suite must contain that compile error, expected and pinned.

4. **`cargo-mutants` in CI (weekly, not per-PR).** It mutates code and fails if tests still pass — which is precisely the detector for "tests that assert the bug." The LSTM test that enshrines `grad is None` as expected behavior (`test_lstm.rs:257-292`) survives every conventional tooling choice and dies the day a mutant of `slice_gate` is proven _equivalent_ under the suite. That test was green infrastructure for a defect; mutation testing is the only listed tool that flags it.

Supporting cast: `insta` snapshot for `state_dict()` round-trips (catches key-format drift between save and load — the `Sequential` prefix-matching bug from Part II is this class), and `static_assertions` the day §6 lands (`assert_impl_all!(Graph: Send)` — a claim, tested).

---

## Sequencing

Ordered by prevention-per-risk; each phase compiles and ships independently.

| Phase                      | Contents                                                                                                                                                 | Prevents                                               | Size                |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ | ------------------- |
| 2 — types for dims and ops | `Shape`/`Axis`/`Strides` (§1.2); specs replace positional args (§1.3); `GradFn` enum (§2.5); `define_ops!` (§5.1); `PadValue` (§2.4); `into_loss` (§2.2) | B4 class, forward/backward/kernel drift, H1            | ~1–2 weeks          |
| 3 — traits and storage     | Sealed + split traits w/ trybuild (§3); `DevicePolicy` (§2.1); storage `Arc`/`OnceCell`/no-Deref (§7); indexing policy + `fetch` (§8)                    | H3/H6/H8, silent fallback, clone economy, silent zeros | ~2 weeks            |
| 4 — endgame, optional      | Arena graph (§6); `Backend` trait (§3.5); proptest DAG checks (§10.2) if not already in phase 2                                                          | cycles, `Send`, memory determinism                     | multi-week, do last |

Phases 0–2 alone retire every prevention target that maps to a Part I blocker.
The arena (§6) is the only genuinely architectural bet,
and it gets cheaper every phase that lands before it.

### done phases

| 0 — mechanical, do immediately | `#[non_exhaustive]` on public enums; `approx`→dev-dep; drop `statrs` (20-line erf); CI `--no-default-features` fix + cargo-hack; cfg-at-definition consolidation (§9); `log` facade for prints | B5 recurrence, dependency bloat, print pollution | 1 day, near-zero risk |
| 1 — the invariant wall | Opaque `Tensor` + §1.4 `edge()` constructor + `NoGrad` (§1.1, 1.4, 1.5); `assert_trainable!` across all layers (§5.3) | B1/B2/B3 class, graph surgery | ~1 week, mechanical churn |

**Phase 1 deviations from this document's original sketch** (all deliberate, made while implementing rather than while planning):

- `edge()`'s `shape`/`grad` params are `Vec<usize>`/`Box<dyn GradFn>`, not the `Shape`/`GradFn`-enum types §1.2/§2.5 describe — those are Phase 2 work that hadn't landed yet. `edge()` also takes an explicit `device: Device` param not in the original sketch: `Storage` doesn't uniquely determine a `Device` value, and every real call site already tracked device separately from the data it produced.
- `NO_GRAD` is a reentrant depth counter (`thread_local! Cell<u32>`), not the sketched `Cell<bool>` — the bool breaks the moment one `no_grad()` scope nests inside another (e.g. a layer's own `no_grad()` running inside `check_gradients`'s wrapping one); the inner guard's `Drop` would incorrectly re-enable grad tracking for the still-active outer scope. It's also `thread_local!`, matching `src/tensor.rs`'s `GLOBAL_RNG` pattern, since a bare `static Cell<_>` doesn't compile (`Cell` isn't `Sync`) — the original sketch's literal `static NO_GRAD: Cell<bool>` would not have compiled either.
- `assert_trainable!` calls `.backward()` directly on `.sum()` rather than going through `.into_loss()` (§2.2's `ScalarLoss` typestate) — that's Phase 2 work and didn't exist to call. It also calls `forward`/`parameters` via UFCS (`<_ as Module>::method(..)`) rather than `.method()`, since some layers (`MultiHeadAttention`, `TransformerBlock`) have an inherent `forward` of different arity alongside `Module::forward`, and inherent methods always win over trait methods in `.method()` resolution regardless of arity.
- Applied to 10 of the 11 parameterized layers; skipped `Embedding`, whose `Module::forward` is a deliberate panic stub (`Embedding::forward` takes `&[usize]`, not `&Tensor`, so it can't implement `Module`'s signature) — `assert_trainable!` calls forward through that trait, so it would always panic there. `Embedding`'s existing `test_embedding_backward_flow` already checks exact per-index accumulated gradient values, stronger than "nonzero."
- `Tensor::borrow()`/`borrow_mut()`/`as_ptr()` are `pub(crate)`, which is the actual opacity guarantee (external crates can't reach `inner`), but it also breaks every `tests/*.rs` integration test and `examples/*.rs` example that touched these directly as the old `Rc<RefCell<RawTensor>>` alias (~195 call sites total) — those were **not** fixed this pass (`examples/TODO.md` was referenced here but never actually created; the fix landed directly). `tests/TODO.md` tracked the remaining gap — 8 sites needing a public `grad`-seeding method — and was resolved and removed once `Tensor::set_grad` was added.

**Phase 2 deviations so far** (Phase 2 is in progress; §1.2 `Shape`/`Axis` threading landed in `31e2edd`, `b12efe5`, `567946f`):

- `RawTensor.shape` is still `Vec<usize>`, and `Tensor::shape()` still returns `Vec<usize>`. The `Shape` boundary is the signatures: `edge()` takes `Shape`, and `RawTensor::try_new`/`try_new_with_storage`, `expand`, `pad`, `broadcast_to`, and `try_broadcast_shape` validate through `Shape::from_dims`. Flipping the field is deferred (DECISIONS.md, "Shape boundary").
- Public constructors (`new`, `zeros`, `randn`, `reshape`, `expand`, ...) keep `&[usize]` parameters, so literal shapes stay ergonomic. They still pass through the one checked constructor.
- `Shape::from_dims` rejects zero dimensions and more than `MAX_NUMEL` (100M) elements, as §1.2 sketches. It cannot assert `data.len() == numel()` because it never sees the data; that check lives in `try_new`/`try_new_with_storage` and reports `ShapeDataMismatch`.
- Added `Shape::derived` (`pub(crate)`, panics), which the sketch doesn't have. Op code builds output shapes from tensors that already exist, so an invalid one is an op bug and should fail loudly.
- `Axis` replaces `usize` on `sum_dim`, `try_sum_dim`, `max_dim`, `mean_dim`, `softmax`, `log_softmax`, and `permute` (including the `TensorOps` trait). `GradFn` structs keep `usize` dims internally.
- `Strides` is defined but unused: `RawTensor::compute_strides` still returns `Vec<usize>`.
- Conv specs landed in `src/spec.rs` (`pub(crate)`): `Conv2dSpec`, `ConvTranspose2dSpec`, and `Window2dSpec`, which the sketch does not name. `Window2dSpec` covers `MaxPool2d` and `im2col`/`col2im`. The eight conv GPU wrappers are now `pub(crate)` and take a spec. Spec accessors return `[usize; 4]`, not `Shape` (DECISIONS.md, "Conv-family specs").
- §1.3's `Channel`, `TensorIndex`, and `NodeIndex` newtypes are deferred: named spec fields cover the transposition risk, and the index types wait for the arena (§6).
- `MatmulSpec` landed in `src/spec.rs` with the signature sketched in §1.2. It adds `a_batch`/`b_batch`/`out_batch` fields, covers only rank >= 2 operands of equal rank (the vector forms keep their own arms), and the three GPU matmul wrappers are `pub(crate)` and take it. This retires B4.
- Still open: `PadValue` (§2.4), then the rest of Phase 2.

---

## Patterns explicitly declined (so the next reviewer knows they were considered)

- **Full device/dtype typestate on `Tensor`** (§2.1) — runtime dynamism is load-bearing here (`to_device` mid-graph); burn/candle/PyTorch concur.
- **`Variable`/`Tensor` split** (§2.2) — PyTorch 0.4's lesson; the bounded `ScalarLoss` version takes the real benefit.
- **Const-generic rank `Tensor<const R: usize>`** — infects every signature, breaks shape-agnostic user code, saves nothing the `Shape` newtype doesn't.
- **Proc-macro op DSL** (§5.5) — dependency + debugging tax unjustified at this op count.
- **`Arc<Mutex>` today** — nothing is `Send`; the arena (§6) is the correct route to threading, and premature `Mutex`ing is how `BufferPool` ended up guarding state one thread can reach (Part II §1).
- **`async` GPU API** — the synchronous-blocking design is coherent; make transfers explicit (§7) before making them async.

The one-sentence summary of this whole document: **Volta's bugs were all writes that shouldn't have been possible, made silently** — so the fix is not more care, it's removing the write, typing the value, and making the silence loud.

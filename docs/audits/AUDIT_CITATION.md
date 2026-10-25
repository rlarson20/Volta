# Audit citation ledger

This is a durable research ledger for the findings in `OPUS_5_EVAL_AUDIT.md` and `notes/TODO.md`.
It distinguishes (1) a locally verified defect from (2) the external contract or reference implementation that establishes the intended behavior.
Sources are primary documentation, specifications, papers, or the upstream implementation where practical.
Accessed 2026-09-06.

## Scope and traceability

`OPUS_5_EVAL_AUDIT.md` is the local evidence source for whether each reported Volta behavior exists (including reproduced outputs and source locations).
This ledger supplies the missing external sources for the intended contract and repair pattern.
The mapping below makes the coverage explicit rather than implying that a framework reference proves a local code observation.

| Audit/TODO area                                                              | Local evidence                                                      | Intended contract / repair source in this ledger               |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------- |
| B1–B3 graph dead ends                                                        | [audit](/Users/rjlarson/src/volta/OPUS_5_EVAL_AUDIT.md) §§B1–B3     | B1–B3                                                          |
| B4 unequal-rank matmul                                                       | [audit](/Users/rjlarson/src/volta/OPUS_5_EVAL_AUDIT.md) §B4         | B4                                                             |
| B5 CPU-only build, CI cfg                                                    | [audit](/Users/rjlarson/src/volta/OPUS_5_EVAL_AUDIT.md) §B5 and §CI | B5                                                             |
| H1–H8 layer/kernel/state defects                                             | [audit](/Users/rjlarson/src/volta/OPUS_5_EVAL_AUDIT.md) §§H1–H8     | H1–H8                                                          |
| Copies, convolution cost, GPU reduction/broadcast, gradcheck, tests          | [audit](/Users/rjlarson/src/volta/OPUS_5_EVAL_AUDIT.md) §3–§5       | Medium findings                                                |
| RNG, dead config, dtype/device, Embedding, Sequential keys                   | [audit](/Users/rjlarson/src/volta/OPUS_5_EVAL_AUDIT.md) §3          | Medium findings                                                |
| RefCell/public-state, traits, errors, unsafe, packaging, preventative design | [audit](/Users/rjlarson/src/volta/OPUS_5_EVAL_AUDIT.md) §7–§15      | Rust architecture and safety findings, plus B5/Medium findings |

The citations name the source used for each normative statement.
Where there is no single standard (for example, a choice between returning an error or implementing a general GPU kernel),
the ledger labels the alternatives and states the invariant that all acceptable solutions must preserve.

## B1–B3 — autograd graph construction through parameterized layers

**General invariant.** [PyTorch's autograd mechanics](https://docs.pytorch.org/docs/main/notes/autograd.html)
states the relevant forward rule explicitly:
record an operation when **at least one input** requires a gradient;
in backward, accumulate only into the leaf inputs whose `requires_grad` is true.
Its [autograd tutorial](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial)
also describes the output as requiring gradients when any one input does.
This is the correct rule for Volta's graph-edge constructor: use
`x.requires_grad || weight.requires_grad || bias.requires_grad`, and have the
backward function independently skip accumulation for non-requiring parents.
Gating an operation solely on the activation therefore incorrectly disconnects
trainable parameter leaves whenever the training data does not require grad.

### B1 — `ConvTranspose2d`

[PyTorch `ConvTranspose2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html)
is the conventional reference operator; it is a parameterized module whose
transposed-convolution forward is differentiable. The general invariant above
is the decisive source here: its `weight` and optional `bias` are differentiable
inputs to the layer's forward, so their gradients must exist even when the data
input is a non-differentiable leaf. The local defect is thus a graph-building
error, not a special property of transposed convolution.

### B2 — `LSTMCell` gate slicing

[PyTorch's `LSTMCell` definition](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html)
gives the four gates and the equations that consume all four gate slices, and
identifies `weight_ih`, `weight_hh`, `bias_ih`, and `bias_hh` as learnable
parameters. A gate-slice operation must preserve an edge to the combined
preactivation (its backward is scatter/add into the corresponding slice);
copying its values into a fresh leaf has the same semantics as a detach and
cannot differentiate those equations to any of the four parameter tensors.
Use an existing differentiable narrow/slice primitive, or implement
`SliceGradFn` with `grad_input[start..end] += grad_output` for each batch row.

### B3 — positional encoding

The general graph invariant establishes that the final addition must be a graph
operation, not raw-data reconstruction. For learned positional encoding, the
[PyTorch `Embedding` contract](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding)
calls `weight` learnable and documents that only `padding_idx` deliberately
does not contribute gradients. The correct learned path is consequently:
generate positions → differentiable embedding lookup → broadcast/add to `x`.
For fixed sinusoidal encoding, keep the table as a non-grad constant but still
perform the final broadcast-add in the graph so the gradient reaches `x`.

## B5 — CPU-only feature build

[The Cargo Book's feature reference](https://doc.rust-lang.org/cargo/reference/features.html)
defines features as conditional-compilation switches and optional-dependency
selectors; [the Rust Reference](https://doc.rust-lang.org/reference/conditional-compilation.html)
states that `#[cfg]` conditionally includes the item to which it is attached.
Accordingly, an advertised optional `gpu` feature requires the complete
non-GPU closure to type-check when the feature is absent. The reference pattern
is to gate GPU-only imports, types, methods, and calls at their definition/use
boundaries, while exposing a separately type-correct CPU path. A CI job that
runs `cargo build --no-default-features` is the direct regression fence.

## H1 — MaxPool2d padding identity

[PyTorch `MaxPool2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.pooling.MaxPool2d.html)
specifies that nonzero padding is **implicit negative infinity** padding. That
is also the mathematically required identity for a maximum: `max(x, -∞) = x`.
Zero is not an identity for negative inputs and changes both the selected value
and its argmax/backward route. Fix this operation specifically—not generic
padding—by either evaluating out-of-input kernel locations as `f32::NEG_INFINITY`
or making padding carry an explicit `PadValue`/fill value.

## H2 — stable binary cross entropy with logits

[PyTorch `BCEWithLogitsLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
states that combining sigmoid and BCE uses the log-sum-exp trick for numerical
stability. Its stable scalar loss is algebraically equivalent to

```text
max(x, 0) - x * target + log1p(exp(-abs(x)))
```

(equivalently, the two-pass formulation in the
[upstream discussion of its implementation](https://discuss.pytorch.org/t/numerical-stability-of-bcewithlogitsloss/8246)).
This avoids overflowing `exp(x)` for a large positive `f32` logit. Probability
space BCE must likewise either validate/clamp its domain deliberately or use
the logits formulation; direct `log(p)` / `log(1-p)` has singular endpoints.

## H3 — unsafe GPU cache and dtype accessors

The local audit establishes the contradiction in the cache's stated invariant;
the relevant external standard is [Rust's safety-comment policy](https://std-dev-guide.rust-lang.org/policy/safety-comments.html):
the comment must name invariants that actually make the block safe. The proper
fix is representation-level: expose a host readback as an owned value or a
cache whose reference lifetime is structurally tied to immutable storage; allow
invalidation only through exclusive mutation. A [`OnceLock`](https://doc.rust-lang.org/std/sync/struct.OnceLock.html)
is appropriate only if the cache is genuinely write-once, since its
`get_or_init` operation is exactly-once initialization.

For GPU dtype access, support the stored/device dtype with a checked conversion
or a fallible `UnsupportedDType` error. Do not transmute/reinterpret `f32`
bytes into `f64`, `f16`, or `bf16`: that is neither numeric conversion nor a
valid tensor value transformation. Removing unreachable accessors is safer
than retaining a public path whose meaning is wrong when it becomes reachable.

## H4 — ordinary-data panic boundaries

- Float NaNs are not a total order. Rust's [`f32` documentation](https://doc.rust-lang.org/stable/core/primitive.f32.html)
  distinguishes `total_cmp` from `PartialOrd`; code that unwraps
  `partial_cmp` is therefore not a valid public max policy for arbitrary
  floating inputs. Pick and document one semantic policy (propagate NaN,
  ignore NaNs as `maxNum`, or return `Result`); do not unwrap.
- Xavier/Glorot initialization is defined per **weight element**. PyTorch's
  [`nn.init` documentation](https://docs.pytorch.org/docs/stable/nn.init.html)
  describes the fan-in/fan-out convention, and [Glorot & Bengio (2010)](https://proceedings.mlr.press/v9/glorot10a.html)
  is the primary source for the scheme. The generator must allocate
  `shape.iter().product()` values; fan counts determine only the distribution
  bound. For convolutional weights, standard fan computation includes
  receptive-field dimensions.
- A stride of zero is invalid input, so validate it before output-size division
  and return the library's structured error. This turns a host-language panic
  into an actionable API failure.

## H5 — GPU rank limits and shader indexing

[WGSL's normative specification](https://www.w3.org/TR/WGSL/) says an
out-of-bounds indexing access generates a dynamic error and makes shader
authors responsible for dynamic-index bounds. A host encoder must therefore
never pass a logical rank greater than the dimensions physically encoded in a
`vec4<u32>`. The correct design is one of: validate `rank <= 4` and return an
error; dispatch a rank-generic/array-backed shader; or use an explicitly
correct CPU fallback. Truncating or left-aligning a rank-5 shape while passing
the original rank has no valid WGSL interpretation.

## H6 — TransformerBlock contract

The original Transformer specification, [_Attention Is All You Need_](https://arxiv.org/abs/1706.03762),
defines encoder/decoder sublayers around multi-head attention, residual
connections, and layer normalization. Its architecture figure and §3 make a
useful acceptance contract: an advertised Transformer block must apply its
attention transform, honor causality when it presents a causal API, preserve
all attention state through state-dict round trips, and apply the documented
residual/norm ordering exactly once. PyTorch's
[`MultiheadAttention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
is a practical reference: `is_causal` applies a causal attention mask. A
placeholder identity module with ignored causal input is not a degraded
implementation of that contract; it is a different layer and should either be
removed from the public Transformer API or replaced with the existing real
attention component after disentangling its residual/norm responsibilities.

## H7 — state-dict loading must preserve destination representation

[PyTorch's `load_state_dict` recipe](https://docs.pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html)
documents the normal behavior: copy checkpoint values into the corresponding
parameter/buffer already owned by the module (normally in-place). That preserves
the destination object's device, dtype, identity, `requires_grad`, and shape
invariants; a shape mismatch is an error rather than an unvalidated storage
replacement. Volta should therefore validate the incoming tensor shape/element
count before mutation and copy/move the incoming values to the existing
parameter's device. It must never overwrite a GPU parameter's storage with a
CPU storage while retaining a GPU device tag.

## H8 — Muon reference algorithm and GPU safety

The primary implementation, [Keller Jordan's Muon repository](https://github.com/KellerJordan/Muon),
defines Muon for hidden matrix weights and routes gains, biases, embeddings, and
heads to AdamW. The original speedrun implementation gives concrete code for
the five-step quintic Newton–Schulz orthogonalization:
[`zeroth_power_via_newtonschulz5`](https://github.com/KellerJordan/modded-nanogpt).
It normalizes by `G.norm() + eps`, transposes tall matrices for the iteration,
uses the tuned coefficients `(3.4445, -4.7750, 2.0315)`, restores orientation,
and returns the original dtype. Therefore Frobenius normalization is not the
defect; it is a convergence-basin bound used by the reference. The local
quadratic iteration and unexplained `0.1` scale are a different algorithm.

For a correct Volta implementation, make the supported parameter/device set
explicit: accept 2-D CPU/GPU matrices and dispatch all algebra and parameter
updates through backend operations; route unsupported parameters to a specified
optimizer or return `Result::Err`. Do not obtain a mutable CPU slice from a GPU
parameter and panic. The upstream optimizer's separate parameter groups are a
concrete model for the policy.

## Medium findings — concrete implementation patterns

### Storage, views, and CPU/GPU ownership

The local deep-copy findings call for an ownership/view redesign, not a
micro-optimization. Make immutable CPU backing shareable (`Arc<[u8]>` or a
typed `Arc<[f32]>`) and use copy-on-write only where mutation is required;
represent broadcast as a view with zero stride (or preserve a compact
broadcast descriptor) until a backend requires materialization. For the GPU
readback cache, a one-time cell is a concrete way to make the stated
"initialized once" invariant executable: [Rust `OnceLock::get_or_init`](https://doc.rust-lang.org/std/sync/struct.OnceLock.html)
guarantees only one successful initialization. If cache invalidation is a real
requirement, model it explicitly with exclusive mutation instead of claiming it
never happens; never hand out a reference whose safety relies on a contradicted
comment.

Every tensor operation should preserve a device policy. A mixed-device binary
operation must either transfer according to a documented explicit policy or
return an error; silently choosing CPU is neither a valid view nor a valid
backend dispatch policy. GPU accessors must expose only the dtypes that the
GPU storage actually stores—do not reinterpret byte slices as a different
numeric type.

### Convolution / pooling / reduction kernels

The standard backward relationship is concrete: input gradient for a
convolution is a transposed convolution with the saved weight, and weight
gradient is a convolution/correlation of saved input with upstream gradient.
PyTorch's [custom convolution backward tutorial](https://github.com/pytorch/tutorials/blob/main/intermediate_source/custom_function_conv_bn_tutorial.py)
shows those formulas. An implementation should iterate the at-most `K_h*K_w`
valid receptive-field contributions for each input-gradient location (or use
`col2im`/a tiled GEMM), not scan every output coordinate and test membership.
Likewise, load each input tile once per workgroup where using shared memory;
the existing 16×16 tiled GPU matmul is a suitable in-repo model.

GPU reduction should use parallel workgroups and a tree/warp reduction, then a
second reduction or atomic/segmented stage as needed; a `workgroup_size(1)`
serial loop is correct only as a fallback. For broadcast-gradient reduction,
map each output coordinate to an input coordinate and use supported atomics or
parallel segmented reduction; scanning all outputs once for every target is
quadratic. The host-side rank/length checks in H5 are prerequisites for either
implementation. This workgroup/shared-memory pattern follows the GPU execution
model documented in NVIDIA's [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)
and is directly portable in structure to WGSL workgroups.

### Numerical and regression testing

[PyTorch's gradcheck note](https://docs.pytorch.org/docs/stable/notes/gradcheck.html)
documents the appropriate oracle: compare analytic backward with central finite
differences, `(f(x+eps)-f(x-eps))/(2*eps)`, elementwise. A gradcheck harness
must clear pre-existing leaf gradients and avoid mutating stateful layers during
probe evaluations (snapshot/restore running statistics or run them in a frozen
mode), otherwise it is not comparing one derivative to one numerical estimate.

Use varied, signed test values and randomized shapes rather than all ones, then
add fixed regression tests for each discovered edge case: non-grad layer input
with trainable parameters (B1–B3); unequal-rank matmul; negative padded maxpool;
rank-5 GPU broadcast rejection/fallback; CPU-only features; Sequential keys
containing both `1.*` and `10.*`; checkpoint load on an existing GPU tensor.
[Proptest](https://github.com/proptest-rs/proptest) is suitable for the broader
shape/value matrix because it generates arbitrary constrained inputs and shrinks
a failure to a minimal reproducer. It supplements—not replaces—those specific
regressions.

### Determinism, serialization, and API validation

A seed promise requires all random consumers to use the same injected RNG (or
deterministically derived child streams). The [`SeedableRng` contract](https://docs.rs/rand/latest/rand/trait.SeedableRng.html)
describes reproducibility for a fixed seed; [`ThreadRng`](https://docs.rs/rand/latest/rand/rngs/struct.ThreadRng.html)
is automatically seeded from system randomness. Therefore a fallback to
`rand::rng()` and DataLoader shuffling outside the seeded source violates the
API promise. Return a borrow/reentrancy error or structure the RNG ownership so
the fallback is impossible.

State-dict key parsing should split at an exact segment boundary (for example,
parse `usize`, then require `.`) rather than use a string prefix. Validate
every deserialized shape before copying data. Remove or compile-test dead
configuration code; a module that is intentionally excluded from the crate has
no build guarantee and is not an API.

Embedding needs a distinct indexed-input contract, rather than implementing a
general `Module<Tensor>` method that deliberately panics. A typed
`IndexedLayer`/`Embedding::forward(&[usize])` interface makes invalid placement
in a tensor-only `Sequential` unrepresentable. The reference embedding API
accepts arbitrary-shaped integer indices and returns the index shape plus
embedding dimension ([PyTorch `Embedding`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding)).

## Rust architecture and safety findings

These findings are primarily local-code observations; the sources below anchor
the proposed Rust remedies rather than pretending an external source can prove
a line count or a specific field write.

**Encapsulation and graph state.** Keep `RawTensor`/autograd node fields private
behind an opaque `Tensor` API, and make state transitions named, validated
methods (construct, attach edge, set device/storage together, accumulate grad).
The goal is to make independent public mutation of shape, storage, graph edges,
and gradient flags impossible. Public fields are API commitments under the
[Rust privacy RFC](https://rust-lang.github.io/rfcs/0001-private-fields.html),
and the Rust API Guidelines explicitly recommend private struct fields
([C-STRUCT-PRIVATE](https://rust-lang.github.io/api-guidelines/checklist.html)).
`RefCell` is not a substitute for that API
boundary: [its documentation](https://doc.rust-lang.org/core/cell/struct.RefCell.html)
states that incompatible dynamic borrows panic, and supplies `try_borrow` for
an explicit fallible case. Thus a long-lived `borrow()` across an operation
is a correctness hazard; clone/copy the data needed, drop the borrow, compute,
then take a short mutable borrow only to publish the result.

**Unsafe boundary.** The [Rust standard-library safety-comment policy](https://std-dev-guide.rust-lang.org/policy/safety-comments.html)
requires every unsafe block to explain the actual invariants used to make it
safe; [the Rust Reference](https://doc.rust-lang.org/reference/unsafe-keyword.html)
likewise requires unsafe functions to document extra safety conditions. For
H3, correct the code/type representation first, then make the comment prove
only invariants actually enforced by the type and borrow signatures. A false
`SAFETY:` statement is not harmless documentation debt: it gives the next
refactorer a false premise for modifying unsafe code.

**Public API evolution.** Apply [`#[non_exhaustive]`](https://doc.rust-lang.org/reference/attributes/type_system.html)
to public, intentionally growable error/config enums and structs; downstream
callers must then leave room for new variants/fields while the crate retains
internal exhaustive matching. Seal traits intended only for crate-provided
implementations, split incompatible input contracts (`Layer`, `IndexedLayer`,
`RecurrentLayer`) instead of using a generic trait plus panic, and consolidate
the operation-construction path. These are structural remedies for the
"three call styles," decorative Module/Embedding implementation, public grad
machinery, and duplicated graph-attach rituals documented in the audit.

**Errors, cfg, and diagnostics.** For public data-dependent failures use one
structured `Result<_, VoltaError>` regime; reserve `panic!`/`assert!` for
internal invariant violations after validated inputs. Feature-specific code
must compile in every advertised feature set per the Cargo sources in B5.
Kernel and fallback paths should return/emit an explicit unsupported-device or
unsupported-rank error rather than silently produce CPU results/zeroes. The
test matrix should compile default, no-default-features, and each supported
feature combination, then execute CPU-safe tests in each relevant configuration.

## B4 — different-rank batched `matmul` backward

**External contract.** [NumPy's `matmul` specification](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)
defines an N-D operand as a stack of matrices in its _last two_ axes and says
that the preceding axes are broadcast together. Thus `[2,3,4] @ [2,1,4,6]`
is a valid operation: the matrix dimensions are `(3,4) @ (4,6)`, and the batch
dimensions `[2]` and `[2,1]` broadcast (right aligned) to `[2,2]`; the result
is `[2,2,3,6]`. This is also the documented `torch.matmul` contract
([PyTorch reference](https://docs.pytorch.org/docs/stable/generated/torch.matmul.html)).

**Derivation of the correct backward.** Write the broadcasted operands as
`A' : B × M × K`, `B' : B × K × N`, and upstream gradient
`G : B × M × N`; `B` here denotes the _fully broadcast result batch shape_,
not one input's raw batch shape. For each broadcasted batch coordinate,

```text
dA' = G @ transpose(B', -1, -2)       # B × M × K
dB' = transpose(A', -1, -2) @ G       # B × K × N
```

Then reduce (`sum_to_shape`) every axis introduced by left-padding a lower-rank
operand and every axis whose original dimension was `1`; reshape to that
operand's original shape. This is the chain rule for `C=A'B'` and the
broadcasting rule above—not an index derived from the other operand's rank.
PyTorch's derivative registry is a concrete upstream reference for the same
pattern: its batched-matrix derivatives form the transposed matmul and its
autograd machinery reduces expanded/broadcast gradients back to the original
sizes ([`derivatives.yaml`](https://github.com/pytorch/pytorch/blob/main/tools/autograd/derivatives.yaml),
[`sum_backward`](https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/FunctionsManual.cpp)).

**Implementation choices.** Either implement that complete `broadcast → matmul
→ sum_to_shape` algorithm, including right-aligned shape validation, or reject
unequal ranks at forward validation until it exists. Merely reading `N`, `M`,
or `K` with the other operand's rank cannot satisfy the stated contract.

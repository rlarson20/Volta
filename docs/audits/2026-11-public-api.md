# 2026-11 Public API audit

Dated record. Original text is never edited or removed; corrections are appended under `## Corrections` (`.claude/rules/audits.md`).

**Date:** 2026-11-17. **Commit audited:** `ebbd683` (`improvements`), crate version 0.3.0. **Issue:** #299. **Finding IDs:** `PA1`–`PA14` (`PA`, not `P`, so they do not collide with the `P0`–`P3` priority labels).

**Scope.** Everything reachable from `src/lib.rs`: what is exported that should not be, internal types in public signatures, naming, panic versus `Result` against invariant 8, semver hazards, and whether the docs are true. Folds in #137 (method names and operators), #237 (`RawTensor` as a namespace), #100 (rank-0 tensors) and #247 (wgpu limits) by reference instead of repeating them.

**Method.** The surface was listed from rustdoc JSON (`cargo +nightly-2026-11-01 doc` with `--output-format json`), once with default features and once with `--no-default-features`, and walked with a script from the crate root through every `pub mod` and `pub use`. `cargo public-api` is not installed here; the JSON is the same data. Behavior claims were probed from an integration test (an external crate, so only the public API is visible), with each call under `catch_unwind`. Probes are in the appendix, not committed (invariant 6).

**What ran.** Linux, debug build, CPU and lavapipe for the one GPU-labelled probe. Nothing here depends on Metal.

**Result.** The surface is documented (rustdoc JSON shows no public item without a doc comment in either feature set) and the `try_` constructors follow invariant 8. The problems are width and leakage: 290 public paths for 123 definitions, an internal struct exported with public fields, wgpu and storage internals in signatures, a feature that removes root functions, and a set of enums and structs that cannot grow without a breaking change. Nothing here is a wrong answer; U1–U3 in the soundness audit are the behavioral cousins of PA1, PA3 and PA7.

| ID | Before `main`? | One line | Evidence |
| --- | --- | --- | --- |
| PA1 | Yes | `RawTensor` is exported with six `pub` fields, including `parents` and `grad`, that no outside code can reach | rustdoc JSON |
| PA2 | Yes | `volta::new_tensor` is a type alias for `RawTensor` that reads as a function | JSON + probe |
| PA3 | Yes | Internals in public signatures: `&wgpu::Buffer`, `GpuBuffer`, `GpuStorage`, `HostBuf`'s `Arc<[T]>` variants, and a `Storage` enum whose `Gpu` variant appears only with the `gpu` feature | JSON, both feature sets |
| PA4 | Yes | 167 re-exports of 123 definitions: most items have two to four public paths, and the root set is uneven | JSON |
| PA5 | Yes | `is_gpu_available`, `gpu_sync` and five more root functions exist only with the `gpu` feature | JSON diff |
| PA6 | Partly | User-supplied arguments panic with no `try_` form on `Tensor` and on every layer constructor; about 20 `Tensor` methods that panic have no `# Panics` | JSON + probe |
| PA7 | Partly | Six constructors accept an argument that fails later or never | probe |
| PA8 | Decide | `Module::load_state_dict` is lenient by default: an empty state dict loads nothing and returns `Ok` | probe |
| PA9 | No (#137) | Naming: three call styles per op, four spellings of "constructor with options", three receivers for `zero_grad` | JSON, reading |
| PA10 | Yes (cheap) | Nine public enums and four public-field structs cannot grow without a breaking change; `Module` and `Optimizer` are open traits | reading |
| PA11 | No | `data()`, `item()` and `grad()` decode every dtype to `f32`: an `i64` 9007199254740993 reads as 9.007199e15 | probe |
| PA12 | Decide | Test helpers in the public API: `assert_trainable!`, `check_gradients`, `check_gradients_simple` | JSON + probe |
| PA13 | Yes (docs) | The crate-level doc and `RawTensor`'s doc describe an older, smaller library | reading |
| PA14 | No | What a user will look for and not find | JSON |

## Findings

### PA1. `RawTensor` is exported with public fields

`volta::RawTensor` (`src/tensor.rs:417`) has six `pub` fields in the rustdoc output: `data`, `shape`, `grad`, `parents`, `requires_grad`, `device`. `Tensor` is an opaque newtype over `Rc<RefCell<RawTensor>>` whose `borrow()` and `borrow_mut()` are `pub(crate)`, and every `RawTensor` constructor returns a `Tensor`, so outside code cannot obtain a `RawTensor` value or reference. The fields are therefore unreachable today.

They are still published surface, and three invariants in `CLAUDE.md` depend on them staying unreachable: only `edge()` writes `parents` (1), a `borrow_mut()` outside the core means a missing method (2), only `accumulate_grad` and `set_grad` write `grad` (3). One convenience accessor that hands out `&mut RawTensor` would void all three without a compile error. `RawTensor::invalidate_gpu_cache(&mut self)` is the same kind of item: a public method no outside caller can invoke.

**Fix.** Make the fields `pub(crate)`. It cannot break a user, because no user can name a `RawTensor` value. Whether `RawTensor` stays the namespace for constructors and ops is #237.

### PA2. `volta::new_tensor` is a type alias

`src/tensor.rs:2377` is `pub use RawTensor as new_tensor;`, re-exported at the root. It is a struct under a lower-case name, next to real functions `zeros`, `ones`, `rand`. Confirmed: `volta::new_tensor::new(vec![1.0, 2.0], &[2], true)` compiles and is `RawTensor::new`; `new_tensor(..)` does not compile. It also doubles `RawTensor`'s 62 methods in the docs.

**Fix.** Delete it, or replace it with `pub fn new_tensor(data, shape, requires_grad) -> Tensor`.

### PA3. Internals in public signatures

| Item | What leaks | Consequence |
| --- | --- | --- |
| `volta::GpuBuffer::buffer() -> &wgpu::Buffer` | A wgpu 30 type | Every wgpu major version becomes a Volta breaking change; and U3 (a user can `destroy` a pooled buffer) |
| `volta::GpuBuffer` (`from_slice`, `zeros`, `to_vec`, `copy_region`), also at `volta::gpu::GpuBuffer` | The device buffer itself | U2 (`copy_region` reads pool slack); no example or test outside `src/gpu/` uses these |
| `volta::Storage::gpu_buffer() -> Option<&GpuBuffer>` | The same, from any tensor's storage | As above |
| `volta::storage::GpuStorage` | A struct with no public methods, public only because `Storage::Gpu(Arc<GpuStorage>)` is a public variant | Representation locked |
| `volta::storage::HostBuf` variants (`F32(Arc<[f32]>)`, ...) | The host representation (`Arc` slices, one variant per dtype) | Changing the buffer type is breaking; `HostBuf` is `#[non_exhaustive]`, which covers new dtypes only |
| `volta::Storage` (`Cpu(HostBuf)`, `Gpu(Arc<GpuStorage>)`) | Not `#[non_exhaustive]`, and `Gpu` exists only with the `gpu` feature | `match s { Storage::Cpu(_) => .. }` compiles without `gpu` and stops compiling when any crate in the build turns `gpu` on. Features are meant to be additive |
| `volta::storage::Element` (`view`, `view_mut`, `wrap` over `HostBuf`; impls for `half::f16`, `half::bf16`) | Sealed, so safe, but its methods are public plumbing, and `half`'s types are in the API | A `half` major bump is breaking (acceptable; say so in the docs) |

**Fix.** `pub(crate)` for `GpuBuffer`'s constructors and `buffer()`, `Storage::gpu_buffer`, `GpuStorage` and `Element`'s three methods (keep `DTYPE`). Make `Storage` an opaque struct, or at least `#[non_exhaustive]` with the `Gpu` variant present in both feature sets. `Storage` is needed publicly (`set_grad`, `grad_storage`, `Storage::cpu`, `from_typed`, `as_host`), its variants are not.

### PA4. Two to four public paths per item

The default-feature surface has 290 top-level public paths: 123 definitions and 167 re-exports. A layer is reachable four ways:

```
volta::Conv2d
volta::nn::Conv2d
volta::nn::layers::Conv2d
volta::nn::layers::conv::Conv2d
```

The same holds for all 19 layers, the three optimizers, the transformer types (three paths each), the op enums (`volta::BinaryOp`, `volta::ops::BinaryOp`, `volta::ops::binary::BinaryOp`), `volta::mapping` beside `volta::io::mapping`, and `volta::data::load_mnist_images` beside `volta::data::mnist::load_mnist_images`. Every path is semver surface: moving `conv.rs` is a breaking change today.

The root is also uneven. It has every layer but not `FeedForward`, `MultiHeadAttention`, `TransformerBlock`, `PositionalEncoding` or `causal_mask` (only under `volta::nn`); it has `autocast` but not `no_grad` (`volta::autograd::no_grad`); `softmax`, `sum_dim` and `max_dim` as free functions but not `mean_dim` or `log_softmax`; `TypedTensorData` but not `TensorData` or `StateDict`; five op enums but not `CmpOp`.

**Fix.** Make the leaf modules private (`mod conv; pub use conv::{Conv2d, ConvAlgo};`), so each item has one canonical path plus, at most, the root re-export. Then pick the root set by rule (for example: everything a README example imports) instead of by accretion.

### PA5. The `gpu` feature removes root functions

With `--no-default-features`, 46 public paths disappear. Most are `GpuBuffer` and should (PA3), but seven are functions a portable program calls: `volta::is_gpu_available`, `gpu_sync`, `gpu_cleanup`, `gpu_compact`, `gpu_pending_count`, `gpu_pool_stats`, `gpu_sync_threshold`, and `volta::gpu::invalidate_all_tensor_caches`. Code that asks `volta::is_gpu_available()` does not compile against a CPU-only build, so every caller needs its own `cfg`, which is what `.claude/rules/gpu.md` forbids inside the crate ("a gated real function plus a not-gpu stub"). `Device::gpu()` already does it right: it returns `None` without the feature.

**Fix.** Stubs for the seven: `is_gpu_available() -> false`, `gpu_sync() -> true`, the stats functions `None` or 0. Then the two feature sets differ only in types that are GPU-specific.

### PA6. User-supplied arguments that panic, and missing `# Panics`

Invariant 8 says a user-supplied shape, axis or range is validated and returned as a `VoltaError`, with the panicking form wrapping a `try_` one. The constructors follow it (`try_new`, `try_zeros`, `try_rand`, `try_from_vec`, optimizers' `try_new`). The rest of the surface is uneven:

| User argument | Panicking form | `try_` form |
| --- | --- | --- |
| reshape, expand | `Tensor::reshape`, `expand` | `Tensor::try_reshape`, `try_expand` |
| ranges, padding, strides | `Tensor::shrink`, `pad_with`, `stride_op` | Only as `RawTensor::try_shrink(&t, ..)`, `try_pad_with`, `try_stride_op`; none on `Tensor` |
| axis of a reduction | `Tensor::sum_dim` | `RawTensor::try_sum_dim` only |
| axis | `Tensor::max_dim`, `mean_dim`, `softmax`, `log_softmax` | None |
| axes | `Tensor::permute` | None |
| layer sizes and rates | `Linear::new`, `Conv2d::new`, `Dropout::new`, `Embedding::new`, `PixelShuffle::new`, `MultiHeadAttention::new`, `PositionalEncoding::new`, `causal_mask`, `GradScaler::with_params` | None |
| index values | `Embedding::forward` (index out of range), `cross_entropy_loss` (class out of range) | None |

All of the panicking forms were confirmed to panic with a message that names the argument (appendix). Two message formats are in use: Display (`Stride: Invalid parameter: stride_op: stride 0 for dimension 0`) and a Debug-printed error (`Invalid shrink ranges: DimensionOutOfBounds { dim: 0, shape: [2, 3, 4] }`, `Dimension out of bounds: DimensionOutOfBounds { .. }`).

About 20 `Tensor` methods that panic carry no `# Panics` section (rustdoc JSON, `[--D]` in the appendix): the binary ops (`add`, `sub`, `elem_mul`, `div`, `max_elem`, `modulo`, the six comparisons, `mulacc`) on a shape, device or strict-dtype mismatch; `matmul`, `transpose`; `permute`, `pad_with`, `shrink`, `stride_op`; `sum_dim`, `max_dim`, `mean_dim`, `softmax`, `log_softmax`. `RawTensor::softmax`, `log_softmax` and `unary_op` have none either. Each is a one-line forward, so the section can point at the `RawTensor` function that has it.

**Before `main`:** the `# Panics` sections (documentation, and the merge gate says the library documents every item truthfully). **Can wait:** the `try_` forms, which are additive.

### PA7. Arguments accepted at construction that fail later or never

Confirmed by probe (CPU):

| Call | Result |
| --- | --- |
| `Conv2d::new(1, 1, 3, 0, 0, true)` (stride 0) | Accepted; `forward` panics `stride must be non-zero` |
| `MaxPool2d::new(0, 1, 0)` (kernel 0) | Accepted |
| `LayerNorm::new(vec![])` | Accepted |
| `LayerNorm::new_with_eps(vec![4], -1.0)` | Accepted, and `forward` runs (a negative epsilon under a square root) |
| `BatchNorm2d::new_with_params(3, 1e-5, 7.0)` (momentum 7) | Accepted |
| `Storage::new_zeros(4, &Device::GPU("nope".into()))` | Accepted on lavapipe: the label is not checked, though `Tensor::to_device` with the same device panics `no such GPU` |

`Device::GPU(String)` is a public variant, so any string is a constructible device; `Device::gpu()` is the checked way to get one. Together with U1 (`set_grad` and length) these are the places where a wrong argument is not caught at the call that received it.

**Fix.** Validate in the constructor, as `Dropout::new`, `Embedding::new` and `PixelShuffle::new` already do. For `Device`, either make the `GPU` variant's payload private or check it wherever a `Device` enters.

### PA8. `load_state_dict` is lenient by default

`Module::load_state_dict` (`src/nn/mod.rs:77-94`) ignores keys the module does not have and leaves parameters with no entry unchanged. Its doc says so, and `io::load_state_dict_checked` returns the diff. Confirmed: `Linear::new(2, 2, true).load_state_dict(&StateDict::new())` returns `Ok(())` and loads nothing.

This is documented behavior, not a bug. It is listed because PyTorch's default is `strict=True`, the project's stated reference is PyTorch, and here the safe form has the longer name and lives in another module. A typo in a key prefix produces a model that silently keeps its random initialization, the failure mode S-series audits spent the most effort on.

**Decide before `main`:** make the trait method strict (missing or unexpected keys are an `Err` listing them) and give the lenient form the explicit name (`load_state_dict_partial`, next to `io::mapping::load_partial`), or keep it and say so in `DECISIONS.md`. Changing it later is a silent behavior change for users.

### PA9. Naming

Folded into #137; recorded here so that issue has the full list.

- **Three call styles per op.** `RawTensor::softmax(&t, Axis(1))`, `t.softmax(Axis(1))`, and `volta::softmax(&t, Axis(1))` are all public, for a different subset of ops each. The `Tensor` methods are the documented style (`.claude/rules/autograd.md`); the other two are extra surface.
- **A second API to every op.** `RawTensor::unary_op(&t, UnaryOp::Exp)`, `binary_op`, `compare`, `reduce_op`, `ternary_op` and the op enums are public, so `t.exp()` and `RawTensor::unary_op(&t, UnaryOp::Exp)` are both supported forever. `MovementOp`'s variants carry public `Vec<usize>` fields.
- **Method names.** `elem_mul`, `elem_eq`, `elem_ne`, `max_elem`, `where_op`, `stride_op`, `max_reduce`, `modulo`, `mulacc` against `add`, `sub`, `div`, `lt`, `sum`, `mean`.
- **Constructors.** `RawTensor::new(data, shape, requires_grad)` takes the flag; `from_vec(data, shape)`, `from_typed(data, shape)` and `constant` do not. "With options" is spelled `new_with_params` (BatchNorm), `new_with_eps` (LayerNorm), `with_params` (GradScaler) and `new_on_device` (Linear, Conv2d).
- **Receivers.** `Module::zero_grad(&mut self)`, `Optimizer::zero_grad(&self)`, `Tensor::zero_grad(&self)`. `Module::to_device(Device)` and `Tensor::to_device(Device)` take the device by value, `Storage::to_device(&Device)` by reference.
- **Positional same-typed arguments.** `SGD::new(params, lr, momentum, weight_decay)` is three `f32`s; `Conv2d::new(in, out, kernel, stride, padding, bias)` is five `usize`s. Swapping two compiles.
- **Shapes.** Every public function takes `&[usize]` (and `LayerNorm::new` a `Vec<usize>`), where `.claude/rules/autograd.md` says not to add loose shape parameters; `Shape` exists and is returned by `Tensor::shape()`.
- **Two structs with the same fields.** `io::TensorData` and `io::TypedTensorData` are both `{ data: Vec<u8>, dtype: DType, shape: Vec<usize> }`.
- **Case.** `GELU`, `ReLU`, `SGD`, `LSTMCell`, `Device::CPU`, `Device::GPU` follow PyTorch, not the Rust API guidelines (`Gelu`, `Sgd`, `Cpu`). Either is defensible; choose once, before `main`, because a rename later is breaking.

### PA10. Types that cannot grow

- **Enums without `#[non_exhaustive]`:** `BinaryOp`, `UnaryOp`, `CmpOp`, `MovementOp`, `ReduceOp`, `TernaryOp` (adding an op breaks any exhaustive match), `Storage` (PA3), `nn::PositionalEncodingType`, `device_policy::Pass`. `DType`, `Device`, `DevicePolicy`, `DTypePolicy`, `ConvAlgo`, `HostBuf` and `VoltaError` already have it.
- **Structs with all-public fields,** which a user can build with a literal, so adding a field is breaking: `io::TensorData`, `io::TypedTensorData`, `io::StateDictDiff`, `device_policy::FallbackEvent`. `TensorData`'s doc already notes that public fields allow an inconsistent value.
- **Open traits.** `Module` (three required methods) and `Optimizer` (three) can be implemented outside the crate, which is the point, so adding a required method is breaking; new methods need defaults. `Layer` is open too; `RecurrentLayer` is sealed.
- **Stringly errors.** `VoltaError::InvalidParameter(String)`, `DeviceError(String)` and `OpError(String)` carry most user-argument failures, so a caller can match the variant but not the cause.
- **Default features.** `default = ["gpu", "accelerate"]` gives every dependent `wgpu` unless it opts out. The README documents the opt-out. For a library, the conservative default is no GPU; this is a product decision, not a defect.
- **Fine as is:** MSRV 1.89 and edition 2024 are declared; `VoltaError::Io` wraps `std::io::Error`; no public signature mentions `safetensors` or `bytemuck` types.

**Before `main`:** the `#[non_exhaustive]` attributes are one line each and free now, breaking later.

### PA11. `data()`, `item()` and `grad()` decode to `f32`

`Tensor::data() -> Vec<f32>` is documented as "decoded to `f32` regardless of underlying dtype", and `grad()` as lossy for f64. Confirmed: an I64 tensor holding 16777217 and 9007199254740993 reads back as `[16777216.0, 9007199000000000.0]`. `to_vec_as::<T>()` is the exact reader and returns an error on a dtype mismatch. `item()` is `f32` as well, with no typed form, so reading an f64 loss exactly takes `to_vec_as::<f64>()` and an index.

**Can wait.** Add `item_as::<T>()`; consider whether `data()` on an integer tensor should exist at all.

### PA12. Test helpers in the public API

`volta::assert_trainable!` is `#[macro_export]`ed (`src/macros/assert_trainable.rs:20`), and it works from an external crate (confirmed with a `Linear`). `check_gradients` and `check_gradients_simple` are at the root and on `RawTensor`. `device_policy::MAX_FALLBACK_EVENTS`, `gpu_pool_stats`, `gpu_pending_count` and `gpu_sync_threshold` are diagnostics.

A gradient checker is a reasonable thing for an educational autograd library to publish; a macro whose semantics are tuned to this repository's tests (it backpropagates a randomly weighted sum, per `.claude/rules/tests.md`) is less clearly so. **Decide:** keep them under a `volta::testing` module, or accept them at the root on purpose.

### PA13. Docs that describe an older library

Every public item has a doc comment. Two of the most-read ones are out of date:

- The crate-level doc (`src/lib.rs:1-13`) describes "a minimal automatic differentiation library" with four bullets (graphs, broadcasting, "matmul, activations, etc.", gradient checking). It does not mention dtypes, the GPU backend and device policy, `nn`, `io`, `amp`, the feature flags, or how to start (`RawTensor::new`, `into_loss().backward()`). It is the front page on docs.rs.
- `RawTensor`'s doc (`src/tensor.rs:406-413`) says `data` is "CPU: flat `Vec<f32>` ... GPU: `Arc<GpuBuffer>`". It has been a `Storage` holding any of eight dtypes since Phase 5.

Also, `Tensor`'s `Debug` output is `Tensor { inner: RefCell { value: Tensor { shape: .., requires_grad: .., has_grad: .., device: .., .. } } }`: the wrapper layers show, and the values do not. Low.

### PA14. What a user will look for and not find

Not defects; a list for `kind:feature` triage, none of it before `main`.

- Operator overloads and scalar arithmetic (`a + b`, `x * 2.0`): #137.
- `abs`, `pow`, `clamp`, `log1p` (N7 wants the last), `squeeze` / `unsqueeze` on `Tensor` (they exist on `Shape`), `cat` / `stack`, `argmax`, indexing.
- A learning-rate getter and setter on the optimizers, and schedulers (N10 says to settle SGD's momentum form first). AdamW.
- `Tensor::numel`, `PartialEq` and `Display` for `Tensor`.
- Rank-0 tensors (#100): `volta::zeros(&[])` is accepted today, so the question there is what it should mean, not whether it exists.
- `try_` forms for layer constructors (PA6).

## Before `main`, and what can wait

**Before `main`** (each is cheap now and a breaking change later):

1. PA1: `RawTensor`'s fields to `pub(crate)`.
2. PA2: remove `new_tensor`.
3. PA3: hide `GpuBuffer`'s constructors and `buffer()`, `Storage::gpu_buffer`, `GpuStorage`; make `Storage` opaque or `#[non_exhaustive]` with a feature-independent variant set. This also closes U2 and U3.
4. PA4: private leaf modules, one path per item.
5. PA5: no-`gpu` stubs for the seven root functions.
6. PA10: `#[non_exhaustive]` on the nine enums; decide the four public-field structs.
7. PA8, PA12, and the case question in PA9: decisions, each recorded in `DECISIONS.md` whichever way it goes.
8. PA6 (the `# Panics` sections only) and PA13: documentation.
9. U1 from the soundness audit (`set_grad` length), which is an API contract.

**Can wait** (additive, or already tracked): the `try_` forms (PA6), constructor validation (PA7), names (PA9, #137), typed readers (PA11), features (PA14).

## Checked and found in order

- No public item lacks a doc comment, with or without `gpu` (rustdoc JSON).
- `Tensor` is opaque: `borrow`, `borrow_mut` and `as_ptr` are `pub(crate)`, and `tests/compile_fail/` pins `backward` on a bare tensor, the guards' `!Send`, `Storage` not dereferencing, and the sealed `RecurrentLayer`.
- `autograd` exports only `ScalarLoss`, `NoGradGuard` and `no_grad`: backward has one entry point.
- Constructors and optimizers have `try_` forms that return `VoltaError` (`try_new`, `try_zeros`, `try_ones`, `try_rand`, `try_randn`, `try_from_vec`, `try_from_typed`, `try_constant`, `try_xavier_uniform`, `try_he_initialization`, `SGD::try_new`, `Adam::try_new`, `Muon::try_new`, `DataLoader::try_new`, `try_normalize`, `Storage::try_from_bytes`, `TensorData::try_to_tensor`).
- The I/O functions return `Result` (`load_safetensors*`, `save_safetensors*`, `load_mnist_*`, `to_one_hot`).
- The policy guards (`with_device_policy`, `with_dtype_policy`, `autocast`, `no_grad`) are RAII and documented.
- `VoltaError` is `#[non_exhaustive]` and derives its messages; `Result<T>` is a plain alias.

## Not covered

- Example and README prose beyond what the doctests compile.
- Ergonomics measured with real users; these are a reviewer's judgments.
- `cargo semver-checks` against 0.2: not installed here, and `main` is 229 commits behind, so nearly everything would be flagged.
- Trait-object and auto-trait surface in detail (which types are `Send`, `Sync`, `UnwindSafe`), beyond `Tensor` being neither `Send` nor `Sync`.

## Appendix A: the exported surface

Default features, generated from rustdoc JSON. Definitions by module, then the root re-exports. Methods are listed for the two core types only; the script's full output is 1,879 lines.

| Module | Items defined there |
| --- | --- |
| `volta::amp` | `AutocastGuard` (struct), `GradScaler` (struct), `autocast` (function), `autocast_dtype` (function) |
| `volta` | `assert_trainable` (macro) |
| `volta::autograd` | `NoGradGuard` (struct), `ScalarLoss` (struct), `no_grad` (function) |
| `volta::data::mnist` | `load_mnist_images` (function), `load_mnist_labels` (function) |
| `volta::data::transforms` | `normalize` (function), `to_one_hot` (function), `try_normalize` (function) |
| `volta::device` | `Device` (enum) |
| `volta::device_policy` | `DevicePolicy` (enum), `DevicePolicyGuard` (struct), `FallbackEvent` (struct), `MAX_FALLBACK_EVENTS` (constant), `Pass` (enum), `device_policy` (function), `take_fallback_events` (function), `with_device_policy` (function) |
| `volta::dtype` | `DType` (enum), `DTypePolicy` (enum), `DTypePolicyGuard` (struct), `dtype_policy` (function), `with_dtype_policy` (function) |
| `volta::error` | `Result` (type_alias), `VoltaError` (enum) |
| `volta::gpu` | `gpu_cleanup` (function), `gpu_compact` (function), `gpu_pending_count` (function), `gpu_pool_stats` (function), `gpu_sync` (function), `gpu_sync_threshold` (function), `invalidate_all_tensor_caches` (function), `is_gpu_available` (function) |
| `volta::io` | `StateDict` (type_alias), `StateDictDiff` (struct), `TensorData` (struct), `TypedTensorData` (struct), `diff_state_dict` (function), `load_safetensors` (function), `load_safetensors_raw` (function), `load_safetensors_with_mapping` (function), `load_state_dict_checked` (function), `save_safetensors` (function), `save_safetensors_typed` (function) |
| `volta::io::mapping` | `StateDictMapper` (struct), `TransformationBox` (type_alias), `create_key_mapping` (function), `load_huggingface_weights` (function), `load_partial` (function), `load_pytorch_linear_weights` (function) |
| `volta::mapping` | `TransformationBox` (type_alias) |
| `volta::nn` | `Layer` (trait), `Module` (trait), `RecurrentLayer` (trait) |
| `volta::nn::layers::batchnorm` | `BatchNorm1d` (struct), `BatchNorm2d` (struct) |
| `volta::nn::layers::conv` | `Conv2d` (struct), `ConvAlgo` (enum) |
| `volta::nn::layers::conv_transpose` | `ConvTranspose2d` (struct) |
| `volta::nn::layers::dropout` | `Dropout` (struct) |
| `volta::nn::layers::embedding` | `Embedding` (struct) |
| `volta::nn::layers::flatten` | `Flatten` (struct) |
| `volta::nn::layers::gelu` | `GELU` (struct) |
| `volta::nn::layers::layernorm` | `LayerNorm` (struct) |
| `volta::nn::layers::linear` | `Linear` (struct) |
| `volta::nn::layers::lstm` | `LSTMCell` (struct) |
| `volta::nn::layers::maxpool` | `MaxPool2d` (struct) |
| `volta::nn::layers::pixelshuffle` | `PixelShuffle` (struct) |
| `volta::nn::layers::relu` | `ReLU` (struct) |
| `volta::nn::layers::sequential` | `Sequential` (struct) |
| `volta::nn::layers::sequential_builder` | `SequentialBuilder` (struct) |
| `volta::nn::layers::sigmoid` | `Sigmoid` (struct) |
| `volta::nn::layers::tanh` | `Tanh` (struct) |
| `volta::nn::optim` | `Optimizer` (trait) |
| `volta::nn::optim::adam` | `Adam` (struct) |
| `volta::nn::optim::muon` | `Muon` (struct) |
| `volta::nn::optim::sgd` | `SGD` (struct) |
| `volta::nn::transformer::attention` | `ScaledDotProductAttention` (struct) |
| `volta::nn::transformer::block` | `TransformerBlock` (struct) |
| `volta::nn::transformer::feed_forward` | `FeedForward` (struct) |
| `volta::nn::transformer::mask` | `causal_mask` (function) |
| `volta::nn::transformer::multi_head_attention` | `MultiHeadAttention` (struct) |
| `volta::nn::transformer::positional_encoding` | `PositionalEncoding` (struct), `PositionalEncodingType` (enum) |
| `volta::ops::binary` | `BinaryOp` (enum) |
| `volta::ops::compare` | `CmpOp` (enum) |
| `volta::ops::movement` | `MovementOp` (enum) |
| `volta::ops::reduce` | `ReduceOp` (enum) |
| `volta::ops::ternary` | `TernaryOp` (enum) |
| `volta::ops::unary` | `UnaryOp` (enum) |
| `volta::shape` | `Axis` (struct), `MAX_NUMEL` (constant), `Shape` (struct), `Strides` (struct) |
| `volta::storage` | `Element` (trait), `GpuStorage` (struct), `HostBuf` (enum), `Storage` (enum) |
| `volta::tensor` | `DataLoader` (struct), `RawTensor` (struct), `Tensor` (struct), `bce_loss` (function), `bce_with_logits_loss` (function), `check_gradients` (function), `check_gradients_simple` (function), `cross_entropy_loss` (function), `kl_divergence_gaussian` (function), `manual_seed` (function), `max_dim` (function), `mse_loss` (function), `nll_loss` (function), `ones` (function), `rand` (function), `randn` (function), `randn_like` (function), `softmax` (function), `sum_dim` (function), `zeros` (function) |
| `volta::utils::progress` | `ProgressBar` (struct) |

Re-exported at the crate root (`volta::X`): `Adam`, `Axis`, `BatchNorm1d`, `BatchNorm2d`, `BinaryOp`, `Conv2d`, `ConvAlgo`, `ConvTranspose2d`, `DType`, `DTypePolicy`, `DTypePolicyGuard`, `DataLoader`, `Device`, `DevicePolicy`, `DevicePolicyGuard`, `Dropout`, `Embedding`, `FallbackEvent`, `Flatten`, `GELU`, `GpuBuffer`, `GradScaler`, `LSTMCell`, `Layer`, `LayerNorm`, `Linear`, `MaxPool2d`, `Module`, `MovementOp`, `Muon`, `Optimizer`, `PadValue`, `Pass`, `PixelShuffle`, `ProgressBar`, `RawTensor`, `ReLU`, `RecurrentLayer`, `ReduceOp`, `Result`, `SGD`, `ScalarLoss`, `Sequential`, `SequentialBuilder`, `Shape`, `Sigmoid`, `Storage`, `Strides`, `Tanh`, `Tensor`, `TernaryOp`, `TypedTensorData`, `UnaryOp`, `VoltaError`, `autocast`, `bce_loss`, `bce_with_logits_loss`, `check_gradients`, `check_gradients_simple`, `cross_entropy_loss`, `device_policy`, `dtype_policy`, `gpu_cleanup`, `gpu_compact`, `gpu_pending_count`, `gpu_pool_stats`, `gpu_sync`, `gpu_sync_threshold`, `is_gpu_available`, `kl_divergence_gaussian`, `load_mnist_images`, `load_mnist_labels`, `load_safetensors`, `load_safetensors_raw`, `load_safetensors_with_mapping`, `manual_seed`, `max_dim`, `mse_loss`, `new_tensor`, `nll_loss`, `normalize`, `ones`, `rand`, `randn`, `randn_like`, `save_safetensors`, `save_safetensors_typed`, `softmax`, `sum_dim`, `take_fallback_events`, `to_one_hot`, `try_normalize`, `with_device_policy`, `with_dtype_policy`, `zeros`.

`Tensor` methods, with `[P E D]` flags for a `# Panics` section, an `# Errors` section and a doc comment: `into_loss[-ED]` `requires_grad[--D]` `dtype[--D]` `ptr_eq[--D]` `shape[--D]` `data[--D]` `to_vec_as[-ED]` `item[P-D]` `device[--D]` `grad_storage[--D]` `set_requires_grad[P-D]` `detach[--D]` `set_grad[P-D]` `add[--D]` `sub[--D]` `elem_mul[--D]` `div[--D]` `max_elem[--D]` `modulo[--D]` `lt[--D]` `le[--D]` `gt[--D]` `ge[--D]` `elem_eq[--D]` `elem_ne[--D]` `neg[--D]` `recip[--D]` `sqrt[--D]` `exp2[--D]` `log2[--D]` `exp[--D]` `log[--D]` `sin[--D]` `cos[--D]` `tanh[--D]` `sigmoid[--D]` `relu[--D]` `erf[--D]` `sum[--D]` `max_reduce[--D]` `mean[--D]` `mulacc[--D]` `where_op[P-D]` `reshape[P-D]` `try_reshape[-ED]` `permute[--D]` `expand[P-D]` `try_expand[-ED]` `pad_with[--D]` `shrink[--D]` `stride_op[--D]` `to_dtype[P-D]` `to_device[P-D]` `matmul[--D]` `transpose[--D]` `grad[--D]` `zero_grad[--D]` `sum_dim[--D]` `max_dim[--D]` `mean_dim[--D]` `softmax[--D]` `log_softmax[--D]`.

`RawTensor` associated functions: `broadcast_shape[P-D]` `try_broadcast_shape[-ED]` `binary_op[P-D]` `compare[P-D]` `matmul[P-D]` `transpose[P-D]` `reshape[P-D]` `try_reshape[-ED]` `permute[P-D]` `expand[P-D]` `try_expand[-ED]` `pad_with[P-D]` `try_pad_with[-ED]` `shrink[P-D]` `try_shrink[-ED]` `stride_op[P-D]` `try_stride_op[-ED]` `reduce_op[P-D]` `ternary_op[P-D]` `unary_op[--D]` `try_empty[-ED]` `empty[P-D]` `try_constant[-ED]` `constant[P-D]` `try_from_vec[-ED]` `from_vec[P-D]` `try_from_typed[-ED]` `from_typed[P-D]` `contiguous[--D]` `try_new[-ED]` `new[P-D]` `try_zeros[-ED]` `zeros[P-D]` `try_ones[-ED]` `ones[P-D]` `try_rand[-ED]` `rand[P-D]` `try_randn[-ED]` `randn[P-D]` `randn_like[P-D]` `try_xavier_uniform[-ED]` `xavier_uniform[P-D]` `try_he_initialization[-ED]` `he_initialization[P-D]` `invalidate_gpu_cache[--D]` `mse_loss[P-D]` `cross_entropy_loss[P-D]` `nll_loss[P-D]` `kl_divergence_gaussian[P-D]` `bce_loss[P-D]` `bce_with_logits_loss[P-D]` `try_sum_dim[-ED]` `sum_dim[P-D]` `to_device[P-D]` `to_dtype[P-D]` `max_dim[P-D]` `softmax[--D]` `log_softmax[--D]` `mean_dim[P-D]` `check_gradients[P-D]` `check_gradients_simple[P-D]` `zero_grad[--D]`.

Present only with the `gpu` feature: `volta::GpuBuffer` and `volta::gpu::GpuBuffer` with their eight methods and `Drop`; `volta::storage::GpuStorage`; the `Storage::Gpu` variant; `Storage::gpu_buffer`, `Storage::invalidate_cpu_cache`, `RawTensor::invalidate_gpu_cache`; and the `volta::gpu` functions with their root re-exports (PA5).

## Appendix B: probes

From an integration test (public API only), each call under `catch_unwind`, first line of the panic message shown.

```
permute(&[Axis(0), Axis(0), Axis(5)])   PANIC assertion `left == right` failed: Invalid permutation axes
shrink(&[(0, 9), (0, 3), (0, 4)])       PANIC Invalid shrink ranges: DimensionOutOfBounds { dim: 0, shape: [2, 3, 4] }
stride_op(&[0, 1, 1])                   PANIC Stride: Invalid parameter: stride_op: stride 0 for dimension 0 (strides must be >= 1)
sum_dim(Axis(7), false)                 PANIC Dimension out of bounds: DimensionOutOfBounds { dim: 7, shape: [2, 3, 4] }
softmax(Axis(7))                        PANIC max_dim: Dimension 7 out of bounds for shape [2, 3, 4]
mean_dim(Axis(7), false)                PANIC mean_dim: Dimension 7 out of bounds for shape [2, 3, 4]
reshape(&[5, 5])                        PANIC Reshape: Invalid parameter: cannot reshape [2, 3, 4] (24 elements) to [5, 5] (25 elements)
x.matmul(&x), x of [2, 3, 4]            PANIC Invalid parameter: matmul dimension mismatch: [2, 3, 4] @ [2, 3, 4] (inner dims 4 vs 3)
x.add(&five_elements)                   PANIC Cannot broadcast shapes [2, 3, 4] and [5] at dimension 2
zeros(&[])                              accepted
zeros(&[0])                             PANIC zeros: Invalid parameter: shape [0] has a zero dimension
Linear::new(0, 3, true)                 PANIC xavier_uniform: Invalid parameter: shape [0, 3] has a zero dimension
Conv2d::new(1, 1, 0, 1, 0, true)        PANIC he_initialization: Invalid parameter: shape [1, 1, 0, 0] has a zero dimension
Conv2d::new(1, 1, 3, 0, 0, true)        accepted (forward: PANIC Invalid parameter: stride must be non-zero, got (0, 0))
MaxPool2d::new(0, 1, 0)                 accepted
Embedding::new(0, 0)                    PANIC vocab_size must be positive
Dropout::new(1.5), (-0.1), (NaN)        PANIC Dropout probability must be in [0, 1]
LayerNorm::new(vec![])                  accepted
LayerNorm::new_with_eps(vec![4], -1.0)  accepted, forward runs
BatchNorm2d::new_with_params(3, 1e-5, 7.0)  accepted
PixelShuffle::new(0)                    PANIC PixelShuffle: upscale_factor must be positive, got 0
MultiHeadAttention::new(5, 2, 0.0)      PANIC embed_dim must be divisible by num_heads
PositionalEncoding::new(5, 10, Sinusoidal)  PANIC embed_dim must be even for sinusoidal positional encoding, got 5
causal_mask(0, Device::CPU)             PANIC new: Invalid parameter: shape [0, 0] has a zero dimension
GradScaler::with_params(1.0, 0.5, 0.5, 10)  PANIC GradScaler: growth_factor must be > 1
Embedding index 9, vocab 4              PANIC Embedding: index 9 out of range for vocab_size 4
x.to_device(Device::GPU("nope"))        PANIC to_device(nope): no such GPU; the available GPU is llvmpipe (..)
Storage::new_zeros(4, &Device::GPU("nope"))  accepted
cross_entropy_loss, class 7 of 3        PANIC cross_entropy_loss: target 7 out of range for 3 classes
Linear.load_state_dict(&StateDict::new())   Ok(())
new_tensor::new(vec![1.0, 2.0], &[2], true) compiles; shape [2]
assert_trainable!(Linear::new(3, 2, true), randn(&[4, 3]))   passes from an external crate
from_typed(vec![16777217i64, 9007199254740993], &[2]).data() [16777216.0, 9007199000000000.0]
```

Surface listing: `RUSTDOCFLAGS="-Z unstable-options --output-format json" cargo +nightly-2026-11-01 doc --no-deps --lib` (and again with `--no-default-features`), walked from the root module through `module` and `use` items, counting a `use` whose target is in the crate as a re-export.

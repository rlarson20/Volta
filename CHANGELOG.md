# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)

Sections drafted with the assistance of [git-cliff](https://git-cliff.org/) (see `cliff.toml`).

## [Unreleased]

## [0.4.0] - 2026-11-24 - The Cleanup Release

The `improvements` branch merged into `main`: an opaque `Tensor`, typed shapes, dtypes and mixed precision, loud failures, and fixes checked against PyTorch.

### `Tensor` and `Autograd`

#### `Tensor` is now opaque; `RawTensor` fields are now hidden.

Changed: `data`, `shape`, `grad`, `requires_grad`, `grad_fn`, `parents` and `device` are no longer public fields. `RawTensor` gained private `version` fields, preventing struct literal construction.
Migration: use the `Tensor` accessor methods.
Source: bfd4ec7, #376; `struct_pub_field_missing` and `constructible_struct_adds_private_field` semver lints.

#### Removed: `RawTensor` op methods

Changed: `add`, `sub`, `elem_mul`, `div`, `max_elem`, `modulo`, `cmplt`, `matmul_raw`, `pad`, `compute_strides`, `sum`, `max_reduce`, `mean`, `mulacc`, `where_op`, `neg`, `recip`, `sqrt`, `exp2`, `log2`, `exp`, `log`, `sin`, `cos`, `tanh`, `sigmoid`, `relu`, `empty`, `backward` and `invalidate_gpu_cache` are removed.
Migration: use `Tensor`'s methods.
Source: #376, `inherent_method_missing` semver lint.

#### Removed: `Tensor::from_raw` (#390) and `new_tensor` (#376)

#### Removed: `TensorOps`

Changed: Its methods were already inherent to `Tensor`.
Migration: drop `use volta::TensorOps`.
Source: 132790f

#### Breaking: `.backward()` requires scalar loss

Changed: `{TensorOps,RawTensor}::backward` were removed. Backward runs only through `t.into_loss()?.backward()`. `into_loss` returns `NotScalar` when `numel != 1` and `NoGradRequired` when the tensor has no grad. The implicit all-ones seed for non-scalar roots is gone.
Migration: If you relied on that seed, call `.sum()` first.
Source: f1c07fe

#### `Tensor::shape()` returns `Shape`, not `Vec<usize>`

Migration: read a dim with `shape.dim(Axis(i))`, get a slice with `truncate_rank()`. `assert_eq!(t.shape(), vec![..])` still compiles.
Source: 840ceaf

#### Dim arguments take `Axis`

Changed: `(try_)sum_dim`, `(max/mean)_dim`, and `(log_)softmax` all take `dim: Axis`. `permute` takes `&[Axis]`. Out of range axes now panic (`try_sum_dim` returns `Err`) instead of defaulting silently.
Migration: Where you passed `n`, pass `Axis(n)`.
Source: 567946f

#### Shape Validation

Changed: `Shape::from_dims` rejects zero dimensions, shapes with more than `MAX_NUMEL` elements, and op outputs go through the same checks. Integer shape defaults that used to turn a rank bug into a size-1 axis now panic.
Source: 31e2edd, 380f563

#### Full reductions return rank-0

Changed: `sum`, `mean` and `max` over all elements, the losses built on them, and 1D dot product return shape `[]` instead of `[1]`. 0-dim operands promote like `torch.result_type`, so `f16[n] * f32[]` remains f16, while `f16[n] * f32[1]` becomes f32.
Migration: `item()` still works. Code that checks `shape == [1]` or indexes dim 0 of a loss breaks.
Source: #438, #436 via 370b1e3

#### Only leaf tensors keep gradients

Changed: Intermediate tensors no longer have `grad` after backward, as in PyTorch.
Source: #89

#### DType rules

Changed: Integer and bool tensors are `NotDifferentiable`. `set_grad` panics on a gradient of another dtype. Mixed-dtype ops promote under `DTypePolicy` instead of reading f64 as f32. `load_weight` returns `DTypeMismatch` for a non-f32 destination.
Source: 5c7fdcc, #19-25

#### `GradFn` is private

Changed: `GradFn` trait and every `*GradFn` struct left the public API when closed `GradFn` enum replaced `Box<dyn GradFn>`. User defined autograd ops are no longer possible at the moment.
Source: 8ce5eae, `{trait,struct}_missing` semver lints

### `Storage`

#### Removed: `Deref` / `Index` / `IntoIterator` on `Storage`

Changed: removed `iter`, `iter_mut`, `as_slice`, `as_mut_slice`, `as_f32_slice(_mut)`, `as_f64_slice`, `as_f16_slice`, `as_bf16_slice`, `cpu_f32`, `cpu_f64`, `cpu_f16`, `cpu_bf16`, `to_dtype`, `gpu_buffer` and `try_as_bytes`.
Migration: `Storage::from_typed`, `as_host::<T>()`, `as_host_mut::<T>()` (copy-on-write), `to_vec`, and `to_f32_vec` (a lossy decode). `from_bytes` and `try_from_bytes` take `&[u8]`.
Source: 4418437, 5c7fdcc.

#### `Storage` is now an opaque struct, not an enum.

Changed: `Storage::{Cpu,Gpu}` no longer exist, and `HostBuf` is crate-private.
Source: #432, `enum_changed_kind` semver lint.

#### `GpuBuffer`, `GpuBuffer::{buffer,copy_region}` gone from package root (#365) (#382)

### Item Paths (#382)

Crate root now only exports 8 core modules: `tensor`, `autograd`, `amp`, `dtype`, `device`, `device_policy`, `shape` and `error`.
The following table shows how everything else was moved:

| Was                                                                                                                                                                                                                                                                           | Now                         |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| `volta::Linear`, `Conv2d`, `Sequential`, `SequentialBuilder`, `ReLU`, `Sigmoid`, `Tanh`, `Dropout`, `Flatten`, `MaxPool2d`, `BatchNorm1d`, `BatchNorm2d`, `Embedding`, `LSTMCell`, `PixelShuffle`, `ConvTranspose2d`, `Module`; `volta::nn::layers::*`, `volta::nn::optim::*` | `volta::nn::X` (flat)       |
| `volta::{SGD,Adam,Muon}`                                                                                                                                                                                                                                                      | `volta::nn::SGD` etc        |
| `volta::BinaryOp`, `UnaryOp`, `ReduceOp`, `MovementOp`, `TernaryOp`; `volta::ops::binary::*`, etc.                                                                                                                                                                            | `volta::ops::X`             |
| `volta::load_safetensors`, `save_safetensors`, `load_safetensors_with_mapping`                                                                                                                                                                                                | `volta::io::X`              |
| `volta::mapping::*`                                                                                                                                                                                                                                                           | `volta::io::mapping::*`     |
| `volta::normalize`, `to_one_hot`, `load_mnist_images`, `load_mnist_labels`; `volta::data::mnist::*`, `data::transforms::*`                                                                                                                                                    | `volta::data::X`            |
| `volta::ProgressBar`, `volta::utils::progress::*`                                                                                                                                                                                                                             | `volta::utils::ProgressBar` |
| `volta::is_gpu_available`, `gpu_sync`, `gpu_cleanup`, `gpu_compact`, `gpu_pending_count`, `gpu_pool_stats`, `gpu_sync_threshold`                                                                                                                                              | `volta::gpu::X`             |
| `volta::Storage`                                                                                                                                                                                                                                                              | `volta::storage::Storage`   |

Added at the root so the rule holds: `no_grad`, `NoGradGuard`, `AutocastGuard`, `autocast_dtype`, `MAX_NUMEL` and `MAX_FALLBACK_EVENTS`.
`assert_trainable!` is no longer exported (#388).

### `nn`

#### `Module` is split

Changed: `Module` keeps `parameters` and `state_dict`, but no `forward`. `Layer : Module` adds `forward` and is held by `Sequential`. `Embedding` is a sealed `IndexedLayer` with `forward_indices`. `LSTMCell` is a sealed `RecurrentLayer` with `initial_{state,step}`, and `forward_step` is removed. Putting an `Embedding` in a `Sequential` is now a compile-time error.
Migration: Implement `Layer` for custom layers, and import `volta::nn::Layer` to call `forward`.
Source: 1a51ab7; `trait_method_missing` semver lint

#### `load_state_dict` returns `Result`, is now strict

Changed: A missing or unexpected key returns `StateKeyMismatch` and nothing is written. The lenient form is now `load_state_dict_partial`, and custom layers implement it. Loads are atomic (plan, then apply).
Source: 4a68601, #420, #439.

#### `Conv2d::set_algo` takes `&mut self`. Migration: `let mut conv = ....` (#335)

#### `TransformerBlock` holds a real `MultiHeadAttention`.

Changed: it used to hold the parameterless `SimpleAttention`, so it now has `attention.*` state-dict keys. Attention no longer adds a residual; the block does.
Source: 1a51ab7, #163

#### Constructors validate their hyperparameters.

Changed: bad values now panic with a named error, each constructor has a `try_new`, and an overflowing parameter returns `Err`.
Source: #440, #470.

#### Losses reject a pred/target shape mismatch instead of broadcasting. (#95, #253)

#### Optimizers refuse non-leaf parameters (#295), a parameter listed twice such as a tied weight (#101), and, for `Adam`, half-precision parameters (#372).

#### `BatchNorm` refuses one value per channel in training mode (#161) and a mismatched input `dtype` (#349).

### Error handling

#### Data from outside the program returns `Result`.

Changed: tensor constructors, reshape and expand have `try_` twins, and the panicking form wraps the twin. `to_one_hot`, `load_weight` and `load_state_dict_checked` return `Result`.
Source: 4a68601

#### Error payloads are structured.

Changed: `VoltaError::InvalidParameter`, `DeviceError` and `OpError` carry `ErrorDetail { op, arg, message }` instead of a `String`.
Migration: read `.detail()` or `detail.message`.
Source: #435

#### The IO loaders return `volta::Result`, `not std::io::Result`.

Changed: this covers `load_safetensors`, `save_safetensors`, `load_safetensors_with_mapping` and the MNIST loaders. A file that can't be decoded is the new `VoltaError::Decode { path, key, detail }`.
Source: #436

#### More types are `#[non_exhaustive]`

Changed: the enums `Device`, `DType`, `BinaryOp`, `UnaryOp`, `ReduceOp`, `MovementOp` and `TernaryOp`, and the structs `TensorData` and `StateDictDiff`. Matches need a _ arm, and the structs need their constructors (`TensorData::from_f32`, `from_typed`, `from_tensor`).
Source: #379, 5b82e42

#### Removed: `BinaryOp::Cmplt`

Changed: Comparisons are now `CmpOp` and return `Bool` tensors.
Source: #25; `enum_variant_missing` semver lint

### IO

#### Removed: `bincode`

Changed: `io::save_state_dict, load_state_dict, load_state_dict_with_mapping` are gone, and old .bin checkpoints no longer load.
Migration: use the `*_safetensors` functions.
Source: #256

#### `TypedTensorData`, `load_safetensors_raw` and `save_safetensors_typed` are removed.

Migration: `TensorData`, `load_safetensors` and `save_safetensors`. `to_storage` now takes the key.
Source: #412

#### The mapping helpers take layer names.

Changed: `load_pytorch_linear_weights(state, linear_layers: &[&str])` takes 2 arguments instead of 1, and `load_huggingface_weights(state, model_prefix, linear_layers)` takes 3 instead of 2, because transposes now go by layer name, not key pattern.
Source: c32fb28

#### `create_key_mapping`, `select_keys` and `exclude_keys` are generic `(impl IntoIterator<Item: Into<String>>)`

Migration: Existing `Vec<String>` callers still compile, so this only breaks turbofish or function-pointer uses.
Source: #376

### `Device` and `GPU`

#### `DevicePolicy::Strict` is the default.

Changed: a GPU input to an op with no GPU kernel panics instead of running silently on the CPU.
Migration: wrap such code in with_device_policy(PreferGpu, ..).
Source: 2040b99

#### A GPU kernel that fails now panics instead of falling back to the CPU (1fc289c)

#### GPU internals left the public API:

- `GpuContext` and `get_gpu_context`, along with `sync_checked`, `sync_with_retry` and `GpuSyncError` (#293)
- The `monitor`, `system_monitor` and `early_warning` modules, with `SystemMonitor`, `MonitorStats`, `EarlyWarningSystem`, `TrendData`, `ResourceStatus`, `HealthStatus`, `check_system_resources` and `get_process_memory_mb` (#293, #338)
- `GpuKernels`, `BufferPool`, `BufferPoolConfig`, `StagingBufferPool`, `StagingPoolStats`, `OptimizerStepParams` and `MovementParams` (55be134)

### Features and build

#### Default features are empty.

Changed: 0.3.0 defaulted to `gpu` and `accelerate`.
Migration: add `features = ["gpu", "accelerate"]` to get the old build.
Source: #422

#### The blas-src feature is gone.

Changed: it was the implicit feature of an optional dependency that did nothing build.rs didn't already do. `accelerate = []` now gates the link in `build.rs`. The implicit `wgpu` and `pollster` features are also no longer on by default.
◦ Source: 5cc8f29; `feature_missing` semver lint

#### Big-endian targets refuse to build (#411)

#### `ProgressBar` is no longer `Sync` or `RefUnwindSafe`

Changed: It now holds a `Cell`.
Source: #273; `auto_trait_impl_removed` semver lint

### Result changes without API changes

Outputs differ from 0.3.0:

- `Max` ties and `NaN` follow torch (#220)
- Modulo is floored, so the result has the divisor's sign (#157)
- `Div`/`Recip` backward stopped squaring the denominator (#284)
- `bce_loss` clamps its logs (#164)
- `Attention` has no residual (#163)
- Transformer dropout is actually applied (#225)
- Pairwise summation changes low-order bits in sums, matmul and conv gradients (#211, #271, #330, #369, #370, #381, #385)
- `Adam` counts steps per parameter (#208)
- `Muon` uses the reference Newton-Schulz implementation (8f3d0d4)

#### Flagged by semver checks but non-breaking:

- `#[must_use]` was added to many functions and methods.
- Stateless layers (`ReLU`, `Sigmoid`, `Tanh`, `Dropout`, `Flatten`, `MaxPool2d`, `PixelShuffle`) now implement `Copy`.

## Cliff notes

### Breaking

Hand-written from the commit log; git-cliff flags only the 11 `!` commits. Each line names the PR, the merge PR, or the commit that made the change.

- `Tensor` is opaque: `RawTensor` fields are hidden and `Tensor::from_raw` is removed. In-place updates go through `Tensor` methods (bfd4ec7, #376, #390, #393).
- `Tensor::shape()` returns `Shape`, dim arguments take `Axis`, and `Shape::from_dims` rejects zero dims (merge of `shape-public-return`, 567946f, 31e2edd).
- The `TensorOps` trait is deleted and its methods are inherent on `Tensor`. `TensorOps::backward` is replaced by `into_loss` / `ScalarLoss` (132790f, f1c07fe).
- `Module` is split into `Module`, `Layer`, `IndexedLayer` and `RecurrentLayer`: `forward` moves to `Layer`, and `LSTMCell::forward_step` becomes `RecurrentLayer::step` (1a51ab7).
- Full reductions (`sum`, `mean`, `max`, the losses, a 1-D dot) return rank-0 tensors (shape `[]`, not `[1]`); zero-dim operands promote like `torch.result_type` (#438, squashed into #436).
- The `GradFn` trait and the `*GradFn` structs are no longer public; the graph uses a closed enum (8ce5eae).
- `Conv2d::set_algo` takes `&mut self` (#335).
- One canonical path per item: the crate root re-exports only the core modules, and layers and optimizers live at `volta::nn::X` (#382).
- Public enums and data structs are `#[non_exhaustive]` (#379, #365).
- `Storage` is an opaque struct with no `Deref`/`Index`; host access is explicit and typed, and CPU storage is copy-on-write. `HostBuf` is crate-private (#4, #5, #20, #432).
- `DevicePolicy::Strict` is the default: a GPU input to an op with no GPU kernel panics instead of running on the CPU. `PreferGpu` opts in to the fallback (2040b99).
- Default features are empty: `gpu` and `accelerate` are opt-in (#422).
- Errors: functions fed by outside data return `Result<_, VoltaError>` and have `try_` forms; file loaders return `volta::Result`, and `VoltaError` variants carry an `ErrorDetail` (#435, #436, #440).
- `Module::load_state_dict` is strict; the lenient form is `load_state_dict_partial` (#420).
- `BinaryOp::Cmplt` is removed; comparisons are `CmpOp` and return Bool (#25).
- `io::mapping::load_pytorch_linear_weights` and `load_huggingface_weights` take the Linear layer names (c32fb28).
- The `blas-src` feature is removed; `accelerate` gates the link itself (5cc8f29).
- Removed: bincode (safetensors is the only state-dict format, #256), `TypedTensorData` and its loader and saver (#412), the GPU monitors, `GpuContext`, `GpuKernels` and the buffer pools from the public API (#293, #338, 55be134), and the public `assert_trainable!` macro (#388).
- Tensors carry a dtype: kernels refuse a dtype they don't support, and mixed dtypes promote under `DTypePolicy` (#19–#25).

### Added

- Implement the backward pass for ConvTranspose2d and correct input permutation in its forward pass. (7dc9522)
- Add GPU-accelerated direct convolution and comprehensive algorithm benchmarks (cb18401)
- Add GPU-accelerated backward pass for direct convolution (455a606)
- Add GPU-accelerated iGEMM convolution with full gradient support (48ed2cf)
- Add GPU-accelerated col2im for complete convolution backward pass (902e39d)
- **nn:** Add LayerNorm implementation for Transformer architectures (a73b604)
- Components for building transformers (05e0eb1)
- Utilizing new transformer components for a simple example (a204ab3)
- Route library diagnostics through the log facade (110512a)
- Finish Phase 1 invariant wall (opaque Tensor, edge(), NoGrad, assert_trainable!) (bfd4ec7)
- Add Tensor::set_grad, closing the Phase 1 opaque-Tensor test gap (6d35880)
- Define Strides and fill in Shape/Axis newtypes (f1cc5ea)
- **spec:** Validated geometry specs for the conv family (2338054)
- **spec:** MatmulSpec for rank >= 2 matrix products (2c80d5d)
- **pad:** PadValue and pad_with, retiring H1 (a74c145)
- **ops:** Define_ops! macro and table-driven UnaryOp, with GPU erf kernels (6db9140)
- **ops:** Table-driven BinaryOp via define_ops!, typed binary GpuKernels (eb2eaa3)
- **autograd:** `into_loss` / ScalarLoss; remove TensorOps::backward (f1c07fe)
- **matmul:** Decide contiguous-only layout contract and harden matmul_raw (8bf3cbd)
- **index:** Loud fetch/slot/span helpers; convert autograd core (ADDENDUM §8) (36d6d6f)
- **tensor:** Add Tensor::item() and use it for scalar reads (670eabf)
- **device:** Add DevicePolicy, its guard, and the fallback event log (3d0b880)
- **device:** `edge()` enforces one input device and applies the policy to forward gaps (395daee)
- **device:** Backward gradients obey the device policy (be0ab61)
- **device:** Strict is the default device policy (2040b99)
- **autograd:** `to_device` is a differentiable op (PyTorch's .to()) (48f6356)
- **errors:** Result where data comes from outside the program (H4) (4a68601)
- **optim:** Muon uses the reference quintic Newton-Schulz and shape scale (H8) (8f3d0d4)
- **optim:** Muon runs on the GPU instead of rejecting GPU parameters (H8) (b817f54)
- **ops:** Per-op dtype check names the op and input (Phase 5 P1 review) (900f809)
- **autograd:** Differentiable to_dtype matching torch casts bit for bit (Phase 5 P2) (d55cfda)
- **ops:** Core ops compute in f64 (Phase 5 P3a) (af2bd70)
- **tensor:** Tensor::to_vec_as, a lossless typed readback (9c2fe39)
- **nn:** Module::to_dtype; load_state_dict casts; gradcheck in the input's dtype (20c0a6d)
- **nn:** Layers and optimizers compute in f64; torch f64 fixtures (Phase 5 P3b) (80c8e0f)
- **dtype:** F16/bf16 tensors compute in f32 and round each output (Phase 5 P4) (508f5f3)
- **dtype:** I64 and Bool tensors as data (Phase 5 P5) (4b5fdcf)
- **dtype:** Implicit promotion with a DTypePolicy (Phase 5 P6) (083dcfb)
- **amp:** Autocast, GradScaler, Optimizer trait (Phase 5 P7) (61fe8b5)
- **gpu:** F16, bf16 and Bool on the GPU (Phase 5 P8) (755b2d1)
- **api:** Mark public enums and data structs #[non_exhaustive] (#345) (#379)
- **ops:** Add `Tensor::gather` (torch.gather); class-index cross entropy uses it (#78) (#430)
- Layer constructors check their hyperparameters; every one has a try_ form (#133) (#440)
- Every public type implements Debug (#116) (#442)

### Changed

- Implement comprehensive error handling with VoltaError (d5543c3)
- Remove TensorOps trait duplication anti-pattern (daba38d)
- Consolidate movement operations with IndexMapper traits (6b891b1)
- **shape:** `edge()` takes Shape; Shape::from_dims rejects zero dims and >100M (31e2edd)
- **shape:** Validate tensor shapes through Shape::from_dims (b12efe5)
- **shape:** Axis on every dim argument (567946f)
- **conv:** Conv2d and its GPU wrappers take Conv2dSpec (82eecba)
- **conv_transpose:** ConvTranspose2d takes ConvTranspose2dSpec (8d6a037)
- **maxpool:** MaxPool2d takes Window2dSpec (aa43c66)
- **autograd:** Closed GradFn enum replaces Box<dyn GradFn> (8ce5eae)
- **gpu:** Delete the orphaned non-safe broadcast-backward pipelines (49c1916)
- **ops:** Strides replaces compute_strides; movement ops end in one edge (d3fcf92)
- Delete TensorOps, make its methods inherent on Tensor (132790f)
- Split Module into Module/Layer/IndexedLayer/RecurrentLayer (1a51ab7)
- **storage:** Remove Deref/Index; explicit host access (4418437)
- **shape:** Make the integer shape defaults loud (380f563)
- **shape:** RawTensor.shape is a Shape, read through typed accessors (91ae3ca)
- **cfg:** Gate GPU wrappers at their definition, not their callers (e51b3ab)
- **gpu:** Hide kernel and pool types; test kernels in-crate (55be134)
- **gpu:** Shaders read and write data buffers through ELEM templates (Phase 5 P8, part 1) (70c3559)
- **gpu:** Delete the test-only resource monitors (#130) (#338)
- **nn:** One BatchNorm core, one causal-mask builder, shared prefix helpers (#119) (#334)
- Delete dead code and hide RawTensor's fields (#118, #176) (#376)
- **tensor:** Delete Tensor::from_raw (#62) (#390)
- **tensor:** In-place updates through Tensor methods, not borrow_mut (#61) (#393)
- **storage:** Replace HostMut's Drop write-back with an explicit with_host_mut closure (#174) (#419)
- **autograd:** `GradFn::backward` returns storages, not tensor handles (#268) (#434)
- One dispatch path, shared conv/window params, length checks (#287)
- Make state-dict loads atomic: plan, then apply (#65) (#439)
- Panic messages name the public op or layer and show the values (#142) (#441)

### Fixed

- Implement comprehensive bounds validation for shrink() operation (3d3b494)
- Resolve GPU bind group mismatches in binary/unary backward shaders (396b30f)
- Use-after-free in Storage GPU cache (991a8b7)
- Fix: replaced statrs::erf with libm::erff (1457ef9)
- **H2:** Make bce_with_logits_loss numerically stable (f0ee3a9)
- **H7:** Load_state_dict preserves device and validates shape (1709f92)
- **matmul:** Read dims from MatmulSpec, retiring B4 (4b5514d)
- **conv:** CPU iGEMM was wrong for batch > 1; convert conv-family indexing (a2c3e40)
- **sequential:** Reject layer names whose state_dict keys collide (e6dd279)
- **conv:** Direct-conv backward falls back to CPU when a GPU gradient kernel misses (30fb3dc)
- **storage:** GPU storage shares a write-once host copy and writes copy-on-write (H3) (0fb46be)
- **device:** A GPU label always means GPU storage; to_device moves grad (893b317)
- **device:** A GPU kernel that returns None panics instead of falling back (1fc289c)
- **device:** Backward through to_device is a transfer, not a gap (2641120)
- **lstm:** Slice gates with shrink so the graph and device survive (B2) (e4b9eef)
- **nn:** PositionalEncoding adds a shrink of its table, not a host copy (B3) (8442aaa)
- **gpu,autograd:** !Send scope guards; checked GPU movement packing (H5) (2e4d86b)
- **io:** Transpose PyTorch Linear weights by layer name, not key pattern (c32fb28)
- **rng,tests:** Seeded DataLoader shuffle, loud with_rng; two blind or racy tests (b3e69a8)
- **examples:** Vae, mnist_cnn, mnist_rnn exit instead of training on fake data (e6bf647)
- **tensor:** Check_gradients puts other leaves' gradients back (4b01b32)
- **examples:** Load_external_mnist compares against PyTorch; dcgan exits without MNIST (33ab612)
- Silent fallbacks in examples, tests, and he_initialization; check-fallbacks matches them (d0e45b9)
- **examples:** Vae trains instead of diverging to NaN (95e88fa)
- **gpu:** Count uploads as pending work so gpu_sync flushes them (8008088)
- **examples:** Dcgan no longer re-normalizes MNIST; examples assert their input range (6e4d2c1)
- **io,dtype:** Reject unsupported safetensors dtypes (S18); promote matches torch (d8e3330)
- **ops:** Copy GPU gradients to the host in MulAcc and rank-5 permute backward (c0ba426)
- **nn:** ConvTranspose2d adds its bias on the host, not to a CPU tensor (audit A1) (9245f2b)
- **gpu:** 1-D kernels cover any size with a 2-D dispatch (audit A2) (8999005)
- Sequential::to_device reaches children; optimizers validate hyperparameters; GradScaler refuses f16 (S13, A3, A4) (418a637)
- **nn:** Norm layers take a half input with f32 params as torch does (audit A5) (0282233)
- **gpu:** Measure process memory on Linux and read real system RAM (9d8aff5)
- **autograd:** Only leaves keep a gradient (S1) (#89)
- **gpu:** Tanh stays finite for large inputs on Metal (S24) (#146)
- **autograd:** Contiguous is a differentiable identity (#46)
- **ops:** Modulo is floored on CPU and GPU, and differentiable (S5) (#157)
- **nn:** BatchNorm rejects one value per channel in training mode (S12) (#161)
- **nn:** Attention adds no residual; the block owns it (S10, S11) (#163)
- **nn:** Bce_loss clamps its logs; cross entropy gathers class indices (S7, S8) (#164)
- **deps:** Migrate to rand 0.10 and pin naga back to wgpu's version (#203)
- **optim:** Allocate optimizer state on first step; count Adam steps per parameter (S17) (#208)
- **autograd:** Backward refuses a graph whose saved inputs were rewritten in place (S9) (#209)
- **ops:** Add sums pairwise on the CPU so a long sum stops stalling (S6) (#211)
- **gpu:** Make erf relatively accurate near zero; unignore the S21 repros (S21, S22) (#215)
- **ops:** Max ties and NaN follow torch, op by op (S2, S3, S4) (#220)
- **io:** Give TensorData a dtype so non-f32 models save (A8) (#228)
- **io:** Reject state-dict key collisions, short DataLoader targets, bad MNIST headers (#226)
- **loss:** Reject float targets that don't match the logits' shape (#224)
- **nn:** Apply transformer dropout, fix intermediate_dim, allow cross-attention (#225)
- **tensor:** Check_gradients fails on NaN and refuses f16/bf16 inputs (#227)
- **gpu:** Hold a reduction's partials until its encoder is submitted (#244) (#246)
- Panics from try_x wrappers point at the caller, not into src/ (#132) (#261)
- **examples:** Time_sequence steps per sequence on the mean loss, no NaN (#88) (#263)
- **nn:** PositionalEncoding accepts an odd embed_dim for the Learned type (#107) (#265)
- **gpu:** GPU upload length check is an assert in every profile (#173) (#267)
- **data:** Try_normalize rejects a zero, negative or non-finite std and a non-finite mean (#135) (#266)
- **device_policy:** Bound the PreferGpu fallback record (#125) (#269)
- **autograd:** Assert one gradient per parent before pairing them (#169) (#270)
- **ops:** Broadcast backward sums each gradient pairwise (#212) (#271)
- **examples:** MNIST batch count includes the partial batch; download hint points at just mnist-data (#126) (#272)
- **optim:** SGD, Adam and Muon reject non-leaf parameters (#102) (#295)
- **ops:** Try_pad_with checks padded sizes for overflow; add try_stride_op (#96, #97) (#279)
- **ops:** Matmul validates its output shape before allocating it (#165) (#280)
- **io:** Save_safetensors_typed returns InvalidData on a length mismatch (#110) (#282)
- **ops:** Div and Recip backward don't square the denominator (#92) (#284)
- **nn:** Embedding and PositionalEncoding no longer derive Clone (#108) (#285)
- **nn:** ConvAlgo::auto_select sizes the real im2col matrix from Conv2dSpec (#106) (#288)
- **tensor:** DataLoader validation, randn_like keeps dtype/device, set_requires_grad(false) refuses non-leaves (#93, #99, #134) (#294)
- **nn:** ConvTranspose2dGradFn skips unneeded grads and fails on a missing bias (#113) (#292)
- **io:** Load_partial drops keys outside the prefix, as documented (#111) (#289)
- **matmul:** Pairwise sums on the CPU rank-1 paths (#213) (#330)
- **tensor:** Drop the graph iteratively so a long chain can't overflow the stack (#91) (#333)
- **gradcheck:** Check_gradients works on GPU tensors (#124) (#351)
- **batchnorm:** Refuse a mismatched input dtype in train and eval mode (#104) (#349)
- **gpu:** Movement launches cover the result buffer, not new_shape (#301) (#357)
- **tensor:** Try_set_grad refuses a gradient of the wrong length (#324) (#363)
- **gpu:** Modulo handles infinite divisors and documents its bound (#302) (#364)
- **api:** Storage is non_exhaustive with one variant set; GPU handles internal (#326, #340, #325) (#365)
- **api:** GPU control functions exist without the gpu feature (#342) (#366)
- **gpu:** Exp2 and its backward use the exp2 builtin (#311) (#367)
- **gpu:** Matmul sums each tile into a partial and the partials pairwise (#319) (#369)
- **conv:** Direct and IGEMM weight gradients sum blocked pairwise (#318) (#370)
- **numerics:** Log_softmax without the max round trip; f16 mean_dim and norm statistics in f32 (#315, #316) (#371)
- **optim:** Adam refuses half parameters (#317) (#372)
- **optim:** Muon's CPU Frobenius norm scales by the largest entry (#322) (#374)
- **gpu:** Conv shaders test padding without u32 wraparound (#314) (#375)
- **gpu:** Launch 16x16 kernels as a flat tile list past 65535 groups; iGEMM computes each output once (#305, #307) (#380)
- **ops:** Expand backward sums each source pairwise over only the broadcast axes (#303, #308) (#381)
- **matmul:** Block Accelerate's cblas_sgemm over k so a long contraction stays accurate (#384) (#385)
- **optim:** Adam's bias-correction scales come from the host in f64 (#304) (#386)
- **gpu:** Adapter buffer limits, exact bindings, unfilled outputs (#306, #310) (#391)
- **spec:** Checked arithmetic in window_out and transpose_out (#281) (#409)
- **examples:** MNIST CNN/RNN measure accuracy on t10k (#262) (#410)
- **storage:** Refuse to build on big-endian targets (#396) (#411)
- **gpu:** Erf within an ulp of the CPU in GELU's negative tail (#323) (#413)
- **io:** Atomic save_safetensors via temp file and rename (#395) (#418)
- **gpu:** Muon Frobenius norm scales by the largest magnitude, as the CPU (#373) (#421)
- **io:** Skip the directory fsync under Miri; run Miri on src/io.rs PRs (#425) (#426)
- **panics:** Report the caller's location through binary/reduce/ternary ops, layers and the mapper (#260) (#428)
- **optim:** GPU SGD and Muon steps allocate no placeholder state buffers (#207) (#429)
- **gpu:** Readbacks and syncs wait on their own submission (#437) (#445)
- **gpu:** Optimizer placeholders are process-wide, never dropped (#433) (#446)
- **io:** Refuse requires_grad on integer and bool state-dict tensors (#458) (#468)
- **io:** Reject zero-dim state-dict entries so StateDictMapper can't hang (#459) (#469)
- **nn:** Layer try_new returns Err when a parameter would overflow (#460) (#470)
- **autograd:** Check_gradients fails an all-zero gradient; weight three degenerate checks (#461) (#471)
- Fix ProgressBar: clamp, redraw on visible change, finish once (#273)
- Track the unscale/step stage per optimizer (#274)
- Loss shapes, tied weights and the MHA mask (#95, #101, #103) (#253)

### Performance

- **storage:** Share CPU storage via copy-on-write Arc<[u8]> (6f3c367)
- **device:** `edge()` checks input devices without cloning (5f874da)
- **conv:** Im2col copies its input once, not once per kernel element (5844a23)
- **conv:** CPU input gradient visits only the windows that cover each pixel (7fe1328)
- **gpu:** Parallel multi-pass reductions with no host readback (#80) (#210)
- **autograd:** Run_backward holds the node's borrow instead of deep-copying its GradFn (#168) (#283)
- **conv_transpose:** Backward through two GEMMs instead of 7-deep loops (#230) (#336)
- **gpu:** Col2im visits only the output windows that cover each input (#309) (#368)

## [0.3.0] - 2026-01-21

Summarized from the commit log; there was no changelog at the time.

### Added

- GPU backward for unary, binary, matmul, reduction and movement ops; GPU Conv2d forward; GPU optimizer state; buffer pooling and throttling.
- Layers Embedding, PixelShuffle, BatchNorm1d, LSTM and ConvTranspose2d; BCE, NLL and KL losses.
- MNIST loader; safetensors and external model loading; Criterion benchmarks.
- Examples: MNIST CNN, DCGAN, VAE, char-LM, super-resolution.

### Changed

- Pedantic clippy cleanup, including `indexing_slicing`.

## [0.2.0] - 2025-11-25

Summarized from the commit log; there was no changelog at the time.

### Added

- Initial wgpu GPU backend; `matrixmultiply` sgemm; batched and broadcast matmul.
- Conv2d with im2col, MaxPool2d, Flatten; Muon optimizer; He initialization; LogSoftmax.
- Layer serialization; seeded RNG; `Module::train` / `eval`; CI.

## [0.1.0] - 2025-10-23

Initial release: a CPU tensor with reverse-mode autograd; unary, binary, reduce, movement and ternary ops and matmul; Linear, Conv, ReLU and Sequential; SGD and Adam; basic IO.

[Unreleased]: https://github.com/rlarson20/Volta/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/rlarson20/Volta/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/rlarson20/Volta/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/rlarson20/Volta/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/rlarson20/Volta/releases/tag/v0.1.0

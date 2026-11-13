# Architecture

A map of where things live and how they fit together, as of Phases 0–3 and 5 (`docs/plans/ADDENDUM.md` § Sequencing). This file says *what* the code does; the reasons, and the alternatives rejected, are in [`DECISIONS.md`](../DECISIONS.md), linked by entry title below. Open defects and gaps live in [GitHub issues](https://github.com/rlarson20/Volta/issues), not here.

## Module map (`src/`)

| Module | Holds |
| --- | --- |
| `tensor.rs` | `Tensor` (the public handle), `RawTensor` (the node), constructors, the user-facing op methods, losses, `sum_dim`/`max_dim` |
| `autograd.rs` | `GradFn`, `edge()`, `no_grad`, `into_loss`/`ScalarLoss`, the backward pass, `accumulate_grad` |
| `ops/` | Forward kernels and per-op grad fns: `unary`, `binary`, `compare`, `reduce`, `ternary`, `matmul`, `movement`, `loss` (fused BCE), `gpu_ops` |
| `storage.rs` | `Storage`, `HostBuf`, the sealed `Element` trait |
| `dtype.rs`, `real.rs`, `amp.rs` | `DType` and `DTypePolicy`; the `Real` trait CPU kernels are generic over; `autocast` and `GradScaler` |
| `shape.rs`, `spec.rs`, `index.rs` | `Shape`, `Axis`, `Strides`; the conv/matmul specs and `PadValue`; the panicking `fetch`/`slot`/`span` accessors |
| `device.rs`, `device_policy.rs` | `Device`; `DevicePolicy` and the fallback log |
| `error.rs` | `VoltaError` and the `Result` alias |
| `nn/` | `Module`/`Layer`/`RecurrentLayer`, `layers/`, `optim/`, `transformer/` |
| `io.rs`, `io/mapping.rs` | State dicts, bincode and SafeTensors load/save, `StateDictMapper` |
| `data/` | MNIST IDX loaders, `normalize`, `to_one_hot` |
| `gpu/` | wgpu backend (feature `gpu`): context, buffers, pools, monitors, kernels, WGSL shaders |
| `macros/` | `define_ops!`, `assert_trainable!`, `param_test!`, gradcheck and `assert_close` helpers |

## Tensors and the graph

```rust
pub struct Tensor { inner: Rc<RefCell<RawTensor>> }   // opaque; Clone bumps the Rc

pub struct RawTensor {
    pub data: Storage,
    pub shape: Shape,
    pub grad: Option<Storage>,
    pub requires_grad: bool,
    pub(crate) grad_fn: Option<GradFn>,
    pub parents: Vec<Tensor>,
    pub device: Device,
}
```

- **`Tensor` is opaque.** Downstream code cannot reach `RawTensor`'s fields or `borrow_mut()` the node; it goes through `Tensor` methods. Inside the crate, `borrow_mut()` outside `autograd.rs` and `ops/` means a method is missing (CLAUDE.md invariant 2). Why: "Opaque `Tensor`".
- **`edge()` is the only constructor of graph edges** (`src/autograd.rs`). Every op and layer output goes through it: it checks that the inputs share a device, applies the device policy to the output, and attaches `parents` and `grad_fn` only when grad mode is on, some input requires grad, the output is floating-point, and the grad fn is not trivial. Why: "`edge()` — single graph-edge constructor".
- **`GradFn` is a closed enum** with one variant per op family (unary, binary, matmul, the reductions, movement, the three conv algorithms, conv-transpose, max-pool, embedding, BCE, `ToDevice`, `Cast`, ...). `wildcard_enum_match_arm` is denied, so a new variant cannot compile without a `backward` arm. Why: "Closed `GradFn` enum".
- **Single-threaded by design.** `Rc<RefCell<..>>`, not `Arc<Mutex<..>>`; thread-local grad mode, device policy, dtype policy and autocast. Why: "`Rc` and Node Liveness", "Stay synchronous".

## Autograd (`src/autograd.rs`)

- **Entry point.** `tensor.into_loss()` checks that the tensor has one element and requires grad, and returns a `Result<ScalarLoss>`; `ScalarLoss::backward()` runs the pass. There is no `backward` on a bare `Tensor`. Why: "`into_loss` / `ScalarLoss`".
- **Order.** An iterative post-order DFS from the root builds a topological order, so a node in a diamond receives every consumer's contribution before its own backward runs. Why: "Topological Sort".
- **Only leaves keep a gradient.** An op output's gradient lives in a per-pass `pending` map and is dropped once consumed, as in PyTorch without `retain_grad`. A second `backward()` adds only its own contribution to each leaf. Why: "Only leaves keep a gradient (S1)".
- **Who writes `grad`.** Only `accumulate_grad` and `set_grad`; only `RawTensor::zero_grad` (called by `Tensor`, `Module` and the optimizers) clears it (invariant 3). `accumulate_grad` is also the backward device-policy check.
- **`no_grad()`** returns a guard over a thread-local depth counter, so scopes nest and unwind correctly. Why: "`NO_GRAD` as a reentrant depth counter".

## Shapes and specs

`RawTensor.shape` is a `Shape`; axes are `Axis`. User-supplied shapes and axes are validated once, at the public boundary (`try_` forms return `VoltaError`), and shapes computed from already valid ones use `Shape::derived`. Conv-family and matmul geometry is computed once into the specs in `src/spec.rs`, and padding values are a `PadValue`, not a bare float. Why: "One validation point, and `Shape::derived`", "`RawTensor.shape` is a `Shape`", "Conv-family specs", "MatmulSpec", "PadValue".

## Storage and dtypes (`src/storage.rs`, `src/dtype.rs`)

```rust
pub enum Storage {
    Cpu(HostBuf),                  // F16 | BF16 | F32 | F64 | I32 | I64 | U8 | Bool, each an Arc<[T]>
    #[cfg(feature = "gpu")]
    Gpu(Arc<GpuStorage>),          // GpuBuffer (F32, F16, BF16 or Bool) + OnceLock<HostBuf> host copy
}
```

- **Copy-on-write.** Cloning storage bumps a refcount; a write copies first if the buffer is shared. On the GPU the host copy is filled once by the first read and shared by every clone, and it cannot go stale because a shared buffer is never written. `Storage` has no `Deref`. Why: "CPU storage: `Arc<[u8]>` copy-on-write", "Typed host buffer", "No `Deref` on `Storage`", "GPU storage: shared write-once host copy".
- **Which dtypes compute.** CPU kernels are generic over the sealed `Real` trait (f32, f64), with two instances per `define_ops!` row. F16 and BF16 run the f32 instance and round each output. Integer and Bool tensors are data: movement, comparisons, `where`, indices and targets. Why: "Core ops in f64", "Half types run the f32 kernels", "I64 and Bool are data".
- **Casts and promotion.** `to_dtype` is a differentiable cast. Mixed-dtype binary ops promote under `DTypePolicy::Promote` and refuse under `Strict`. `Module::to_dtype` converts a whole model, buffers included. Why: "`to_dtype` is a differentiable cast", "Implicit promotion with a `DTypePolicy`", "Layers and optimizers in f64".
- **Mixed precision** (`src/amp.rs`). Inside `autocast(dtype)`, `matmul`, `Linear`, `Conv2d` and `ConvTranspose2d` cast their float inputs to F16 or BF16 and the losses cast theirs to F32, as explicit `to_dtype` nodes, so parameters stay f32; `GradScaler` scales the loss, unscales gradients through the `Optimizer` trait, and skips steps with non-finite gradients. Why: "Autocast with f32 parameters, `GradScaler`, an `Optimizer` trait".

## Devices (`src/device.rs`, `src/device_policy.rs`)

`Device` is `CPU` or `GPU(adapter_name)`; `Device::gpu()` returns the one adapter, and `to_device` to any other name panics. `to_device` is a differentiable op (`GradFn::ToDevice`), and `Module::to_device` moves every parameter.

An op's output lives on its inputs' device and nothing moves silently. Mixed-device inputs panic. A GPU input that reaches an op (or a case of one) with no GPU kernel is a *gap*: `DevicePolicy::Strict`, the default, panics, and `DevicePolicy::PreferGpu` (scoped with `with_device_policy`) runs it on the CPU, moves the result back, and records a `FallbackEvent` (`take_fallback_events()`). A kernel that exists and returns nothing panics under either policy. Why: "`DevicePolicy`: the device label is a promise", "`to_device` is a differentiable op". The current gaps are listed in [#67](https://github.com/rlarson20/Volta/issues/67).

## Errors (`src/error.rs`)

A user-supplied argument (shape, axis, range, label, bytes, state dict) is validated before any allocation and returned as a `VoltaError`; the panicking form wraps the `try_` one. A broken contract between tensors the program built panics, with a `# Panics` section. Kernels index through `crate::index` and panic out of bounds in every profile. Why: "Panic inside the program, `Result` where data comes from outside", "Indexing policy: panic, not NaN".

## Ops (`src/ops/`)

- **Elementwise unary and binary ops are tables.** `define_ops!` (`src/macros/define_ops.rs`) turns one row per op into the enum variant, the f32 and f64 forward and backward, and the GPU kernel names. Unary: `neg, recip, sqrt, exp2, exp, log2, log, sin, cos, tanh, sigmoid, relu, erf`. Binary: `add, sub, mul, div, max, mod` (floored, as `torch.remainder`), with broadcasting. Why: "`define_ops!` for unary and binary ops".
- **Comparisons** (`compare.rs`, `CmpOp`: `lt, le, gt, ge, eq, ne`) return Bool and carry no gradient.
- **Reductions** (`reduce.rs`): whole-tensor `sum`, `max`, `mean`; the per-axis `sum_dim`/`max_dim` are in `tensor.rs`.
- **Ternary** (`ternary.rs`): fused multiply-accumulate (`mulacc`) and `where_op`.
- **Matmul** (`matmul.rs`): matrix, vector and batched forms on contiguous inputs, with Accelerate's `cblas_sgemm` on macOS (feature `accelerate`, linked by `build.rs`) and `matrixmultiply` elsewhere. Why: "Matmul layout: contiguous-only".
- **Movement** (`movement.rs`, `MovementOp`): reshape, permute, expand, pad, shrink, stride; one `MovementGradFn` undoes each.
- **Losses**: `mse_loss`, `cross_entropy_loss`, `nll_loss`, `bce_with_logits_loss` are compositions of the ops above; `bce_loss` has a fused grad fn (`ops/loss.rs`) for PyTorch's gradient.

Adding an op, step by step: `dev-docs/DEVELOPMENT.md`.

## Neural networks (`src/nn/`)

- **Traits.** `Module` holds what every layer shares (parameters, state dict, `to_device`, `to_dtype`, train/eval, `zero_grad`). `Layer: Module` adds `forward(&Tensor) -> Tensor` and stays open for user layers. `RecurrentLayer` (input plus carried state) is sealed. Why: "Split layer traits; `Module` and `Layer` stay open".
- **Layers** (`layers/`): `Linear`, `Conv2d`, `ConvTranspose2d`, `MaxPool2d`, `BatchNorm1d`/`2d`, `LayerNorm`, `Dropout`, `Embedding` (an integer index tensor in), `LSTMCell`, `PixelShuffle`, `Flatten`, and the activations `ReLU`, `Sigmoid`, `Tanh`, `GELU`. `Sequential` and `SequentialBuilder` compose them with collision-free names. Why: "`Sequential` layer names can't collide".
- **Conv2d algorithms** (`ConvAlgo`): `Direct`, `Im2col`, `IGEMM` (implicit GEMM), and `Auto`, which picks by the estimated im2col size and the device (`ConvAlgo::auto_select`). Each has its own grad fn and a GPU forward and backward. Details: `dev-docs/CONV2D_IMPLEMENTATION.md`; why: "Convolution — im2col / GEMM".
- **Transformer** (`transformer/`): `ScaledDotProductAttention`, `MultiHeadAttention`, `FeedForward`, `PositionalEncoding` (a `Layer`), `TransformerBlock`, `causal_mask`. Why: "Attention owns no residual", "`PositionalEncoding` is a `Layer`".
- **Optimizers** (`optim/`): `SGD` (momentum, weight decay), `Adam` (bias correction, weight decay), `Muon` (Newton–Schulz orthogonalization, CPU and GPU), all behind the `Optimizer` trait. Why: "Muon follows the reference iteration".

## Serialization (`src/io.rs`)

A `StateDict` is a `BTreeMap<String, TensorData>`, each entry the bytes of one tensor in its own dtype. It saves and loads as bincode or SafeTensors; the eight dtypes map one-to-one to SafeTensors dtypes, and any other is a load error. `StateDictMapper` renames, filters and transposes keys for PyTorch and Hugging Face checkpoints, and `load_state_dict_checked` reports missing and unexpected keys. Why: "Linear weights are transposed by name, not by key pattern".

## GPU backend (`src/gpu/`, feature `gpu`)

- **Context.** One lazily built `GpuContext` per process (`static GPU_CONTEXT: OnceLock<Option<GpuContext>>`; `None` without an adapter). It counts pending submissions and syncs past a threshold (`gpu_sync_threshold`) for back-pressure.
- **Buffers.** `GpuBuffer` carries its dtype and returns itself to a power-of-two bucketed `BufferPool` on drop (8 per bucket, 64 total by default). GPU→CPU reads borrow `MAP_READ` buffers from a separate staging pool (also capped at 64). `monitor`, `system_monitor` and `early_warning` report memory and queue pressure.
- **Kernels.** `kernels.rs` wraps the WGSL shaders in `shaders/`. `shader_template.rs` expands each shader per element type (f32, native f16, and f16/bf16 packed two per `u32`), so one source serves every GPU dtype. Why: "GPU f16, bf16 and Bool". GPU code is gated at the definition site with a non-GPU stub, so call sites carry no `cfg`. Why: "GPU Dispatch".
- **Coverage.** Unary and binary ops (equal shapes), comparisons and `where`, casts, whole-tensor reductions, 2-D matmul, movement up to rank 4, the three conv algorithms, and the optimizer steps (including Muon) have kernels. What does not is [#67](https://github.com/rlarson20/Volta/issues/67). Rules for GPU work: `.claude/rules/gpu.md`.

## Performance

Correctness first. CPU kernels are plain single-threaded loops; matmul uses Accelerate on macOS or `matrixmultiply` elsewhere. No speed claim in this repository holds without a bench run on the target hardware (`just bench-matmul`).

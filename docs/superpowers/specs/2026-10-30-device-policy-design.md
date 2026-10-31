# DevicePolicy — design

ADDENDUM §2.1, the last Phase 3 item. Retires the silent GPU→CPU fallback (`notes/TODO.md` Phase 3, and "Mixed-device binary ops silently produce CPU output").

## Why

Volta quietly changes where a tensor lives. A GPU op whose kernel is missing or returns `None` computes on the CPU and labels its output `Device::CPU`, so the rest of the graph also runs on the CPU without saying so. A test can report "GPU works" when nothing ran on the GPU. That breaks the project rule that numeric code fails loudly (invariant 5) and that nothing is claimed working that wasn't exercised (invariant 7).

Nobody depends on the current GPU behavior, so breaking it is acceptable. The errors this produces become the list of missing kernels.

## What mature libraries do (the model)

PyTorch: an op's output is on its inputs' device; mixed devices raise; a missing kernel raises `NotImplementedError`; the only fallback is opt-in (`PYTORCH_ENABLE_MPS_FALLBACK`), runs unimplemented ops on the CPU with a warning, and returns the result to the original device. A kernel that runs and fails (e.g. OOM) is never retried on the CPU. candle and JAX also error on mixed devices. None of them silently changes the device.

## Goals

1. **The device label is a promise.** A tensor's `device` says where its storage is, and an op's output is on its inputs' device. No op returns `Device::CPU` for GPU inputs.
2. **`DevicePolicy::Strict` is the default.** Under it, each of these panics with a message naming the op:
   - inputs on different devices;
   - a GPU input reaching an op, or a case of an op, with no GPU kernel (a *gap*);
   - a GPU kernel that exists and returns `None` (a *failure*);
   - a backward gradient that would land on a different device from its parameter.
3. **`DevicePolicy::PreferGpu` is the opt-in.** Gaps only: the op runs on the CPU, its result is uploaded to the inputs' GPU, and a `FallbackEvent` is recorded. Mixed devices and kernel failures still panic.
4. **No GPU means `to_device(GPU)` panics**, under every policy, as do `Storage::gpu` and `Storage::new_zeros(_, GPU)`. A GPU-labeled tensor never holds CPU storage.
5. Backward obeys the same rules as forward.

## Non-goals

- Writing missing kernels (batched matmul, broadcasting binary forward, `sum_dim`, `max_dim`, maxpool, embedding, conv-transpose, and so on).
- `Result`-returning device errors. Errors are panics for now; the H4 error-policy decision revisits this.
- Moving the GPU layer from `Option` to `Result<_, GpuError>` (TODO, Rust/errors).
- The `Backend` trait (Phase 4).
- `CpuOnly`. Keeping tensors on the CPU already gives that.

## Anti-goals

- Any path where the policy is consulted and the device still changes silently.
- `PreferGpu` hiding a broken kernel: a kernel that exists and returns `None` must panic under every policy.
- An exception for CPU scalar constants (PyTorch allows 0-dim CPU tensors next to CUDA tensors). Volta's scalars are shape `[1]` and the exception would reopen the mixed-device hole; library code builds constants on its input's device instead.

## Constraints

- Ops return `Tensor`, not `Result`; device errors panic.
- `edge()` stays the only graph constructor (invariant 1), and `accumulate_grad`/`set_grad` stay the only `grad` writers (invariant 3).
- The crate is single-threaded (`Rc<RefCell<_>>`); thread-local state follows the `NO_GRAD_DEPTH` pattern.
- `--no-default-features` builds and tests keep working. Without the `gpu` feature, no tensor can be on the GPU, so the policy only affects `to_device(GPU)`, which panics.

## How it works (decided)

**Two enforcement layers.**

1. **Backstops at the choke points**, which catch every site whether or not it was edited:
   - `edge()` checks that all inputs share one device (else panic, every policy). If the output device it was given differs from the inputs' device, that is a gap: `Strict` panics; `PreferGpu` uploads the storage to the inputs' device, records a `FallbackEvent`, and builds the output there. It also asserts the storage matches the device label.
   - `Tensor::accumulate_grad` already moves a gradient to its tensor's device with `to_device`. That move becomes the backward gap check: if the incoming storage is on a different device from the tensor, `Strict` panics and `PreferGpu` uploads and records.
   - `RawTensor::new_with_storage` asserts storage and label agree.
2. **Per-site edits where a kernel exists.** Today `if let Some(s) = gpu_x(..) { .. }` falls through to the CPU when `gpu_x` returns `None`. When the inputs are on the GPU and a kernel exists for this case, that `None` is a failure: replace the fall-through with a panic. The CPU branch then runs only for CPU inputs, or for gaps, where it passes `Device::CPU` and the `edge()` backstop applies the policy. Delete the "falling back to CPU" `log::warn!`s.
   - Sites that compute on the CPU and then call `.to_device(&device)` themselves (`sum_dim`, `max_dim`, the scalar reductions' CPU branch) hide a fallback from the backstop. They pass `Device::CPU` instead, so `edge()` sees the gap.
   - `GpuKernels` returns `None` only through `get_gpu_context()?` and buffer `?`s, never on purpose for unsupported inputs, so a `None` from a wrapper for GPU inputs is a failure. Shape cases a kernel doesn't cover (differing binary shapes, batched matmul) are already decided at the call site before dispatch, so they are gaps.

**Layers build constants on their input's device.** Masks, positional encodings, LSTM zero states, GELU and attention constants, and similar tensors are currently created on the CPU and would hit the mixed-device panic in both policies. Each such site creates its constant on the device of the tensor it combines with.

**`to_device` moves the gradient too.** Today it copies `grad` unchanged ("keep gradients on CPU for now"), which breaks goal 1.

**Public API** (in a new `src/device_policy.rs`, re-exported from the crate root):

- `#[non_exhaustive] pub enum DevicePolicy { Strict, PreferGpu }`, `Default` = `Strict`.
- `pub fn device_policy() -> DevicePolicy`.
- `pub fn with_device_policy(policy) -> DevicePolicyGuard`: sets it for the guard's lifetime and restores the previous one on drop, so scopes nest (same shape as `no_grad()`).
- `pub struct FallbackEvent { pub op: String, pub pass: Pass }` with `Pass::{Forward, Backward}`, and `pub fn take_fallback_events() -> Vec<FallbackEvent>`, which drains this thread's log.

**Panic messages** name the op and the fix:
- mixed: `"{op}: expected all inputs on one device, found {a} and {b}"`;
- gap under `Strict`: `"{op}: no GPU kernel for this case; wrap the call in with_device_policy(DevicePolicy::PreferGpu) to run it on the CPU"`;
- failure: `"{op}: GPU kernel returned no result"`.

The op name comes from a new `GradFn::op_name()` (variant name, plus the inner `BinaryOp`/`UnaryOp`/`MovementOp` where there is one).

## Calls made without asking (check these)

- `CpuOnly` dropped.
- A guard (`with_device_policy`) instead of the sketch's bare `set_device_policy()`, for nesting.
- `FallbackEvent`s are pulled from a thread-local log rather than returned from each op (ops return `Tensor`, so they can't return the event). `PreferGpu` also emits `log::warn!` the first time each op falls back on a thread, not on every call, so training loops don't flood the log.
- No CPU-scalar exception.
- `to_device(GPU)` without a GPU panics under every policy, not only `Strict`.
- "Same device" means equal `Device` values, names included, so a future multi-GPU `Device` (`cuda:0` vs `cuda:1`) needs no change to the check. To make that safe today, `to_device(Device::GPU(name))` panics unless `name` is the adapter's name (`Device::gpu()`), the way PyTorch rejects `cuda:7` on a one-GPU machine. Test helpers and examples stop inventing names like `GPU("ParityTest")` and use `Device::gpu()`. Multi-GPU support itself is a non-goal.
- `PreferGpu` is the thread's starting policy until the final task flips it to `Strict`, so every intermediate commit passes `just ci`. Only the finished change has to meet the spec.

## Left to the builder

- The order of per-site edits and how to group commits.
- How existing GPU tests and examples that hit gaps are fixed: wrap in `PreferGpu` and assert the expected `FallbackEvent`s, or switch to an op that has a kernel. A test that only "passed" because of a silent fallback must assert the fallback now, not hide it.
- The wording of the DECISIONS.md entry and the ADDENDUM §2.1 / Phase 3 deviation notes.

## Done when

- Under `Strict`, every GPU test either passes with no fallback or is rewritten to expect a panic or a `FallbackEvent`.
- New tests cover: mixed devices panic under both policies; a gap panics under `Strict` and records exactly one event under `PreferGpu` with GPU output equal to the CPU result; backward under `PreferGpu` leaves GPU gradients on GPU parameters; nested guards restore the previous policy; `to_device(GPU)` panics with `--no-default-features`.
- No library code passes `Device::CPU` to `edge()` for GPU inputs without going through the backstop; no `"falling back to CPU"` log lines remain.
- `just ci` passes. TODO.md Phase 3 `DevicePolicy` and the mixed-device line are checked off, the stale H3 `storage.rs` line is ticked, and a DECISIONS.md entry records this design.

# DevicePolicy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace every silent GPU→CPU fallback with a `DevicePolicy` that panics by default (`Strict`) and, when opted into (`PreferGpu`), runs only known kernel gaps on the CPU, returns the result to the GPU, and records a `FallbackEvent`.

**Architecture:** A new `src/device_policy.rs` holds the policy, its RAII guard, and the event log. Enforcement sits at the three choke points (`edge()` for forward, `Tensor::accumulate_grad` for backward, `RawTensor::new_with_storage` for label/storage agreement), so every op is covered without editing each one. Per-site edits then turn "a kernel exists but returned `None`" into a panic under every policy.

**Tech Stack:** Rust 2024, wgpu (feature `gpu`), `log`.

**Spec:** `docs/superpowers/specs/2026-10-30-device-policy-design.md`

## Global Constraints

- Device errors are panics. No `Result` API (deferred to H4).
- `edge()` is the only graph constructor; `accumulate_grad`/`set_grad` are the only `grad` writers (CLAUDE.md invariants 1, 3).
- Library code never prints; use `log::` (invariant 4).
- "Same device" is `Device` equality, names included (multi-GPU-ready). `to_device(Device::GPU(name))` with a name other than the adapter's panics: `"to_device({device}): no such GPU; the available GPU is {Device::gpu()}"`.
- Panic messages, verbatim:
  - mixed: `"{op}: expected all inputs on one device, found {a} and {b}"`
  - gap under `Strict`: `"{op}: no GPU kernel for this case; wrap the call in with_device_policy(DevicePolicy::PreferGpu) to run it on the CPU"`
  - failure: `"{op}: GPU kernel returned no result"`
  - no GPU on `to_device`: `"to_device({device}): no GPU is available"`
- `DevicePolicy` is `#[non_exhaustive]` with exactly `Strict` and `PreferGpu`.
- Both `cargo test` (default features) and `cargo test --no-default-features` pass at every commit; `just ci` before each commit. The pre-commit `cargo-test` hook has known flaky gradient checks (`notes/TODO.md`, "Flaky random-input gradient checks"): rerun before suspecting the change.
- GPU tests start with `if skip_if_no_gpu() { return; }` (`tests/common/mod.rs`).

## Review Focus

1. **Invented GPU names.** `to_device(Device::GPU("ParityTest"))` must panic instead of producing a tensor that later fails the mixed-device check against a `Device::gpu()` tensor far from the cause. Test in Task 2.
2. **Gaps under `no_grad()`.** The `edge()` backstop must run whether or not grad is attached; output still lands on the GPU and one event is recorded. Test in Task 3.
3. **A panic inside a policy scope.** The guard must restore the previous policy on unwind, or one failed test poisons the next on the same thread. Test in Task 1.
4. **A gap in the middle of a graph.** GPU kernel op → gap op → GPU kernel op: backward must give values equal to the all-CPU run, with gradients on the GPU. Test in Task 4.
5. **Event draining.** `take_fallback_events()` returns forward and backward events in order and empties the log; a second call returns nothing. Test in Task 4.

---

### Task 1: `device_policy` module

**Files:**
- Create: `src/device_policy.rs`
- Modify: `src/lib.rs` (add `pub mod device_policy;` beside `pub mod device;` and `pub use device_policy::{DevicePolicy, DevicePolicyGuard, FallbackEvent, Pass, device_policy, take_fallback_events, with_device_policy};`)

**Interfaces:**
- Produces:
  - `#[non_exhaustive] #[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum DevicePolicy { Strict, PreferGpu }`; `Default` returns `Strict`.
  - `pub fn device_policy() -> DevicePolicy`
  - `#[must_use] pub fn with_device_policy(policy: DevicePolicy) -> DevicePolicyGuard`; the guard stores the previous policy and restores it in `Drop`.
  - `#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum Pass { Forward, Backward }`
  - `#[derive(Clone, Debug, PartialEq, Eq)] pub struct FallbackEvent { pub op: String, pub pass: Pass }`
  - `pub fn take_fallback_events() -> Vec<FallbackEvent>`
  - `pub(crate) fn on_gap(op: &str, pass: Pass)`: `Strict` panics with the gap message; `PreferGpu` pushes an event and emits `log::warn!` the first time each `op` falls back on this thread (keep a thread-local `HashSet<String>`).
  - `pub(crate) fn kernel_failed(op: &str) -> !`: panics with the failure message.
  - `pub(crate) fn common_device<'a>(op: &str, devices: impl IntoIterator<Item = &'a Device>) -> Option<Device>`: returns the first device, or `None` when empty; panics with the mixed message when two are not equal (`!=`, names included).

Thread-local state follows `NO_GRAD_DEPTH` (`src/autograd.rs`): `thread_local!` `Cell<DevicePolicy>` and `RefCell<Vec<FallbackEvent>>`. Until Task 6, the thread-local **starts at `PreferGpu`**, with `// FIXME(Task 6): start at Strict` on the line. `Default` is `Strict` from the start.

- [ ] **Step 1: Write the failing unit tests** in a `#[cfg(test)] mod tests` in `src/device_policy.rs` (CPU-only; no GPU needed):
  - `default_is_strict`: `assert_eq!(DevicePolicy::default(), DevicePolicy::Strict)`.
  - `guards_nest_and_restore`: under `with_device_policy(Strict)`, open `with_device_policy(PreferGpu)`, assert `device_policy() == PreferGpu`, drop the inner guard, assert `Strict`.
  - `guard_restores_on_panic`: set `Strict`, run `std::panic::catch_unwind(|| { let _g = with_device_policy(PreferGpu); panic!() })`, assert `device_policy() == Strict`.
  - `strict_gap_panics`: `#[should_panic(expected = "Foo: no GPU kernel for this case")]` with `Strict` set, `on_gap("Foo", Pass::Forward)`.
  - `prefer_gpu_gap_records_and_drains`: under `PreferGpu`, `take_fallback_events()` (clear), `on_gap("Foo", Forward)`, `on_gap("Bar", Backward)`; assert the take equals `[FallbackEvent{op:"Foo",pass:Forward}, FallbackEvent{op:"Bar",pass:Backward}]`; assert a second take is empty.
  - `mixed_devices_panic`: `#[should_panic(expected = "Foo: expected all inputs on one device")]`, `common_device("Foo", [&Device::CPU, &Device::GPU("x".into())])`.
  - `different_gpus_panic`: `#[should_panic(expected = "Foo: expected all inputs on one device")]`, `common_device("Foo", [&Device::GPU("a".into()), &Device::GPU("b".into())])`.

- [ ] **Step 2:** `cargo test --no-default-features --lib device_policy`. Expected: compile failure (module missing).
- [ ] **Step 3:** Implement the module and the re-exports. Document every public item (the merge gate counts missing docs).
- [ ] **Step 4:** `cargo test --no-default-features --lib device_policy`. Expected: 7 passed.
- [ ] **Step 5:** `just ci`, then commit: `feat(device): add DevicePolicy, its guard, and the fallback event log`.

---

### Task 2: Device labels never lie

**Files:**
- Modify: `src/storage.rs` (`Storage::gpu` ~line 152, `Storage::new_zeros` ~line 186, `Storage::to_device` ~line 503)
- Modify: `src/tensor.rs` (`RawTensor::to_device` ~line 922, `RawTensor::new_with_storage`)
- Modify: every invented GPU name (23 sites: `src/lib.rs`, `src/tensor.rs`, `tests/{gpu_unary_test,gpu_smoke_test,gpu_test,improvements_test}.rs`, `tests/common/mod.rs`, `examples/gpu.rs`). `tests/common::gpu_device()` returns `Device::gpu().expect("checked by skip_if_no_gpu")`.
- Test: `tests/device_policy.rs` (create), plus a `#[cfg(test)]` unit test in `src/storage.rs` for the no-GPU build

**Interfaces:**
- Produces: `Storage::to_device`, `Storage::gpu`, and `Storage::new_zeros(_, &Device::GPU(_))` panic with the no-GPU message when no GPU is available (feature off, or `is_gpu_available()` false); they no longer return CPU storage. `RawTensor::to_device(Device::GPU(name))` panics with the no-such-GPU message when `Some(Device::GPU(name)) != Device::gpu()`. `RawTensor::to_device` also moves `grad` to the new device. `new_with_storage` asserts `storage.is_gpu() == device.is_gpu()` with message `"tensor labeled {device} but its storage is on the {CPU|GPU}"`.

- [ ] **Step 1: Write the failing tests.**
  - `src/storage.rs`, `#[cfg(all(test, not(feature = "gpu")))]`: `to_device_gpu_without_gpu_panics`, `#[should_panic(expected = "no GPU is available")]`, `Storage::cpu(vec![1.0]).to_device(&Device::GPU("x".into()))`.
  - `tests/device_policy.rs` (GPU): `to_device_moves_grad`: a CPU tensor with `set_requires_grad(true)` and `set_grad(Storage::cpu(vec![1.0; 4]))`, then `.to_device(gpu_device())`; assert `t.grad_storage().unwrap().is_gpu()` and values unchanged (`t.grad().unwrap() == vec![1.0; 4]`).
  - `tests/device_policy.rs` (GPU, Review Focus 1): `unknown_gpu_name_panics`, `#[should_panic(expected = "no such GPU")]`, `RawTensor::new(vec![1.0], &[1], false).to_device(Device::GPU("ParityTest".into()))`.
- [ ] **Step 2:** `cargo test --no-default-features --lib to_device_gpu_without_gpu` and `cargo test --test device_policy`. Expected: both fail.
- [ ] **Step 3:** Implement. Delete the "keep gradients on CPU for now" comment along with the behavior it describes. Fix any existing test that relied on a GPU-labeled CPU tensor; that is a test of the old lie, so `--no-default-features` tests that call `to_device(GPU)` either become `#[cfg(feature = "gpu")]` or assert the panic.
- [ ] **Step 4:** `cargo test` and `cargo test --no-default-features`. Expected: all pass.
- [ ] **Step 5:** `just ci`, commit: `fix(device): a GPU label always means GPU storage; to_device moves grad`.

---

### Task 3: Forward backstop in `edge()`, and layer constants on the input's device

**Files:**
- Modify: `src/autograd.rs` (`GradFn::op_name`, `edge()` ~line 120)
- Modify (constants built on the CPU and combined with an input; find them by running the GPU tests once the backstop is in): `src/nn/transformer/{mask.rs, attention.rs, multi_head_attention.rs, positional_encoding.rs}`, `src/nn/layers/{lstm.rs, gelu.rs, batchnorm.rs, layernorm.rs, dropout.rs, linear.rs, embedding.rs}`, and any other site the tests surface
- Test: `tests/device_policy.rs`

**Interfaces:**
- Consumes: `common_device`, `on_gap`, `Pass` (Task 1).
- Produces: `pub(crate) fn op_name(&self) -> String` on `GradFn`: the variant name, plus the inner op's `Debug` where one exists (`"Binary(Add)"`, `"Unary(Exp)"`, `"Movement(Permute)"`), otherwise the bare variant (`"MatMul"`, `"SumDim"`). `edge()` behavior: `common_device(&grad.op_name(), inputs' devices)`. If the inputs are on the GPU and the `device` argument is CPU, call `on_gap(.., Pass::Forward)`, move `data` to the inputs' device, and use that device. CPU inputs with a GPU `device` argument panic (an op bug).

`op_name()` builds a `String` on every `edge()` call. Build it only on the mismatch and gap paths: pass a closure or check kinds first.

- [ ] **Step 1: Write the failing tests** in `tests/device_policy.rs`:
  - `gap_strict_panics`: `#[should_panic(expected = "Binary(Add): no GPU kernel for this case")]`; under `with_device_policy(Strict)`, GPU `[2,3]` + GPU `[3]`.
  - `gap_prefer_gpu_runs_and_records`: under `PreferGpu`, drain events; GPU `[2,3]` + GPU `[3]` gives an output with `device().is_gpu()`, data equal to the CPU result (`assert_close`, tol 1e-6), and `take_fallback_events() == [FallbackEvent { op: "Binary(Add)".into(), pass: Pass::Forward }]`.
  - `batched_matmul_gap_prefer_gpu`: same shape of test for `[2,3,4] @ [2,4,5]`, op `"MatMul"`, tol `TOL_MATMUL_FWD`.
  - `mixed_devices_panic_under_prefer_gpu`: `#[should_panic(expected = "expected all inputs on one device")]`, CPU `[3]` + GPU `[3]` under `PreferGpu`.
  - `gap_under_no_grad_still_placed` (Review Focus 2): under `PreferGpu` and `no_grad()`, the `[2,3]` + `[3]` output is on the GPU and one event is recorded.
  - `gpu_layers_keep_constants_on_device`: under `PreferGpu`, a forward pass with GPU inputs and parameters through `MultiHeadAttention` (causal mask), `PositionalEncoding`, `LSTMCell`, `GELU`, `LayerNorm`, `BatchNorm`, and `Dropout` (training) completes without the mixed-device panic, and every output is on the GPU.
- [ ] **Step 2:** `cargo test --test device_policy`. Expected: the new tests fail.
- [ ] **Step 3:** Implement `op_name` and the `edge()` backstop.
- [ ] **Step 4:** `cargo test` (default features). Every mixed-device panic is a library constant built on the CPU: build it on the device of the tensor it combines with (`x.device()`). `causal_mask` already takes a `Device`, so fix its callers. Repeat until clean.
- [ ] **Step 5:** `cargo test` and `cargo test --no-default-features`. Expected: all pass.
- [ ] **Step 6:** `just ci`, commit: `feat(device): edge() enforces one input device and applies the policy to forward gaps`.

---

### Task 4: Backward backstop

**Files:**
- Modify: `src/tensor.rs` (`Tensor::accumulate_grad` ~line 204), `src/autograd.rs` (the backward loop's call ~line 283)
- Modify: `src/ops/gpu_ops.rs` (`gpu_binary_backward` ~line 225)
- Test: `tests/device_policy.rs`

**Interfaces:**
- Consumes: `on_gap`, `Pass`, `GradFn::op_name`.
- Produces: `pub(crate) fn accumulate_grad(&self, g: Storage, op: &GradFn)`. When `g` is on the CPU and the tensor is on the GPU: `on_gap(&op.op_name(), Pass::Backward)`, then move. When `g` is on the GPU and the tensor is on the CPU: panic (an op bug). It no longer calls `to_device` unconditionally. `RawTensor::accumulate_grad` (the storage-level add, `src/autograd.rs` ~line 351) calls `kernel_failed("accumulate_grad")` when both are on the GPU and `gpu_add` returns `None`; its CPU path runs only when both are on the CPU.
- `gpu_binary_backward` returns `None` (a gap, CPU path) for differing shapes instead of calling `gpu_binary_backward_broadcast_shapes`. That path is broken (wgpu validation panic, `notes/TODO.md` "GPU broadcast backward is broken, but unreachable"), and Task 3 just made it reachable. Leave a `// Gap until the broadcast backward bind groups are fixed (TODO.md)` comment. `gpu_binary_backward_broadcast_matches_cpu` stays `#[ignore]`d.

- [ ] **Step 1: Write the failing tests** in `tests/device_policy.rs`:
  - `backward_gap_prefer_gpu_grads_on_gpu`: under `PreferGpu`, `x` GPU `[2,3,4]` and `w` GPU `[2,4,5]`, both requiring grad; `x.matmul(&w).sum().into_loss().unwrap().backward()`. Both `grad_storage()` are GPU, values equal the CPU run (tol `TOL_MATMUL_BWD`), and the events include `FallbackEvent { op: "MatMul", pass: Backward }`.
  - `backward_broadcast_add_prefer_gpu`: `[2,3]` + `[3]` on GPU, `.sum()`, backward under `PreferGpu`. No wgpu panic, the grads are on the GPU, and `b`'s grad is `[2.0, 2.0, 2.0]`.
  - `gap_mid_graph_backward_matches_cpu` (Review Focus 4): `x.exp()` → batched matmul with `w` → `.relu()` → `.sum()`, GPU under `PreferGpu` versus all-CPU: grads equal (tol `TOL_MATMUL_BWD`), grads on the GPU.
  - `events_ordered_and_drained` (Review Focus 5): after one `PreferGpu` forward+backward with a single gap op, the take is `[(MatMul, Forward), (MatMul, Backward), ...]` with every `Forward` before every `Backward`, and a second take is empty.
- [ ] **Step 2:** `cargo test --test device_policy backward`. Expected: the new tests fail.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** `cargo test` and `cargo test --no-default-features`. Expected: all pass.
- [ ] **Step 5:** `just ci`, commit: `feat(device): backward gradients obey the device policy`.

---

### Task 5: A kernel that exists and fails is a panic

**Files (each GPU attempt that falls through to the CPU today):**
- `src/ops/unary.rs` (forward ~line 120, backward ~line 75)
- `src/ops/binary.rs` / `src/ops/gpu_ops.rs` (`gpu_binary_forward` with same-shape GPU inputs; `gpu_binary_backward` same-shape; delete the `log::warn!` at ~line 240)
- `src/ops/matmul.rs` (2-D forward ~line 245, delete the warn; the matmul backward GPU calls)
- `src/ops/reduce.rs` (`Sum`/`Max`/`Mean` forward GPU branches and their backward GPU branches)
- `src/ops/ternary.rs` (`MulAcc` ~line 195, delete both warns)
- `src/ops/movement.rs` (forward GPU wrappers and the five backward "CPU fallback" branches ~lines 415–590, 709, 887)
- `src/nn/layers/conv.rs` (im2col ~905 and col2im backward ~160, direct conv forward and backward including the branches from `30fb3dc`, igemm forward ~1195 and backward ~490–560)
- `src/tensor.rs` (`try_sum_dim` ~883 and `max_dim` ~1001: compute on the CPU and pass `Device::CPU` to `edge()` instead of calling `.to_device(&device)` themselves); likewise the scalar `reduce_op` tail in `src/ops/reduce.rs` ~line 204, which keeps uploading because its GPU branch did run a kernel

**Interfaces:**
- Consumes: `kernel_failed` (Task 1).
- The rule at each site: when the inputs are on the GPU and this case has a kernel, `gpu_x(..).unwrap_or_else(|| kernel_failed(<op name>))`. The CPU branch remains for CPU inputs and for gaps, and gaps pass `Device::CPU` so the backstops apply the policy. A `None` from a `gpu_*` wrapper for GPU inputs is a failure: `GpuKernels` returns `None` only through `get_gpu_context()?` and buffer `?`s.

- [ ] **Step 1: Write the failing tests** in `tests/device_policy.rs`. A real kernel failure can't be induced, so test the reclassified gaps instead:
  - `sum_dim_is_a_gap`: `#[should_panic(expected = "SumDim: no GPU kernel")]` under `Strict`, GPU `[2,3]` `.sum_dim(Axis(1), false)` (today it uploads silently).
  - `max_dim_is_a_gap`: the same for `max_dim`, expected `"MaxDim: no GPU kernel"`.
- [ ] **Step 2:** `cargo test --test device_policy _is_a_gap`. Expected: both fail (no panic).
- [ ] **Step 3:** Convert every site in the file list.
- [ ] **Step 4:** Verify:
  - `rtk proxy grep -rn -i "falling back to CPU" src` prints nothing.
  - `rtk proxy grep -rn "to_device(&device)" src/tensor.rs src/ops` shows only the scalar `reduce_op` tail.
  - `cargo test` and `cargo test --no-default-features` pass.
- [ ] **Step 5:** `just ci`, commit: `fix(device): a GPU kernel that returns None panics instead of falling back`.

---

### Task 6: `Strict` by default, GPU tests and examples, docs

**Files:**
- Modify: `src/device_policy.rs` (the FIXME: start at `Strict`)
- Modify, as `cargo test` surfaces them: `tests/gpu_*.rs`, `tests/parity_*.rs`, `tests/improvements_test.rs`, `tests/load_state_dict_device.rs`, `tests/movement_backward_test.rs`, GPU tests in `src/`, `examples/gpu.rs`, `examples/gpu_training.rs`, `benches/gpu_comparison.rs`
- Modify: `DECISIONS.md`, `docs/plans/ADDENDUM.md` (Phase 3 deviation bullets under "done phases", and move Phase 3 into the done table), `notes/TODO.md`, `CLAUDE.md` ("Current state")

- [ ] **Step 1:** Flip the starting policy to `Strict` and delete the FIXME. Add a unit test, `thread_starts_strict`, in a fresh thread: `std::thread::spawn(device_policy).join().unwrap() == DevicePolicy::Strict`.
- [ ] **Step 2:** `cargo test`. Each failure is a GPU test or example that hit a gap. Fix each one in one of three ways:
  - If the test's purpose is GPU parity on an op with a kernel and it hit a gap through a helper, rewrite it to avoid the gap op.
  - If the test exercises a gap op on the GPU, wrap it in `with_device_policy(PreferGpu)` and assert the exact `take_fallback_events()` it expects.
  - Examples and benches wrap their GPU section in `PreferGpu` and `log` the event count at the end.

  Never weaken an assertion to get a test passing. Record the list of gap ops that surfaced (the op names from the panics); it goes into TODO.md.
- [ ] **Step 3:** `just ci`. Expected: pass, including `--no-default-features` and examples.
- [ ] **Step 4:** Update the docs:
  - `DECISIONS.md`: a "DevicePolicy" entry. Cover why `Strict` is the default (PyTorch parity, invariant 5), gap vs. failure, why `PreferGpu` never covers failures, why `CpuOnly` was dropped, full `Device` equality with unknown GPU names rejected (multi-GPU-ready), the absence of a CPU-scalar exception, and panics until H4.
  - ADDENDUM Phase 3 deviations: the guard instead of `set_device_policy()`, the event log instead of a returned event, and `CpuOnly` dropped.
  - `notes/TODO.md`: check off Phase 3 `DevicePolicy` and the "Mixed-device binary ops" line, tick the stale H3 `storage.rs` line (pointing to `0fb46be`), and add one line per surfaced gap op ("no GPU kernel: <op>; `Strict` panics").
  - `CLAUDE.md` "Current state": Phase 3 done; next is the merge gate (or Phase 4).
- [ ] **Step 5:** `just ci`, commit: `feat(device): Strict is the default device policy`.

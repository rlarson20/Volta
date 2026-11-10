# Post-Phase-5 audit (2026-11-04)

A whole-codebase audit after the Phase 5 stack (dtypes and mixed precision, PRs #19–#28), on `audit/2026-11-post-phase5` stacked on `dtype/p8-gpu-half` (`de41abb`). It asks four questions: what is dealt with, what is newly exposed, what is fragile, and what claims don't match the code. New defects get an `A<n>` ID and an `#[ignore = "A<n>: ..."]` repro in `tests/audit_repros.rs`; confirmed sweep items keep their `S<n>`. Like the other files in this directory, this is a frozen record: live status is in `notes/TODO.md` [corrected 2026-11-10: see C2]. Original text is never edited or removed; corrections are appended under `## Corrections` at the end (`.claude/rules/audits.md`).

## Method

- **Everything run.** The full suites under both feature sets, the default one with the GPU tests on lavapipe (Mesa 25.2 software Vulkan, `shaderFloat16`; see `dev-docs/CLOUD.md`), and **every ignored test** (`cargo test [--no-default-features] -- --ignored`), to find defects whose repro now passes silently. All examples were built in release and the CPU-runnable ones run with a 150 s limit.
- **Mechanical checks** of invariants 1–8 and the area rules, with a script that greps library code only (it stops at each file's `#[cfg(test)] mod`).
- **About 270 cross-feature probes** in a scratch harness (not committed): every layer and loss in f64, f16 and bf16 on the CPU; every layer in f32, f16 and bf16 on the GPU under `Strict` and `PreferGpu`; autocast with `DTypePolicy::Strict`; state dicts; optimizers after `to_dtype`/`to_device`; the sweep items marked "not yet run"; and tensors at the GPU workgroup limit.
- **Torch checks** (2.15.0) for each behaviour classed as a divergence below.
- **Docs against code:** `notes/TODO.md` [corrected 2026-11-10: see C2] item by item, README, `CLAUDE.md`, `.claude/rules/`, `dev-docs/`, `DECISIONS.md`.

Not covered: Metal (`just test-gpu` on the Mac), performance, and the WASM target.

## 1. Dealt with

- **Phase 5 delivers what ADDENDUM §11 promised**, P0–P8, each against PyTorch fixtures (DECISIONS.md, one entry per PR). The TODO item "The dtype system is decorative" is resolved by the stack and removed here.
- **B1–B5 and H1–H8** stay fixed: their regression tests pass in both feature sets.
- **Every 2026-11 sweep defect still reproduces.** 29 ignored repros fail on the CPU and 7 GPU-only ones fail on lavapipe, as they should until fixed. One exception: **S24** (GPU `tanh` NaN for x >= 45) [corrected 2026-11-10: see C1] passes on lavapipe, because its overflow is Metal's `tanh`; it stays ignored, with the note.
- **Every runnable example completes:** regression, showcase, layernorm_demo, char_language_model, char_transformer_model (loss 3.24 to 0.035), vae, super_resolution, load_external_mnist, readme1 and readme2. time_sequence and mnist_rnn run past 150 s; mnist_cnn, dcgan and the GPU examples were not run.
- **Phase 5's own claims hold up under the probes.** Every layer and loss runs forward and backward in f64, f16 and bf16 on the CPU, with outputs and gradients in the input's dtype. Every layer runs in f32, f16 and bf16 on the GPU under `PreferGpu` (except A1), with GPU outputs and gradients. The optimizers step half and f64 parameters, GPU promotion and casts stay on the device, and GPU Bool comparisons and `where` broadcast through the gap path.

## 2. Exposed: defects

| ID | Severity | Defect | Since | Repro |
| -- | -------- | ------ | ----- | ----- |
| A1 | High | `ConvTranspose2d` with a bias on the GPU panics under **every** device policy ("expected all inputs on one device"): `forward` builds its pre-bias output as CPU storage and adds the GPU bias to it. `PreferGpu` promises that a layer with no kernel still runs. No GPU test builds a `ConvTranspose2d` (`tests/device_policy.rs` omits it), which is how it survived. Without a bias it works. | before Phase 5 | `conv_transpose2d_with_bias_runs_on_the_gpu_under_prefer_gpu` |
| A2 | High | GPU elementwise kernels dispatch `len / 256` workgroups without checking wgpu's 65535 limit, so a tensor of 16,776,961 elements or more aborts in wgpu's error handler with `wgpu error: Validation Error`. A 4096 × 4096 matrix (16,777,216) is past it. The P8 kernels (cast, compare, where, `GradScaler` unscale) go through `workgroups_for`, which refuses the same sizes with a message naming "GPU Muon buffer", whatever the kernel. Autocast casts every weight, so a 4096 × 4096 `Linear` under GPU autocast cannot run. | before Phase 5 (elementwise); P8 inherits it | `gpu_elementwise_and_cast_cover_a_4096_square` |
| A3 | Medium | `SGD` accepts a negative momentum. The CPU step tests `momentum > 0.0`, so it silently runs plain SGD; the GPU step tests `!= 0.0` and panics on an `unwrap()` (`sgd.rs`), because no velocity was allocated. PyTorch refuses it: `ValueError: Invalid momentum value: -0.5`. Closes the S17 "not yet run" note. | before Phase 5 | `sgd_refuses_a_negative_momentum` |
| A4 | Low | `GradScaler` unscales the gradients of f16 parameters. PyTorch refuses (`ValueError: Attempting to unscale FP16 gradients.`), because such a gradient can under- or overflow in f16 itself; loss scaling is for autocast's f32 parameters. | P7 | `grad_scaler_refuses_f16_gradients` |
| S13 | Medium | Confirmed (was "not yet run"): `Sequential::to_device` moves only parameters, so `PositionalEncoding`'s sinusoid table stays on the CPU and `forward` on a GPU input panics. P3b gave `Sequential` a `to_dtype` override that reaches each child; `to_device` has none. BatchNorm in a `Sequential` is unaffected, because it keeps its running statistics on the host by design and copies them to the input's device. | before Phase 5 | `sequential_to_device_moves_child_buffers` |

**Design gaps** (behaviour as specified, but the combination doesn't work; TODO entries, no repro):

- **A5: `DTypePolicy::Strict` and autocast don't mix with norm layers.** Autocast's own casts are explicit, but a `LayerNorm` or `BatchNorm` meeting bf16 activations with f32 parameters promotes implicitly, so Strict panics (`Binary(Mul): DType mismatch: expected BF16, got F32 ... under DTypePolicy::Strict`). An autocast MLP without norms runs under Strict. This is the same root as the TODO "Norm layers under autocast return f32": torch's norm kernels accept the mix and return the input's dtype.
- **A6: `check_gradients` on f16/bf16 reports failure for a correct gradient,** because a finite-difference step can't resolve at half precision. Torch's `gradcheck` fails too, but warns first ("not a double precision floating point"). Volta says nothing, and the doc doesn't mention half inputs.
- **A7: class-index `cross_entropy_loss` builds a dense `[N, C]` one-hot constant.** It is exact, but costs O(N·C) memory: about 100 MB for a batch of 512 over a 50k vocabulary. On the GPU, N·C >= 16.7M reaches A2.
- **A8: half and f64 models can't be saved, and there is no public way to build a `TypedTensorData` from a `Tensor`.** `state_dict()` panics for f64, f16 and bf16, and `save_safetensors_typed` takes `TypedTensorData`, so a user has to serialize the bytes by hand. This extends the existing TODO.

## 3. Vulnerable to breakage

- **A9: 43 bare `.unwrap()`s in library code** (optimizers, conv backward, pools, `max_reduce`). They fail loudly, as invariant 5 wants, but without a message; A3's crash is one of them. `clippy::unwrap_used` is not enabled, and `check-fallbacks` looks only for `unwrap_or` fallbacks.
- **A2's limit is the general case:** any GPU path dispatching `len / 256` one-dimensional workgroups has the same ceiling, including the P8 kernels and any new one copied from them.
- **The packed 16-bit stores (P8) rely on a contract:** no two threads of one dispatch write the same element. Each half-word store is an atomic clear plus an atomic set, which is safe against a neighbour writing the other half, not against two writers of one element. Every current kernel keeps the contract (f32 has the same race); a future accumulating kernel must too.
- **Shader validation covers only the shaders named in its list** (`shader_template::tests::SHADERS`, complete today at 25). A new `.wgsl` file left out of the list is neither validated nor checked for direct buffer access. Comparing the list with the directory in the test would close that.
- **The shader validation and the GPU half tests run in CI only on the macOS lane.** They live in `gpu`-gated modules, and the Linux lane is `--no-default-features`. The P8 summary said the validation "runs in every CI lane"; that was wrong. The Linux lane could run the full default-feature suite on lavapipe with one `apt-get install mesa-vulkan-drivers` step.
- **The `Strict` default makes most GPU models panic.** With GPU parameters, `Linear` (its bias add), every norm layer, GELU, PixelShuffle, attention and the transformer blocks hit a gap (`Binary(Add|Div|Mul)` on broadcast shapes, `SumDim`, `MaxDim`, batched `MatMul`, rank-6 permute), as do `MaxPool2d`, `Embedding` and class-index cross-entropy. This is the documented gap list, so it is not a defect, but a new user's first GPU MLP needs `PreferGpu`.
- **`gpu_stress_tests::test_auto_sync_threshold`** races other tests on the global pending-submission counter (already in TODO).

## 4. Mismatches between claims and code

**Fixed in this PR** (text only; no behaviour changed):

- `compute_dtype`'s panic said "ops compute in f32 or f64". It now names f32, f64, f16 and bf16, and says that integer and bool tensors are data.
- `Tensor::where_op`'s doc said that GPU tensors panic "until P8" and that branches of different dtypes panic; both stopped being true in P8 and P6.
- `DECISIONS.md`: six entries superseded by Phase 5 now say so in place. They cover `Cmplt` in the binary table, `IndexedLayer` and Embedding-in-Sequential being a compile error, "GPU storage is f32-only", GPU `to_dtype` panicking, and `compute_dtype` admitting only F32/F64.
- `dev-docs/ARCHITECTURE.md` was corrected:
  - It called the grad fns "trait objects"; `GradFn` is a closed enum.
  - It described the GPU storage comment as f32-only, which it no longer is.
  - It listed `cmplt` among the GPU binary ops.
  - It called binary ops "fully GPU-accelerated"; only equal shapes are.
  - It called "Linear backward" CPU-only; the gap is the forward bias broadcast.
- README: "GradFn trait dispatch" is now "`GradFn` dispatch".
- `notes/TODO.md` [corrected 2026-11-10: see C2]:
  - The decorative-dtype item is gone.
  - The GPU gap list now says `Where` has a kernel for equal shapes, and that `ConvTranspose2d` with a bias panics (A1).
  - The "non-f32 model cannot be saved" item covers half models and the missing constructor (A8).
  - S13 and S17's "not yet run" notes are resolved.
  - New items for A1–A9 and the CI lane.

**Open** (recorded, not fixed):

- **Invariant 2 debt:** 17 constructor sites in `src/nn/layers/` write `requires_grad` through `borrow_mut()`, though `Tensor::set_requires_grad` exists.
- **Invariant 3 debt:** `zero_grad` writes `grad = None` directly in `nn/mod.rs`, `Adam`, `SGD`, and `RawTensor`. The invariant says only `accumulate_grad` and `set_grad` write `grad`. Either route clearing through `set_grad` or a `clear_grad`, or name `zero_grad` in the invariant.
- **Lint exemption without a reason:** `src/utils/progress.rs` opts out of the print lint with a module-wide `#![allow(clippy::print_stderr)]` that has no `reason`. `.claude/rules/tests.md` asks for the narrowest item and a reason.
- **About 15 doc examples are marked ` ```ignore ` but run and pass** with an adapter (`src/gpu/*`, `storage.rs`, `tensor.rs`, `define_ops!`). So they never run in CI and can rot unseen; `no_run` or a guarded body would keep them compiled.
- **The `cmplt` WGSL entry point and its pipeline are dead** since P5, and P8 compiles them once per element type.
- **README "Project status"** still lists a CIFAR example and three missing example commands (S23, already tracked). Its "~223 pedantic lints remaining" figure is unverified.

## 5. Invariant check results

| Invariant | Result |
| --------- | ------ |
| 1 `edge()` builds every output | Holds, except the known `RawTensor::contiguous` (TODO). `ConvTranspose2d`'s CPU intermediate is not an output, but it is A1's cause. |
| 2 no `borrow_mut()` outside autograd/ops | 17 `requires_grad` writes in layers, plus the optimizers' in-place updates (open, above). |
| 3 only `accumulate_grad`/`set_grad` write `grad` | `zero_grad` in four places (open, above). |
| 4 library code never prints | Holds; `progress.rs` opts out without a reason. |
| 5 fail loudly | `check-fallbacks` passes; 43 bare `unwrap()`s fail loudly but without messages (A9). |
| 6 no tests asserting bugs | Holds; every defect repro is `#[ignore]`d with an ID. |
| 7 don't overclaim | ARCHITECTURE.md and two doc comments overclaimed; fixed here. |
| 8 errors by data origin | Holds for the probes run. A3's momentum is a user argument and should be validated. |
| GPU rule: gate at the definition site | The 11 `cfg(feature = "gpu")` lines Phase 5 added outside `src/gpu/` are gated definitions plus stubs, the existing pattern. |

## 6. Recommended order

1. A1 and A2: crashes on supported GPU paths at ordinary sizes. A2 needs a 2-D dispatch or a strided loop in every 1-D kernel and a shared helper; A1 needs the bias add on the output's device, plus a `ConvTranspose2d` entry in `tests/device_policy.rs`.
2. S13 and A3: small fixes (a `Sequential::to_device` override; validate momentum at construction, as torch does).
3. A5: decide whether norm layers return the input's dtype under mixed inputs, as torch does. That fixes both A5 and the TODO norm item.
4. The Linux CI lane on lavapipe, so the GPU half tests and shader validation run on every PR.
5. Hygiene: A9 (`unwrap_used`), the invariant 2/3 debts, the ignored doc examples.

## Reproduce

```bash
sudo apt-get install -y mesa-vulkan-drivers   # lavapipe, for the GPU tests (cloud: see dev-docs/CLOUD.md)
cargo test -- --ignored                        # every repro, with the GPU ones on the adapter
cargo test --no-default-features -- --ignored  # CPU repros only
cargo test --test audit_repros -- --ignored --nocapture   # this audit's repros (GPU-RAN markers)
```

## Corrections

Appended after the original text, which is unchanged (`.claude/rules/audits.md`).

### C1 (2026-11-10): the S24 range in § 1

- **Said** (§ 1, "Every 2026-11 sweep defect still reproduces"): "**S24** (GPU `tanh` NaN for x >= 45) passes on lavapipe, because its overflow is Metal's `tanh`; it stays ignored, with the note."
- **True:** the lavapipe result stands. The range is wrong: on Metal, GPU `tanh` is 0 for about 43.7 to 44.35 and NaN from 44.37, not NaN from 45. See C1 in `2026-11-sweep.md` for the cause and the evidence.
- **Fix:** PR #146 (`Fixes #49`) removes the `#[ignore]` once merged, so "it stays ignored" will no longer hold.

### C2 (2026-11-10): `notes/TODO.md` no longer exists

- **Said** (header; § Method "Docs against code"; § 4 list of files): "live status is in `notes/TODO.md`", and `notes/TODO.md` as a file audited item by item.
- **True:** PR #87 (`b20a43f`) retired `notes/TODO.md`. Live status is in the GitHub issues, one per defect ID: search the `S<n>` or `A<n>`. The file's last version is `git show 9d8aff5:notes/TODO.md`. The mentions in § Method and § 4 describe what was audited and are accurate as history; only the pointer to live status in the header is wrong, and the file cannot be opened from the working tree.

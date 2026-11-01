---
paths:
  - "src/gpu/**"
  - "src/storage.rs"
  - "src/device.rs"
  - "tests/gpu_*.rs"
---

# GPU backend (wgpu)

- CI has no GPU adapter, and GPU tests skip silently when none is found. "Tests pass" does not mean GPU-verified. Run `just test-gpu` on real hardware and say explicitly whether you did.
- **Device policy** (`src/device_policy.rs`, DECISIONS.md "`DevicePolicy`"): never fall back to the CPU from a GPU path. If a kernel covers the case and returns `None`, call `kernel_failed(op)`. If no kernel covers the case (a gap), compute on the CPU and pass `Device::CPU` to `edge()` (forward) or return a CPU gradient (backward); `edge()` and `accumulate_grad` then apply the policy. Never call `to_device` on an op's own result to hide a gap.
- **Close a GPU gap by writing the kernel.** Never add an assert or panic that refuses GPU inputs where the CPU path works, and never record "CPU only" as the fix (Muon's first H8 fix did, and was redone). Compose existing kernels (`matmul`, `permute`, `optimizer_step`) where they fit.
- **Backward:** compute every required gradient or panic. Returning `vec![None, gy]` silently freezes a parameter (the P0-2 class).
- Build a layer's constants with `x.constant_like(..)`, on its input's device; a CPU constant next to a GPU input is a mixed-device panic.
- Log levels: a `PreferGpu` fallback at `warn` (once per op per thread), init at `info`, sync lifecycle at `debug`, pool churn at `trace`. Debug with `RUST_LOG=volta::gpu=trace`.
- GPU storage is f32-only. Assert that invariant; never cast-slice f32 buffers to other dtypes.
- Gate features at the *definition* site: a gated real function plus a not-gpu stub. A `cfg(feature = "gpu")` outside `src/gpu/` seeds B5 (ADDENDUM §9).
- After editing `src/gpu/shaders/*.wgsl`, run `just validate-shaders`.

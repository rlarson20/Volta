---
paths:
  - "src/gpu/**"
  - "src/storage.rs"
  - "src/device.rs"
  - "tests/gpu_*.rs"
---

# GPU backend (wgpu)

- GPU tests skip silently without an adapter. CI's macOS lane has an Apple GPU, its Linux lane none; a cloud session can install lavapipe (`dev-docs/CLOUD.md`), a software Vulkan device that runs them but is not Metal. "Tests pass" means GPU-verified only on what actually ran: say which (lavapipe, CI macOS, `just test-gpu` on the Mac).
- **Device policy** (`src/device_policy.rs`, DECISIONS.md "`DevicePolicy`"): never fall back to the CPU from a GPU path. If a kernel covers the case and returns `None`, call `kernel_failed(op)`. If no kernel covers the case (a gap), compute on the CPU and pass `Device::CPU` to `edge()` (forward) or return a CPU gradient (backward); `edge()` and `accumulate_grad` then apply the policy. Never call `to_device` on an op's own result to hide a gap.
- **Close a GPU gap by writing the kernel.** Never add an assert or panic that refuses GPU inputs where the CPU path works, and never record "CPU only" as the fix (Muon's first H8 fix did, and was redone). Compose existing kernels (`matmul`, `permute`, `optimizer_step`) where they fit.
- **Backward:** compute every required gradient or panic. Returning `vec![None, gy]` silently freezes a parameter (the P0-2 class).
- Build a layer's constants with `x.constant_like(..)`, on its input's device; a CPU constant next to a GPU input is a mixed-device panic.
- Log levels: a `PreferGpu` fallback at `warn` (once per op per thread), init at `info`, sync lifecycle at `debug`, pool churn at `trace`. Debug with `RUST_LOG=volta::gpu=trace`.
- GPU buffers carry their dtype (F32, F16, BF16, Bool as one `u32` per element; ADDENDUM §11 P8). A kernel wrapper picks pipelines with `ctx.pipelines_for(input.dtype())`, allocates outputs with `GpuBuffer::zeros_of(dtype, n)`, and binds one element type (`assert_same_dtype`); never reinterpret a buffer's bytes as another dtype. Shaders declare data buffers `array<ELEM>` and touch them only through `ld_`/`st_`/`len_` (`src/gpu/shader_template.rs`), so the packed 16-bit expansions stay correct.
- Gate features at the *definition* site: a gated real function plus a not-gpu stub. A `cfg(feature = "gpu")` outside `src/gpu/` seeds B5 (ADDENDUM §9).
- After editing `src/gpu/shaders/*.wgsl`, run `just validate-shaders` (the naga test of every shader under every element type; add a new shader to its `SHADERS` list).

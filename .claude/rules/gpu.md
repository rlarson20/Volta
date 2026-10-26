---
paths:
  - "src/gpu/**"
  - "src/storage.rs"
  - "src/device.rs"
  - "tests/gpu_*.rs"
---

# GPU backend (wgpu)

- CI has no GPU adapter, and GPU tests skip silently when none is found. "Tests pass" does not mean GPU-verified. Run `just test-gpu` on real hardware and say explicitly whether you did.
- **Backward fallback rule:** return early from a GPU path only if *every* required gradient was computed. Otherwise fall through to the CPU path. Returning `vec![None, gy]` silently freezes a parameter (the P0-2 class).
- A GPU→CPU fallback logs at `warn`. Other log levels: init at `info`, sync lifecycle at `debug`, pool churn at `trace`. Debug with `RUST_LOG=volta::gpu=trace`.
- GPU storage is f32-only. Assert that invariant; never cast-slice f32 buffers to other dtypes.
- Gate features at the *definition* site: a gated real function plus a not-gpu stub. A `cfg(feature = "gpu")` outside `src/gpu/` seeds B5 (ADDENDUM §9).
- After editing `src/gpu/shaders/*.wgsl`, run `just validate-shaders`.

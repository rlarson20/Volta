# 2026-11 Unsafe and soundness audit

Dated record. Original text is never edited or removed; corrections are appended under `## Corrections` (`.claude/rules/audits.md`).

**Date:** 2026-11-17. **Commit audited:** `ebbd683` (`improvements`). **Issue:** #300. **Finding IDs:** `U1`–`U5`.

**Scope.** Every `unsafe` block, `unsafe fn` and `unsafe impl`; the Accelerate FFI and `build.rs`; every place the crate reinterprets bytes (`bytemuck` casts, the `Pod` kernel-parameter structs, `Storage`'s host buffers, `try_from_bytes`, safetensors loading); GPU buffer mapping, the two buffer pools, in-place kernels and the unsharing write path; interior mutability (`Rc<RefCell<RawTensor>>`, thread-locals, the global GPU context, `F16_EMULATION`); and whether safe public calls can reach undefined behavior, a wgpu validation panic, or a silently broken invariant. GPU kernel logic and the shape of the public API are out of scope (the GPU kernel audit and #299).

**Method.** `grep` for every `unsafe`, cast, global and lock; read each site and its callers; then probe the public surface through integration tests on the CPU and on lavapipe, and run Miri on the modules that hold or feed the `unsafe` blocks. The probes are in the appendix, not committed as tests (invariant 6: an `#[ignore]` needs an issue ID, and this audit files none).

**What ran.** CPU and lavapipe (Mesa 25.2.8, software Vulkan, wgpu 30.0.1), debug build, Linux. Miri (`nightly-2026-11-01`, the nightly the justfile pins, `--no-default-features --lib`, filters `ops::matmul storage::`): 36 tests, 21 in `ops::matmul` and 15 in `storage`, passed with no undefined-behavior report in 320 s. **Nothing ran on Metal or on macOS**, so the Accelerate `cblas_sgemm` block was read, not executed. Each finding says "confirmed" (with the backend) or "by reading".

**Result.** No undefined behavior found. The crate has four `unsafe` sites, all in `src/ops/matmul.rs`, all with a `// SAFETY:` comment that states a true and sufficient invariant, and safe code cannot break any of them. There is no `unsafe impl` and no `unsafe fn`. The findings are safe-API soundness: one silent partial update (U1), two ways the public `GpuBuffer` surface breaks an invariant the rest of the crate relies on (U2, U3), and two low items (U4, U5).

| ID | Severity | One line | Evidence |
| --- | --- | --- | --- |
| U1 | Silent wrong update | `Tensor::set_grad` does not check the gradient's length: the CPU optimizers update a prefix of the parameter and say nothing | CPU + lavapipe |
| U2 | Stale data read | Public `GpuBuffer::copy_region` has no bounds check: it returns another tensor's old bytes from the pool's slack, or a wgpu validation panic | lavapipe |
| U3 | Panic at a distance | Public `GpuBuffer::buffer()` hands out the `wgpu::Buffer`; `destroy()` on it poisons the pool and an unrelated tensor panics later | lavapipe |
| U4 | Low (resource) | The staging pool is keyed by exact byte size and never evicts: 64 distinct sizes fill it for the life of the process | reading |
| U5 | Informational | The Accelerate block is the one `unsafe` site no tool here or in CI's sanitizer jobs executes | reading |

## Every `unsafe` site

| Site | What it is | Invariant relied on | `// SAFETY:` | Can safe code break it? | Verdict |
| --- | --- | --- | --- | --- | --- |
| `src/ops/matmul.rs:105` | `unsafe extern "C"` declaration of `cblas_sgemm` (macOS with `accelerate`) | The signature matches Accelerate's LP64 CBLAS: enums and dimensions are C `int`, scalars `float`, three pointers | n/a (a declaration; the lint covers blocks) | No | Sound by reading. `i32` for the enum arguments is right for the LP64 symbol; the ILP64 variant has a different symbol name, which this does not link |
| `src/ops/matmul.rs:131` | Call to `cblas_sgemm` | `a` holds `m*k` floats, `b` holds `k*n`, `result` holds `m*n`; leading dimensions `k`, `n`, `n`; no aliasing | Yes, complete | No: `gemm_lens` computes the three products with `checked_mul`, both input lengths are asserted against them before the block, each dimension goes through `i32::try_from(..).expect(..)`, and `result` is a fresh `Vec` | Sound by reading (U5) |
| `src/ops/matmul.rs:159` | Call to `matrixmultiply::sgemm` | Same lengths; row strides `k`, `n`, `n`, column strides 1; `c` does not alias `a` or `b` | Yes, complete | No: same asserts; strides go through `index::signed`, which panics instead of wrapping | Sound; Miri clean |
| `src/ops/matmul.rs:205` | Call to `matrixmultiply::dgemm` | As above, in `f64` | Yes, complete | No | Sound; Miri clean |

`clippy::undocumented_unsafe_blocks` is denied in `Cargo.toml`, so a new block without a comment fails `just ci`. `beta` is 0 and the output is zero-initialized, so neither gemm reads uninitialized memory. `build.rs` links `Accelerate` only when the `accelerate` feature is on and the *target* is macOS (it reads `CARGO_CFG_TARGET_OS`, not the host's `cfg!`), which matches the `cfg` on the extern block, so there is no configuration where the symbol is declared and not linked.

## Findings

### U1. `set_grad` accepts a gradient of the wrong length

`Tensor::set_grad` (`src/tensor.rs:385`) checks the gradient's dtype and moves it to the tensor's device. It does not compare `grad.len()` with the tensor's element count. What happens next depends on the backend and the optimizer.

Confirmed, a 4-element parameter of ones, `lr = 0.5`, gradient of ones:

| Gradient length | SGD, CPU | Adam, CPU | SGD and Adam, lavapipe |
| --- | --- | --- | --- |
| 2 (short) | `[0.5, 0.5, 1.0, 1.0]`, no error | `[0.5, 0.5, 1.0, 1.0]`, no error | panic: `GPU optimizer_step: buffer lengths [4, 2, 4, 4] differ` |
| 6 (long) | `[0.5, 0.5, 0.5, 0.5]`, tail ignored, no error | panic: `index 4 out of bounds for len 4 (index-mapping bug)` | panic: `buffer lengths [4, 6, 4, 4] differ` |

The CPU loops `zip` the parameter with the gradient (`src/nn/optim/sgd.rs:234-243`) or run to the gradient's length (`adam.rs:242`), so a short gradient updates a prefix and stops. This is the silent-prefix failure invariant 5 exists to prevent, reached through a safe public method with one wrong argument. A later `backward()` does catch the mismatch (`Gradient size mismatch during accumulation`), but an optimizer step before it does not.

**Fix.** Validate in `set_grad`: the length is a user-supplied argument, so by invariant 8 it is a `VoltaError` from a `try_set_grad`, with the panicking form wrapping it. Independently, the CPU optimizer loops should assert equal lengths, as the GPU kernel wrapper does (`assert_same_len`).

### U2. `GpuBuffer::copy_region` reads past the buffer's elements

`GpuBuffer` is public (`volta::gpu::GpuBuffer`), and `copy_region(offset, len)` (`src/gpu/buffer.rs:319`) copies `len` elements from `offset` without checking `offset + len <= self.len()`. The underlying `wgpu::Buffer` comes from a pool in power-of-two size classes, so it is usually larger than the tensor, and the slack holds whatever the previous owner wrote.

Confirmed on lavapipe. An 8-element buffer of 7.0 is created and dropped (it returns to the 32-byte class); a 5-element buffer `[1, 2, 3, 4, 5]` then takes it:

| Call | Result |
| --- | --- |
| `copy_region(0, 8)` | `[1, 2, 3, 4, 5, 7, 7, 7]`, no error |
| `copy_region(3, 5)` | `[4, 5, 7, 7, 7]`, no error |
| `copy_region(0, 9)` | wgpu validation panic in `Copy Buffer Region Encoder` |

So a safe call returns three elements of a tensor that no longer exists. This is not memory unsafety (the bytes belong to the same pooled buffer), but it is stale data presented as tensor contents, and past the size class it is a raw wgpu panic with no Volta message. Every internal caller passes an in-range region (`storage.rs:850`, `kernels.rs:1033-1034`, `:1185`, `:1665`).

**Fix.** Assert `offset + len <= self.len` at the top, with a message. Whether `copy_region` should be public at all is a question for #299.

### U3. `GpuBuffer::buffer()` exposes the `wgpu::Buffer`

`GpuBuffer::buffer()` (`src/gpu/buffer.rs:293`) and `Storage::gpu_buffer()` (`src/storage.rs:831`) are public, so user code can hold a `&wgpu::Buffer` to a tensor's device memory. `wgpu::Buffer::destroy` takes `&self`. A destroyed buffer still goes back to the pool when its `GpuBuffer` drops, and the pool hands it to the next allocation of that size class.

Confirmed on lavapipe: create a 100-element buffer, call `buffer().destroy()`, drop it, then build an unrelated 100-element tensor. The new tensor's upload panics with a wgpu validation error in `Queue::write_buffer`. The panic is in code that did nothing wrong, with no pointer back to the `destroy`.

The same handle is what the "a shared buffer never changes" invariant rests on (`Storage::host_buf` caches the host copy on that basis, `src/storage.rs:416`). A user cannot write through `&wgpu::Buffer` today, because the device and queue are `pub(crate)`, so `destroy` is the reachable case. Low likelihood: it takes a deliberate call.

**Fix.** Make `buffer()` and `gpu_buffer()` `pub(crate)`; no example or test outside `src/gpu/` needs them (`grep` finds no `.buffer()` call outside `src/gpu/`). This is an API-surface change, so it belongs with #299.

### U4. The staging pool never evicts

`StagingBufferPool` (`src/gpu/staging_pool.rs`) keeps read-back buffers in a map keyed by the exact byte size, with one cap of 64 buffers across all sizes. `acquire` only matches the exact size. A program that reads back 64 different tensor sizes fills the pool; after that every release is dropped, the 64 pooled buffers are kept for the life of the process whether or not their size is ever read again, and a new size never benefits. By reading. The cost is retained device memory (64 buffers of up to the largest sizes read) and lost reuse, not a wrong result.

**Fix.** Use the main pool's size classes (`BufferPool::size_to_bucket`) with a per-class cap, so nearby sizes share buffers; `download` already slices the staging buffer to the bytes it needs.

### U5. Nothing executes the Accelerate block except the Mac

The sanitizer workflow (#257) runs Miri and AddressSanitizer on Linux, where the `cfg` selects `matrixmultiply`. Miri cannot call foreign functions in any case. So the `cblas_sgemm` declaration and call are covered only by the functional tests in the macOS CI job, which check values, not memory safety. The block is short and its argument order was checked here against the CBLAS prototype (`Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc`: the call passes `m, n, k` in that order with `lda = k`, `ldb = n`, `ldc = n`), so this is a coverage note, not a defect.

**Option.** Run the macOS test job once under AddressSanitizer (`-Zsanitizer=address` supports `aarch64-apple-darwin`); it instruments the Rust side and would catch an out-of-bounds write into `result`.

## Checked and found sound

- **`Pod` kernel-parameter structs** (`src/gpu/kernels.rs:16-300`): each is `#[repr(C)]` with only `u32`, `f32` and arrays of them, and `bytemuck`'s derive rejects a struct with padding at compile time. Explicit `_padding` fields bring each to a multiple of 16 bytes. `shared_structs_match_their_rust_twins` checks the two shared structs against the WGSL; the others are matched by convention ("Must match ..."), which is a layout-drift risk, not a soundness one. By reading.
- **Host buffers** (`src/storage.rs:30-47`): `HostBuf` holds a typed `Arc<[T]>` per dtype, so alignment is the type's own and there is no byte-buffer reinterpretation on the read path. `as_bytes` casts typed slices to `u8` (always valid). By reading.
- **`try_from_bytes`** (`storage.rs:329`): checks the length is a multiple of the element size, copies through `pod_collect_to_vec` (so an unaligned input is fine; `from_bytes_accepts_unaligned_input` tests it), and builds `Bool` only from bytes 0 and 1, returning an error otherwise. No `bool` is ever made from an arbitrary byte. Miri ran the `storage::` tests clean.
- **GPU Bool read-back** (`buffer.rs:256-263`): each `u32` word is asserted to be 0 or 1 before it becomes a `bool`, so a kernel bug is a panic, not an invalid value.
- **`bits` / `from_bits`** (`storage.rs:600-624`): assert the carrier's width equals the dtype's before `cast_slice`; alignment follows because the carrier and the element have the same size and the slice came from a typed buffer. By reading.
- **Safetensors loading**: parsing and offset validation are the `safetensors` crate's; Volta converts each tensor's bytes through `try_from_bytes` and checks the shape product with `checked_mul` (`io.rs:192-200`). By reading (CI's `just miri` filter includes `io::`; this audit's Miri run did not).
- **Buffer mapping** (`buffer.rs:172-228`): `download` copies into a staging buffer, submits, maps, blocks on `poll`, copies the mapped range into a `Vec`, drops the view, unmaps, and only then releases the staging buffer to its pool. No mapped buffer reaches the pool, and no view outlives the unmap. By reading; every lavapipe probe in the three audits went through it.
- **Buffers bound in an unsubmitted encoder** (the #244 class): every kernel submits through `run_passes` (`kernels.rs:605`), which borrows the `wgpu::Buffer`s and submits before it returns, so a bound `GpuBuffer` cannot drop first. `reduce_to_scalar` holds every level's buffers until after the call (`kernels.rs:408-411`). By reading.
- **In-place kernels**: `optimizer_step` and the Muon step take their written buffers from `Storage::gpu_buffer_mut` (`storage.rs:845`), which copies a shared buffer first and clears the host cache. Confirmed on the CPU and on lavapipe: a `detach()` snapshot taken before an SGD step keeps the old values (`[1, 2]` against the stepped `[0.5, 1.0]`).
- **Locks**: both pools take their two mutexes in one fixed order (`buckets` then `total_count`; `current_count` then `pools`), `clear` never holds both, and a poisoned lock is recovered with `into_inner`. No lock is held across a wgpu call. By reading.
- **Globals**: `GPU_CONTEXT` is a `OnceLock`; the pending-submission counter is an atomic; `F16_EMULATION` is an `AtomicBool` whose setter is `#[cfg(test)]`, and both element types share a byte layout, so flipping it cannot invalidate a buffer. `NO_GRAD_DEPTH`, the device and dtype policies, autocast and the RNG are thread-locals. By reading.
- **Threads**: `Tensor` is `Rc<RefCell<..>>`, so it is neither `Send` nor `Sync`, and `tests/compile_fail/guards_not_send.rs` pins the guards. `Storage` is `Send + Sync` through `Arc`; eight threads reading one GPU `Storage` at once returned the same data (lavapipe).
- **Re-entrant borrows**: `x + x`, `x @ x` and `where(x > 0, f(x), x)` with backward gave the right gradient on the CPU; a step between forward and backward is caught by the version check with a clear message on both backends; a second `backward()` through the same graph accumulates; an `i64` storage passed to `set_grad` is rejected. No `BorrowMutError` was reached.
- **Empty buffers**: `GpuBuffer::from_slice(&[])`, `GpuBuffer::zeros(0)` and `Storage::gpu(vec![])` return empty results on lavapipe without a validation error. An f16 buffer read with `GpuBuffer::to_vec` panics with a dtype message, not a reinterpretation.

## Not covered

- Metal and macOS, including the Accelerate call (U5).
- Dependencies' own `unsafe` (`matrixmultiply`, `bytemuck`, `half`, `wgpu`, `safetensors`), beyond calling them within their documented contracts.
- AddressSanitizer here (CI runs `just asan` on the CPU suite; it was not re-run for this audit).
- Device loss and out-of-memory on the GPU: wgpu reports both as errors that panic, and no probe forced one.
- A concurrency stress of the pools beyond the eight-thread read; `just test-repeat-threads` exists for that.

## Appendix: probes

Integration tests against the public API, run with `VOLTA_REQUIRE_GPU=1 cargo test --test <file> -- --nocapture --test-threads=1`. Each call is wrapped in `catch_unwind` and prints the value or the first line of the panic.

**U1.**

```rust
let w = leaf(vec![1.0; 4], &[4], gpu);                  // requires_grad, CPU or lavapipe
w.set_grad(Storage::cpu(vec![1.0; 2]));                 // or 6
let mut o = SGD::new(vec![w.clone()], 0.5, 0.0, 0.0);   // or Adam::new(.., 0.5, (0.9, 0.999), 1e-8, 0.0)
o.step();
w.data()                                                // CPU, length 2: [0.5, 0.5, 1.0, 1.0]
```

**U2.**

```rust
drop(GpuBuffer::from_slice(&[7.0; 8]).unwrap());        // leaves 7.0s in the 32-byte class
let b = GpuBuffer::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
b.copy_region(0, 8).map(|c| c.to_vec());                // Some([1, 2, 3, 4, 5, 7, 7, 7])
b.copy_region(0, 9);                                    // wgpu validation panic
```

**U3.**

```rust
let t = GpuBuffer::from_slice(&[9.0; 100]).unwrap();
t.buffer().destroy();
drop(t);
let u = RawTensor::new(vec![1.0; 100], &[100], false).to_device(Device::gpu().unwrap());
// panics: wgpu error: Validation Error, In Queue::write_buffer
```

**Snapshot isolation, threads.**

```rust
let w = leaf(vec![1.0, 2.0], &[2], gpu);
w.set_grad(Storage::cpu(vec![1.0, 2.0]));
let snap = w.detach();
SGD::new(vec![w.clone()], 0.5, 0.0, 0.0).step();
(w.data(), snap.data())                                 // ([0.5, 1.0], [1.0, 2.0]) on both backends

let s = Storage::gpu(vec![1.0, 2.0, 3.0]);
let hs: Vec<_> = (0..8).map(|_| { let s = s.clone(); std::thread::spawn(move || s.to_vec()) }).collect();
```

**Miri.** `MIRIFLAGS="-Zmiri-disable-isolation" cargo +nightly-2026-11-01 miri test --no-default-features --lib -- ops::matmul storage::` (the `just miri` recipe with a narrower filter).

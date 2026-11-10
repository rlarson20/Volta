# 2026-11 Rust idiom audit: anti-idiomatic Rust and borrow-checker workarounds

Dated record. Original text is never edited or removed; corrections are appended under `## Corrections` (`.claude/rules/audits.md`).

Audit of the tip of `improvements` at commit `8e5fa4e`, 2026-11-10. Report only: no code was changed and no issues were opened. Nothing here is status; findings were not deduplicated against the GitHub issues (section 6). The audit was done by a read-only agent pass; a second pass then checked the cited lines for R1 to R4, R8 to R10, R14 to R20, R23, R26 and R31 against the tree and re-ran the repros for R1, R15, R16, R18 and R19 (appendix A), with the same results.

## 1. Scope and method

**Scope.** Library code in `src/` (39,799 lines, of which about 25,300 are library code and about 14,600 are `src/lib_tests/` plus trailing `#[cfg(test)] mod` blocks). `tests/` (40 files), `examples/` (17), `benches/` (4) and `build.rs` were grepped for the structural items only.

**Read for context.** `CLAUDE.md`, `.claude/rules/{autograd,gpu,nn-layers,serialization}.md`, `clippy.toml`, the `[lints]` table of `Cargo.toml`, `src/index.rs`, and these `DECISIONS.md` entries: "Only leaves keep a gradient", "`Rc` and Node Liveness", "Closed `GradFn` enum", "Opaque `Tensor`", "`edge()`", "`NO_GRAD` as a reentrant depth counter", "CPU storage: `Arc<[u8]>` copy-on-write", "No `Deref` on `Storage`", "GPU storage: shared write-once host copy", "Panic inside the program, `Result` where data comes from outside", "Stay synchronous". The other entries were read by heading only. `docs/audits/*.md` were skimmed by heading and keyword-searched for overlap.

**Counting.** Counts come from a throwaway script (`<scratch>/count.py`, not committed; `<scratch>` is a directory outside the repository). It walks `src/**/*.rs`, skips lines that start with `//`, and splits each file at the first top-level `#[cfg(test)]` followed by `mod`. Where a list below says "library code", files under `src/lib_tests/` were also excluded with `grep -v lib_tests`. Approximate equivalents:

```
grep -rn '\.clone()' src --include='*.rs' | grep -v lib_tests | grep -vE ':[0-9]+:\s*//'
grep -rnw 'unsafe' src build.rs tests examples benches --include='*.rs'
grep -rnE '\bstatic\b|thread_local!|OnceLock|OnceCell|LazyLock|Box::leak|mem::forget|transmute|MaybeUninit|set_len|from_raw_parts|get_unchecked' src build.rs --include='*.rs'
grep -rnE 'RefCell<|Mutex<|RwLock<|Cell<|Atomic[A-Z]\w+|Weak<|Rc<|Arc<' src --include='*.rs'
grep -rnE '#!?\[(allow|expect)\(' src build.rs benches examples tests --include='*.rs'
```

A nested (non-trailing) test module would be miscounted as library code. I did not find one, but did not prove there is none.

**Clippy run (evidence only).** `cargo clippy --no-default-features --lib` with `CARGO_TARGET_DIR` pointed at the scratch directory and these extra warnings: `needless_pass_by_value redundant_clone unnecessary_to_owned needless_collect large_enum_variant clone_on_ref_ptr needless_range_loop trivially_copy_pass_by_ref unused_self option_option mem_forget rc_buffer panic_in_result_fn unwrap_in_result arithmetic_side_effects cast_precision_loss float_cmp` (and a few more; log at `<scratch>/clippy.log`). It finished with 404 warnings and no errors. The GPU feature was off, so `src/gpu/` and the gated half of `src/ops/gpu_ops.rs` were not linted.

**Repros.** A throwaway crate at `<scratch>/repro` depends on Volta by path with `default-features = false`, builds into a scratch target directory, and has three binaries (`alias`, `deep`, `userin`). Each finding says "reproduced" (with build profile) or "not reproduced, by reading". Machine: Linux x86_64, 8 MiB main-thread stack, no GPU.

## 2. Summary of counts

Library code vs test code inside `src/` (script split as described above):

| Pattern | Library | Test code in `src/` |
| --- | ---: | ---: |
| `.clone()` | 359 | 150 |
| `.to_vec()` | 70 | 134 |
| `.to_owned()` | 0 | 0 |
| `.into_owned()` | 2 | 0 |
| `.collect()` (with or without turbofish) | 85 | 47 |
| `.borrow()` | 133 | 420 |
| `.borrow_mut()` | 23 | 20 |
| `try_borrow*` | 1 | 0 |
| `.unwrap()` | 0 | 263 |
| `.expect(` | 90 | 333 |
| `unwrap_or_else(\|..\| panic!` | 39 | 7 |
| `panic!` | 77 | 29 |
| `assert!` / `assert_eq!` / `assert_ne!` | 134 | 1335 |
| `debug_assert*` | 8 | 2 |
| `unreachable!` | 16 | 2 |
| `todo!` / `unimplemented!` | 0 | 0 |
| `unsafe` (keyword) | 4 | 0 |
| `as usize` | 5 | 2 |
| `as u32` / `as isize` | 0 | 0 |
| `as i64` / `as i32` / `as u8` | 2 / 2 / 2 | 0 |
| `as u64` | 4 | 0 |
| `as f32` / `as f64` | 27 / 11 | 32 / 3 |
| `#[allow(` | 17 | 0 |
| `#[expect(` | 8 | 5 |
| `mem::take/replace/swap` | 2 | 0 |
| `Option::take()` | 3 | 0 |
| `.lock().unwrap()` / `.lock().expect(` | 0 | 0 |

`.clone()` by file (library part, top ten): `tensor.rs` 50, `ops/movement.rs` 43, `ops/matmul.rs` 31, `nn/layers/conv.rs` 29, `ops/ternary.rs` 28, `ops/gpu_ops.rs` 19, `io.rs` 15, `autograd.rs` 14, `ops/reduce.rs` 13, `ops/binary.rs` 9. Most are cheap: a `Tensor` handle (`Rc` bump), a `Storage` (`Arc` bump), a `Shape` (`Vec<usize>`), or a `Device`. The deep or pointless ones are R7 to R13.

`.borrow_mut()` in library code, all 23: `tensor.rs` 8 (7 on `Tensor.inner`, 1 on the RNG thread-local), `nn/layers/batchnorm.rs` 4, `device_policy.rs` 3 (thread-locals), `nn/optim/sgd.rs` 2, `nn/optim/adam.rs` 2, `nn/optim/muon.rs` 1, `autograd.rs` 1, `nn/mod.rs` 1, `gpu/mod.rs` 1.

`.expect(` in library code by file (top): `tensor.rs` 9, `nn/optim/adam.rs` 9, `gpu/buffer.rs` 8, `ops/gpu_ops.rs` 7, `ops/ternary.rs` 7, `nn/layers/conv.rs` 7, `nn/optim/sgd.rs` 6, `gpu/kernels.rs` 5. `panic!` by file (top): `tensor.rs` 14, `storage.rs` 12, `index.rs` 5, `ops/binary.rs` 5, `ops/mod.rs` 4.

Structural items, exhaustive:

| Item | Count | Where |
| --- | ---: | --- |
| `unsafe` blocks | 3 | `src/ops/matmul.rs:108`, `:135`, `:178` |
| `unsafe extern` blocks | 1 | `src/ops/matmul.rs:83` (`cblas_sgemm`, macOS + `accelerate` only) |
| `unsafe impl Send/Sync`, `transmute`, `from_raw_parts`, `get_unchecked`, `set_len`, `MaybeUninit`, `static mut`, `Box::leak`, `mem::forget`, `ManuallyDrop` | 0 | |
| `unsafe` in `tests/`, `examples/`, `benches/`, `build.rs` | 0 | |
| `Rc<RefCell<..>>` types | 1 | `src/tensor.rs:34` (`Tensor.inner`) |
| `Weak<..>` | 0 | |
| `thread_local!` statics | 7 | `device_policy.rs:54,55,57` (`POLICY`, `EVENTS`, `WARNED`), `autograd.rs:323` (`NO_GRAD_DEPTH`), `dtype.rs:207` (`DTYPE_POLICY`), `amp.rs:34` (`AUTOCAST`), `tensor.rs:418` (`GLOBAL_RNG`) |
| Process-wide statics | 4 (+1 test) | `gpu/mod.rs:34` (`GPU_CONTEXT: OnceLock<Option<GpuContext>>`), `gpu/context.rs:23` (`F16_EMULATION: AtomicBool`), `gpu/monitor.rs:149` (`Once`), `gpu/monitor.rs:174` (`OnceLock<Option<u64>>`); test only: `lib_tests/gpu_half_tests.rs:25` (`MODE_LOCK: Mutex<()>`) |
| `Mutex<..>` fields | 6 | `gpu/pool.rs:62,66`, `gpu/staging_pool.rs:41,45`, `gpu/context.rs:93`, `gpu/system_monitor.rs:43` |
| `RwLock` | 0 | |
| `OnceLock` fields | 2 | `storage.rs:181` (GPU host copy), `gpu/context.rs:91` (three half pipelines) |
| `Cell` fields | 1 | `nn/layers/conv.rs:150` (`algo`) |
| Atomic fields | 6 | `gpu/context.rs:99,103`, `gpu/system_monitor.rs:35,37,39,41` |

Findings by severity: High 3 (one conditional on a 32-bit target and not reproduced), Medium 14, Low 18, Info 6.

## 3. Findings

Severity scale as requested: High = soundness hole, UB, or a runtime panic/abort reachable from the safe public API; Medium = real ownership or design debt, a measurable needless cost, or a breach of the repo's own invariant; Low = style or idiom; Info = deliberate and documented.

### Category 1: `Rc<RefCell<..>>` and shared ownership

#### R1 (High): dropping a deep graph overflows the stack. Reproduced.

`src/tensor.rs:398-413`; there is no `impl Drop for RawTensor` (`grep -n 'impl Drop' -r src` lists only guards, `GpuBuffer`, `HostMut`, `ProgressBar`, and a test `TempFile`).

```rust
pub struct RawTensor {
    ...
    pub(crate) grad_fn: Option<GradFn>,
    /// Inputs this tensor was computed from (the graph's edges).
    pub parents: Vec<Tensor>,
```

A node owns its parents through `Vec<Tensor>` (`Rc`), so dropping the last handle to a long chain runs the compiler-generated drop glue recursively, one set of frames per node. `run_backward` was deliberately made iterative for this reason (`src/autograd.rs:412-413`: "We simulate recursion with a stack to avoid recursion limit issues on deep graphs"; DECISIONS.md "Topological Sort"), but the destructor was not. The result is a stack overflow, which aborts the process and cannot be caught, including on the wasm32 target that DECISIONS.md "Panic inside the program" names as the first outside consumer.

Repro (`repro/src/bin/deep.rs`): `let mut y = x.clone(); for _ in 0..n { y = y.add(&x); }`, then `y.sum().into_loss()?.backward()`, then drop. Backward succeeds at every size tried. The drop aborts with "thread 'main' has overflowed its stack" at n = 100,000 in a debug build (20,000 is fine) and at n = 300,000 in a release build (100,000 is fine), on an 8 MiB main stack. A spawned thread has 2 MiB by default, so the threshold there is lower (not measured). An RNN that carries state across many steps without `detach()` builds exactly this shape.

Direction: give `RawTensor` an iterative `Drop` that moves `parents` onto an explicit worklist and only descends into nodes whose strong count is 1, or move to the arena graph (ADDENDUM Phase 4), which removes the recursion.

#### R2 (Medium): `borrow_mut()` outside `autograd.rs` and `ops/`, against Invariant 2. Partly tracked.

Eleven sites on `Tensor` handles: `src/nn/optim/sgd.rs:137`, `:190`; `src/nn/optim/adam.rs:150`, `:188`; `src/nn/optim/muon.rs:196`; `src/nn/layers/batchnorm.rs:174`, `:175`, `:398`, `:399`; `src/nn/mod.rs:112`; `src/gpu/mod.rs:266`.

```rust
// src/nn/mod.rs:112-116
let mut p = param.borrow_mut();
// `Storage::to_device` panics when the GPU is unavailable ...
p.data = p.data.to_device(&device);
p.device = device.clone();
```

```rust
// src/gpu/mod.rs:264-267
pub fn invalidate_all_tensor_caches(tensors: &[&crate::Tensor]) {
    for tensor in tensors {
        tensor.borrow_mut().invalidate_gpu_cache();
```

CLAUDE.md Invariant 2 says a `borrow_mut()` outside those two places "means `Tensor` is missing a method". Each of these writes `data` (and in `nn/mod.rs` the `device` label, two fields that must stay in agreement) directly on the `RawTensor`. `Module::to_device` is the clearest case: it sets storage and label separately, the one pair `new_with_storage` asserts on (`src/tensor.rs:589-594`). No double borrow is reachable from these by reading: each takes one `RefMut` on one tensor and calls only `Storage` methods under it, and the `alias` repro (same parameter listed twice in `SGD::new`, two steps) ran clean. Overlap: `docs/audits/2026-11-post-phase5.md` section 4 and section 5 row 2 record "17 `requires_grad` writes in layers, plus the optimizers' in-place updates (open)". The 17 `requires_grad` writes are gone at this commit; the optimizer, BatchNorm, `Module::to_device` and `invalidate_all_tensor_caches` writes remain.

Direction: add the missing `Tensor` methods (an in-place parameter update taking a closure over `&mut Storage`, a `move_to_device`, an `invalidate_host_copy`) and delete the raw borrows.

#### R3 (Medium): `run_backward` deep-copies every node's `GradFn` on every pass. Deliberate, documented, cost not stated. By reading.

`src/autograd.rs:441-449`:

```rust
let (grad_fn, parents, shape, device) = {
    let t = tensor.borrow();
    (
        t.grad_fn.clone(),
        t.parents.clone(),
        t.shape.clone(),
        t.device.clone(),
    )
};
```

DECISIONS.md ("`Rc` and Node Liveness", "Closed `GradFn` enum") records the clone as the way to release the `RefCell` borrow before `backward`. It is a borrow-checker workaround with a real cost: `GradFn` is `#[derive(Clone)]` and some variants own buffers proportional to the tensor, so the clone is a deep copy each pass. `MaxPool2dGradFn { max_indices: Vec<usize> }` (`src/nn/layers/maxpool.rs:27-30`), `EmbeddingGradFn { indices: Vec<usize>, .. }` (`src/nn/layers/embedding.rs:185-192`), `WhereGradFn { mask: Vec<bool>, .. }` (`src/ops/ternary.rs:83-90`), plus the `Vec`s inside `MatmulSpec` and `MovementOp`. The same block also clones `parents` (a `Vec` allocation), `shape` (a `Vec<usize>`) and `device` per node, and `:426` clones `parents` a second time for the DFS. By reading, no `GradFn::backward` takes a mutable borrow of the node whose `grad_fn` it is (they receive `&RawTensor` for the gradient and `&[Tensor]` for the parents), so a shared `Ref` held across the call would not conflict; I did not test that change.

Direction: hold the `Ref` across `grad_fn.backward(..)`, or store `grad_fn` behind an `Rc` so the clone is a refcount bump.

#### R4 (Medium): gradients are wrapped in fresh `Tensor` handles only to be unwrapped, and the pairing with parents is an unchecked `zip`. By reading.

`src/autograd.rs:466-474`:

```rust
let parent_grads = grad_fn.backward(&grad_out, &parents);
...
for (parent_grad, parent_ref) in parent_grads.into_iter().zip(parents.iter()) {
    if let Some(g) = parent_grad {
        let grad_storage = g.borrow().data.clone();
```

Every `backward` returns `Vec<Option<Tensor>>`. Each gradient is built as a full `Rc<RefCell<RawTensor>>` (with its own `Shape` vector, `Device` clone, empty `parents`), and the engine immediately borrows it, clones the `Storage` out, and drops the handle. The type carries nothing the engine uses beyond the storage. Second, gradients and parents are parallel sequences matched by position with `zip`, and nothing asserts `parent_grads.len() == parents.len()`. A `backward` that returns too few entries silently leaves a parent without a gradient, which is the "silently freezes a parameter" class that `.claude/rules/gpu.md` warns about and the opposite of the "zip after asserting equal lengths" rule Invariant 5 sets for tests. I found no current `backward` that returns the wrong length.

Direction: return the gradient storages directly (for example `Vec<Option<Storage>>` or a small fixed-arity type), and assert the length before the `zip`.

#### R5 (Low): node identity is a raw pointer used as a map key.

`src/autograd.rs:389`, `:410` (keys inserted at `:401`, `:422`, `:480`), and `src/tensor.rs:76`:

```rust
let mut pending: HashMap<_, Storage> = HashMap::new();
...
let mut visited = HashSet::new();
```

Both are keyed by `Tensor::as_ptr() -> *const RawTensor`. The pointer is never dereferenced, and every keyed node is kept alive by `topo_order` for the whole pass, so there is no reuse hazard as written. It is an id-in-place-of-reference pattern forced by `Rc<RefCell<..>>` having no `Hash`/`Eq`, and `Tensor` exposes no `ptr_eq`, so code outside the crate cannot ask whether two handles alias (see R30).

Direction: a `NodeId` newtype (or `Rc::ptr_eq` behind a `Tensor::ptr_eq`) would name the intent; the arena graph makes it an index.

#### R6 (Info): the `Tensor` handle design itself.

`src/tensor.rs:31-35`:

```rust
#[derive(Clone, Debug)]
pub struct Tensor {
    /// The shared node; private so downstream code cannot reach `RawTensor`'s fields.
    inner: Rc<RefCell<RawTensor>>,
}
```

Deliberate (DECISIONS.md "Opaque `Tensor`", "Stay synchronous"). Consequences, stated for completeness: `Tensor` and everything holding one (every layer, optimizer, `DataLoader` batch) is `!Send` and `!Sync`; all mutation goes through `&self` (`set_requires_grad`, `set_grad`, `zero_grad`, `load_weight`, `cast_in_place`); accessors return owned copies because a reference cannot outlive the `Ref` (`shape()` clones a `Vec<usize>` at `:101`, `device()` clones a `Device` at `:219`, `data()` copies the whole buffer at `:150`). No reference cycle is constructible by reading: `parents`/`grad_fn` are attached only in `edge()` (`src/autograd.rs:288-292`) to a tensor created on the previous line, so edges always point from newer to older nodes. There is no `Weak` and none is needed for cycles; a retained output does keep its whole history alive until dropped or `detach()`ed, which is PyTorch's behavior too.

The doc comment at `src/tensor.rs:29-30` ("For multi-threading, replace with `Arc<Mutex<RawTensor>>`") contradicts DECISIONS.md "Stay synchronous", which names the arena graph as the route and calls premature `Arc<Mutex>` a past mistake.

### Category 2: copies and clones

#### R7 (Medium): elementwise binary ops copy both operands in full even when no broadcast is needed. Known and documented as not changed.

`src/ops/binary.rs:199-201` and `:364-367`:

```rust
if from_shape == to_shape {
    return data.to_vec();
}
```

```rust
let bc_data_a =
    Self::broadcast_to(&data_a.host::<T>(), shape_a.truncate_rank(), &out_shape);
let bc_data_b =
    Self::broadcast_to(&data_b.host::<T>(), shape_b.truncate_rank(), &out_shape);
```

For `a.add(&b)` with equal shapes the CPU path makes two full copies of the inputs, collects the result into a `Vec`, then `Storage::from_compute` copies that `Vec` into an `Arc<[T]>` (`src/storage.rs:305-306`, `Arc::from(Vec<T>)` reallocates). The backward path (`src/ops/binary.rs:87-96`) repeats the two operand copies, computes both gradients whether or not each parent requires one, and `sum_over_broadcast_dims` returns `grad.to_vec()` when shapes match (`:253-255`). `broadcast_to` returning `Cow<[T]>` (or the caller checking shape equality) would remove the copies in the common case. DECISIONS.md "CPU storage" says "op outputs still allocate, and broadcasts still materialize", so the materialization is a recorded limitation; `docs/audits/OPUS_5_EVAL_AUDIT.md:352` also notes it. The same helper is used by `matmul` (`src/ops/matmul.rs:331`, `:336`, `:443`, `:468`), `compare` (`src/ops/compare.rs:106-107`) and `where` (`src/ops/ternary.rs:272-283`). Not measured.

Direction: return `Cow` from `broadcast_to`/`sum_over_broadcast_dims`, or branch on equal shapes at the call sites.

#### R8 (Medium): kernels copy a host buffer they already own a handle to. By reading.

Fourteen sites of `<borrowed host view>.to_vec()` where the result is only read:

- `src/ops/movement.rs:859`, `:965`, `:1061`, `:1210`, `:1280`: `let cpu_data = data.bits::<B>().to_vec();`
- `src/nn/layers/conv.rs:522`, `:523`, `:570`, `:571`, `:930`, `:1057`, `:1058`, `:1217`, `:1218`, for example:

```rust
// src/nn/layers/conv.rs:1057-1058
let x_data_cpu = x_data.host::<T>().to_vec();
let w_data_cpu = w_data.host::<T>().to_vec();
```

In each case `data` / `x_data` / `w_data` is a local `Storage` clone (an `Arc` bump) that lives to the end of the function, and `bits()` returns `&[B]` while `host()` returns `Cow<'_, [T]>`. Binding the slice or the `Cow` directly gives the same read access with no copy; `.to_vec()` on the `Cow` copies even when it is already borrowed. For convolution this is a full copy of the input and of the weights on every forward call, and of the output gradient on every backward call. I did not measure the cost. In the same movement kernels, `index_to_coords` (`src/ops/movement.rs:822-830`) and `vec![0; axes.len()]` allocate a `Vec` per output element inside the loop.

Direction: bind `let x = x_data.host::<T>();` and index through `&x`; keep `.to_vec()` only where the buffer is mutated (as in `muon.rs:143`, `:264`, which are legitimate).

#### R9 (Medium): `StateDictMapper` deep-copies weights in order to mutate the map while iterating. Same loop as S19. By reading.

`src/io/mapping.rs:86-96` (also `:108-118`, `:130-140`):

```rust
let mut updates = Vec::new();
for (key, value) in state.iter() {
    if key.starts_with(&old_prefix) {
        let suffix = &key[old_prefix.len()..];
        updates.push((key.clone(), format!("{new_prefix}{suffix}"), value.clone()));
    }
}
```

`value` is a `TensorData { data: Vec<f32>, shape: Vec<usize> }`, so `value.clone()` copies the tensor's whole weight buffer; the original is then removed and dropped. `add_prefix` (`:130-135`) does this for every entry, so peak memory is about twice the state dict. DECISIONS.md "Panic inside the program" rules out exactly that doubling for `load_state_dict` ("peak memory about twice the weights, gigabytes for a large model, in a browser tab"). The clone exists only because the loop cannot remove while it holds `state.iter()`; collecting the keys and moving the value out with `state.remove(&old_key)` needs no copy. `docs/audits/2026-11-sweep.md` S19 records a separate correctness bug in this same remove/insert loop. Line `:89` slices a `str` by byte range where the next function (`:111`) already uses `strip_prefix`.

Direction: collect `(old_key, new_key)` pairs only, then move each value with `state.remove(&old)` into a second map, which also addresses S19's ordering problem.

#### R10 (Low): clone-then-drop and an index loop in `SGD::step` and `Adam::step`.

`src/nn/optim/sgd.rs:106-114` (same shape at `src/nn/optim/adam.rs:117-127`):

```rust
for i in 0..self.params.len() {
    let param = self.params.get(i).expect("SGD: a parameter per index");
    let p = param.borrow();

    // Skip parameters without gradients
    let _grad = match &p.grad {
        Some(g) => g.clone(),
```

`_grad` is cloned (an `Arc` bump) and never used; `p.grad.is_none()` is the test. The loop is by index, and `step_cpu_param(&mut self, i)` / `step_gpu_param(&mut self, i)` (`sgd.rs:133`, `:188`) re-fetch `self.params.get(i).expect(..)` and clone the gradient again, because a `&mut self` helper cannot be called while `self.params` is borrowed by an iterator. `Muon::step` in the same directory shows the direct form: `for (param, velocity) in self.params.iter().zip(self.velocity.iter_mut())` (`src/nn/optim/muon.rs:195`). `adam.rs:118-119` adds a `// SAFETY:` comment and a `debug_assert!(i < self.params.len())` to a loop whose bound is that length (see R34).

Direction: iterate `params` zipped with the state vectors and pass the hyperparameters by value into a free function.

#### R11 (Low): `Device` is a heap string, cloned on every op.

`src/device.rs:9-15`:

```rust
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum Device {
    #[default]
    CPU,
    GPU(String),
}
```

`grep -rn 'device\.clone()' src --include='*.rs' | grep -v lib_tests | wc -l` gives 79. The op prologue pattern `(s.data.clone(), s.shape.clone(), s.device.clone())` (23 occurrences of the `data.clone(), shape.clone()` pair in library code) therefore allocates a `String` per GPU op and a `Vec<usize>` per op on either device, purely to end the `RefCell` borrow. Device equality (`src/autograd.rs:306`, `t.borrow().device != *first`) is a string comparison of the adapter name, yet there is one global context (R27), so the name selects nothing. Not measured.

Direction: make `Device` `Copy` (an enum with an index or an interned id) and keep the adapter name on the context.

#### R12 (Low): `check_gradients` copies the whole input twice per element. By reading.

`src/tensor.rs:1585-1588`:

```rust
let perturbed = |i: usize, up: bool| {
    let mut data = original_data.clone();
    {
        let mut host = data.host_mut::<T>();
```

`original_data.clone()` shares the buffer and `host_mut` then triggers `Arc::make_mut`, a full copy, for each of the 2N perturbed tensors, so the checker is quadratic in the input size before it runs the loss. It is a test utility on small inputs; it is listed because it is public API (`volta::check_gradients`).

Direction: none needed unless it is used on larger inputs; perturb one buffer in place and restore.

#### R13 (Low): two redundant clones and six unused `self` receivers reported by clippy.

`clippy::redundant_clone`: `src/nn/layers/conv_transpose.rs:426` (`weight: weight.clone(),`, the last use of `weight`) and `src/nn/layers/maxpool.rs:182` (`input_shape.clone()` in the final expression). `clippy::unused_self`: `src/nn/optim/adam.rs:268`, `src/nn/optim/sgd.rs:249` (the non-GPU stubs), `src/nn/transformer/attention.rs:143`, `src/nn/transformer/multi_head_attention.rs:225`, `src/ops/matmul.rs:23`, `src/ops/ternary.rs:29`. `clippy::needless_pass_by_value` (allowed crate-wide, with a rationale in `Cargo.toml`) fires at only three places: `src/ops/compare.rs:34` (two arguments) and `src/tensor.rs:366` (`set_grad(grad: Storage)`, which the rationale covers).

Direction: remove the two clones; the allow on `needless_pass_by_value` could be narrowed now that only three sites need it.

### Category 3: `unsafe`

#### R14 (High, conditional; not reproduced, by reading): the `sgemm`/`dgemm` SAFETY argument relies on `m * n` not overflowing, which is only guaranteed on 64-bit targets.

`src/ops/matmul.rs:131-136` (same at `:104-108` for cblas and `:174-178` for f64):

```rust
let mut result = vec![0.0; m * n];
// SAFETY: the asserts above fix `a.len() == m*k` and `b.len() == k*n`, and `result`
// holds m*n floats. The row strides (k, n, n) and unit column strides keep every
// access in bounds, and the three buffers are distinct allocations.
unsafe {
    matrixmultiply::sgemm(
```

All three `unsafe` blocks have a `// SAFETY:` comment and the stated invariant is correct when `m * k`, `k * n` and `m * n` are the true products. The two input products are bounded because `a` and `b` are existing tensors (`MAX_NUMEL` is 100,000,000, `src/shape.rs:11`). The output product is not validated before the call: `matmul` runs `T::gemm(..)` at `src/ops/matmul.rs:313-319` and only afterwards `Shape::derived(spec.out_dims())` at `:323`; `MatmulSpec::new` (`src/spec.rs:370-414`) does not bound `m * n`. On a 64-bit target `m * n <= 1e16` cannot wrap, so the worst case is R17 (allocation failure). On a 32-bit target built without overflow checks (the release default), `m = n = 65_536`, `k = 1` gives `m * n == 0` after wrapping: `vec![0.0; 0]`, and then `sgemm` is asked to write 2^32 floats through a dangling pointer. Both operands are 65,536-element tensors, which any caller can build. DECISIONS.md names wasm32 as the first outside consumer. I did not build for a 32-bit target, so whether the crate and its dependencies compile there, and the exact behavior, are unverified; in a debug build the multiplication panics instead.

Direction: compute the output length with `checked_mul` (or validate `spec.out_dims()` through `Shape::derived` before calling `gemm`) so the SAFETY comment's premise is established on every target.

Other `unsafe` checks, clean by reading: the FFI declaration at `:83-99` matches the CBLAS `cblas_sgemm` prototype in argument order and types for an LP64 build (`i32` for `int`, enum values 101/111), dimensions go through `i32::try_from(..).expect(..)` at `:102`, and `build.rs` links the framework only for macOS with the feature on. I could not compile or run that path here. `HostBuf` is typed `Arc<[T]>` (`src/storage.rs:31-45`) and byte input goes through `pod_collect_to_vec` (`:206-208`), which copies, so no `bytemuck` cast runs on a possibly misaligned byte buffer. `Storage::bits` (`:591-599`) uses `cast_slice` from the typed buffer's bytes to an unsigned carrier of the same width after asserting the width, which preserves alignment.

### Category 4: panics measured against Invariant 8

Library totals: 0 `unwrap()` (enforced by `#![cfg_attr(not(test), deny(clippy::unwrap_used))]` at `src/lib.rs:23`), 90 `.expect(`, 39 `unwrap_or_else(|..| panic!(..))` (the documented "panicking form wraps the `try_` one" pattern and `index.rs`), 77 `panic!`, 134 `assert*!`, 16 `unreachable!`, 8 `debug_assert*!`. `missing_panics_doc` and `missing_errors_doc` are denied, and my CPU-only clippy run reported no errors, so functions with a direct panic in their body carry a `# Panics` section. That lint does not see panics reached through a callee, which is where the breaches below are.

#### R15 (High): `try_pad_with` panics in debug builds and returns a wrong shape in release builds for a large padding. Reproduced.

`src/ops/movement.rs:1041-1045`:

```rust
let new_shape: Vec<usize> = old_shape
    .iter()
    .zip(padding)
    .map(|(d, (l, r))| d + l + r)
    .collect();
```

`padding` is a user-suppliable argument by DECISIONS.md's own list ("padding, ranges, and bytes can come from outside"), and this is the `try_` form. The sum is unchecked. On a `[2, 2]` tensor, `RawTensor::try_pad_with(&t, &[(usize::MAX, 1), (0, 0)], PadValue::ZERO)`:

- debug build: panics at `movement.rs:1044:32`, "attempt to add with overflow".
- release build: returns `Ok` with shape `[2, 2]` (the sum wrapped back to the input size). With `(usize::MAX - 1, 1)` it returns `Ok` with shape `[1, 2]` holding 2 elements, a smaller tensor from a pad.

So the function that exists to turn bad padding into a `VoltaError` either panics or silently produces a wrong tensor, depending on profile. `Shape::from_dims` on `:1047` cannot catch it because the wrapped dims are valid. The same unchecked pattern is in `src/spec.rs:39` (`let padded = input + 2 * pad;`, reached from `Conv2d`/`MaxPool2d` with a user-supplied padding) and `src/spec.rs:246` (`(input - 1) * stride + kernel`); those two were not reproduced. The clippy run with `arithmetic_side_effects` reports 308 unchecked integer operations in the CPU-only library (115 in `nn/layers/conv.rs`, 30 each in `ops/matmul.rs` and `nn/layers/conv_transpose.rs`, 24 in `ops/movement.rs`, 23 in `tensor.rs`, 11 in `spec.rs`); most operate on already-validated dims, and I did not triage them individually.

Direction: use `checked_add` and return `InvalidParameter` in every `try_` path and spec constructor that combines a user-supplied extent with a dim.

#### R16 (Medium): the MNIST loader trusts the file header for arithmetic and allocation. Reproduced. Listed as "not run" in the sweep audit.

`src/data/mnist.rs:27-33`:

```rust
let num_images = u32::from_be_bytes([header[4], header[5], header[6], header[7]]) as usize;
...
let num_pixels = num_images * rows * cols;
let mut pixels = vec![0u8; num_pixels];
```

File contents are outside data. A 16-byte file with the right magic and a count of `0xFFFFFFFF` images of 28 by 28 aborts the process in both profiles, with and without a memory limit: "memory allocation of 3367254359280 bytes failed" (exit 134; an allocation failure is an abort, not an unwind, which DECISIONS.md "Validate, then allocate" calls out). With all three fields `0xFFFFFFFF` the debug build panics at `mnist.rs:32:22` ("attempt to multiply with overflow") and the release build wraps and returns `Err(UnexpectedEof)`. `load_mnist_labels` (`:62-65`) allocates up to 4 GiB from the header in the same way (not run). Both functions return `std::io::Result`, not `VoltaError` (R37). `docs/audits/2026-11-sweep.md:74` lists "MNIST header overflow (`mnist.rs:32`)" under "Not run this round".

Direction: `checked_mul`, compare against the remaining file length (or `MAX_NUMEL`) before allocating, and return an error.

#### R17 (Medium): `matmul` allocates the output before its size is validated. Reproduced under a memory limit.

`src/ops/matmul.rs:313-323` (call order) with `:131` (`vec![0.0; m * n]`). `RawTensor::zeros(&[100_000, 1]).matmul(&RawTensor::zeros(&[1, 100_000]))` asks for 10^10 output elements from two 100,000-element operands. Under `ulimit -v 4000000` both profiles abort with "memory allocation of 40000000000 bytes failed" before `Shape::derived` gets to reject the shape. Without a limit the outcome depends on overcommit; I did not run it unlimited. The operands are program-built tensors, so by DECISIONS.md a panic would be the accepted outcome; an abort is not a panic, and the documented rule is to validate before allocating. The batched path and the vector forms were not checked for the same ordering.

Direction: build and validate the output `Shape` first, then allocate from `shape.numel()`. This also closes R14.

#### R18 (Medium): `DataLoader::new` validates nothing. Reproduced. Partly tracked as S20.

`src/tensor.rs:2037-2045`:

```rust
pub fn new(
    data: Vec<f32>,
    targets: Vec<f32>,
    data_shape: &[usize],   // e.g., [28, 28] for MNIST
    ...
    let num_samples = data.len() / data_shape.iter().product::<usize>();
```

Dataset buffers and shapes are user data; there is no `try_new` and no `# Panics` section. Three reproduced panics, identical in debug and release:

- `DataLoader::new(vec![], vec![], &[0], &[1], 1, false)` panics at `tensor.rs:2045:27` ("attempt to divide by zero").
- `batch_size = 0` panics on the first `next()` at `tensor.rs:544` ("shape [0, 1] has a zero dimension").
- Four samples with one target panics on the second `next()` at `tensor.rs:2127:45` with "range 1..2 out of bounds for len 1 (index-mapping bug)", an `index::span` message that blames a kernel for a caller's input.

A trailing partial sample (`data.len()` not a multiple of the sample size) is dropped by the integer division (by reading). Overlap: `docs/audits/2026-11-sweep.md` S20 (targets not checked) and line 74 (`batch_size = 0`, "not run").

Direction: a `try_new` that checks the shape product, the two lengths, and `batch_size > 0`, with `new` wrapping it.

#### R19 (Medium): saving and rebuilding from a `TensorData` panics on inconsistent `shape`/`data`. Reproduced.

`src/io.rs:380-382` (same at `:408-414`) and `src/io.rs:103-105`:

```rust
let view =
    TensorView::new(Dtype::F32, td.shape.clone(), bytemuck::cast_slice(&td.data))
        .expect("Failed to create TensorView");
```

```rust
pub fn to_tensor(&self, requires_grad: bool) -> Tensor {
    crate::RawTensor::new(self.data.clone(), &self.shape, requires_grad)
}
```

`TensorData` has two public fields (`:22-27`) and derives `Decode`, so a state dict read by `load_state_dict` (bincode) or built by hand can have `shape` and `data` that disagree. `save_safetensors` returns `Result` but panics at `io.rs:382:22` ("Failed to create TensorView: InvalidTensorView(F32, [2, 2], 12)") for a 3-element entry labeled `[2, 2]`; the `# Panics` section documents it, but the input is a state dict, which Invariant 8 names as data to validate. `TensorData::to_tensor` panics at `tensor.rs:544` through `RawTensor::new` ("Shape mismatch: tensor has 4 elements but data length is 3"), has no `# Panics` section (the lint cannot see through the call), and has no `try_` form; it also clones the whole buffer. `TypedTensorData::numel` (`:330-332`) multiplies unvalidated dims with `iter().product()` (by reading).

Direction: map the `TensorView` error into the returned `Result`; add `TensorData::try_to_tensor` (or validate on decode) and document or remove the panicking form.

#### R20 (Medium): silent zero defaults in the movement index mappers, outside what `check-fallbacks` matches. By reading.

`src/ops/movement.rs:59`, `:81`, `:164`, `:193`:

```rust
let pad_left = self.padding.get(dim).map_or(0, |p| p.0);
```

```rust
let range_start = self.ranges.get(dim).map_or(0, |r| r.0);
```

Invariant 5: "Never use `.get(i).unwrap_or(0.0)` ... Silent zeros are how wrong gradients pass tests." These four use `map_or(0, ..)`, which `scripts/check-fallbacks.sh` does not match (its patterns are `unwrap_or(..)` and `let Some(..) = ...get(`), so the gate passes. The callers check `padding.len() == rank` first (`src/ops/movement.rs:1033-1039`), so the default is unreachable today by reading; if a mapper were ever built with a short vector, the forward and backward index maps would shift to offset 0 without any signal, which is the B4 failure shape. A related skip, in GPU code I could not compile: `src/gpu/kernels.rs:3365-3366` computes an output size with `(0..params.rank as usize).filter_map(|i| params.new_shape.get(i))`, which drops an out-of-range dim instead of failing.

Direction: use `fetch(&self.padding, dim).0`, and add `map_or(` and `filter_map(..get(..))` after a positional `.get(` to the script.

#### R21 (Medium; not verified, no GPU): a GPU readback panics on a poll timeout or a failed map.

`src/gpu/buffer.rs:195-204`:

```rust
ctx.device()
    .poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(Duration::from_secs(5)),
    })
    .expect("Failed to poll device");
```

`download()` is behind every host read of a GPU tensor (`Tensor::data()`, `item()`, `Storage::to_host_f32`, `PartialEq for Storage`). A five-second timeout under load or a lost device panics here, while the kernels in the same module return `Option` and the doc on the sync function says "A timeout doesn't necessarily mean the device is lost" (`src/gpu/mod.rs:64-66`). The read accessors are infallible by signature, so there is no error path to use. Also `:164` `get_gpu_context().expect("GPU context should exist if buffer exists")` and `:277` (see R25). All by reading; nothing here was run.

Direction: decide whether a readback failure is a `VoltaError::DeviceError` on a `try_` accessor, and at minimum retry or lengthen the wait before panicking.

#### R22 (Low): layer and loader constructors take user-suppliable hyperparameters and only panic.

For example `Linear::new(in_features, out_features, ..)` (`src/nn/layers/linear.rs:56-58`) reaches `RawTensor::xavier_uniform`, which panics on a zero dim; `Dropout::new(p)` (`src/nn/layers/dropout.rs:22-27`) asserts the range; `permute` and `stride_op` take axes/strides and assert (`src/ops/movement.rs:882-890`, `:1249-1257`). DECISIONS.md "Where the line was drawn" and "Not changed" list the `Axis` ops as deliberately left panicking, and the optimizers do have `try_new` (`src/nn/optim/sgd.rs:48`, `adam.rs:43`). Layer constructors are not mentioned on either side of that line, so this is a gap in the stated rule and not a clear breach. I did not count every constructor.

Direction: state in DECISIONS.md which side layer hyperparameters fall on; add `try_new` where a model definition can come from a file.

#### R23 (Low): `debug_assert!` where the repo's rule is to fail in every profile.

Eight library sites: `src/ops/ternary.rs:31`, `:96`; `src/nn/optim/adam.rs:119`, `:218`, `:219`; `src/autograd.rs:382`; `src/gpu/buffer.rs:84`, `:276`. Most are followed by a real `expect` or cannot fail. `src/gpu/buffer.rs:84` (`debug_assert_eq!(bytes.len(), gpu_bytes(dtype, len));`) guards an upload size and has no release-mode twin that I could see by reading; `src/autograd.rs:382` guards `requires_grad` on the root and its comment explains the one way it can fail. `src/tensor.rs:253-260` argues in a doc comment for unconditional `assert!` on exactly these grounds.

Direction: make `buffer.rs:84` an `assert_eq!`; delete the ones shadowed by an `expect`.

### Category 5: index and take workarounds

Covered above: R5 (pointer ids), R9 (clone to iterate while mutating), R10 (index loop forced by `&mut self` helpers). Remaining:

#### R24 (Low): two mutexes guard one invariant in each GPU pool, and one lint about it is suppressed with a "will come back" note. By reading.

`src/gpu/pool.rs:60-67` (same design at `src/gpu/staging_pool.rs:41-45`):

```rust
pub struct BufferPool {
    /// Free buffers organized by size bucket (key = log2(size))
    buckets: Mutex<HashMap<u32, VecDeque<wgpu::Buffer>>>,
    ...
    total_count: Mutex<usize>,
}
```

`total_count` must equal the number of buffers in `buckets`, but they are locked separately, so every method takes both. The order is consistent inside each type (buckets then count in `pool.rs:135-146`, `:175-182`; count then pools in `staging_pool.rs:72-80`, `:107-125`), so I see no deadlock by reading. It is still a parallel-state design: one `Mutex<PoolState>` would make the invariant structural. `src/gpu/pool.rs:165-168` carries `#[expect(clippy::significant_drop_tightening, reason = "Attempts to fix by merging the constructon with its single usage leads to other errors, will come back to fix this.")]`. DECISIONS.md "Stay synchronous" already says `BufferPool` guards state only one thread can reach. Poisoning is handled uniformly with `unwrap_or_else(PoisonError::into_inner)` and a comment explaining why (`pool.rs:56-59`).

Direction: one mutex over a struct holding both, or a `RefCell` if the pool stays single-threaded.

#### R25 (Low): `Option` wrapper on `GpuBuffer.buffer` only so `Drop` can move it out. By reading.

`src/gpu/buffer.rs:37-38`, `:275-278`, `:344-353`:

```rust
/// The actual GPU buffer (Option to allow taking on Drop)
buffer: Option<wgpu::Buffer>,
```

The standard `Option::take` in `Drop` idiom. Its cost is that every read goes through `self.buffer.as_ref().expect("Buffer should not be taken")` (`:277`) on a state that only exists during `drop`. `Drop` returns the buffer to the pool keyed by `gpu_bytes(self.dtype, self.len)`, which panics for a non-GPU dtype (`:53-57`); a panic inside `drop` during unwinding aborts. By reading, a `GpuBuffer` cannot be constructed with such a dtype.

Direction: acceptable as is; `ManuallyDrop` would remove the `expect` at the price of one `unsafe`, which this crate reasonably avoids.

#### R26 (Low): `HostMut` writes back in `Drop`. By reading.

`src/storage.rs:944-950`:

```rust
impl<T: crate::real::Real> Drop for HostMut<'_, T> {
    fn drop(&mut self) {
        if let Self::Wide { wide, dst } = self {
            **dst = Storage::from_compute(dst.dtype(), std::mem::take(wide));
```

For half-precision storage `host_mut` decodes the whole buffer to `f32` (`:580`), hands out the copy, and re-encodes on drop. The write therefore happens at an invisible point, is skipped by `mem::forget`, and runs during unwinding if the caller panics mid-update, committing a half-finished buffer. `from_compute` can panic (`:631`), though not for the dtype pairs `host_mut` creates. Each optimizer step on f16/bf16 parameters pays two full conversions. The type is `pub(crate)`, so exposure is internal.

Direction: an explicit `commit()` (or a closure-taking `with_host_mut`) would make the write visible and skip it on unwind.

### Category 6: lifetimes and globals

#### R27 (Medium; by reading): the GPU context is a process-wide singleton whose failure is cached forever.

`src/gpu/mod.rs:34-40`:

```rust
static GPU_CONTEXT: OnceLock<Option<GpuContext>> = OnceLock::new();

/// Get the global GPU context, initializing it if necessary
/// Returns None if GPU is not available
pub fn get_gpu_context() -> Option<&'static GpuContext> {
    GPU_CONTEXT
        .get_or_init(|| match GpuContext::new() {
```

Every GPU buffer, pool and pipeline hangs off this `&'static`. Consequences: a failed initialization is stored as `None` for the life of the process with no retry; the device is never dropped; `Device::GPU(String)` carries an adapter name but cannot select a context (R11); tests cannot isolate GPU state, which is why `src/gpu/context.rs:23` adds a second global, `static F16_EMULATION: AtomicBool`, whose setter is `#[cfg(test)]` (`:26-29`) but whose `load` runs in the production path at `:464` on every f16 pipeline lookup, and why `src/lib_tests/gpu_half_tests.rs:25` needs a `static MODE_LOCK: Mutex<()>` to serialize tests around it. I found no DECISIONS.md entry for the singleton, but searched headings only.

Direction: record the decision, or thread a context handle through `Device::GPU`; gate the `F16_EMULATION` read with `cfg(test)` so release builds do not carry a test switch.

#### R28 (Info): thread-local mode switches.

Seven `thread_local!` statics (listed in section 2) hold the device policy, fallback log, warned-ops set, no-grad depth, dtype policy, autocast dtype and RNG. All are `Cell`/`RefCell` behind guard types that are `!Send` via `PhantomData<*const ()>` (`src/autograd.rs:350`, `src/device_policy.rs:88`, `src/dtype.rs:236`, `src/amp.rs:70`); `tests/compile_fail/guards_not_send.rs` exists. Deliberate (DECISIONS.md "`NO_GRAD` as a reentrant depth counter"). The one re-entrancy hazard, `with_rng`, is handled loudly: `src/tensor.rs:440-442` uses `try_borrow_mut().expect("with_rng called reentrantly ...")`, it is `pub(crate)`, and none of its eight library call sites (`tensor.rs:683`, `:708`, `:758`, `:806`, `:2050`, `:2098`, `nn/layers/embedding.rs:60`, `nn/layers/dropout.rs:63`) nests another call by reading. `NoGradGuard::drop` does `d.set(d.get() - 1)` (`src/autograd.rs:354`), which can only underflow if a guard is forged, and its field is private. `WARNED` is a `HashSet<String>` keyed by an op-name string (`src/device_policy.rs:57`, `:114`), touched only on a fallback.

#### R29 (Low): small signature and export oddities.

- `src/tensor.rs:2157`: `pub use RawTensor as new_tensor;` exports a type under a snake_case, function-like name.
- `src/tensor.rs:398-413`: `RawTensor` is public with public fields (`data`, `shape`, `grad`, `requires_grad`, `parents`, `device`) and a public `invalidate_gpu_cache(&mut self)` (`:873`), but `Tensor::borrow`/`borrow_mut` are `pub(crate)` and I found no public function that hands out a `RawTensor` or a reference to one (grep of `pub fn` signatures), so the fields and that method appear unreachable from outside. The type is public because the constructors and ops are associated functions on it (`RawTensor::new(..) -> Tensor`, suppressed with `#[allow(clippy::new_ret_no_self)]` at `:537`).
- `src/io.rs:176`, `:184`: `save_state_dict(.., path: &str)` and `load_state_dict(path: &str)` take `&str` where the safetensors functions in the same file take `P: AsRef<Path>` (`:264`, `:374`).
- `src/io/mapping.rs:227`, `:238`, `:348`: `select_keys(keys: Vec<String>)`, `exclude_keys(keys: Vec<String>)`, `create_key_mapping(mappings: Vec<(&str, &str)>)` take owned vectors; the first two store them in a closure, the third only iterates.
- `src/storage.rs:886` and `src/shape.rs:263`: `fn eq(&self, other: &Vec<f32>)` / `&Vec<usize>` are forced by `impl PartialEq<Vec<_>>` and are fine.

Direction: rename or drop the alias; make `RawTensor`'s fields `pub(crate)`; take `impl AsRef<Path>` consistently.

#### R30 (Low): `Tensor` has no identity or equality API.

`Tensor` derives only `Clone` and `Debug` (`src/tensor.rs:31`). There is no `PartialEq` (reasonably, since value equality on a graph node is ambiguous) and no public `ptr_eq`, while the crate itself needs identity and gets it through the `pub(crate)` `as_ptr` (R5; `src/autograd.rs:516`). Downstream code that wants to deduplicate a parameter list or check aliasing has no tool. The `alias` repro shows the same parameter can be registered twice in an optimizer and is accepted without complaint.

Direction: a public `Tensor::ptr_eq(&self, other: &Self) -> bool`.

### Category 7: interior mutability behind `&self`

#### R31 (Medium): `Layer::forward(&self)` mutates BatchNorm's running statistics. By reading.

`src/nn/mod.rs:169` and `src/nn/layers/batchnorm.rs:173-175` (same at `:398-399`):

```rust
fn forward(&self, x: &Tensor) -> Tensor;
```

```rust
let _no_grad = crate::autograd::no_grad();
let mut rm = self.running_mean.borrow_mut();
let mut rv = self.running_var.borrow_mut();
```

`train(&mut self, ..)`, `load_state_dict(&mut self, ..)` and `to_device(&mut self, ..)` take `&mut self` (`src/nn/mod.rs:80-145`), so the trait presents `&self` as "does not change the layer". A training-mode forward changes state that eval-mode forwards read. The repo already documents one consequence: `check_gradients` cannot restore it (`src/tensor.rs:1512-1517`, "`BatchNorm` in training mode keeps updating its running statistics"). The mutation is possible only because the buffers are `Tensor` handles (R6), and it is done with raw `borrow_mut()` (R2). `RecurrentLayer::step(&self, x, state: &mut Self::State)` (`src/nn/mod.rs:180`) shows the honest alternative in the same file: mutable state passed explicitly.

Direction: either `forward(&mut self, ..)` for layers with buffers, or keep `&self` and document on `Layer::forward` that it may update non-parameter state; route the write through a `Tensor` method.

#### R32 (Low): `Conv2d::set_algo(&self)` through a `Cell`. Noted in earlier audits.

`src/nn/layers/conv.rs:149-150`, `:871-873`:

```rust
/// Algorithm used by `forward`.
algo: std::cell::Cell<ConvAlgo>,
```

```rust
pub fn set_algo(&self, algo: ConvAlgo) {
    self.algo.set(algo);
}
```

A configuration setter on `&self`. `&mut self` would be the honest signature and would remove the `Cell`. `Conv2d` is `!Sync` regardless because of its `Tensor` fields. Overlap: `docs/audits/OPUS_5_EVAL_AUDIT.md:663` and `docs/audits/glm-53-adversarial-audit.md:260-261`.

Direction: `pub fn set_algo(&mut self, ..)` and a plain field.

#### R33 (Info): lazily synced GPU host copy.

`src/storage.rs:181`, `:405-411`: `host: OnceLock<HostBuf>` filled by `shared.host.get_or_init(|| shared.buffer.to_host())` behind `&self` readers. Deliberate and documented at length (DECISIONS.md "GPU storage: shared write-once host copy"), write-once, and cleared only through `&mut self` on an unshared `Arc` (`:845`, `:869`). The `unsafe` cache it replaced is gone. One side effect to know, by reading: `impl PartialEq for Storage` (`:876-883`) calls `as_bytes()`, so comparing GPU storages performs device readbacks.

### Category 8: other

#### R34 (Low): `// SAFETY:` comments on code that is not `unsafe`.

`src/ops/ternary.rs:30`, `:95`; `src/nn/optim/adam.rs:118`, `:217`:

```rust
// SAFETY: MulAcc operation always creates 2 or 3 parents (x, y, optional w)
debug_assert!(
    parents.len() >= 2,
```

The repo denies `undocumented_unsafe_blocks`, which gives `// SAFETY:` a specific meaning; using it for ordinary preconditions dilutes a grep for real safety arguments (seven hits in library code, three real). `ternary.rs:30` also disagrees with the `expect` message a few lines below ("three parents").

Direction: reword to plain comments.

#### R35 (Low): macros hide catch-all enum arms and a duplicated body.

`src/real.rs:198-212`:

```rust
macro_rules! dispatch_real {
    ($dtype:expr, $T:ident => $body:expr) => {
        match $dtype {
            ...
            other => unreachable!("dispatch_real on {other}: compute_dtype admits floats only"),
```

`dispatch_real!`/`dispatch_bits!` are used at 51 library sites (`grep -rn 'dispatch_real!\|dispatch_bits!' src --include='*.rs' | grep -v lib_tests | grep -v macro_rules | wc -l`). The crate denies `wildcard_enum_match_arm` so that a new enum variant is a compile error, and DECISIONS.md leans on that for `GradFn`. The `other =>` arm here on `DType`, and the `_ => None` arms on `HostBuf` inside the `element!` macro (`src/storage.rs:130`, `:136`), were not reported by clippy in my run, so a new `DType` or `HostBuf` variant would compile and reach `unreachable!` or `None` at runtime. The `dispatch_real!` body is pasted once per arm with a local `type $T = ..;`, so each kernel is compiled twice by textual duplication instead of by a generic function, and a `return` written in the body (`src/ops/matmul.rs:303`, `:320`) returns from the enclosing function from inside a macro argument. `dispatch_bits!` matches on `size_of` with `n => unreachable!` (`:243-246`).

Direction: list the remaining variants explicitly in the macros; where a body is long, a generic `fn kernel<T: Real>(..)` makes control flow visible.

#### R36 (Low): `Storage` equality compares bytes in one impl and floats in the other. By reading.

`src/storage.rs:876-891`:

```rust
impl PartialEq for Storage {
    fn eq(&self, other: &Self) -> bool {
        ...
        self.as_bytes() == other.as_bytes()
```

```rust
impl PartialEq<Vec<f32>> for Storage {
    ...
        self.to_host_f32() == other.as_slice()
```

The first makes `NaN == NaN` true (for identical bit patterns) and `0.0 == -0.0` false; the second follows IEEE (`NaN != NaN`, `0.0 == -0.0`). The same contents can therefore compare equal as a `Storage` and unequal as a `Vec<f32>`, or the reverse. Not run.

Direction: pick one semantics, document it on the type, and name the other (`bit_eq`).

#### R37 (Low): two error types, and stringly-typed variants.

`src/io.rs:10` and `src/data/mnist.rs:2` import `std::io::Result`, so `load_state_dict`, `save_state_dict`, `load_safetensors*`, `save_safetensors*`, `load_mnist_images`, `load_mnist_labels` return `io::Error` with messages built by `Error::other(..)` / `io::Error::new(InvalidData, format!(..))` (8 sites in library code), while the rest of the crate returns `VoltaError`, which has `Io(#[from] std::io::Error)` (`src/error.rs:48`). A caller gets a decode failure as an opaque `io::Error` string and cannot match on it. Inside `VoltaError`, `InvalidParameter(String)` is constructed at 46 library sites, `DeviceError(String)` at 5 and `OpError(String)` at 1, against structured variants such as `StateShapeMismatch { key, expected, actual }`. `BroadcastError(Vec<usize>, Vec<usize>)` is a tuple variant with two same-typed fields (`:40`).

Direction: return `VoltaError` from the io and data functions; split the most common `InvalidParameter` messages into variants as callers need to match on them.

#### R38 (Low): lint suppressions in `src/`, complete list.

`#[allow(..)]` (17). Unlike `#[expect]`, none of these fails when the lint stops firing:

| Site | Lint | What it hides |
| --- | --- | --- |
| `src/nn/layers/lstm.rs:16` | `dead_code` | `input_size` field never read |
| `src/nn/layers/sequential.rs:190` | `dead_code` | a private example `impl Sequential { fn build_mlp }` compiled into the library ("Usage example") |
| `src/nn/layers/conv_transpose.rs:293-296` | `clippy::tuple_array_conversions` | reason: "just don't know hot to fix right now" |
| `src/nn/transformer/block.rs:39`, `:42`, `:45`, `:48` | `dead_code` | `dropout`, `embed_dim`, `num_heads`, `expansion_factor` stored and never read |
| `src/nn/transformer/multi_head_attention.rs:39` | `dead_code` | `dropout` "Stored but not applied" |
| `src/nn/transformer/feed_forward.rs:31` | `dead_code` | `dropout` never read |
| `src/real.rs:62`, `:69` | `cast_possible_truncation`, `cast_precision_loss` | inside the `real!` macro, with reasons |
| `src/tensor.rs:537` | `clippy::new_ret_no_self` | `RawTensor::new` returns `Tensor` |
| `src/tensor.rs:548`, `:582` | `dead_code` | on `try_new_with_storage` and `new_with_storage`; the second is called throughout the crate, so that attribute is stale |
| `src/gpu/pool.rs:221`, `:245` | `dead_code` | `BufferPool::stats` and `BufferPoolStats`, both `pub` |
| `src/utils/progress.rs:6` | `clippy::print_stderr` | module-wide, with a reason |

`#[expect(..)]` in library code (8 by the script, plus one the split counts differently): `src/ops/matmul.rs:709`, `src/storage.rs:215`, `:274`, `:658`, `src/amp.rs:233`, `:319`, `src/tensor.rs:1632` (all narrowing casts with a reason), `src/ops/gpu_ops.rs:407` (`dead_code` on `gpu_binary_backward_broadcast_shapes`, reason cites issue #60), `src/gpu/pool.rs:165` (R24). Benches carry one `#[allow(missing_docs, clippy::missing_docs_in_private_items)]` each (four files); `tests/common/mod.rs:13` and `tests/torch_ref/mod.rs:9` have `#![allow(dead_code)]`; `examples/char_transformer_model.rs:118` has `#[allow(dead_code)]`.

The three `dropout` suppressions are the visible edge of a public constructor argument that does nothing (`MultiHeadAttention::new(embed_dim, num_heads, dropout)`, `FeedForward::new`, `TransformerBlock::new`); that defect is `docs/audits/2026-11-sweep.md` S14.

Direction: convert the `dead_code` allows to `#[expect(.., reason = "S14 ...")]`, drop the stale one in `tensor.rs`, move `build_mlp` to a doc example.

#### R39 (Info): `Box<dyn Layer>` in `Sequential` and `Box<dyn Fn(&mut StateDict)>` in `StateDictMapper`.

`src/nn/layers/sequential.rs:12`, `:124`; `src/io/mapping.rs:6`, `:249-251`. Open-ended heterogeneous collections, which is what trait objects are for. The layer case is a recorded decision (DECISIONS.md "Split layer traits; `Module` and `Layer` stay open", read by heading). The mapper's `F: Fn(&mut StateDict) + 'static` bound is required by the box. `GradFn` went the other way, to a closed enum, also by decision. No finding.

#### R40 (Info): crate-wide `allow` of `needless_pass_by_value` and `cast_precision_loss`.

`Cargo.toml` `[lints.clippy]`, both with a written rationale. With the allow lifted, `needless_pass_by_value` fires at three places (R13) and `cast_precision_loss` at 17 in the CPU-only library, all integer-to-float for counts and means. `Storage::from_typed(data: Vec<T>)` (`src/storage.rs:305-306`) takes ownership as the rationale says, then `Arc::from(data)` copies the elements into a new allocation, so handing over a freshly built buffer still costs one copy per op output; DECISIONS.md "CPU storage" acknowledges the constructor copy.

#### R41 (Info): test code inside `src/` still writes through `borrow_mut()`.

`src/nn/layers/conv.rs:2158`, `:2163`, `:2313`, `:2316`, `:2381`, `:2384`, `:2567`, `:2569`, `:2847`, `:2850`, `:2851`, `:2997`, `:3004`, `:3315`, `:3317` assign `X.borrow_mut().data = Storage::cpu(..)` or `Storage::gpu(..)` in `#[cfg(test)]` code. Invariant 2 does not exempt tests. The `Storage::gpu` lines replace storage without touching `device`, the mismatch `new_with_storage` asserts against. Recorded in `docs/audits/glm-53-adversarial-audit.md:249` and `OPUS_5_EVAL_AUDIT.md:637` against an older tree; the sites are still present.

## 4. What the codebase does well

- `unwrap()` is absent from library code and denied by attribute (`src/lib.rs:23`); `todo!`, `unimplemented!`, `dbg!`, printing, undocumented `unsafe`, and `missing_panics_doc`/`missing_errors_doc` are denied in `Cargo.toml`, and my CPU-only clippy run produced no errors under that configuration.
- `indexing_slicing` is denied and `src/index.rs` gives loud, `#[track_caller]` replacements; I found no slice indexing in library code (one `str` range slice, R9) and only the `map_or(0, ..)` defaults of R20.
- Truncating and wrapping casts are denied. Library code has no `as u32` or `as isize`, five `as usize` (four `u32 as usize` in `mnist.rs`, one at `gpu/kernels.rs:3365`), four `as u64` from `usize` in `gpu/buffer.rs`, and the narrowing float/int casts sit under `#[expect(.., reason)]`. GPU kernel parameters go through `pack_u32` (`src/gpu/kernels.rs:3387-3390`).
- `unsafe` is three blocks in one file, each with a SAFETY comment preceded by the asserts it cites.
- `Storage` is copy-on-write over typed `Arc<[T]>` buffers with no `Deref`, so a host transfer is always spelled out, and the old `unsafe` host cache is gone.
- The closed `GradFn` enum plus `edge()` as the only graph constructor removes the scattered `borrow_mut().grad_fn = ..` writes the older audits found.
- Mutex poisoning is handled deliberately and uniformly; there is no `lock().unwrap()`.
- Guards for thread-local modes are `!Send` by construction.
- Conv and matmul geometry is validated once into spec structs (`src/spec.rs`) and stored in the `GradFn`; `window_out` rejects a kernel larger than the padded input with an error.

## 5. Checked and found clean

- **Double borrows.** By reading, every `borrow_mut()` on a `Tensor` in library code holds one `RefMut` on one tensor and calls only `Storage`/`Shape` methods under it; op forward and backward paths take shared borrows only. The `alias` repro ran without a panic in a debug build: `x.add(&x)`, `elem_mul(&x)`, `matmul(&x)`, `mulacc(&x, &x)`, `div(&x)`, `max_elem(&x)`, `y.add(&y)`, `softmax`, two `backward()` calls on one loss, `SGD` with the same parameter listed twice (two steps, momentum and weight decay on), `check_gradients_simple` with a closure capturing the tensor under test, `set_grad` with the tensor's own `grad_storage()`, `zero_grad`. CPU only; GPU paths not run.
- **Reference cycles.** None constructible through `edge()`; no public API sets `parents`.
- **`try_borrow`.** One use, `with_rng`, which panics with a message instead of falling back (the silent fallback the older audits describe is gone).
- **`transmute`, raw-pointer dereference, `from_raw_parts`, `get_unchecked`, `set_len`, `MaybeUninit`, `static mut`, `Box::leak`, `mem::forget`, `unsafe impl`.** Zero occurrences in `src/`, `tests/`, `examples/`, `benches/`, `build.rs`.
- **`&Vec<T>` / `&String` / `&Box<T>` parameters.** Only the two `PartialEq<Vec<_>>` impls, which require that signature.
- **Wildcard match arms.** `grep -rnE '^\s*_ => '` in library code gives 15 hits; 13 are on tuples, integers, slices or `Option`, and the two on an enum are the macro arms in R35 (`storage.rs:130`, `:136`).
- **Lock ordering** in `BufferPool` and `StagingBufferPool`: consistent within each type, by reading.
- **`Tensor::data()` in library hot paths.** Two library uses on a `Tensor` (`autograd.rs:581`, `tensor.rs:206`), neither in a loop; the two `io.rs` hits are the `safetensors` accessor.
- **`mem::take` / `Option::take`.** Five uses (`device_policy.rs:100`, `storage.rs:845`, `:869`, `:947`, `gpu/buffer.rs:347`); each moves a value out for a real reason (R25 and R26 discuss two).
- **`build.rs`.** Reads `CARGO_FEATURE_*`/`CARGO_CFG_TARGET_OS` (target, not host) and emits one link line; nothing to report.
- **`tests/`, `examples/`, `benches/`.** No `unsafe`. `unwrap()` counts: tests 109, examples 12, benches 13 (CLAUDE.md asks examples and benches for `.expect(..)`; I did not check what those 25 are applied to).

## 6. Not verified, and limits

- **No GPU.** Nothing under `src/gpu/` or the `gpu`-gated code in `src/ops/gpu_ops.rs`, `storage.rs`, the optimizers and conv was compiled or run. R21, R25, R27 and the GPU remarks in R20, R24 and R33 are by reading only. The clippy evidence covers the CPU-only feature set.
- **No macOS.** The `cblas_sgemm` FFI path was read, not compiled.
- **No 32-bit build.** R14 is an argument from the code; I did not confirm that the crate builds for wasm32 or observe the behavior.
- **R17** was reproduced only under `ulimit -v 4000000`. R16's abort was reproduced with and without the limit.
- **R3, R4, R20, R36** are by reading; no test was written for them.
- **Performance claims** (R3, R7, R8, R9, R11, R12) are copy counts read from the code, not measurements.
- **Counts** depend on the test/library split heuristic described in section 1 and on line-based regexes. A multi-line `.lock()` followed by `.unwrap()` would be missed by the last table row, so I checked the `src/gpu/` lock sites by hand (all use `unwrap_or_else(PoisonError::into_inner)`).
- **`arithmetic_side_effects`** (308 sites) was not triaged beyond the reproduced cases.
- **DECISIONS.md** was read in full only for the entries named in section 1; a finding rated Medium or Low might be covered by an entry I read only by heading. `docs/plans/ADDENDUM.md` was not read.
- **Deduplication.** Findings were not checked against the GitHub issue tracker. Against `docs/audits/` I only keyword-searched; overlaps I noticed are cited inline: R2 (post-phase5 sections 4 and 5), R7 (OPUS_5_EVAL_AUDIT line 352), R9 (S19), R16 and R18 (sweep S20 and its "not run" list), R32 and R41 (OPUS_5_EVAL_AUDIT, glm-53 audit), R38 (S14). Other findings may already be tracked.
- `git status --short` printed nothing after the last command; `HEAD` was `8e5fa4e`. All build output went to the scratch directory through `CARGO_TARGET_DIR`.


## Appendix A: repro sources

The repro crate is not committed. Its `Cargo.toml` depends on Volta by path with `default-features = false`, edition 2024. `<scratch>` is any directory outside the repository.

`src/bin/deep.rs` (R1; argument is the chain length):

```rust
use volta::{RawTensor, Tensor};
fn main() {
    let n: usize = std::env::args().nth(1).expect("n").parse().expect("usize");
    let x: Tensor = RawTensor::new(vec![1.0], &[1], true);
    let mut y = x.clone();
    for _ in 0..n { y = y.add(&x); }
    let l = y.sum().into_loss().expect("loss");
    l.backward();
    println!("backward ok over {n} nodes, grad={:?}", x.grad());
    drop(l);
    drop(y);
    println!("drop ok");
}
```

`src/bin/userin.rs` (R15 to R19; argument is the case name):

```rust
use volta::{RawTensor, PadValue};
use std::collections::BTreeMap;
fn main() {
    let which = std::env::args().nth(1).expect("case");
    let t = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false);
    match which.as_str() {
        "pad" => { let r = RawTensor::try_pad_with(&t, &[(usize::MAX, 1), (0, 0)], PadValue::ZERO); println!("pad -> {:?}", r.map(|t| t.shape())); }
        "pad2" => { let r = RawTensor::try_pad_with(&t, &[(usize::MAX - 1, 1), (0, 0)], PadValue::ZERO); println!("pad2 -> {:?}", r.map(|t| (t.shape(), t.data().len()))); }
        "loader0" => { let _ = volta::DataLoader::new(vec![], vec![], &[0], &[1], 1, false); println!("no panic"); }
        "loaderb" => { let mut d = volta::DataLoader::new(vec![0.0; 4], vec![0.0; 4], &[1], &[1], 0, false); let b = d.next(); println!("no panic {}", b.is_some()); }
        "loadert" => { let mut d = volta::DataLoader::new(vec![0.0; 4], vec![0.0; 1], &[1], &[1], 2, false); let _ = d.next(); let b = d.next(); println!("no panic {}", b.is_some()); }
        "save" => {
            let mut sd = BTreeMap::new();
            sd.insert("w".to_string(), volta::io::TensorData { data: vec![1.0; 3], shape: vec![2, 2] });
            let r = volta::io::save_safetensors(&sd, "<scratch>/x.safetensors");
            println!("save -> {r:?}");
        }
        "totensor" => { let td = volta::io::TensorData { data: vec![1.0; 3], shape: vec![2, 2] }; let _ = td.to_tensor(false); println!("no panic"); }
        "mnist" => {
            let p = "<scratch>/bad.idx";
            let mut h = vec![0, 0, 8, 3]; h.extend([0xFF, 0xFF, 0xFF, 0xFF]); h.extend([0xFF, 0xFF, 0xFF, 0xFF]); h.extend([0xFF, 0xFF, 0xFF, 0xFF]);
            std::fs::write(p, &h).expect("write");
            let r = volta::load_mnist_images(p); println!("mnist -> {:?}", r.map(|v| v.len()));
        }
        "mnist2" => {
            let p = "<scratch>/bad2.idx";
            let mut h = vec![0, 0, 8, 3]; h.extend([0xFF, 0xFF, 0xFF, 0xFF]); h.extend([0, 0, 0, 28]); h.extend([0, 0, 0, 28]);
            std::fs::write(p, &h).expect("write");
            let r = volta::load_mnist_images(p); println!("mnist2 -> {:?}", r.map(|v| v.len()));
        }
        "matmul" => {
            let a = RawTensor::zeros(&[100_000, 1]); let b = RawTensor::zeros(&[1, 100_000]);
            let c = a.matmul(&b); println!("matmul -> {:?}", c.shape());
        }
        _ => panic!("unknown case"),
    }
}
```

`src/bin/alias.rs` (section 5, double-borrow check; ran clean):

```rust
use volta::{RawTensor, Tensor, Axis};
fn main() {
    let x: Tensor = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true);
    let y = x.add(&x).elem_mul(&x).matmul(&x).mulacc(&x, &x).div(&x).max_elem(&x);
    let y = y.add(&y);
    let z = y.softmax(Axis(1)).sum();
    let l = z.into_loss().expect("loss");
    l.backward();
    l.backward();
    println!("alias ok grad={:?}", x.grad());
    let mut opt = volta::nn::optim::SGD::new(vec![x.clone(), x.clone()], 0.1, 0.9, 0.01);
    opt.step();
    opt.step();
    println!("dup-param step ok {:?}", x.data());
    let ok = RawTensor::check_gradients_simple(&x, |t| t.elem_mul(&x).add(t).sum());
    println!("check_gradients with captured self: {ok}");
    x.set_grad(x.grad_storage().expect("g"));
    x.zero_grad();
    println!("all ok");
}
```

# Known broken: integration tests vs. the opaque `Tensor` wrapper

As of the Phase 1 "invariant wall" work (ADDENDUM.md §1.1), `Tensor` is
`struct Tensor { inner: Rc<RefCell<RawTensor>> }` with `inner` private and
`borrow()`/`borrow_mut()` only `pub(crate)`. That's intentional — it's what
makes `x.borrow_mut().requires_grad = true` impossible from outside the crate.

But these integration tests compile as separate binaries that only see the
crate's *public* API, the same as any downstream user of the `volta` crate.
About 128 call sites across 9 files here (`coverage_fill.rs`,
`gpu_smoke_test.rs`, `gpu_stress.rs`, `gpu_test.rs`, `gpu_unary_test.rs`,
`improvements_test.rs`, `movement_backward_test.rs`, `parity_movement.rs`,
`test_lstm.rs`, `test_optimizers.rs`) call `.borrow()`/`.borrow_mut()`
directly on `Tensor` values (reading `.shape`/`.data`/`.device`/`.grad`,
and in a few places writing `.requires_grad`/`.grad` directly) — none of
that compiles anymore.

`tests/common/mod.rs` (the shared helper module `#[path = "common/mod.rs"]`'d
into every `parity_*.rs` file) has another 17 call sites of its own. That
means `parity_binary.rs`, `parity_reduce.rs`, `parity_unary.rs`, and
`parity_matmul.rs` — which have *zero* direct `.borrow()` calls — still fail
to build, transitively, because the shared module they include doesn't
compile. Fixing `common/mod.rs` is part of the same fix as the rest of this
list, not a separate one.

**Not fixed yet** — deliberately deferred rather than rewritten under time
pressure. The fix needs a real decision, not just glue:

- Add a small public read API on `Tensor` (`shape()`, `data()`, `device()`,
  something like `grad_storage()` for the GPU-specific tests that need
  `Storage::is_gpu()` rather than just `TensorOps::grad()`'s `Vec<f32>`),
  plus a narrow, clearly-labeled test-fixture setter for injecting a
  synthetic gradient (`test_optimizers.rs`/`coverage_fill.rs` do this to
  drive optimizer steps without a real `backward()`) — then rewrite the
  ~128 call sites to use it instead of raw `.borrow()`.
- Decide whether `set_requires_grad` should become `pub` (it already has a
  debug-assert invariant check) so `x.borrow_mut().requires_grad = true`
  has a real named replacement.

Until that lands, `cargo test` will fail to build these files. `cargo check`
(no `--tests`) and `cargo check --tests` for files *not* in the list above
are unaffected.

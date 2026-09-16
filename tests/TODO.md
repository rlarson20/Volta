# Known broken: `set_grad`-dependent tests

As of the Phase 1 "invariant wall" work (ADDENDUM.md §1.1), `Tensor` is
`struct Tensor { inner: Rc<RefCell<RawTensor>> }` with `inner` private and
`borrow()`/`borrow_mut()` only `pub(crate)`. That's intentional — it's what
makes `x.borrow_mut().requires_grad = true` impossible from outside the crate.

The mechanical fallout of that change — every integration test and example
calling `.borrow()`/`.borrow_mut()` directly on a `Tensor` — has been fixed
by adding a small public read/write API to `Tensor` (`shape()`, `data()`,
`device()`, `grad_storage()`, `detach()`, and a now-`pub` `set_requires_grad()`
guarded by an unconditional `assert!`) and rewriting every call site in
`tests/` and `examples/` to use it instead of raw `.borrow()`.

**Still deliberately broken:** `test_optimizers.rs` (7 sites) and
`coverage_fill.rs` (1 site) use `t.borrow_mut().grad = Some(Storage::cpu(vec![...]))`
to inject a specific synthetic gradient value and drive an optimizer step
without a real `backward()` pass. Doing this from outside the crate needs a
public grad-injection method, which doesn't exist yet — this is deliberate,
not an oversight, so the exact scope of what's still missing stays visible
and measurable rather than being papered over with a hastily-designed
setter. Each broken call site is marked with a `// TODO(set_grad): ...`
comment pointing back here.

Note: *clearing* a gradient (`x.borrow_mut().grad = None`) is a different,
already-solved case — `TensorOps::zero_grad(&self)` is already public, so
those call sites (e.g. in `gpu_unary_test.rs`) were fixed normally.

Until a `set_grad`-style setter is designed and added, `cargo test` will
fail to build `test_optimizers.rs` and `coverage_fill.rs` specifically.
Every other file in `tests/` (including `tests/common/mod.rs`, shared by
the `parity_*.rs` files) builds and passes on both `--no-default-features`
and the default (`gpu`+`accelerate`) feature set.

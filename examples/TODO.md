# Known broken: examples vs. the opaque `Tensor` wrapper

Same root cause as `tests/TODO.md`: Phase 1 (ADDENDUM.md §1.1) made `Tensor`
an opaque wrapper (`inner` private, `borrow()`/`borrow_mut()` `pub(crate)`
only). Examples build as separate binaries against the crate's public API
only, so every example here that calls `.borrow()`/`.borrow_mut()` directly
on a `Tensor` no longer compiles. That's all 16 examples, ~67 call sites
total (`gpu_training.rs` and `layernorm_demo.rs` are the heaviest, at 9 and
11 respectively).

**Not fixed yet** — see `tests/TODO.md` for the actual plan (a small public
read API on `Tensor`: `shape()`, `data()`, `device()`, etc.) — once that
lands, these call sites get the same mechanical rewrite the integration
tests need.

`cargo build --examples` / `cargo run --example <name>` will fail until then.
Plain `cargo build`/`cargo check` (library only) are unaffected.

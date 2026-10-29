---
paths:
  - "tests/**"
  - "examples/**"
  - "benches/**"
  - "src/lib.rs"
  - "src/macros/**"
---

# Tests, examples, benches

- `tests/`, `examples/`, and `benches/` see only the public API. If one of them needs internals, either the test belongs in a `#[cfg(test)]` module in `src/`, or `Tensor` is missing a public method. Ask before adding one.
- Gradient and trainability tests use random, non-constant inputs. Constant inputs produce mathematically zero gradients for normalization and attention layers.
- `assert_trainable!` takes its input from `crate::macros::trainable_input(&shape)`, which is seeded and random. Don't hand-build constant inputs for it. The macro backprops a randomly weighted sum, because a plain `.sum()` zeroes BatchNorm's gamma gradient for any input.
- Read a loss or other single-element tensor with `Tensor::item()`, not `data().first()`; `item()` panics if the tensor has more than one element. Inside `src/`, `#[cfg(test)]` code reads slices with `crate::index::fetch`. In `tests/` and any `#[test]` fn, index with `v[i]` (`clippy.toml` sets `allow-indexing-slicing-in-tests`; it does not cover helpers outside a `#[test]` fn). When comparing expected against actual, `zip` them and assert the lengths are equal first, because `zip` stops at the shorter side. `examples/` and `benches/` use `.get(i).copied().expect(..)`. Never `unwrap_or(0.0)` (`just check-fallbacks` rejects it).
- Never encode a bug in an assertion. Use `#[ignore = "<defect ID>: <why>"]`.
- Examples must not route around broken components. If an example needs a workaround, file the defect in `notes/TODO.md` instead.
- Run `cargo clippy --all-targets`, not just `--all`, so tests and examples get linted.
- GPU integration tests are `tests/gpu_*.rs`. CPU/GPU parity tests are `tests/parity_*.rs`.
- Document tests, examples, and benches like the library: a `//!` crate doc saying what the target covers, and `///` on every item. Don't silence `missing_docs` or `clippy::missing_docs_in_private_items` to avoid writing docs. Exceptions need the user's say-so, and an `allow` must be scoped to the narrowest item and carry a `reason`.
- The one standing exception is `criterion_group!` (criterion 0.5), which generates an undocumented `pub fn benches`. A lint attribute on the macro call is ignored, so wrap the call in a `mod group { .. }` that carries `#[allow(missing_docs, clippy::missing_docs_in_private_items, reason = "..")]`, with targets as `super::bench_x`, and `use group::benches;`. See `benches/tensor_ops.rs`.

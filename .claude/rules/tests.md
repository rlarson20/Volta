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
- Never encode a bug in an assertion. Use `#[ignore = "<defect ID>: <why>"]`.
- Examples must not route around broken components. If an example needs a workaround, file the defect in `notes/TODO.md` instead.
- Run `cargo clippy --all-targets`, not just `--all`, so tests and examples get linted.
- GPU integration tests are `tests/gpu_*.rs`. CPU/GPU parity tests are `tests/parity_*.rs`.

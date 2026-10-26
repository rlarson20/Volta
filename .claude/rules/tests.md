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
- `assert_trainable!` uses a plain `.sum()` loss. Under that loss, BatchNorm's gamma gradient is exactly zero for *any* input, because normalized activations sum to zero. Until the macro switches to a randomly weighted loss, that BN failure is a false positive.
- Never encode a bug in an assertion. Use `#[ignore = "<defect ID>: <why>"]`.
- Examples must not route around broken components. If an example needs a workaround, file the defect in `notes/TODO.md` instead.
- Run `cargo clippy --all-targets`, not just `--all`, so tests and examples get linted.
- GPU integration tests are `tests/gpu_*.rs`. CPU/GPU parity tests are `tests/parity_*.rs`.

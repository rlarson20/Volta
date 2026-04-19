# Volta Test Coverage Analysis

**Well-covered:** Conv2d/ConvTranspose2d/MaxPool2d (~136 tests), Transformer components (~41), GPU smoke/stress tests, positional encoding, state-dict diff tooling.

**Critical gaps (zero or near-zero direct tests):**

| Area                                   | Gap                                                                             |
| -------------------------------------- | ------------------------------------------------------------------------------- |
| `src/ops/binary.rs`                    | Add/Sub/Mul/Div/Max — no isolated unit tests; broadcasting-in-backward untested |
| `src/ops/unary.rs`                     | exp/log/abs/sin/cos — no numerical-stability tests (log(0), exp(large))         |
| `src/ops/reduce.rs`                    | sum/mean/max reductions not directly tested                                     |
| `src/ops/movement.rs`                  | reshape/transpose/slice — only `movement_backward_test.rs`                      |
| `src/ops/matmul.rs`                    | 28KB file with no direct unit tests; only Linear/GPU integration                |
| `src/nn/layers/linear.rs`              | No direct forward/backward tests; no init verification                          |
| `src/nn/layers/dropout.rs`             | Only p=0.0 trivial case; no mask/train-vs-eval test                             |
| `src/nn/layers/batchnorm.rs`           | No running-stats, epsilon-stability, or mode-switch tests                       |
| `src/nn/layers/lstm.rs`                | Gate math, state propagation, gradient flow all untested                        |
| `src/nn/layers/{relu,sigmoid,tanh}.rs` | Only indirect coverage via Sequential                                           |
| `src/nn/optim/adam.rs`, `sgd.rs`       | Adam m/v accumulation + bias correction untested; SGD weight-decay untested     |
| `src/error.rs`                         | VoltaError variants and error paths never exercised                             |
| DType F16/BF16                         | Only shape/size metadata tested; no actual compute                              |
| GPU↔CPU parity                         | Only matmul verified; broadcasting & unary parity missing                       |
| Serialization round-trip               | Missing for BatchNorm, LSTM, ConvTranspose2d                                    |

**Proposed priorities:**

1. (DONE!!!) **Foundations (P1):** Unit tests for `ops/binary.rs`, `ops/unary.rs`, `ops/reduce.rs`, `ops/matmul.rs` including numerical-stability cases and broadcast-in-backward. These sit under everything else — bugs here poison every layer.
2. (DONE!!!) **Core layers (P1):** Direct tests for Linear, Dropout (train vs eval), BatchNorm (running stats, ε), and activations. Sequential-only coverage hides bugs in individual components.
3. (DONE!!!) **Optimizers (P2):** Adam bias-correction + m/v accumulation, SGD weight-decay, plus convergence tests on a convex problem.
4. (DONE!!!) **LSTM (P2):** Critical functionality currently with zero tests — gate math and state propagation. *Note: gradient flow test documented that `slice_gate` breaks the autograd graph.*
5. (DONE!!!) **Error paths (P3):** Exercise each `VoltaError` variant (shape/dtype/device mismatches) to lock in error messages and prevent regressions.
6. (DONE!!!) **GPU/CPU parity matrix (P3):** Parametric test harness comparing every op (unary/binary/reduce/movement) across devices on shared inputs, including broadcast shapes and F16/BF16 dtypes.
7. (DONE!!!) **Serialization round-trips (P3):** Save→load→compare for every layer type, not just Conv2d/Sequential.

(DONE!!!) A good next step is a parametric CPU/GPU parity harness plus P1 ops tests — those two together would catch the largest class of silent correctness bugs.

**All priorities (P1-P3) are now complete.** 39 new tests added across 5 test files:
- `tests/test_optimizers.rs` (7 tests) — Adam/SGD internals + convex convergence
- `tests/test_lstm.rs` (8 tests) — gate math, state propagation, gradient flow documentation
- `tests/test_error_paths.rs` (8 tests) — all 8 VoltaError variants exercised
- `tests/test_serialization.rs` (8 tests) — BatchNorm2d, LSTMCell, ConvTranspose2d round-trips
- `tests/parity_movement.rs` (8 tests) — GPU/CPU parity for transpose, reshape, permute, flatten

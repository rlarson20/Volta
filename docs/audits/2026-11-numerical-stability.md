# 2026-11 Numerical stability audit

Dated record. Original text is never edited or removed; corrections are appended under `## Corrections` (`.claude/rules/audits.md`).

**Date:** 2026-11-17. **Commit audited:** `ebbd683` (`improvements`). **Issue:** #298. **Finding IDs:** `N1`–`N11`.

**Scope.** Losses (`src/ops/loss.rs`, the loss functions in `src/tensor.rs`), `softmax` and `log_softmax`, BatchNorm and LayerNorm, the optimizers (SGD, Adam, Muon), the reductions and every backward pass that accumulates, and f32 accumulation on both backends. Half-precision (F16, BF16) behavior of the same code is in scope, because every half op computes in f32 and rounds once on store, so the rounding points decide the result.

**Method.** Read the code, then run each suspicion through the public API against two references: an f64 computation of the same formula, and PyTorch 2.15.0 (the version `scripts/torch_ref/gen_fixtures.py` pins, CPU build `+cu132`, installed with `uv` for this audit). The probes are in the appendix. They are not committed as tests, because an `#[ignore]` needs an issue ID (invariant 6) and this audit files no issues (triage is the maintainer's).

**What ran.** The CPU kernels and lavapipe, from one debug build with the `gpu` feature on Linux (lavapipe: Mesa 25.2.8, software Vulkan, wgpu 30.0.1). GPU rows ran under `DevicePolicy::PreferGpu`, so an op with no kernel ran on the CPU; each finding names which ops those were. **Nothing here ran on Metal.** Each finding says "confirmed" (a probe printed the numbers quoted, with the backend) or "by reading".

**Result.** One precision loss with a one-line fix that PyTorch does not have (N1), one NaN in pure-f16 models where PyTorch is finite (N2), one optimizer whose f16 behavior differs between CPU and GPU by five orders of magnitude (N3), three f32 chain sums left after #212 (N4–N6), and five low-severity or informational items (N7–N11). No finding is a wrong formula; every one is about where rounding happens.

| ID | Severity | One line | Evidence |
| --- | --- | --- | --- |
| N1 | Precision, PyTorch is exact | `log_softmax` adds the max back before subtracting: error up to half an ulp of the largest logit, inherited by `cross_entropy_loss` | CPU + lavapipe, torch |
| N2 | NaN where PyTorch is finite | `mean_dim` rounds the sum to f16 before dividing: BatchNorm and LayerNorm with f16 parameters return NaN once a sum passes 65504 | CPU + lavapipe, torch |
| N3 | Silent wrong step | f16 Adam: the second moment underflows; the CPU step grows as sqrt(t), the GPU step is `lr*m/eps` (100 per step in the probe) | CPU + lavapipe, torch |
| N4 | Precision (both backends) | `expand` backward is one f32 chain; it is also every Conv2d and ConvTranspose2d bias gradient: 4.0% low at 2^22 elements | CPU + lavapipe, torch |
| N5 | Precision | Conv2d weight gradient (Direct, IGEMM) is one f32 chain over all output positions: 4.0% low at 2^22 | CPU, torch |
| N6 | Precision | GPU matmul's inner product is one f32 chain over `k`: 4.0% low at k = 2^22, stalls at 2^24; the CPU is 1.5e-4 | lavapipe |
| N7 | Low | `bce_with_logits_loss` uses `ln(1 + exp(-abs(x)))`, not `log1p`: the tail term is 17% off at x = 15 and 0 from x = 17 (absolute error under 6e-8); gradient is right | CPU + lavapipe, torch |
| N8 | Low | Optimizer hyperparameters are f32, so f64 training carries f32-accurate `lr` and betas | CPU, torch |
| N9 | Low | Muon's Frobenius norm overflows for entries near 1e20: the update is silently zero | CPU + lavapipe |
| N10 | Informational | SGD folds `lr` into the velocity; equal to PyTorch while `lr` is constant, and there is no way to change `lr` today | CPU + lavapipe, torch |
| N11 | Informational | GELU's negative tail loses all relative precision by x = -6, exactly as PyTorch's f32 does; the GPU differs by 4.4e-5 at x = -3 | CPU + lavapipe, torch |

## Findings

### N1. `log_softmax` adds the max back before subtracting

`src/tensor.rs:1571-1578` computes `x - (ln(sum(exp(x - max))) + max)`. The inner sum `ln(..) + max` is rounded to the spacing of `max`, so the small term is lost before the subtraction. The stable order is `(x - max) - ln(sum(exp(x - max)))`: `x - max` is exact or nearly so, and the result keeps full relative precision. `cross_entropy_loss` (`src/tensor.rs:983`) calls `log_softmax`, so the loss carries the same error.

Confirmed on the CPU; the lavapipe run printed the same numbers (`MaxDim`, `SumDim` and the broadcast `Sub` have no kernel, so those steps ran on the CPU there).

| logits (one row) | Volta f32 | PyTorch 2.15 f32 | f64 |
| --- | --- | --- | --- |
| `[1000, 1000]` | -0.69317627 | -0.69314718 | -0.693147181 |
| `[10000.5, 10000, 9999]`, element 0 | -0.6044922 | -0.60413063 | -0.604130605 |
| `[-3000.25, -3000, -2999.5]`, element 0 | -1.4819336 | -1.4818375 | -1.481837567 |
| `[100.3, 100.1, 99.9, 0]`, element 0 | -0.91189575 | -0.91189957 | -0.911899516 |

The error is absolute, up to half an ulp of the largest logit: 3e-5 at 1000, 3.6e-4 at 10000, 3.8e-6 at 100. PyTorch's error in the same rows is under 6e-8. The gradient is unaffected in the probes (`softmax - target` to 7 digits), because the backward pass is built from `exp(x - max)` and its sum, which are not rounded this way; only the value is off. It matters where the loss value is read: early stopping, a reported validation loss, or a loss that should reach 0 and plateaus at 1e-5 instead.

**Fix.** Return `shifted.sub(&log_sum)`. One line, no new op, and the backward graph gets one node shorter.

### N2. `mean_dim` rounds the sum to f16 before dividing

`mean_dim` (`src/tensor.rs:1589-1594`) is `sum_dim(..).div(n)`: two ops, and a half tensor rounds after each. The f16 maximum is 65504, so the intermediate sum overflows to inf although the mean is small. `mean()` does not have this problem: it divides in f32 and rounds once.

BatchNorm (`src/nn/layers/batchnorm.rs`) and LayerNorm (`src/nn/layers/layernorm.rs`) take both statistics with `mean_dim`, so with f16 parameters (`Module::to_dtype(DType::F16)`) the variance step `mean((x - mean)^2)` overflows as soon as the count times the variance passes 65504. For BatchNorm2d on a 64x1x32x32 batch that is a variance above 1.

Confirmed on the CPU and on lavapipe (same numbers; `SumDim` has no kernel).

| Probe | Volta F16 | PyTorch 2.15 float16 |
| --- | --- | --- |
| `mean_dim` of 1024 values of 100.0 | inf | 100.0 |
| `mean()` of the same | 100.0 | 100.0 |
| BatchNorm2d, input 64x1x32x32, mean 0.50, variance 2.99, first three outputs | NaN, NaN, NaN | -1.3555, -0.8149, 1.3389 |
| LayerNorm(1024), row `100 + 0.01*i`, first output | NaN | -1.7305 |

BF16 is fine in all four (its range is f32's). The autocast path is fine too: with f32 parameters, `nn::norm_mixed` (`src/nn/mod.rs:28`) casts a half input up, normalizes in f32 and casts back, and the probe (f32 Conv2d and BatchNorm2d under `amp::autocast(DType::F16)`, and an f16 input into an f32 LayerNorm) returned the right values on both backends. So this needs a model whose norm parameters are f16.

**Fix.** Make `mean_dim` one kernel that divides before rounding, as `mean` is. That alone does not save the variance when `sum((x - mean)^2)` is formed by f16 ops upstream of the mean; the norm layers should compute their statistics in f32 for half parameters too (extend `norm_mixed` to "half input, any parameter dtype"), which is what PyTorch's norm kernels do.

### N3. f16 Adam: the second moment underflows, and CPU and GPU disagree

Adam keeps `m` and `v` in the parameter's dtype (`AdamState::new`, asserted at `src/nn/optim/adam.rs:229-235`). The update `v = beta2*v + (1 - beta2)*g^2` (`adam.rs:247-248`) with the default `beta2 = 0.999` writes a value near `1e-3 * g^2`, and the smallest f16 subnormal is 6e-8. For `|g|` under about 5.5e-3 the stored `v` rounds to zero every step.

Probe: one f16 parameter `w = 1`, loss `sum(w * 1e-3)`, `Adam(lr = 1e-3, betas = (0.9, 0.999), eps = 1e-8)`, three steps. The right answer is a step of `lr` each time.

| Backend | `w` after steps 1, 2, 3 |
| --- | --- |
| Correct (f32 Volta, CPU) | 0.9990000, 0.9980000, 0.9970000 |
| Volta F16, CPU | 0.99902, 0.99756, 0.99561 (steps of 2, 3, 4 f16 ulps) |
| Volta F16, lavapipe | -99, -199, -299 |
| PyTorch 2.15 float16 | -inf, -inf, -inf |

On the CPU the step computes in f32 and rounds `v` only when the guard writes back, so within a step `v` is right, but it restarts from zero each step while the bias correction `1/(1 - beta2^t)` shrinks: the step is `lr * sqrt(t)`, growing toward `lr / sqrt(1 - beta2)` = 31.6 `lr`. On lavapipe the step is `lr * m_hat / eps` from the first step (1e-3 * 1e-3 / 1e-8 = 100), which says the kernel reads `v` after it was rounded; I did not trace which line. PyTorch is wrong here too, differently: its `eps` is cast to f16 and rounds to zero, so the division is by zero.

No backend is right, so this is not a parity bug against PyTorch. It is a silent, finite, wrong step on the CPU and a CPU-versus-GPU disagreement of five orders of magnitude. With `eps = 1e-4` and the same gradient PyTorch steps by 0.01 (ten times `lr`), the same failure in a milder form. BF16 does not underflow, but its 8-bit significand cannot represent `1 - 1e-3`, so `w` stays at 1.0 on both backends, as in PyTorch.

**Fix options.** Keep optimizer state (`m`, `v`, SGD and Muon velocity) in f32 for half parameters, which departs from PyTorch and is right; or reject half parameters in `Adam::try_new` with a message that points to f32 parameters under `amp::autocast` (the supported mixed-precision path, and what `GradScaler`'s FP16 message already says). Either way the CPU and GPU kernels should agree on when `v` is rounded.

### N4. `expand` backward is one f32 chain, and it is the conv bias gradient

The CPU `expand` backward adds each output gradient into its source element in index order (`src/ops/movement.rs:573`); the GPU kernel does the same per source element (`src/gpu/shaders/movement_backward.wgsl:254`). The GPU audit reported the stall at 2^24 as G3 and handed the CPU half here (G17). Two things are new.

First, the error is large well below 2^24. With an upstream gradient of 0.1 in each of 2^22 elements (exact sum 419430.4):

| Path | Result | Relative error |
| --- | --- | --- |
| Volta `expand` backward, CPU | 402740.78 | -4.0% |
| Volta `expand` backward, lavapipe | 402740.78 | -4.0% |
| PyTorch 2.15 `expand` backward | 419430.47 | 1.6e-7 |
| Volta `sum` / `sum_dim` forward (pairwise, #212) | 419430.8 | 1e-6 |

Second, this is not an exotic op. `Conv2d::forward` adds its bias as `bias.reshape([1, C, 1, 1]).expand(output_dims)` (`src/nn/layers/conv.rs:1445-1451`), and ConvTranspose2d does the equivalent, so every conv bias gradient is this chain over batch x height x width. Probe: a 1x1 Conv2d on a 16x1x512x512 input, loss `sum(conv(x) * 0.1)`, all three algorithms on the CPU: bias gradient 402740.78 (-4.0%). PyTorch's own conv bias gradient for the same input is 423567.375 (+0.99%), so PyTorch is not exact here either, but it is four times closer. With 2^25 ones the Volta bias gradient is 16,777,216 and PyTorch's is 33,554,432. On lavapipe the conv forward for this input size panics first (G5), so the GPU row for conv was not measured; the `expand` kernel it would use was.

A constant-sign gradient is the worst case. Mixed-sign gradients cancel and the relative error against `sum(abs(g))` is smaller, but a bias gradient late in training is exactly a sum of many small same-sign terms.

**Fix.** Sum each source element's contributions with `pairwise_sum_by` (`src/ops/reduce.rs`), as the broadcast backward does since #212, and give the GPU kernel the tree reduction `reduce.wgsl` uses (the same change removes G8's quadratic cost).

### N5. Conv2d weight gradient (Direct, IGEMM) is one f32 chain

The direct and IGEMM weight gradients add `x * dout` into each weight over every output position in index order (`src/nn/layers/conv.rs:451`, `:773`). Same probe as N4 (1x1 conv, 16x1x512x512 input of ones, upstream 0.1):

| Algorithm (CPU) | Weight gradient | Relative error |
| --- | --- | --- |
| Direct | 402740.78 | -4.0% |
| IGEMM | 402740.78 | -4.0% |
| Im2col (through gemm) | 419365.66 | -1.5e-4 |
| PyTorch 2.15 | 419424.5 | -1.4e-5 |

With 2^25 ones both the weight and the bias gradient stall at 16,777,216. So which algorithm `ConvAlgo::Auto` picks changes the weight gradient by 4% for a large feature map. ConvTranspose2d's weight and input gradients accumulate the same way (`src/nn/layers/conv_transpose.rs:226`, `:283`); that is by reading, not probed.

**Fix.** Accumulate per weight in blocks (a partial sum per output row or per batch item, then a pairwise sum of the partials). This keeps the loop order and costs one small buffer.

### N6. GPU matmul's inner product is one f32 chain over `k`

`src/gpu/shaders/matmul.wgsl:69` adds `a[i,k] * b[k,j]` into one `sum` across all tiles, and `matmul_backward.wgsl:78`, `:142` do the same. Confirmed on lavapipe:

| `(1,k) @ (k,1)` | lavapipe | CPU | PyTorch 2.15 | Exact |
| --- | --- | --- | --- | --- |
| k = 2^22, a = 0.1, b = 1 | 402740.78 | 419365.66 | 423567.375 | 419430.4 |
| k = 2^25, a = b = 1 | 16,777,216 | 33,554,432 | 33,554,432 | 33,554,432 |

The CPU gemm blocks its sum, so the two backends differ by 4% at k = 2^22. For the `k` of a typical layer (hundreds to a few thousand) the chain's worst-case relative error is `k * 6e-8`, under 1e-4, which is inside the parity tolerances (`TOL_MATMUL_FWD = 1e-4`); this only bites for a very long contraction, such as a flattened feature map into a Linear layer.

**Fix.** Keep one partial sum per tile and add the partials at the end; the tile loop is already there.

### N7. `bce_with_logits_loss` uses `ln(1 + e)`, not `log1p(e)`

`src/tensor.rs:1163` computes the tail term as `log(1 + exp(-abs(x)))`. For `abs(x)` above about 16.6, `1 + exp(-abs(x))` rounds to 1 and the term is 0; just below, it is quantized to multiples of 1.2e-7. Confirmed on the CPU and on lavapipe (same to 7 digits), target 1:

| x | Volta | PyTorch 2.15 f32 | f64 | Relative error |
| --- | --- | --- | --- | --- |
| 10 | 4.5417706e-5 | 4.5398901e-5 | 4.5398899e-5 | 4.1e-4 |
| 12 | 6.198864e-6 | 6.144193e-6 | 6.144193e-6 | 8.9e-3 |
| 15 | 3.576278e-7 | 3.059023e-7 | 3.059023e-7 | 17% |
| 17 | 0 | 4.139937e-8 | 4.139938e-8 | 100% |

The absolute error never exceeds 6e-8 per element, and the gradient is right at every row (for example -3.059022e-7 at x = 15 against -3.0590223e-7), because the backward of `log` divides by the rounded `1 + e`, which is 1 to f32 precision. The effect is a loss that reads exactly 0 for confident predictions while PyTorch reports 1e-8. Low.

The comment at `src/tensor.rs:1158` says a tie at `x == 0` gives `relu'(0) = 1`. `max_elem` now splits a tie in half (`src/ops/max_rule.rs`, `maximum_backward`), so that comment is stale. The net gradient at 0 is still `0.5 - y`, because the two `max_elem` calls on the next line split the same way and cancel; by reading.

**Fix.** There is no `log1p` op. Adding one (a `define_ops!` row with a GPU kernel) is the clean fix; it is a new op, so it is a feature decision.

### N8. Optimizer hyperparameters are f32

`lr`, `eps`, `weight_decay`, momentum and the betas are `f32` fields, widened to the kernel's type (`T::from_f64(f64::from(self.lr))`, `src/nn/optim/adam.rs:203-209`, `sgd.rs:199-201`). For an F64 parameter the kernel is f64 but `lr = 0.1` is 0.100000001490116. PyTorch holds these as Python doubles for every dtype.

Confirmed on the CPU, one parameter `w = 1`, loss `w^2`:

| Probe | Volta | PyTorch 2.15 |
| --- | --- | --- |
| SGD f64, `lr = 0.1`, 1 step | 0.7999999970197678 | 0.8 |
| Adam f64, `lr = 1e-3`, 1 step | 0.9989999999575025 | 0.999000000005 |
| Adam f64, 1000 steps | 0.25766519364500867 | 0.257665027571658 |
| Adam f32, 1 step and 10 steps | 0.9990000128746033, 0.990003228187561 | identical, both |
| Adam f32, 1000 steps | 0.2576664686203003 | 0.25766611099243164 |

So an f64 run agrees with PyTorch to about 1e-8 per step and 6e-7 after 1000 Adam steps, not to the 1e-10 that `tests/torch_ref_f64.rs` uses for single ops. The f32 run drifts 1.4e-6 over 1000 steps for the same reason: `1 - beta2` computed from the f32 0.999 is 1.3e-5 off relative. Low: it is a parity limit, not an instability.

**Fix.** Store the hyperparameters as `f64` (constructors can keep taking `f32` or gain `f64` forms; that is an API decision for #299).

### N9. Muon's Frobenius norm overflows

`newton_schulz_with` normalizes by `sqrt(sum(v*v))` (`src/nn/optim/muon.rs:145`), an unscaled sum of squares. An entry near 1e20 squares to inf in f32, the inverse norm is 0, and the update is zero. Confirmed on the CPU and on lavapipe: a 2x3 gradient scaled by 1e20 leaves the weights unchanged, where the scale-1 gradient moves them by about 0.6 (Muon's update is meant to be scale invariant). The reference implementation (`X / (X.norm() + 1e-7)` in PyTorch f32) overflows the same way.

A gradient norm far below `NS_EPS = 1e-7` gives a near-zero update as well (scale 1e-21: weights move 1e-11). That is the epsilon doing its job, the same in the reference, and not a finding. The sum of squares is also a sequential f32 sum (`Iterator::sum`), so a matrix over 2^24 elements of similar size would underestimate its norm and start Newton-Schulz above singular value 1; by reading, and an 8192x8192 weight is outside this library's range. Low.

**Fix.** Scale by the largest absolute entry before squaring (`m * sqrt(sum((v/m)^2))`), and use `pairwise_sum_by`.

### N10. SGD folds `lr` into the velocity

`v = mu*v - lr*g; p += v` (`src/nn/optim/sgd.rs:230-236`, and the GPU kernel) against PyTorch's `v = mu*v + g; p -= lr*v`. The two are the same sequence while `lr` is constant: the probe (`lr = 0.1`, momentum 0.9, loss `w^2`) printed 0.8 and 0.46 on the CPU and on lavapipe, PyTorch's values. They differ after `lr` changes, because Volta's stored velocity keeps the old `lr`. No optimizer has an `lr` setter or a scheduler today, so the difference cannot be observed. Informational: decide the form before a scheduler lands (PyTorch's, if fixtures are to match).

### N11. GELU's negative tail

`x * 0.5 * (1 + erf(x / sqrt(2)))` (`src/nn/layers/gelu.rs`) cancels in `1 + erf(..)` for negative `x`. On the CPU the results are bit-identical to PyTorch 2.15 f32 at all five probe points, including the bad ones:

| x | Volta CPU = PyTorch f32 | f64 | Relative error |
| --- | --- | --- | --- |
| -3 | -0.0040496886 | -0.0040496941 | 1.4e-6 |
| -4 | -1.2671947e-4 | -1.2668497e-4 | 2.7e-4 |
| -5 | -1.4901161e-6 | -1.4332579e-6 | 4.0% |
| -5.5 | -1.6391277e-7 | -1.0444259e-7 | 57% |
| -6 | -0.0 | -5.92e-9 | 100% |

The absolute error is under 6e-8 everywhere. On lavapipe the forward at -3 is -0.0040498674 (4.4e-5 relative to the CPU), from the shader's polynomial `erf`; the other four points match the CPU. Under `PreferGpu` the scalar `div`, `add` and `mul` of GELU are CPU gaps (`Binary(Div)`, `Binary(Add)`, `Binary(Mul)` broadcast), so only `erf` ran on the device. Informational: matching PyTorch is the project's stated target, and an `erfc`-based form would be more accurate than PyTorch.

## Checked and found correct

Each of these was probed, with the backend named, unless it says "by reading".

- **`softmax` and `log_softmax` at the extremes** (CPU and lavapipe): an all `-inf` row and a `+inf` logit give NaN, and `[3e38, -3e38]` gives `[1, 0]`, all as PyTorch. With a `-inf` mask entry, `[0, -inf, 1]`, the gradients of `-log_softmax[0]` and `softmax[0]` are `[-0.7310586, 0, 0.7310586]` and `[0.19661194, 0, -0.19661194]`, PyTorch's values, with no NaN from the masked position.
- **`bce_loss`** (CPU and lavapipe): the `-100` log floor and the `1e-12` gradient clamp give the same loss (216.63553 summed) and gradients (1e12, -1e12, 16777216) as PyTorch for `p = [1, 0, 1 - 6e-8]`.
- **BatchNorm and LayerNorm in f32 with a large mean** (CPU and lavapipe): inputs `10000 + 0.01*k`. Volta's first output is -1.3359019 against -1.3306193 in f64, a 0.4% error; PyTorch f32 gives -1.3402599 (LayerNorm) and -1.3365479 (BatchNorm1d). The error comes from the f32 inputs' own spacing of 1e-3, the two-pass variance is the stable form, and Volta is at least as close as PyTorch. Running statistics use the unbiased variance, as PyTorch.
- **Forward reductions** (CPU and lavapipe): `sum`, `mean`, `sum_dim`, `mean_dim` of 2^25 ones are exact, and 2^22 values of 0.1 sum to 419430.8 (1e-6). The broadcast backward of a binary op is exact at 2^25 (pairwise since #212).
- **CPU matmul** forward and backward with a contraction of 2^25 ones: exact.
- **Embedding backward**: 2^25 lookups of one row stall at 16,777,216, and so does PyTorch's. Parity, not a finding.
- **Adam in f32 on the CPU**: bit-identical to PyTorch after 1 and 10 steps. (The GPU's `pow` bias correction is G4.)
- **CPU `modulo`**: `[1e10 mod 3, 5.5 mod inf, -5.5 mod inf]` is `[1, 5.5, inf]`, PyTorch's `remainder`. (The GPU's is G2.)
- **`sigmoid`** (CPU and lavapipe): 0 at -90 and below, 1 at 20 and above, as PyTorch. The backward is `s*(1 - s)`, PyTorch's formula, which is 0 once `s` rounds to 1 (PyTorch printed 0 at 20 and 40; Volta's gradient there is by reading).
- **KL divergence**: `exp(logvar)` overflows to inf at `logvar = 89`, as the same formula does in PyTorch. By reading against the torch output.
- **`div` and `recip` backward**: `-g*a*(1/b)*(1/b)` multiplies left to right, so an intermediate overflows only when the result does. By reading, and `tests/div_recip_grad_range.rs` covers it.
- **`GradScaler`**: the scale is a power of two, so scaling and unscaling are exact in f32 and f16 until overflow, which `step` detects. By reading.
- **`accumulate_grad`**: one add per contribution, half types add in f32 and round. A tensor used 2^24 times would stall, which no model does. By reading.

## Not covered

- Metal: nothing ran there.
- `check_gradients` (step size and tolerances of the finite-difference checker).
- LSTM, attention and the transformer blocks end to end. Their building blocks (`sigmoid`, `tanh`, `softmax` with `-inf` masks, LayerNorm) are covered above.
- ConvTranspose2d and MaxPool2d accumulation, beyond the reading note in N5.
- Initializers and the RNG.
- Release builds. The probes ran in the debug profile; the kernels use explicit `mul_add` and no fast-math flags, so the numbers should not move, but that was not run.

## Appendix: probes

The probes are integration tests against the public API (`tests/*.rs`, `#![cfg(feature = "gpu")]`), run with `VOLTA_REQUIRE_GPU=1 cargo test --test <file> -- --nocapture --test-threads=1`. GPU passes hold `with_device_policy(DevicePolicy::PreferGpu)`. Shared helper:

```rust
fn leaf(data: Vec<f32>, shape: &[usize], gpu: bool, dt: DType) -> Tensor {
    let mut t = RawTensor::new(data, shape, false);
    if dt != DType::F32 { t = t.to_dtype(dt).detach(); }
    if gpu { t = t.to_device(Device::gpu().expect("gpu")); }
    t.set_requires_grad(true);
    t
}
```

**N1.**

```rust
let x = leaf(vec![10000.5, 10000.0, 9999.0], &[1, 3], gpu, DType::F32);
println!("{:?}", x.log_softmax(Axis(1)).data());        // [-0.6044922, -1.1044922, -2.1044922]
// torch: F.log_softmax(torch.tensor([[10000.5, 10000.0, 9999.0]]), 1)
//        [-0.60413063, -1.10413063, -2.10413074]
```

**N2.**

```rust
let h = leaf(vec![100.0; 1024], &[1, 1024], gpu, DType::F16);
h.mean_dim(Axis(1), false);                              // [inf]; h.mean() is [100.0]
// 64*32*32 values 3*u + 0.5, u from a 64-bit LCG in [-1, 1): mean 0.5003, variance 2.9873
let x = leaf(data, &[64, 1, 32, 32], gpu, DType::F16);
let mut bn = BatchNorm2d::new(1);
bn.to_dtype(DType::F16);
bn.forward(&x);                                          // NaN everywhere
// torch: torch.nn.BatchNorm2d(1).half()(x.half()) -> [-1.3555, -0.8149, 1.3389, ..]
```

**N3.**

```rust
let w = leaf(vec![1.0], &[1], gpu, DType::F16);
let c = leaf(vec![1e-3], &[1], gpu, DType::F16); c.set_requires_grad(false);
let mut o = Adam::new(vec![w.clone()], 1e-3, (0.9, 0.999), 1e-8, 0.0);
for _ in 0..3 {
    o.zero_grad();
    w.elem_mul(&c).sum().into_loss().unwrap().backward();
    o.step();                                            // CPU 0.99902, 0.99756, 0.99561
}                                                        // lavapipe -99, -199, -299
```

**N4, N5.**

```rust
let (b, hw) = (16, 512);
let n = b * hw * hw;                                     // 2^22
let c = Conv2d::new(1, 1, 1, 1, 0, true);
c.set_algo(algo);                                        // Direct, Im2col, IGEMM
let x = RawTensor::new(vec![1.0; n], &[b, 1, hw, hw], false);
let k = RawTensor::new(vec![0.1; n], &[b, 1, hw, hw], false);
c.forward(&x).elem_mul(&k).sum().into_loss().unwrap().backward();
// [weight, bias] gradient: Direct and IGEMM [402740.78, 402740.78]; Im2col [419365.66, 402740.78]
// torch: 419424.5, 423567.375. Exact: 419430.4.

let s = leaf(vec![1.0], &[1], gpu, DType::F32);
let kk = RawTensor::new(vec![0.1; 1 << 22], &[1 << 22], false); // moved to the GPU for that pass
s.expand(&[1 << 22]).elem_mul(&kk).sum().into_loss().unwrap().backward();
// s.grad(): 402740.78 on the CPU and on lavapipe; torch 419430.47
```

**N6.**

```rust
let k = 1 << 22;
let a = leaf(vec![0.1; k], &[1, k], gpu, DType::F32);
let b = leaf(vec![1.0; k], &[k, 1], gpu, DType::F32);
a.matmul(&b).data();                                     // lavapipe [402740.78], CPU [419365.66]
```

**N7.**

```rust
let x = leaf(vec![15.0], &[1], gpu, DType::F32);
let y = RawTensor::new(vec![1.0], &[1], false);
RawTensor::bce_with_logits_loss(&x, &y).item();          // 3.576278e-7; torch 3.0590226e-7
```

**N8.**

```rust
let w = leaf(vec![1.0], &[1], false, DType::F64);
let mut o = SGD::new(vec![w.clone()], 0.1, 0.0, 0.0);
w.elem_mul(&w).sum().into_loss().unwrap().backward();
o.step();                                                // 0.7999999970197678; torch 0.8
```

**N9.**

```rust
let w = leaf(vec![0.0; 6], &[2, 3], gpu, DType::F32);
let g = RawTensor::new([1.0f32, 2.0, 3.0, -1.0, 0.5, 4.0].map(|v| v * 1e20).to_vec(), &[2, 3], false);
let mut o = Muon::new(vec![w.clone()], 1.0, 0.0, false, 5);
w.elem_mul(&g).sum().into_loss().unwrap().backward();
o.step();                                                // w stays [0; 6]; at scale 1 it is
                                                         // [-0.62, -0.68, -0.30, 0.56, 0.28, -0.68]
```

**PyTorch side.** One script under `uv` with `torch==2.15.0`, mirroring each probe with the same inputs (`F.log_softmax`, `F.binary_cross_entropy_with_logits`, `torch.optim.SGD` / `Adam`, `F.gelu`, `torch.nn.BatchNorm2d(1).half()`, `torch.nn.Conv2d(1, 1, 1)`, `expand`, `@`). Its printed values are the "PyTorch 2.15" columns above.

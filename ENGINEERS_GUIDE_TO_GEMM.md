# An Engineer's Guide to GEMM, applied to Volta

Source: Pete Warden, [*An Engineer's Guide to GEMM*](https://petewarden.com/2015/10/25/an-engineers-guide-to-gemm/) (2015).
Repo state: branch `improvements` at `24ba17e` (Phase 2 closed). The CPU-path claims below come from reading the code, plus one reproduction for §6. GPU claims come from reading the WGSL and the wrappers. No GPU test was run, because this environment has no adapter.

## TL;DR

- **Pete's thesis:** storage order, transposition, and argument order each change a GEMM's result, and their errors can cancel ("Errors % 2 == 0"). So code can be wrong in two places and still pass. His defenses are hand-worked, non-square unit tests; a naive reference GEMM; and a clear model of leading dimensions (`lda`/`ldb`/`ldc`).
- **Volta handles the core hazards well.** It uses one storage order (dense row-major) everywhere. `MatmulSpec` names and validates `m, k, n` once. The GPU wrappers take the spec instead of loose `usize`s. Non-square hand-calculated tests exist on both CPU and GPU, and the CPU gradient checks use a randomly weighted loss specifically so a transposed dim can't hide.
- **Gap 1 (confirmed, unsound): `RawTensor::matmul_raw` is a safe `pub fn` that reads out of bounds.** It passes raw pointers and caller-supplied `m, k, n` to BLAS without checking slice lengths. I reproduced a heap over-read from safe code (§6). This is the "dims and pointers that disagree" failure that `lda` and its siblings invite.
- **Gap 2 (silent wrong weights): square matrices are Volta's blind spot, exactly as Pete warns.** `load_weight` checks shapes exactly, so a missing or extra transpose fails loudly *unless the matrix is square*. Every `MultiHeadAttention` projection is a square `Linear(embed_dim, embed_dim)`. The `transpose_pattern("weight")` helpers also transpose 2-D weights that already match PyTorch's layout (`Embedding`, `LSTMCell`).
- **Gap 3 (weak tests): several tests can't see a transpose.** `test_batched_matmul` uses all-ones inputs. The GPU parity helper backprops a plain `.sum()`, so every element of the incoming gradient is identical. `test_chained_transformations` asserts only the shape of a 2×2 transpose, and its comment calls `[1,2,3,4]` "symmetric".
- **Nothing in the plan addresses leading dimensions yet.** Phase 3/4 brings strided, stride-0 views (`notes/TODO.md:34`). After that, "every buffer is dense row-major" stops being true, and `matmul_raw`'s hardcoded `lda = k` becomes a correctness hazard. No reference GEMM exists or is planned. The planned proptest DAG gradient checks (ADDENDUM §10.2) and `cargo-mutants` (§10.4) would strengthen the cancellation defenses.

## Scorecard

| # | Pete's pitfall | Volta's mechanism | Status |
|---|---|---|---|
| 1 | Row- vs column-major confusion | One convention: dense row-major throughout. BLAS called with `CblasRowMajor`; `matrixmultiply` given explicit `(row, col)` strides | ✅ Covered (by convention, not by type) |
| 1b | Different code expects different conventions | `Linear` stores `[in, out]`; `LSTMCell` stores PyTorch's `[4H, in]`; PyTorch `Linear` is `[out, in]` | ⚠️ Partial: two conventions in one crate |
| 2 | A square matrix in the wrong order is silently read as its transpose | `load_weight` checks shapes exactly, which catches non-square mistakes only | ❌ Gap for square weights |
| 3 | Argument order / non-commutativity / `C' = B'A'` | `MatmulSpec` (validated `m, k, n`, inner-dim check); spec-typed GPU wrappers; no "swap the arguments" hacks | ✅ Covered for `Tensor::matmul`; ⚠️ `matmul_raw` is still positional |
| 4 | Errors that cancel (`Errors % 2 == 0`) | Hand-calculated non-square tests; FD gradient checks with a random weighted loss; `tests.md` rule on weighted losses | ⚠️ Partial: some tests use symmetric data or a plain sum. Planned: proptest DAG (§10.2), `cargo-mutants` (§10.4) |
| 5 | No simple reference GEMM to compare against | Conv has one (direct conv cross-checked against im2col/iGEMM). Matmul does not: the CPU oracle is BLAS or `matrixmultiply`, depending on platform | ❌ Gap for matmul; not planned |
| 6 | Leading dimensions (`lda`/`ldb`/`ldc`) misunderstood | None needed today, since everything is contiguous. But `matmul_raw` doesn't check `len ≥ m·k` and is reachable from safe code | ❌ Unsound today; ⚠️ hazard grows with Phase 3 views |
| 7 | Low-precision offsets kept out of the inner loop (from the post's comments) | f32 only; bf16 autocast is a backlog item | — Not applicable yet |

---

## 1. Row-major vs column-major

**Pete:** layout is "an implementation detail that should not affect the underlying math," but there is "no widespread agreement" on which order is canonical, "so you'll need to pay attention to what convention the code you're interacting with expects." Coming from image processing, he assumed row-major and tripped over the column-major gemmlowp.

**What Volta does**

- It has one convention. `Strides` is documented as "Row-major memory strides" (`src/shape.rs:22`), and `dev-docs/ARCHITECTURE.md:32` states "Row-major flat vectors". Every op materializes a fresh contiguous buffer (`notes/TODO.md:34`: "Every op deep-copies storage"), so there are no views, and therefore no non-contiguous or column-major buffers anywhere.
- Both CPU GEMM back ends receive the convention explicitly rather than by default:
  - Accelerate: `cblas_sgemm(101 /* CblasRowMajor */, NoTrans, NoTrans, m, n, k, …, lda = k, ldb = n, …, ldc = n)` (`src/ops/matmul.rs:84-99`). I checked the argument order against the CBLAS signature `(M, N, K)`, and it is correct.
  - `matrixmultiply::sgemm(m, k, n, …, a, rsa = k, csa = 1, b, rsb = n, csb = 1, …, c, rsc = n, csc = 1)` (`src/ops/matmul.rs:108-121`). The audit also checked these strides (`docs/audits/OPUS_5_EVAL_AUDIT.md:805`).
- The WGSL kernels index row-major (`a[row * k + col]`, `b[row * n + col]`, `c[row * n + col]` in `src/gpu/shaders/matmul.wgsl`).

**Assessment:** covered, because there is only one convention to get wrong. Volta enforces it by construction (no views), not by type: no `Layout` or contiguity marker reaches `matmul_raw`. That is fine today, but §6 explains why it stops being fine in Phase 3.

### 1b. Conventions *within* the crate

Pete's real complaint is about the boundary between code that assumes different conventions. Volta has such a boundary internally:

| Layer | Stored weight | Forward |
|---|---|---|
| `Linear` | `[in, out]` (`src/nn/layers/linear.rs:51`) | `x.matmul(&w)` |
| `LSTMCell` | `[4H, in]`, `[4H, H]` (`src/nn/layers/lstm.rs:34-35`) | `x.matmul(&w.transpose())` (`lstm.rs:87-88`) |
| `Embedding` | `[vocab, dim]` | row gather |
| PyTorch `nn.Linear` | `[out, in]` | `x @ w.T` |

`LSTMCell` and `Embedding` already match PyTorch, and `Linear` does not. Neither choice is wrong, but the difference is undocumented outside `io/mapping.rs`, and it is what makes §2's loader hazard possible.

## 2. Transpose: the silent square-matrix failure

**Pete:** "if you screw it up and pass a square row-major matrix into a library that expects column major, or vice-versa, it will be read as the transpose of the actual matrix."

**What protects Volta**

- `Tensor::load_weight` asserts `src.shape == t.shape` exactly, then checks numel (`src/tensor.rs:98-114`, fixed as part of H7). For a **non-square** weight, a missing or extra transpose is therefore a loud panic, and `load_state_dict_checked` reports and skips it.
- Inside the op graph, transposition is always explicit: `transpose_2d` on the CPU, and transposed indexing in `matmul_backward.wgsl`. No code relies on a layout flip to produce a transpose implicitly.

**Where Volta is exposed**

1. **Square `Linear` weights load silently transposed.** `MultiHeadAttention` builds `q_proj`, `k_proj`, and `v_proj` as `Linear::new(embed_dim, embed_dim, false)` (`src/nn/transformer/multi_head_attention.rs:61-63`). If a user loads PyTorch weights and forgets `.transpose(..)`, every projection loads with the right shape and the wrong values. Nothing fails; the model is just wrong.
2. **`transpose_pattern("weight")` is too broad.** `load_pytorch_linear_weights` and `load_huggingface_weights` (`src/io/mapping.rs:261`, `:275`) transpose *every* 2-D tensor whose key contains `"weight"`. That includes `Embedding.weight` and `LSTMCell.weight_ih`/`weight_hh`, which already use PyTorch's layout (§1b). For non-square shapes this is loud, since `load_weight` rejects it. For a square case (`Embedding` with `vocab == dim`, or any square `Linear`-like weight that already matches PyTorch's layout) it is silent. This is Pete's failure in its purest form: the convention is chosen by a substring match on a key name.
3. **One test cannot observe the property it claims to check.** `test_chained_transformations` (`src/io/mapping.rs:490-510`) transposes a 2×2 `[1,2,3,4]` and asserts only `shape == [2, 2]`, with the comment "Transposed (but 2x2 is symmetric)". The matrix `[[1,2],[3,4]]` is not symmetric, and transposing a square matrix leaves its shape unchanged. The test therefore passes whether or not the transpose runs. (`test_transpose_2d` at `:353` does check values on a non-square matrix, so the transform itself is covered.)

**Assessment:** a gap. The shape check is a good mechanism, but it is blind exactly where Pete says to look.

## 3. Argument ordering and `C' = B' · A'`

**Pete:** matrix multiplication isn't commutative, the inner dims must agree, and `C' = B'A'` means a swapped argument order can masquerade as a layout change.

**What Volta does**

- **`MatmulSpec`** (`src/spec.rs:304-358`) is built once from the two operand `Shape`s. It rejects rank < 2, unequal rank, mismatched inner dims (`"inner dims {k} vs {k2}"`), and batch dims that don't broadcast. It then exposes named `m`, `k`, `n`, and `MatMulGradFn` stores it, so the backward pass never re-derives dims from the wrong tensor. That was B4's root cause (DECISIONS.md, "MatmulSpec").
- **The GPU wrappers take `&MatmulSpec`** instead of three positional `usize`s (`notes/TODO.md:71`). They assert buffer lengths (`a.len() == m*k`, `b.len() == k*n`) and refuse batched specs (`src/gpu/kernels.rs:901-912`, `998-1013`, `1099-1114`).
- **Volta never uses the `C' = B'A'` trick.** The backward pass computes `dA = dC · Bᵀ` and `dB = Aᵀ · dC` literally, by materializing the transpose (`src/ops/matmul.rs:364`, `:379`) or by reading it transposed in the kernel. Pete's bug was exactly this kind of argument swap used to paper over a layout mismatch, and none exists here.
- **There is a test aimed at this.** `test_gpu_matmul_non_square_uses_the_right_dims` (`src/gpu/kernels.rs:2863`) uses `(2×3)·(3×4)`, where "m, k, n are all different, so a swapped pair gives wrong values."

**Residual risk**

- `RawTensor::matmul_raw(a, b, m, k, n)` is still positional `usize` (`src/ops/matmul.rs:59`). It sits between two libraries with *different* argument orders: CBLAS takes `(M, N, K)` and `matrixmultiply` takes `(m, k, n)`. Both mappings are correct today. Its non-`Tensor` callers are Muon (`src/nn/optim/muon.rs:73`, `:93`) and the backward helpers, and I checked both Muon calls by hand: `(cols, rows, cols)` and `(rows, cols, cols)` are right. Nothing but care keeps them right.
- The comments on `BackwardParams` in `src/gpu/shaders/matmul_backward.wgsl:12-16` are wrong for `backward_b`. They describe `m` as "cols of grad" and `k` as "rows of grad"; `grad` is `[M, N]`. The code is correct, because it uses the spec's `m, k, n` meanings. But Pete's method is "explain it to someone else," and comments that explain it wrongly are how the next edit goes wrong.

**Assessment:** covered for the public `Tensor::matmul` path. `matmul_raw` is the one place where the three-way ordering hazard is still live.

## 4. Errors % 2 == 0: cancelling bugs

**Pete:** storage order, transposition, and input order "can mimic and cancel each other out." His defense is to work small examples by hand and turn them into unit tests "with the hand calculations in the comments."

Two kinds of cancellation are worth separating:

- **Exact cancellation** gives correct numbers for every input. No test can detect it, by definition. Pete's code was right but unreadable, and a reviewer caught it. Volta's single convention and spec-named dims (§1, §3) are what remove the opportunity.
- **Partial cancellation** is correct only for square shapes, symmetric data, or uniform gradients. It *is* detectable, but only by tests with non-square shapes and asymmetric, random data. This is where test design matters.

**Mechanisms in place**

| Mechanism | Where | Pete-aligned? |
|---|---|---|
| Hand-calculated, non-square forward test, worked in comments | `matmul_2d_2d_value` (`src/lib.rs:4693`): `(2×3)·(3×2)`. A column-major misread of B gives `[1, 6]`, not `[4, 5]`, so it would be caught | ✅ |
| Same, on the GPU | `test_gpu_matmul` (2×2, worked in comments), `test_gpu_matmul_non_square_uses_the_right_dims` | ✅ |
| FD gradient check with a **random weighted** loss over 2-D, batched, and broadcast-batched shapes | `assert_matmul_gradients` (`src/ops/matmul.rs:614-637`): "A plain sum would be blind to some transposed-dim bugs" | ✅ |
| Rule that gradient tests use random, non-constant inputs and weighted losses | `.claude/rules/tests.md`; `.claude/rules/autograd.md` (FD check for every op, with unequal-rank and broadcast cases) | ✅ |
| Rule against encoding a known bug in a test | Invariant 6 | ✅ |

**Tests that cannot see a transpose**

- `test_batched_matmul` (`src/lib.rs:665`) multiplies all-ones by all-ones. Every layout, transpose, or ordering bug gives the same `3.0`.
- `assert_matmul_parity` (`tests/common/mod.rs:206-208`) backprops `c.sum()` for both CPU and GPU, so the incoming gradient `dC` is all ones. A GPU backward kernel that indexes `dC` wrongly (swapped row and column, wrong row stride) reads the same value from every element, and the test passes. It still catches a misread of `B`/`A`, because random operands make row sums differ from column sums. The parity shapes (`3×4·4×5` and `8×8`) are also all smaller than one 16×16 tile, so the multi-tile `k` loop and partial-tile edges are never checked against the CPU. CPU-side `assert_matmul_gradients` already does this right (random weighted loss); the parity helper doesn't.
- `matmul_*_gradcheck` in `src/lib.rs:4705-4760` use a plain `.sum()` on small fixed inputs. The randomized `assert_matmul_gradients` supersedes them for the matrix forms, but the vector forms rely on them.
- `test_chained_transformations` is covered in §2.

**Planned**

- Property-based random-DAG FD checks that include `matmul` and `permute` (ADDENDUM §10.2, Phase 4). Random shapes put square and non-square operands through every composition.
- `cargo-mutants`, weekly (ADDENDUM §10.4). A mutant that swaps `m`/`n` or drops a `transpose_2d` and still passes would flag the blind tests above directly.

**Assessment:** partial. The CPU matmul tests are designed with Pete's lesson in mind. The parity helper and a few legacy tests are not.

## 5. A reference GEMM

**Pete:** keep "a simple reference GEMM function that's unoptimized, but that I can easily step through and add logging statements to," and compare the optimized versions against it.

**What Volta has**

- **Convolution follows this pattern.** Direct convolution is the naive reference. `test_conv2d_direct_conv_inference` (`src/nn/layers/conv.rs:1955`) compares Direct against im2col on values, and `test_conv2d_igemm_forward_correctness` and `test_conv2d_igemm_gradient_match_direct` (`:2307`, `:2363`) compare iGEMM against it.
- **Matmul does not.** The "reference" for GPU parity is the CPU path, which is `matrixmultiply` on Linux and Accelerate on macOS under default features. So the oracle is an optimized third-party kernel, and which one you get depends on the platform. The hand-calculated tests (§4) are the only ground truth, and they cover four small shapes.

**Assessment:** a gap, and nothing in the roadmap addresses it. A `#[cfg(test)]` triple loop with explicit `(rs, cs)` strides per operand would take about 15 lines. Comparing `matmul_raw` and the GPU kernels against it over random non-square shapes (including sizes above 16 and not multiples of 16) would close §4's parity gap too. Because it takes strides, it would also be the oracle for strided views when Phase 3 lands (§6).

## 6. Leading dimensions (`lda`/`ldb`/`ldc`)

**Pete:** the `ld*` arguments are strides, "the number of values you moved forward in memory when you reached the end of a row … or column." They let you work on sub-tiles, and are otherwise just the row or column count.

**Today:** Volta has no leading-dimension concept, because nothing is ever a sub-view. `matmul_raw` hardcodes `lda = k`, `ldb = n`, `ldc = n`. Batches are materialized with `broadcast_to` and split with `chunks_exact(m*k)` (`src/ops/matmul.rs:273-292`), and the backward pass copies `Bᵀ`/`Aᵀ` instead of passing transposed strides. That is slow but consistent.

### Confirmed defect: `matmul_raw` reads out of bounds from safe code

`RawTensor` is re-exported at the crate root (`src/lib.rs:70`), and `matmul_raw` is a safe `pub fn`. It hands `a.as_ptr()`, `b.as_ptr()`, and the caller's `m, k, n` to `cblas_sgemm` or `matrixmultiply::sgemm` inside `unsafe` blocks, **without checking `a.len() >= m*k` or `b.len() >= k*n`**. Reproduction, run against this checkout with `--no-default-features`, so the `matrixmultiply` path:

```rust
// A and B declared 4×4, but only 2 values supplied each.
let a = vec![1.0f32; 2];
let b = vec![1.0f32; 2];
let c = volta::RawTensor::matmul_raw(&a, &b, 4, 4, 4);
// → no panic; len=16 c=[1.0, 1.0, 1.14e-43, 0.0, …]
```

`1.14e-43` is a denormal read from adjacent heap memory. This is undefined behavior reachable from safe Rust, so the API is unsound. On the macOS path, `m as i32` and the other casts also truncate silently for dims above `i32::MAX`. `Shape`'s 100M `MAX_NUMEL` cap bounds dims that come through `Tensor`, but `matmul_raw` takes raw `usize`s and bypasses that cap. All current internal callers pass consistent lengths, so this is latent rather than live. It is still exactly the failure Pete's section describes: a pointer, a size, and a stride that the callee trusts and nothing checks.

### Why it gets worse in Phase 3/4

The roadmap introduces strided storage and stride-0 broadcast views (`notes/TODO.md:34`; `dev-docs/IMPROVEMENT_PLAN.md:133`; ADDENDUM §7). After that:

- "Every buffer is dense row-major" stops being an invariant, and a transposed or sliced view handed to `matmul_raw` with `lda = k` computes the wrong product silently.
- The upside is that `matrixmultiply::sgemm` already takes a separate row stride and column stride per operand. That is strictly more general than `lda`: it expresses a transpose as swapped strides with no copy, and it accepts sub-tiles. Using it would retire the `transpose_2d` copies in the backward pass and the materialized batch broadcasts.

**Planned:** nothing in ADDENDUM, `notes/TODO.md`, or `docs/backlog.md` mentions matmul's layout contract under views. The Phase 3 storage work is the natural place for it.

## 7. Low-precision GEMM (from the post's comments)

Pete's follow-up comments describe gemmlowp keeping the zero-point offsets "in the O(n²) outer loops, leaving the O(n³) inner loop as pure eight-bit multiplies," with 32-bit accumulators. Volta computes only in f32 (`notes/TODO.md`: "The dtype system is decorative"), and the closest roadmap item is matmul/conv bf16 autocast (`docs/backlog.md:244-271`). When that lands, the hazards are the accumulator width and where the cast and offset happen relative to the inner loop. The reference GEMM from §5, parameterized by accumulator type, is the right oracle for it.

---

## Recommendations

Ordered by correctness impact. Per this repo's scope rule, each would be a line in `notes/TODO.md`, not an unprompted change. I have not filed them.

1. **Make `matmul_raw` sound.** Either make it `pub(crate)` (its only callers are in-crate) or add `assert!(a.len() >= m*k && b.len() >= k*n)` plus checked `i32::try_from` casts. Add a test that the reproduction above panics.
2. **Fence the square-weight hazard in the loaders.** Narrow `transpose_pattern` (for example, to an explicit list of `Linear` keys, or skip keys whose destination layer already uses PyTorch's layout), and document the per-layer layout table from §1b next to `StateDictMapper`. Add a loader round-trip test on a **square, asymmetric** `Linear` that checks values against a PyTorch-computed forward output, not just shapes.
3. **Fix the blind tests.** Make `test_chained_transformations` assert data. Give `test_batched_matmul` asymmetric data. Make `assert_matmul_parity` backprop a random weighted sum (reuse the `assert_matmul_gradients` pattern), and add a parity shape larger than one tile and not a multiple of 16 (for example, `37×19·19×23`).
4. **Add a strided reference GEMM** under `#[cfg(test)]` (§5) and compare CPU and GPU matmul against it over random non-square shapes.
5. **Write down matmul's layout contract before Phase 3 views land.** Either `matmul` requires contiguous inputs and asserts it, or `MatmulSpec` carries per-operand `(rs, cs)` passed straight to `matrixmultiply`, with a bounds check of `(rows-1)·rs + (cols-1)·cs < len`.
6. **Correct the `BackwardParams` comments** in `matmul_backward.wgsl`.

## Bottom line

Volta has already absorbed Pete's main lesson. Its approach is one convention, dims named once in a validated spec, and tests designed so transposed dims can't hide, which is a stronger version of what Pete did by hand. The remaining risk is at the edges: a public raw-pointer GEMM that trusts its caller, a weight loader whose only guard is blind to square matrices, and a parity suite that can't see an incoming-gradient indexing bug. None of these is planned for yet. The most important one to fix before Phase 3 is the layout contract, because strided views remove the invariant that currently makes leading dimensions a non-issue.

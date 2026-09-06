# OPUS_5_EVAL_AUDIT

## Intro

Evaluation of every claim in the **"new notes"** section of `notes/TODO.md`, checked against the
code at commit `75188fa` (branch `improvements`).

**Method.** Claims were checked three ways: (1) reading the cited code, (2) counting with
ripgrep, (3) running the code. For behavioral claims I wrote a throwaway integration test
(`tests/zz_audit_probe.rs`, 11 probes) and ran it; the file has been removed from the repo and
parked at
`/private/tmp/claude-501/-Users-rjlarson-src-volta/ccadcedc-.../scratchpad/audit_probe.rs`.
Raw probe output is quoted inline below. One external fact (Muon's reference algorithm) was
checked against published sources.

**Verdict legend**

| Tag                          | Meaning                                                          |
| ---------------------------- | ---------------------------------------------------------------- |
| ✅ **CONFIRMED**             | Claim is accurate, including cited line numbers where given      |
| 🟡 **CONFIRMED, correction** | Defect is real; a detail (number, line, mechanism, blame) is off |
| 🟠 **PARTLY WRONG**          | Some of the claim holds, some does not                           |
| ❌ **WRONG**                 | Claim does not hold against the code                             |
| 🔵 **JUDGMENT**              | Design opinion, not a factual claim — evaluated on merits        |

**Headline.** The claims are unusually accurate. Of ~120 checkable assertions: **~92 confirmed
outright**, **~17 confirmed with a numeric/attribution correction**, **~6 partly wrong**, and
**3 wrong**. All five "blocking defects" reproduce exactly as described, and four of them
reproduce with the specific mechanism the notes name. The document's self-flagged uncertainty
markers ("double check above", "check again", "need to investigate further ngl") land on
precisely the claims that needed correcting — the calibration is good.

---

## 1. Blocking defects

### B1 — ConvTranspose2d weights get no grad in normal training config ✅ CONFIRMED

Every sub-claim holds.

`src/nn/layers/conv_transpose.rs:417` is `if x.borrow().requires_grad {` — the sole gate on attaching `parents` + `ConvTranspose2dGradFn`.
`conv_transpose.rs:410` seeds the output with `RawTensor::new(output_data, &output_shape, x.borrow().requires_grad)`, so the weight's own `requires_grad` never enters the decision.

Empirical, exactly as predicted:

```
B1 input.requires_grad=false: weight.grad=NONE,             bias.grad=Some(len=3, nonzero=3)
B1 input.requires_grad=true:  weight.grad=Some(len=54, nz=54), bias.grad=Some(len=3, nonzero=3)
```

The bias-trains-anyway detail is right and the mechanism is worth writing down: bias is applied via `output = output.add(&bias_reshaped)` (line 413), a real graph op, _before_ the gate.
When `x` has no grad the `AddGradFn` survives and feeds the bias.
When `x` does have grad, line 424 `output.borrow_mut().grad_fn = Some(Box::new(ConvTranspose2dGradFn{..}))` **overwrites** the add's grad_fn — so the two paths reach the bias by completely different routes.

- "gradfn already checks per-parent req_grad" — ✅, `ConvTranspose2dGradFn::backward` gates each parent independently, so the proposed minimum fix (gate on `{x,weight,bias}.requires_grad`) is sound with no other change.
- "dcgan, vae, super_resolution's generators' fc and both deconv layers have frozen weights" — ✅ structurally: those examples feed plain data tensors into the generator. The `fc` claim is indirect (a frozen `ConvTranspose2d` earlier in the chain doesn't freeze a `Linear` _before_ it — the `Linear` sits upstream of the deconv and still receives grad through the deconv's input path _only if_ the deconv attached a grad_fn, which it does not when the input to the generator is plain noise). Net effect is the same: those generators do not train.

### B2 — LSTMCell severs the graph at the gate slice ✅ CONFIRMED

`src/nn/layers/lstm.rs:122–146` (`slice_gate`) copies floats out of `gates.borrow().data` and returns `RawTensor::new(gate_data, &[batch, hidden], gates.borrow().requires_grad)` — the flag is propagated but no `grad_fn` and no `parents`. Backward stops there.

```
B2 lstm param[0..3] grad = NONE (all four: weight_ih, weight_hh, bias_ih, bias_hh)
B2 lstm input grad     = NONE
```

"Known and enshrined" ✅ — `tests/test_lstm.rs:257–296`, `test_lstm_gradient_flow`, matches on `grad` and takes the `None` arm with `println!("Known limitation confirmed: slice_gate breaks autograd graph.")`. The test cannot fail either way. The comment above it (line 259) explicitly says "If this test starts passing without the catch, the autograd issue has been fixed" — so it is documentation, not a regression fence. The characterization "worse than no test, launders defect" is fair: `cargo test` reports it green.

### B3 — `PositionalEncoding::forward` severs the graph in both modes ✅ CONFIRMED

`src/nn/transformer/positional_encoding.rs`: both arms end at
`RawTensor::new(result_data, &[batch, seq, embed], false)` (lines 155 and 183) after
`result_data.push(x_val + enc_val)` over raw `to_vec()` copies. Hardcoded `false`, not even the
input's flag.

```
B3 learned:     out.requires_grad=false grad_fn=false parents=0
B3 sinusoidal:  out.requires_grad=false grad_fn=false parents=0
```

- "learned mode: pos embedding table can never receive a gradient" ✅ — `emb.forward(&positions)`
  builds a real graph node, then its data is copied out and discarded.
- "char_trans_model hand-rolls pos-enc instead of using component" ✅ —
  `examples/char_transformer_model.rs:153–160`, `add_positional_encoding`, computes sin/cos
  inline. The example's own header comment (line 16) rationalizes it: "The positional encoding is
  added outside the graph (no requires_grad), which is correct for sinusoidal PE."
- "0 for PosEnc grads" ✅ — the module's 11 tests check shapes, parameter counts, and
  "encoding modifies the input"; none touches `backward`.

### B4 — different-rank batched matmul backward is wrong and silent 🟡 CONFIRMED, numbers differ

The mechanism is exactly as described.

`src/ops/matmul.rs:510` (grad wrt A): `let rank = x.shape.len();` then
`let n = *y.shape.get(rank.wrapping_sub(1)).unwrap_or(&1);` — **x's rank indexing y's shape**.
Mirror at `matmul.rs:623` (grad wrt B): `let rank = y.shape.len();` then
`let m = *x.shape.get(rank.wrapping_sub(2))` and `let k = *x.shape.get(rank.wrapping_sub(1))` —
**y's rank indexing x's shape**. For `x`=rank 3, `y`=rank 4 the second one reads
`x.shape.get(3)` → `None` → `unwrap_or(1)`, so `k` silently becomes 1.

Forward acceptance ✅: `[2,3,4] @ [2,1,4,6]` returns shape `[2,2,3,6]` with no error.

Silence mechanism ✅ and the cited line is exact: `src/ops/binary.rs:519–523`,
`if let Some(&src_val) = data.get(from_idx) && let Some(slot) = result.get_mut(i) { *slot = src_val; }`
over a `vec![0.0; to_size]` — an out-of-range source index leaves the destination at 0.0 instead
of failing.

**Correction — the error figures.** Finite-difference check on `[2,3,4] @ [2,1,4,6]`:

|                                                    | max rel err | mean rel err | pass      |
| -------------------------------------------------- | ----------- | ------------ | --------- |
| grad wrt A, unequal rank                           | 0.380       | 0.364        | **false** |
| grad wrt B, unequal rank                           | 1.000       | 0.846        | **false** |
| grad wrt A, equal rank `[2,3,4]@[1,4,6]` (control) | 0.0031      | 0.0009       | true      |

So 36% mean / 38% max, not 29%; and the control passes at 3.1e-3, not 4.8e-4. Both figures are
data-dependent, so this is a reproduction-detail discrepancy, not a substantive one. **The notes
understate the bug**: grad wrt B is far worse than grad wrt A — 18 of 48 weight-gradient entries
come back _identically zero_ against true values of ~6.6–7.8. The `.get(rank-1)` on a
shorter shape collapsing to `unwrap_or(1)` is why.

Both proposed fixes are correct. Rejecting unequal rank at forward time is the cheaper one and
loses nothing — NumPy/PyTorch semantics for `[2,3,4] @ [2,1,4,6]` would broadcast the batch dims,
which this forward pass does _not_ implement correctly either way.

### B5 — CPU-only build does not compile ✅ CONFIRMED, exactly

`cargo build --no-default-features` → **exactly 5 errors**, in exactly the places named:

```
src/ops/gpu_ops.rs:7:12   E0432 unresolved import `crate::gpu`
src/nn/layers/conv.rs:164 E0599 no associated function `gpu_col2im`
src/nn/layers/conv.rs:336 E0308 `if` and `else` have incompatible types (Rc<RefCell<RawTensor>> vs Option<_>)
src/nn/layers/conv.rs:413 E0308 same, weight-gradient arm
src/nn/optim/sgd.rs:87    E0599 no method `step_cpu_param` on `&mut SGD`
```

- "ops/gpu_ops.rs use crate::gpu::GpuKernels unconditionally" ✅ — `use crate::gpu::GpuKernels;`
  at line 7 is ungated while the _neighbouring_ imports at lines 9–12 are `#[cfg(feature = "gpu")]`.
- "conv.rs unguarded gpu_col2im, cfg-gated if/else arms w incomp types" ✅ — `conv.rs:258–337`:
  the `#[cfg(feature="gpu")] if use_gpu {...} else {...}` block yields `Option<Tensor>`, the
  `#[cfg(not(feature="gpu"))] { ... }` block yields a bare `Tensor`, and the shared `else { None }`
  can only match one of them.
- "sgd calling step_cpu_param, inverted-cfg'd out at sgd / GPU feature flag on CPU path: Adam did
  right, not SGD" ✅ and precise — `sgd.rs:189` puts `#[cfg(feature = "gpu")]` on
  `fn step_cpu_param`, so the _CPU_ path vanishes in the _CPU-only_ build. `adam.rs:139` leaves
  the same function ungated. Textbook inversion.
- "cargo.toml says gpu is optional, claude.md tells reader to use --no-default-features" ✅ —
  `Cargo.toml` has `wgpu = { ..., optional = true }` and `default = ["gpu", "accelerate"]`.
- "hasn't been run tbh" ✅ — see §CI below; no CI step passes `--no-default-features`.

---

## 2. High-severity

### H1 — MaxPool2d pads with 0.0 instead of −inf ✅ CONFIRMED, and the suggested check reproduces

`src/nn/layers/maxpool.rs:64` calls the generic `x.pad(&[(0,0),(0,0),(ph,ph),(pw,pw)])`;
`src/ops/movement.rs:898` allocates `vec![0.0; new_size]` — the fill value is an implementation
detail of `pad`, not a parameter.

The exact verification the notes propose:

```
H1 maxpool(2,2,pad=1) over all(-5.0) input:
   shape=[1,1,3,3]  out=[0.0, 0.0, 0.0, 0.0, -5.0, 0.0, 0.0, 0.0, 0.0]
```

Eight of nine outputs are 0.0 for an input whose every real value is −5.0. PyTorch returns −5.0
everywhere. "Wrong when pooling precedes signed activations" ✅ — and also wrong in the backward
pass, since the argmax lands on a padding cell that has no input to route gradient to.

`test_maxpool2d_gradcheck_with_padding` (maxpool.rs:426) passes because it uses `randn` input,
where positives dominate the padded windows.

### H2 — `bce_with_logits_loss` overflows, doc comment lies 🟡 CONFIRMED, threshold off by one

`src/tensor.rs:452–464`: `logits.exp()` → `one.add(&exp_logits)` → `.log()`. Literally
`log(1+exp(x))`. Header at line 444: `/// Binary Cross Entropy with logits (numerically stable)`.
✅ on both.

```
H2 bce_with_logits(x=80)  loss=0    grad=Some(0.0)
H2 bce_with_logits(x=88)  loss=0    grad=Some(-5.96e-8)
H2 bce_with_logits(x=89)  loss=inf  grad=Some(NaN)
H2 bce_with_logits(x=100) loss=inf  grad=Some(NaN)
```

**Two corrections.** (a) The threshold is x ≥ 89, not 88 — `exp(88.72)` is the f32 limit.
(b) It is not `|x| ≥ 88`: negative logits are fine (`exp(-100) → 0`, `log(1+0) = 0`, correct
answer). The failure is one-sided, on large positive logits — which is the DCGAN discriminator's
saturating regime, so the practical exposure is real.

- "bce_loss has same issue w log(p)/log(1-p)" ✅ — `tensor.rs:428–441` takes `pred.log()` and
  `(1-pred).log()` directly. `bce_loss(p=[0,1], t=[0,1])` → `[NaN]`.
- Proposed fix `max(x,0)+log1p(exp(-|x|))` ✅ correct — that is the standard stable softplus and
  matches PyTorch's `BCEWithLogitsLoss` formulation (`max(x,0) - x*z + log(1+exp(-|x|))`).

### H3 — unsafe code has a violated invariant ✅ CONFIRMED, all three parts

`src/storage.rs:245–249`, the SAFETY comment on `try_as_f32_slice`:

> ```
> // 1. The cache is Some(Vec<u8>) after the block above
> // 2. We never clear the cache once populated
> // 3. The Vec inside is stable (won't move/reallocate)
> ```

Point 2 is falsified at `src/storage.rs:627`:
`pub fn invalidate_cpu_cache(&mut self) { if let Self::Gpu { cpu_cache, .. } = self { cpu_cache.borrow_mut().take(); } }`
✅ — and the line number in the note is exact.

"&mut borrow discipline saves this, not stated invariant" ✅ — `invalidate_cpu_cache` takes
`&mut self`, so it cannot coexist with a live `&[f32]` handed out by `&self`. The code is sound;
the _documented reason_ it is sound is false. That's the worse failure mode, since a future
refactor reads the comment, not the signature.

"as_{dtype}_slice on GPU (302-385) reinterp f32 as f64/16" ✅ and the range is right —
`as_f64_slice` (302), `as_f16_slice` (333), `as_bf16_slice` (364) all do, for the `Gpu` arm,
`buffer.to_vec()` (a `Vec<f32>`) → `cast_slice` to bytes → `cast_slice` to `&[f64]`/`&[f16]`.
Pure byte reinterpretation of f32 data.

"dead wrong but unreachable" ✅ — every `Storage::Gpu` construction site (`storage.rs:150`,
`storage.rs:184`, `gpu_ops.rs:111`) hardcodes `dtype: DType::F32`, and `Clone` (line 55) preserves
it, so the `*dtype == DType::F64` guards can never fire.

### H4 — library panics on ordinary data ✅ CONFIRMED, all three, with exact lines

```
H4 max() with NaN:                 PANIC  src/ops/reduce.rs:219  Option::unwrap() on None
H4 xavier_uniform(&[2,3,4,5]):     PANIC  src/tensor.rs:154      ShapeDataMismatch { shape: [2,3,4,5], elements: 120, len: 6 }
H4 stride_op(&[0]):                PANIC  src/ops/movement.rs:1102  attempt to divide by zero
```

- Max/NaN: `reduce.rs:199/210/219` all use `.max_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap())`.
- xavier: `tensor.rs:254–262` takes `fan_in = shape[0]`, `fan_out = shape[1]`, generates
  `fan_in * fan_out` samples, then constructs with the full `shape`. For `[2,3,4,5]` that's 6
  values for a 120-element tensor. The claim "gens fan_in*fan_out elems regardless of shape prod"
  is exactly right.
- stride: `movement.rs:1102` is `.map(|(d, s)| d.div_ceil(*s))` — the cited line is exact.
- "ships VoltaError enum and deny-heavy clippy, panics on ordinary inputs" ✅ — `src/error.rs`
  defines a 9-variant `VoltaError` via `thiserror`; `Cargo.toml` denies `indexing_slicing`,
  `missing_panics_doc`, `missing_errors_doc`. All three of these panics are reachable from public
  API with well-formed arguments.

### H5 — GPU broadcast backward silently truncates rank > 4 🟡 CONFIRMED, wrong function cited

The defect is real and the consequence is stated correctly, but the citation points at the wrong
helper.

`src/gpu/kernels.rs:2954` **is** `fn pad_to_u32_4` with `values.iter().take(4)` ✅ — the line
number is exact. But `pad_to_u32_4` is used only by the _movement_ ops (`kernels.rs:1704, 1735,
1788, 1815, 1839, 1865, 1888`). The broadcast-backward path does its own padding inline at
`kernels.rs:769–800`, writing into `[1u32; 4]` arrays via `.get_mut(idx)` with
`idx = 4usize.saturating_sub(shape.len()) + i`.

That inline version is **worse** than `take(4)`: for rank 5, `saturating_sub` gives offset 0, so
dims 0–3 land left-aligned in the vec4 (destroying the NumPy right-alignment the code comment
claims) and dim 4 is dropped by the failed `get_mut(4)`.

"passes true rank to shaders that index `vec4<u32>` dynamically" ✅ exactly —
`kernels.rs:802–810` sets `out_rank: out_shape.len() as u32` etc. unmodified, and
`src/gpu/shaders/binary_backward.wgsl:25–58` runs `strides[i]`, `coords[i] = remaining % shape[i]`,
`padded[i + offset] = shape[i]` with `i` bounded by that rank. "OOB vector indexing in WGSL" ✅.
"silent wrong grads for rank-5+ tensors instead of cpu fallback" ✅ — there is no rank check
anywhere on the path.

### H6 — TransformerBlock is decorative ✅ CONFIRMED, every bullet

`src/nn/transformer/block.rs`:

- "attn SimpleAttention placeholder" ✅ — line 66 `attention: Rc::new(SimpleAttention::new(...))`;
  line 216 `impl Module for SimpleAttention { fn forward(&self, x) -> Tensor { x.clone() } }`
  with `fn parameters() -> Vec<Tensor> { vec![] }`. It is the identity function.
- "causal param ignored" ✅ — line 83, `pub fn forward(&self, x: &Tensor, _causal: bool)`.
- "load_state_dict skips attention state silent" ✅ — lines 187–190, a comment explaining that
  `Rc<dyn Module>` can't be `&mut`-borrowed, and nothing else. `state_dict` _does_ emit
  `attention.*` keys (line 143), so a save/load round-trip is asymmetric by construction.
- "char_transformer_model bypasses" ✅ — the example imports `FeedForward, MultiHeadAttention`
  and assembles its own block (`examples/char_transformer_model.rs:28, 64–77`).
- "doc'ed as todo admitting MultiHeadAttention makes block non-standard" 🟡 — such a TODO exists,
  but in the _example_, not the block: `char_transformer_model.rs:11–14`, "TODO: handle these
  concerns / MultiHeadAttention::forward internally does x.add(&output) — so calling
  norm1.forward(mha.forward(x)) is Pre-LN applied to the post-residual output, which is slightly
  non-standard (Post-LN)."
- "has dead dropout field" ✅ — `block.rs:40–41`, `#[allow(dead_code)] dropout: f32`, plus three
  more `#[allow(dead_code)]` config fields.
- "real MultiHeadAttention exists and is correct in structure / embeds residual + opt LayerNorm
  (whole block, not layer)" ✅ — `multi_head_attention.rs:105–152`: applies `self.norm` if
  present at the top and returns `x.add(&output)` at the bottom. It is a block wearing a layer's
  name, which is exactly why `TransformerBlock` can't just drop it in.

### H7 — load_state_dict demotes GPU to CPU ✅ CONFIRMED

Every layer's `load_state_dict` writes `t.data = Storage::cpu(w.data.clone())` and then
`t.shape.clone_from(&w.shape)` as two independent field assignments:

`linear.rs:~130`, `conv_transpose.rs:456–470`, `conv.rs`, `batchnorm.rs`. No device read, no
`to_device`, no `len() == shape.product()` check.

- "no device preservation" ✅
- "element count validation" ✅ absent — **with one exception the notes miss**:
  `embedding.rs:145–150` does `assert_eq!(weight_data.shape, vec![self.vocab_size, self.dim], "Weight shape mismatch")`.
  Embedding is the only layer that validates.
- "load checkpoint into GPU-resident model, params move to CPU mid-training w no err" ✅ — the
  tensor's `device` field is left saying `GPU` while its `Storage` is now `Cpu`, so subsequent
  ops take whichever path the stale `device` flag selects.

### H8 — Muon is a toy and panics on GPU params 🟠 PARTLY WRONG (one bullet is unfair)

`src/nn/optim/muon.rs`:

- "uses un-tuned quadratic Newton-Schulz" ✅ — `newton_schulz` (line 46) implements
  `X ← 0.5·X·(3I − XᵀX)`, the classic quadratic iteration. The reference Muon uses a **quintic**
  `p(x) = 3.4445x − 4.7750x³ + 2.0315x⁵`, coefficients tuned to maximize convergence slope at
  zero so 5 steps suffice. Volta's version converges much more slowly per step and is a different
  algorithm.
- "frob-norm 'approx spectral' scaling" ❌ **this one is unfair.** Normalizing by
  `‖X‖_F` before the iteration (`muon.rs:50–56`) is what the reference implementation does too —
  it's the standard trick, since `‖X‖_F ≥ ‖X‖_2` guarantees the spectral radius lands in the
  convergence basin. This is the one part of the implementation that follows the paper.
- "magic 1/10 w stability comment" ✅ — `muon.rs:126`,
  `*w -= self.lr * u * 0.1; // Scaling factor often needed for Muon stability vs Adam`.
  Reference Muon instead scales by `sqrt(max(1, rows/cols))`, which this omits entirely — a
  better complaint than the magic constant.
- "mutates via storage::iter_mut, panics for GPU backed params" ✅ — `muon.rs:125`
  `for (w, u) in p.data.iter_mut().zip(update.iter())`, and `storage.rs:643–647`:
  `pub fn iter_mut(&mut self) -> IterMut<'_, f32> { self.as_f32_slice_mut().expect("Mutable iteration requires F32 CPU storage").iter_mut() }`
  where `as_f32_slice_mut` returns `None` for the `Gpu` arm (`storage.rs:279`). Guaranteed panic.

Not in the notes but worth adding: `Muon` is not re-exported from the crate root — `src/lib.rs:47`
exports `Adam` and `SGD` only. It is reachable as `volta::nn::optim::muon::Muon`.

---

## 3. Medium / nice-to-have

### "every op deep copies" ✅ CONFIRMED

- "storage::clone is a `Vec<u8>` copy" ✅ — `storage.rs:48–61`, the `Cpu` arm is `data.clone()`
  on a `Vec<u8>`. (The `Gpu` arm is an `Arc` bump, so the problem is CPU-path-specific, as the
  notes say later.)
- "every op in un/bin/mm/move/tern leads w `s.data.clone()`" ✅ — the borrow-clone-drop preamble
  is verbatim in `binary.rs:608–615`, `matmul.rs:137–154`, `movement.rs:1085–1091`,
  `unary.rs`, `ternary.rs`.
- "broadcasts are materialized (broadcast_to, no stride-0 view)" ✅ — `binary.rs:480–526`
  allocates `vec![0.0; to_size]` and fills every element.
- "elementwise op allocates ~3x output size" ✅ in the broadcasting case: two `broadcast_to`
  buffers plus the result.
- "WhereGradFn keeps full broadcast-sized cond for graph lifetime" ✅ — `ternary.rs:78`,
  `condition: Vec<f32>, // broadcasted condition (matches out_grad shape)`, populated at line 274
  from `cond_bc`.

### "im2col copies whole tensor per kernel elem" ✅ CONFIRMED, exact

`src/nn/layers/conv.rs:1213`: `let cpu_data = data.to_vec();` sits inside
`for b … for oh … for ow … for c … for kh_idx … for kw_idx {` — the innermost loop. One full
input-tensor copy per output element written. "O(BHWCK²) full tensor copies" ✅.

### "conv input grad is quadratic" ✅ CONFIRMED, both cited ranges

- `conv.rs:427–518` (`DirectConvGradFn::compute_input_gradient_cpu`): for each
  `(b, ic, ih, iw)` it loops `for oc … for oh in 0..h_out … for ow in 0..w_out` and tests
  membership in the receptive field. ✅
- `conv.rs:822–912` (`IGEMMGradFn::igemm_backward_input`): identical structure — "Accumulate from
  all output positions that use this input" followed by the same triple loop and an
  `h_offset >= 0 && h_offset < kernel_h` test. ✅
- "at 64x64, K=3, ~450x wasted inner loop work" ✅ — `h_out·w_out = 62·62 = 3844` versus
  `K² = 9`, a factor of ~427. The estimate is right.
- "tiled iGEMM (1679-1782) has no tile reuse" ✅ — `igemm_tiled` at `conv.rs:1719` tiles the
  _loop bounds_ (`step_by(config.output_channels)` etc.) but the innermost body reads
  `x_data.get(in_idx)` / `w_data.get(w_idx)` straight from the full arrays with a fresh bounds
  check each time. No shared tile buffer is ever allocated. It is direct convolution with extra
  index arithmetic, and "'good balance of speed and memory' is label, not property" is accurate.

### "gpu reductions are single threaded" ✅ CONFIRMED, all three

- `src/gpu/shaders/reduce.wgsl` — all three entry points are `@compute @workgroup_size(1)` with
  `for (var i = 0u; i < params.input_size; i++)`. A one-thread sequential loop on a GPU.
- "gpu_max_reduce downloads to CPU to find argmax (gpu_ops.rs:131)" ✅, exact line —
  `let cpu_data = data.to_vec();` then a linear `find`. (Also fragile: it matches on
  `(v - max_val).abs() < 1e-6`, so ties and near-ties pick the first.)
- "every GPU op builds fresh uniform buffer, bind group, encoder, submit" ✅ — the pattern
  `create_buffer_init` → `create_bind_group` → `create_command_encoder` → `queue().submit()`
  repeats per call in `binary_backward_broadcast` (kernels.rs:810–872), `movement_op`
  (kernels.rs:2965–3020), and every other wrapper.

### "2-pass 'race free' broadcast backward's pass 2 is O(targets × out_size)" ✅ CONFIRMED

`src/gpu/shaders/binary_backward_safe.wgsl:171–210` — each pass-2 thread owns one gradient
element and scans **the entire output** looking for entries whose stored target index equals its
own:

```wgsl
for (var i: u32 = 0u; i < out_size; i = i + 1u) {
    let target_a = u32(temp_buffer[i * 3u]);
    if (target_a == thread_idx) { sum_a = sum_a + temp_buffer[i * 3u + 2u]; }
}
```

Brute-force scan per target element, exactly as claimed. Race-free, but it converts an
atomic-add into a quadratic gather.

### "check_grads doesn't 0 pre-existing grads" ✅ CONFIRMED, exact lines

`src/tensor.rs:899–902`:

```rust
let loss = loss_fn(tensor);
loss.backward();
let analytical_grad = tensor.grad().expect("Tensor must have gradient");
```

No `tensor.borrow_mut().grad = None` first. Reproduced:

```
M checkgrad: grad after first backward = Some([2.0, 4.0, 6.0])
M checkgrad after prior backward: max_err=0.500 mean_err=0.500 pass=false
M checkgrad fresh:                max_err=7.3e-5 mean_err=6.4e-5 pass=true
```

A correct gradient reported as a 50% failure. **Correction:** the note says the probe measured
1.02; I measured 0.50. Both are just `(k·g − g)/(k·g)` for whatever k accumulations happened —
0.50 for one prior backward, ~0.67 for two, and >1.0 only if the prior grads have opposite sign.
The defect is identical either way.

### "committed flakey test on conv.rs" ✅ CONFIRMED, with a sharpening

- "systematically loosened tolerances (ep=5e-2, tol in [2,5]e-2)" ✅ exactly — 14 call sites in
  `conv.rs` use `check_gradients(&x, |t| conv.forward(t).sum(), 5e-2, 2e-2)`; `conv.rs:2342` uses
  `5e-2, 5e-2`. The rationale comment at `conv.rs:2018–2019` reads "relaxed for conv's numerical
  complexity."
- "GPU forward-consistency tests use all-ones ins, weights" ✅ — e.g.
  `test_conv2d_gpu_forward_match_cpu` (conv.rs:2931) sets weights to `vec![1.0; weight_size]`,
  bias to zeros, and input to `vec![1.0; 128]`. With constant weights _and_ constant input the
  result is invariant under any permutation of the receptive-field index map, which is precisely
  the class of bug these tests exist to catch. `vec![1.0; …]` appears 46 times in `conv.rs`.
- "exception: col2im tests use varied data" 🟡 **half right.**
  `test_conv2d_gpu_col2im_cpu_consistency` (conv.rs:3715) uses varied _input_
  (`(0..256).map(|i| i as f32)`) but still sets weights to `vec![1.0; weight_size]`. Varied on one
  axis only.

### "Sequential::load_state_dict prefix match is greedy" ✅ CONFIRMED

`src/nn/layers/sequential.rs:66`: `if key.starts_with(&prefix)` with `prefix = format!("{}.", i)`.
For an 11-layer Sequential:

```
M seq keys: ["0.bias","0.weight","1.bias","1.weight","10.bias","10.weight","2.bias", …]
```

Layer 1 builds `prefix = "1."`, which matches `"10.weight"`, and
`let sub_key = &key[prefix.len()..]` yields `"0.weight"`. Layer 1's sub-state therefore contains
`{bias, weight, 0.bias, 0.weight}` — exactly the described leak.

Consequence, sharpened: today it is latent, because `Linear::load_state_dict` only reads the
literal keys `"weight"`/`"bias"` and ignores the rest. It becomes a real corruption the moment
layer 1 is itself a `Sequential` or any nested module, since `"0.weight"` is then a meaningful
key belonging to a different layer. Also note this same loop uses raw byte slicing
`&key[prefix.len()..]` on a `String` — UB-free but panics on non-ASCII layer names.

### "RNG repro holes" ✅ CONFIRMED, both

- "dataloader shuffles with rand instead of seeded global RNG" ✅ — `tensor.rs:1239` and
  `tensor.rs:1284` (in `reset`, so it recurs every epoch) both call
  `indices.shuffle(&mut rand::rng())`, bypassing `GLOBAL_RNG` / `with_rng` entirely.
- "with_rng silently falls back to unseeded thread RNG under reentrant borrow" ✅ —
  `tensor.rs:62–75`: `if let Ok(mut borrow) = rng_cell.try_borrow_mut() && let Some(rng) = ...
{ return f(rng); }` then unconditionally `f(&mut rand::rng())`. The `Err` from
  `try_borrow_mut` is discarded without a warning. Two failure modes share one silent path:
  never-seeded, and re-entrant.
- "seeded runs not repro over epochs" ✅ follows from the `reset` call site.

### "io/config is dead file of non-compiling pseudocode" ✅ CONFIRMED, exactly

`src/io/config.rs`, in full:

```rust
#[derive(Deserialize)]
pub struct ModelConfig { pub architecture: String, pub num_layers: usize, pub hidden_size: usize }

pub fn load_config(path: &str) -> Result<ModelConfig>;
pub fn build_from_config(config: &ModelConfig) -> Result<Box<dyn Module>>;
```

Bodiless fn declarations ✅. `#[derive(Deserialize)]` with `serde` not in `Cargo.toml` ✅.
`rg "mod config" src/` returns nothing — the file is never declared, so `rustc` never sees it ✅.

### "dtype system is decorative" ✅ CONFIRMED

- "dtype supports 8 types" ✅ exactly — `src/dtype.rs`: F16, BF16, F32, F64, I32, I64, U8, Bool.
- "every compute path is f32" ✅ — `Storage::iter`, `iter_mut`, `Deref<Target=[f32]>`,
  `Index<usize>` all funnel through `as_f32_slice`, which errors on any other dtype.
- "to_device downcasts to f32" ✅ — `storage.rs:~580`, the `Device::GPU` arm is
  `Self::gpu(self.to_f32_vec())` with the comment "Convert to f32 for GPU (GPU ops are f32 only
  for now)".

### "embedding::forward takes `&[usize]`" ✅ CONFIRMED

`embedding.rs:76` `pub fn forward(&self, indices: &[usize]) -> Tensor`, and `embedding.rs:120–127`
`impl Module for Embedding { fn forward(&self, _x: &Tensor) -> Tensor { panic!("Embedding layer should be called with Embedding::forward(&indices), not Module::forward") } }`.

"a seq holding one detonates at runtime" ✅ reproduced:

```
M embedding in Sequential: PANIC at src/nn/layers/embedding.rs:124
```

Compiles cleanly, panics on first forward.

### "mixed device bin ops silent make cpu out" ✅ CONFIRMED

`binary.rs:628`: `if Self::common_gpu_device(&[self_t, other]).is_some() && ...` — returns `None`
for mixed devices, so control falls straight through to the CPU path with no warning and no
device annotation on the result. Same shape in `matmul.rs:137–154`, where `_dev_a` and `_dev_b`
are destructured and then discarded — the underscore prefixes make the discard deliberate.

### "f64 on gpu reinterprets bytes (see H3)" ✅ — see H3.

---

## 4. WGPU backend assessment ✅ CONFIRMED (accurate on both sides of the ledger)

Positive claims:

- "genuine context w ~70 pipelines" ✅ — `src/gpu/context.rs` declares **68**
  `wgpu::ComputePipeline` fields across 20 WGSL shader files.
- "pooled buffers, staged readbacks" ✅ — `gpu/pool.rs` (power-of-2 bucketed `BufferPool`),
  `gpu/staging_pool.rs`.
- "properly tiled 16x16 shared-mem matmul" ✅ — `shaders/matmul.wgsl:22–29`:
  `const TILE_SIZE: u32 = 16u; var<workgroup> tile_a: array<f32, 256>; ... @compute @workgroup_size(16,16)`
  with `workgroupBarrier()` on both sides of the accumulation. This is textbook and correct.
- "working conv fw/bw kernels w CPU parity tests" ✅ — `conv_backward_input.wgsl`,
  `conv_backward_weight.wgsl`, `igemm_backward_*.wgsl`, plus parity tests in `conv.rs`.
- "GPU optimizer step whose adam matches (valid) CPU one" ✅ — `shaders/optimizer_step.wgsl`,
  `adam.rs:98`, `tests/test_optimizers.rs`.

Negative claims:

- "reductions single threaded theater" ✅ — see above.
- "broadcast forward falls back to CPU" ✅ — `binary.rs:628` requires a common GPU device _and_ a
  matching same-shape kernel; broadcasting shapes take the CPU path.
- "broadcast backward caps at rank 4 (H5)" ✅.
- "arrayLength-sized shaders overrun into 2^n bucket padding" ✅ mechanism confirmed:
  `pool.rs:87–112` rounds every allocation up to the next power of two
  (`usize::BITS - (size-1).leading_zeros()`, then `1 << bucket`), while `elementwise.wgsl`,
  `unary.wgsl`, `unary_backward.wgsl`, and `binary_backward.wgsl` bound their loops with
  `arrayLength(&…)`, which reports the _physical_ buffer length. A 1000-element tensor in a
  1024-element bucket exposes 24 elements of stale data.
- "harmless only bc of unenforced invariant holding by accident: buffers always rewritten
  [0..len) before logical reads" ✅ — nothing enforces this; it is a property of current call
  ordering.

---

## 5. Tests section ✅ CONFIRMED, with a count correction

- "this is not no tests" ✅.
- "641 passing tests" 🟡 — measured **671 passing, 13 ignored, 0 failing** across the lib target
  and 17 integration binaries (`cargo test`, default features, on this machine). The claim's
  magnitude is right; the exact figure is stale or was taken with a different feature set.
- "17 integration test files" ✅ **exactly** — `ls tests/*.rs | wc -l` = 17.
- "finite-diff grad checks across ops, cross-algo equiv, GPU/CPU parity w varied data" ✅ —
  `tests/parity_{binary,matmul,movement,reduce,unary}.rs` do use randomized ranges
  (e.g. `(-2.0, 2.0)` in `parity_binary.rs`), so the "varied data" credit is deserved _for the
  parity suite_ even though the in-file conv GPU tests are all-ones.
- "0 for multi-consumer accumulation (flagged soft spot, deserves regression test)" 🟡 mostly —
  there is no test named for diamond/multi-consumer accumulation, but `src/lib.rs:1690–1710`
  (`test_embedding_gradient_accumulation`) and `src/lib.rs:4080` do exercise accumulation
  incidentally ("a should accumulate gradient from all 1000 positions"). No test targets the
  diamond topology directly.
- "0 tests diff w.r.t weights thru ConvTranspose (only input)" ✅ **exactly** — all six
  `check_gradients` calls in `conv_transpose.rs` (lines 738, 759, 775, 791, 807, 823) pass `&x`
  and differentiate `|t| conv.forward(t).sum()`. There are two weight-gradient assertions
  (`conv_transpose.rs:718`, `:843`) but **both set `x.borrow_mut().requires_grad = true` first**
  (lines 707 and 840) — i.e. both run in the one configuration where B1 does not fire. This is
  the sharpest finding in the whole document: the tests are aimed at the bug and miss by one
  boolean.
- "test asserting LSTM bug as expected" ✅ — B2 above.
- "0 for PosEnc grads" ✅ — B3 above.
- "0 for unequal rank matmul" ✅ — no test anywhere constructs a rank-mismatched matmul.
- "examples bypassing broken components so tests stay green" ✅ — `char_transformer_model.rs`
  hand-rolls both PosEnc and the block.

---

## 6. "Finding I'd lead with" 🔵 JUDGMENT — supported

"engine is correct, layers aren't, and silently" — supported by the evidence pattern: `autograd.rs`
topological backward, `binary.rs`/`unary.rs`/`reduce.rs` grad functions, and the `matmul` 2D/1D
cases all pass finite-difference checks; the three grad-dead components (B1–B3) are all _layers_
that reach for `RawTensor::new` instead of a graph op, and the one broken _op_ (B4) is the branch
of matmul that no layer exercises.

"diamond/accumulation code and adam flagged soft spots couldn't be broken" — I did not attempt to
break these either; `autograd.rs:~120–190` does do proper consumer counting before visiting a
node, and Adam's CPU/GPU parity test passes.

"test discipline that fixes whole class of issues" ✅ — the proposed
`assert!(weight.grad.is_some_and(|g| g.iter().any(|v| v != 0.0)))` with a **non-grad input**
would have caught B1, B2, and B3. Verified: my B1/B2/B3 probes are that assertion, and all three
fail. Note the assertion as written needs `!= 0.0` to be `.abs() > 0.0` or it will also accept
`-0.0`, and `is_some_and` consumes the `Option`, so it needs `p.grad().is_some_and(...)` on a
cloned `Storage`.

"single test, added to all 19 layers" — layer count: `src/nn/layers/` has 17 layer modules plus
4 transformer modules; 19 is the right order of magnitude.

---

## 7. Rust issues

### Framing ✅/🔵

"modern-edition codebase wrapped behind pre-1.0 style interior mut design" ✅ — `edition = "2024"`,
`rust-version = "1.89.0"`, let-chains used throughout, alongside `pub type Tensor = Rc<RefCell<RawTensor>>`
with all seven fields `pub`.

"3 competing error strats, winner being silence" ✅ — see §error strategy.
"allow-riddled lint regime documenting hopes not practice" ✅ — see §abstraction quality.
"CI w step that doesn't run config it's named after" ✅ — see §CI.

### Type model

- "Rc RefCell around struct of data, shape, grad, req_grad, grad_fn, parents, dev / all relied on
  invariants are convention" ✅ — `tensor.rs:38–45`, all seven fields `pub`, no constructor
  enforcement beyond `try_new`'s length check, which `new` bypasses by unwrapping.
- "`x.borrow_mut().requires_grad = true;` documented way to make tensor trainable" ✅ — **63
  occurrences** repo-wide (60 in `src/`, 3 in `tests/`+`examples/`), including inside library
  constructors: `Embedding::new` (embedding.rs:57), `ConvTranspose2d::new` (conv_transpose.rs:41,
  45). The library uses the escape hatch on itself.
- "im2col swaps storage without touching device / pattern appears in 10 tests" 🟡 — the pattern
  `X.borrow_mut().data = Storage::cpu(...)` / `Storage::gpu(...)` appears **15+ times** in
  `conv.rs`'s test module (lines 2597, 2602, 2745, 2748, 2808, 2811, 2942, 2944, 2953, 2954,
  3040, 3047, 3120, 3121, 3129…). Undercount, same point.
- "load_state_dict writes t.data and t.shape.clone_from as independent assigns, no elem count
  check / linear, conv, batchnorm" ✅ — see H7 (and note Embedding is the exception).
- "B1 and H7 are direct R/W of pub fields from wrong place" ✅ — B1 reads `x.borrow().requires_grad`
  from inside a layer; H7 writes `t.data`/`t.shape` from inside a layer. Both would be
  unexpressible behind a private-field newtype.

### Accidental RefCell discipline ✅ CONFIRMED

- "each op does borrow-clone-drop dance before compute" ✅ — `binary.rs:608–615` is the canonical
  instance; the scoped block exists solely to drop the borrow.
- "a.add(&a) and x.elem_mul(&x) work today" ✅ — because of that dance; both borrows are released
  before compute.
- "BinaryGradFn::backward holds x/y_ref.borrow across entire grad comp" ✅ — `binary.rs:88–89`
  takes `let x_val = x_ref.borrow(); let y_val = y_ref.borrow();` and they stay live for the
  whole ~250-line function.
- "no structural prevention of re-entrant op … from panicking w BorrowMutError" ✅ — nothing
  guards this; calling a forward op inside a backward (a legitimate VJP pattern) on one of the
  parents will panic.

### "3 incompatible flavors of interior mut" ✅ CONFIRMED, exactly three

1. `RefCell<RawTensor>` — `tensor.rs:23`.
2. `std::cell::Cell<ConvAlgo>` — `conv.rs:134`, `algo: std::cell::Cell<ConvAlgo>`, so
   `set_algo(&self)` mutates through a shared reference.
3. `RefCell<Option<Vec<u8>>>` inside `Storage::Gpu` — `storage.rs:43`.

"set algo mean 'immutable' module fw pass can change behavior btwn calls" ✅.

### RawTensor's Clone ✅ CONFIRMED, all four

`tensor.rs:78–88`: `grad: self.grad.clone()` (clones gradient state), `parents: self.parents.clone()`
(Rc-clones the parents vector — duplicating graph topology), `grad_fn: self.grad_fn.as_ref().map(|gf| gf.clone_box())`
(boxes a fresh grad function). "cheap write, expensive misunderstand" ✅ — the clone yields a node
that _looks_ like a new tensor but shares the entire ancestry.

### "not send/sync but has mutexes" ✅ CONFIRMED

`Mutex` in `gpu/pool.rs:57,61`, `gpu/staging_pool.rs:36,40`, `gpu/system_monitor.rs:37`. Meanwhile
`Tensor = Rc<RefCell<…>>` makes every graph `!Send`. "future proofing part that doesnt need it /
actual blocker Rc is untouched" ✅.

### "repro is best effort" ✅ CONFIRMED — see "RNG repro holes" above; every sub-claim holds.

---

## 8. Trait design / public API

- "TensorOps 140 lines of boilerplate / 40 methods decl / impl for type alias by forwarding each
  to RawTensor static" ✅ — `tensor.rs:1009` `pub trait TensorOps`, `tensor.rs:1071`
  `impl TensorOps for Tensor`, closing at 1208. The impl block is 138 lines; the range 1009–1208
  in the note is exact. Method count is **42**, not 40 — near enough.
  "now every example uses use volta::TensorOps" ✅.
- "3 ways to call shit: static / trait / free funcs re-exported from crate root" ✅ —
  `RawTensor::softmax(&t, 1)`, `t.softmax(1)`, and `volta::softmax(&t, 1)` all exist; `lib.rs:57`
  re-exports 19 free functions.
- "module trait broken by own implementors":
  - "impl Module for Embedding panics by design" ✅ — embedding.rs:120.
  - "seq stores boxed dyn mod and calls trait method (dyn dispatch) / embedding in seq is runtime
    bomb compiling fine" ✅ — reproduced.
  - "module for LSTMCell returns only h, silent discards cell state" ✅ — `lstm.rs:148–152`,
    `let (h, _c) = self.forward_step(x, None); h`.
  - "linear has inherent fw, and trait impl whose body is self fw / resolves to inherent method by
    precedence rules / works, reads like recursion" ✅ **exactly** — `linear.rs:15–18`:
    `impl Module for Linear { fn forward(&self, x: &Tensor) -> Tensor { self.forward(x) } }`, with
    the inherent `pub fn forward` at line 110.
- "grad machinery is public / lib exports gradfn traits / internals no user can construct
  meaningfully" ✅ — `lib.rs:43` `pub use autograd::GradFn;` and `lib.rs:66–69` re-exports
  `BinaryGradFn, MatMulGradFn, MaxReduceGradFn, MeanGradFn, MovementGradFn, MulAccGradFn,
SumGradFn, UnaryGradFn, WhereGradFn`.
- "conv2d weight is private, linear/embedding weight are public / no coherent rule" ✅ —
  `conv.rs:130` `weight: Tensor`, `linear.rs:12` `pub weight: Tensor`, `embedding.rs:32`
  `pub weight: Tensor`.
- "check grad returns (f32, f32, bool) / nothing stops caller from ignoring failure" ✅ —
  `tensor.rs:889`. Not `#[must_use]`.
- "use RawTensor as new_tensor: constructor shaped alias for struct itself / private vocab" ✅ —
  `tensor.rs:1346`, `pub use RawTensor as new_tensor;`. This surfaces in compiler diagnostics:
  errors on `Tensor` render as `Rc<RefCell<new_tensor>>`, which is actively confusing.
- "try_/panic twin pattern exists for 4 funcs: new, sum_dim, shrink, he_init / mm, bc_shape,
  reshape, perm … panic only" 🟠 **PARTLY WRONG.** There are **six** `try_` twins on tensor ops:
  `try_new` (tensor.rs:111), `try_randn` (223), `try_he_initialization` (276), `try_sum_dim` (612),
  `try_broadcast_shape` (**binary.rs:438**), `try_shrink` (movement.rs:1015) — plus three on
  `Storage`. `bc_shape` is explicitly listed as panic-only but _does_ have a twin. `matmul`,
  `reshape`, and `permute` are correctly identified as panic-only.

---

## 9. Error strategy

- "result with well-formed voltaerror, used by the try_ cons and nothing else" ✅ — `src/error.rs`
  defines 9 variants via `thiserror`; the only producers are the six `try_` functions and
  `Storage`'s three.
- "panics: asserts for every shape, broadcast, dim, channel check in op layer" ✅ — e.g.
  `matmul.rs:163` `assert_eq!(n, n2, "Matmul dimension mismatch: ...")`,
  `conv_transpose.rs:371` `assert_eq!(w_ch, in_channels, "Channel mismatch: ...")`.
- "458 non-test unwrap/expect occurrences in src/" 🟡 — measured **456** occurrences of
  `.unwrap()` / `.expect(` across `src/` **including** the large in-file `#[cfg(test)]` modules
  (which is most of `lib.rs`'s 5000 lines and 2000 lines of `conv.rs`). The true non-test figure
  is lower, likely 250–300. Magnitude correct, "non-test" qualifier overstated.
- "biggest one: silence" ✅ — supported by B4, mixed-device ops, GPU fallbacks, H5.

### "lint config made correctness worse" ✅ CONFIRMED, and the count is exact

- "indexing_slicing = deny pushes kernel code to 77 sites with unwrap_or style access in math
  kernels" ✅ — `rg "unwrap_or\(0\.0\)" src/` = **77 occurrences across 10 files**. Exact match.
- "bounds violation (usually index map bug) produces silent 0 instead of panic" ✅.
- "not hypothetical: B4 stayed silent bc broadcast_to maps OOR source indices through
  data.get(from_idx) => default (check bin.rs:519-523)" ✅ — the cited range is exact; see B4.
- "curr policy: ships green behind [641] test suite" ✅.
- "lint turning corruption into wrong numbers is shit for numerical code" 🔵 — I agree, and the
  proposed `fetch()` returning `f32::NAN` in §"loud by default" is the right shape: NaN
  propagates through the FD comparison and fails loudly; 0.0 produces a plausible number.

### "GPU layer returns option for everything" ✅ CONFIRMED with a refinement

- "callers convert none into warnings plus silent cpu fallback" ✅.
- "eprintln req but failed in mm, bin, tern, conv" 🟡 — reading this as "warnings exist in mm,
  tern, conv but the fallback still happens silently to the user": `matmul.rs:187`
  (`"Warning: GPU matmul requested but failed; falling back to CPU"`), `conv.rs:1510`
  (`"[WARNING] GPU Direct Convolution failed, falling back to CPU"`), `ternary.rs:216,220` all
  warn. **`binary.rs` has no warning at all** — the most-called op family falls back with zero
  signal. Worth calling out separately.
- "users op silently changes dev, kernel class, threading chars / only signal is stderr may be
  buried in logs" ✅.
- "mixed dev bin ops dont warn: devices are unwrapped and discarded and computed on CPU
  (mm:137-154)" ✅ — line range is exact: `matmul.rs:137–154` binds `_dev_a` and `_dev_b` and
  never reads them.

### "library prints" 🟡 CONFIRMED, count high

- "47 print/eprintln sites in non-test source" — measured **49 total**, of which **12 are inside
  doc comments** (`/// eprintln!(...)` in `gpu/monitor.rs`, `gpu/early_warning.rs`, `gpu/mod.rs`),
  leaving **37 real call sites**, 4 of which are in `lib.rs`'s test module → **33 in real
  non-test library code**. Overstated by ~40%, but the substance holds:
  `gpu/context.rs` alone has 15 `eprintln!` sites including a 4-line advice block
  (`context.rs:375–378`, "Consider: - Reducing batch size - Adding more frequent sync points…")
  emitted from inside a sync retry loop.
- "per-call fallback warnings that will spam in training loop" ✅ — `matmul.rs:187` and
  `conv.rs:1510` are inside per-op code paths with no rate limiting or once-flag.
- "printing to stdout unprompted is a defect for a crates.io library" ✅ agreed. (All 37 are
  `eprintln!`/stderr except the ones in test code, which slightly softens it.)

### "one good thing: missing docs are denied and honored" ❌ **WRONG**

`missing_docs` appears **nowhere** — not in `[lints.clippy]`, not in `[lints.rust]` (there is no
`[lints.rust]` table), and not as `#![deny(missing_docs)]` in `src/lib.rs`. What exists is
`missing_panics_doc = "deny"`, `missing_errors_doc = "deny"`, and `doc_markdown = "deny"`.

The _observation_ underneath is still fair — doc coverage is genuinely good and the
`# Panics` / `# Errors` sections are present and mostly accurate. But they are enforced by
different lints, and those lints are being _satisfied in form_: `storage.rs:298`
`/// # Panics\n/// Wrong storage datatype` sits above `as_f64_slice`, which returns `Option` and
does not panic at all. So even the honored lints are being fed placeholder text. The conclusion
"documentation discipline is real" needs qualifying.

---

## 10. Unsafe audit

- "volume is low" ✅ — 4 `unsafe` blocks in `storage.rs` (lines 252, 317, 348, 379), 1 `unsafe`
  in `storage.rs:418`, 1 `unsafe extern "C"` + call in `matmul.rs:59–90`, plus
  `matrixmultiply::sgemm`.
- "hand decl extern C cblas_sgemm w manual i32 casts / works, but blas-src is a dep, supplies
  decl / hand rolled extern next to bindings crate is drift waiting to happen" ✅ —
  `matmul.rs:59–81` declares `unsafe extern "C" { fn cblas_sgemm(...) }` by hand while
  `Cargo.toml` carries `blas-src = { version = "0.14", features = ["accelerate"], optional = true }`.
  The dependency exists purely for linkage.
- "matrixmultiply::sgemm unsafe reqd by API, args, strides correct" ✅ — verified the row-major
  stride arguments.
- "bytemuck casts through storage.rs, tool used right" ✅.

### "storage block" ✅ CONFIRMED — every sub-claim; see H3.

The one addition: "false safety is worse than none bc refactor trusts it" is not hypothetical
here. The comment's point 3, "The Vec inside is stable (won't move/reallocate)", is _also_ only
true because nothing currently pushes to the cache — a future incremental-population scheme would
silently invalidate the outstanding `&[f32]`.

---

## 11. Ownership and clone economy

- "own notes count 400+ clone ops across 45 files" ✅ — measured **417** `.clone()` (432 including
  `clone_from`) across **34 files in `src/`**; **520 across 49 files** counting `tests/`,
  `examples/`, `benches/`. The 45-file figure matches the wider sweep, not `src/` alone.
- "load bearing pattern repeated by hand every op: deep copy / compute into fresh vec /
  borrow mut clone t into output parents and define grad fn in a box" ✅ — verbatim in
  `binary.rs`, `unary.rs`, `matmul.rs`, `movement.rs`, `ternary.rs`, `reduce.rs`.
- "cpu storage clone is deep VecU8 copy, clone economy isn't Rc-pointer economy at all on CPU
  path" ✅ — `storage.rs:50–53`.
- "borrow dance never abstracted / no with_tensor helper, no op builder / same 5 lines appear
  dozens of times" ✅.
- "deref impl for storage is deref polymorphism over diff types / data.iter can panic, block and
  iter/deref_mut panic on gpu storage / exact mechanism behind muon panicking (H8)" ✅ —
  `storage.rs:652–665`, `Deref<Target = [f32]>` calling `as_f32_slice()` (which `expect`s), and
  `DerefMut` calling `as_f32_slice_mut().expect("Mutable deref requires F32 CPU storage")`, which
  returns `None` for GPU. `Index`/`IndexMut` (lines 668–685) have the same shape. Confirmed as
  the H8 mechanism.
- "explicit try_as_host_f32 would put cost and failure at call site instead of 3 layers below it"
  🔵 agreed — and `try_as_f32_slice` already exists (`storage.rs:213`); it is simply bypassed by
  the `Deref` convenience.

---

## 12. Abstraction quality

- "BatchNorm1d/2d are duplicated, differing in reduction axis / already forced || edits (same
  bessel-correction logic had to be typed 2 times)" ✅ — `batchnorm.rs`, 347 lines, two structs
  (`BatchNorm2d` at line 6, `BatchNorm1d` at line 195). The Bessel correction is typed twice:
  line 113 `let unbiased_var = bv_val * num_pixels / (num_pixels - 1.0);` and line 279
  `let unbiased_var = bv_val * batch_size / (batch_size - 1.0);` — same expression, different
  variable name.
- "15 allow too many lines" ✅ **exactly 15** across 11 files.
- "9 expects with reasons being TODO confessions" ✅ **exactly 9** `reason = "…"` attributes
  (a mix of `#[allow]` and `#[expect]`, so "expects" is loose). The reasons, verbatim:
  `"will handle idiomatically after i finish the rest of these"` (×2, ternary.rs),
  `"can't figure out right now"` / `"can't figure out rn"` (×3, reduce.rs),
  `"Don't know how to fix right now"` (sequential.rs),
  `"just don't know hot to fix right now"` (conv_transpose.rs),
  `"Attempts to fix by merging the construction with its single usage leads to other errors, will come back to fix this."` (pool.rs).
  Only one is a real justification (`binary.rs:296`, `"∂(x/y)/∂y = -x/y²"`). The characterization
  is exactly right.
- "crate level allow float_cmp, similar_names, cast_precision_loss" ✅ — `lib.rs:15`, verbatim.
  There are 5 more file-level `#![allow(...)]` blocks (`storage.rs:7`, `adam.rs:1`,
  `conv_transpose.rs:1`, `conv.rs:1`, `gpu/kernels.rs:6`).
- "60 line deny table plus scoped exceptions ⟹ aspiration, not code" ✅ — the `[lints.clippy]`
  table is ~60 deny entries.
- "16 pos usize args on GPU kernel wrappers / WGSL side uses Im2colParams and MovementParams
  structs / had good idea in shaders, didnt use in rust" ✅ —
  `gpu_ops.rs:670 gpu_conv_backward_input` takes 2 `&Storage` + **14 positional `usize`** = 16
  parameters, twelve of which are mutually substitutable integers. Meanwhile `kernels.rs:60,73,109`
  define `MovementParams`, `Im2colParams`, `DirectConvParams` as `#[repr(C)]` structs matching the
  WGSL side. 104 bare `: usize,` parameter lines in `kernels.rs`.
- "cfg triplication / DirectConvGradFn dupes each cpu call across cfg arms / structure that
  produced 2/5 no features compile errors (B5) / correct config hygiene would compute arms once"
  ✅ — see B5; `conv.rs:336` and `conv.rs:413` are precisely those 2 of 5.
- "config.rs dead file of pseudocode / survives bc it doesn't get pulled in" ✅ — see above.
- "lib.rs: 83 lines of exp, ~5000 lines of test, alongside 17 integration test files / follows no
  rule (comment of lib:4094 admits convention) / blob makes crate root largest file in project"
  ✅ **exactly** — `src/lib.rs` is 5051 lines; the first `#[cfg(test)]` is at line 83; there are
  19 separate `#[cfg(test)] mod` blocks; 17 integration test files; and 5051 > `conv.rs`'s 3861,
  so it is the largest file in the project.
- "diagnostic tooling exported as API / pub on early_warning, system_monitor, monitor" ✅ —
  `gpu/mod.rs:11,13,15,16`: `pub mod early_warning; pub mod monitor; pub mod staging_pool; pub mod system_monitor;`.
- "system_monitor shelling to ps is test scaffolding not library surface" 🟡 — the `ps` shell-out
  is in **`gpu/monitor.rs`**, not `system_monitor.rs`: `monitor.rs:113`,
  `Command::new("ps").args(["-o","rss=","-p", &pid])`. `system_monitor.rs:64` consumes it via
  `get_process_memory_mb()`. Both modules are `pub`, so the conclusion stands; the attribution is
  one module off. (`monitor.rs:141` also hardcodes `const SYSTEM_MEMORY_MB: usize = 24 * 1024` —
  this machine's RAM, baked into a library.)

---

## 13. Deps and packaging

- "default has heavy wgpu by default / downstream crate wanting cpu autograd pays full wgpu
  compile unless it opts out" ✅ — `default = ["gpu", "accelerate"]`, `gpu = ["wgpu", "pollster"]`.
  And per B5, opting out **does not compile**, so there is currently no escape at all.
- "readme frames GPU as opt-in, contradicting / claude admits mismatch: docs know, manifest
  doesn't" ✅ on the manifest side; the README/CLAUDE.md side is confirmed by `CLAUDE.md`'s own
  "Feature flags gotcha" note.
- "statrs pulled in for erf / 20 line Abramowitz-Stegun or libm would delete several dependencies"
  ✅ — `src/ops/unary.rs:3` `use statrs::function::erf::erf as statrs_erf;` and exactly one call
  site, `unary.rs:273`, `UnaryOp::Erf => statrs_erf(f64::from(x)) as f32`. That is the entire
  usage of `statrs` in the crate. (`libm::erff` would be the smaller swap; A&S 7.1.26 gives
  ~1.5e-7 absolute error, adequate for f32.)
- "approx in dependencies but only used in tests, so it's a dev dependency" ✅ — every
  `approx::` use is inside a `#[cfg(test)] mod` (`lib.rs:169, 3040, 3041, 3118, 3317, 4227, 4423,
4535, 4871` and `nn/transformer/mask.rs:49`). It is in `[dependencies]`, not `[dev-dependencies]`.
- "blas-src dep only to link Accelerate / actual cblas decl handwritten / linkage and decl can
  drift independently" ✅ — see §unsafe audit.
- "it is modern: 2024, 1.89 vers, let-chains, crate metadata, best maintained dim of project" ✅ —
  `edition = "2024"`, `rust-version = "1.89.0"`, `license`, `readme`, `repository`, `keywords`,
  `categories` all present; let-chains used pervasively.
- "repo hygiene: no tracking target, but pub repo carries LLM-session artifacts at top level /
  responses/*.md / sys prompts" ✅ — `.gitignore` has `/target`; `git ls-files` shows
  **`responses/` (7 tracked .md) and `sys-prompts/` (5 tracked .md)** at the repo root. There are
  also four more top-level LLM artifacts the note doesn't mention: `ADDENDUM.md` (41 KB),
  `DECISIONS.md`, `glm-53-adversarial-audit.md` (35 KB), and `coverage.txt`.
- "unfinished minigrad thing / not shipped in crate, but is repo pub face" 🟠 — `minigrad-react/`
  is **gitignored** (`.gitignore` line: `#hiding this for now\nminigrad-react/`) and untracked.
  It exists in the working tree only, so it is not part of the repo's public face.
- "node_modules at crate root invites accidents" 🟡 — `node_modules/` exists locally but is
  gitignored and untracked. The "invites accidents" concern is about local tooling
  (`cargo package` excludes it via `.gitignore`), so the risk is lower than stated but nonzero.

---

## 14. CI ✅ CONFIRMED, exactly

`.github/workflows/ci.yml`:

```yaml
- name: Clippy (no features)
  run: cargo clippy --all-targets -- -D warnings # ← default features: gpu + accelerate

- name: Test (no features) #TODO: improve CI setup/reorganize...
  run: cargo test # ← same

# - name: Check (gpu feature) # Runners lack GPUs
#   run: cargo check --features gpu
```

- "no-features doesn't check no-features" ✅ — neither step passes `--no-default-features`, and
  `default = ["gpu","accelerate"]`, so both steps compile **with** GPU. The step names are
  inverted from reality.
- "-D warnings caused allow with excuses culture instead of fixing shit / warnings were errors,
  every issue gets a waiver rather than change" ✅ — supported by the 9 TODO-confession `reason =`
  strings and 15 `too_many_lines` allows.
- "GPU feature check commented out (runners tend to lack GPUs)" ✅ — lines 41–42, with that
  comment verbatim.
- "cfg gated code paths had no compiler looking from any direction" 🟡 — one direction _is_
  covered: because the "no features" steps actually run with `gpu` on, every
  `#[cfg(feature = "gpu")]` arm is compiled. What is never compiled is every
  `#[cfg(not(feature = "gpu"))]` arm — which is exactly where all 5 of B5's errors live. The
  conclusion is right; "from any direction" is one direction too strong.

---

## 15. Preventative architecture 🔵 JUDGMENT

This section is design proposal, not factual claim. Evaluated on merit against what I found.

### Defect → class → prevention table

The mappings are sound. Two notes:

- **"B4 → dims derived from wrong tensor rank → shape/axis newtypes."** A `Shape` newtype alone
  would _not_ have prevented B4 — `y_shape.dim(Axis(x_rank - 1))` is still expressible and still
  wrong. What prevents it is the `MatmulSpec::new(a: &Shape, b: &Shape)` idea in §1.2, which
  derives `m,k,n,batch` **once** at forward time. That's the load-bearing half; the newtype is
  scaffolding for it. Worth reordering the framing.
- **"Silent zeros → bounds failures become data → loud indexing policy."** ✅ this is the
  highest-leverage item in the document and my evidence supports it directly: B4's 36% gradient
  error is invisible precisely because `binary.rs:519` turns an out-of-range index into a 0.0.

### Opaque tensor newtype ✅ well-founded

"every time you write borrow_mut outside autograd and src/ops, found a missing method on tensor"
is a good rule and the data supports the payoff: 63 `borrow_mut().requires_grad` sites and 15+
`borrow_mut().data =` sites would each become a named method or a compile error. B1 (reads
`requires_grad` from a layer) and H7 (writes `data`/`shape` from a layer) both become
unexpressible. "roughly a weekend / prereq for nearly everything else / do first or don't bother"
— the churn estimate looks right given 34 files touch `.borrow()`.

One caveat: `pub type Tensor = Rc<RefCell<RawTensor>>` is in the public API and every example
does `x.borrow_mut()`, so this is a breaking change for any downstream user. At 0.3.0 that's fine.

### Shape/Axis newtypes ✅ sound, with the reordering noted above

`Shape::from_dims` as the single checked constructor also subsumes the `MAX_ALLOC` cap that is
currently copy-pasted at `movement.rs:754`, `movement.rs:849`, and `conv.rs:1181`.

### One graph-edge constructor ✅ the strongest proposal in the document

"25 copies of it" — I count graph-attach rituals in `binary.rs`, `unary.rs`, `matmul.rs`,
`reduce.rs`, `ternary.rs`, `movement.rs` (×6 movement ops), `conv.rs` (×4 algorithms),
`conv_transpose.rs`, `maxpool.rs`, `pixelshuffle.rs`, `embedding.rs` — 20–25 is right.

"b2/3 dead: both needed rawtensor new because graph path harder than raw path / with rawtensor new
demoted and edge being easy as raw / easy and correct become same way" — this is the correct
diagnosis. `slice_gate` and `PositionalEncoding::forward` both reached for `RawTensor::new`
because composing `shrink` + `broadcast` + `add` required knowing three op names and their
gradient semantics, while `RawTensor::new(vec, shape, flag)` required knowing one.

`grad_is_trivial(&grad)` handling `Mod`/`Cmplt` in one place is a real simplification — today
`binary.rs:618–623` special-cases them in a match arm that every future non-differentiable op
would have to remember to extend.

### RAII no_grad guard ✅ sound

Directly fixes the `check_gradients` contamination I reproduced. The BatchNorm running-stat point
is also real: `batchnorm.rs:100–116` mutates `running_var` inside `forward`, so every finite-
difference probe evaluation perturbs the statistics that the next probe evaluation reads. That is
a second, independent source of noise in the conv/batchnorm FD tests, and plausibly part of why
the tolerances at `conv.rs:2018` had to be loosened to 5e-2/2e-2.

### Typestate section 🔵 well-calibrated

"Decline: full device typestate" — correct call; PyTorch, Burn, and Candle all keep device
runtime-dynamic for the reason given.

"Decline: Variable/Tensor typestate" — the historical claim is right: PyTorch merged `Variable`
into `Tensor` in 0.4 (2018) for essentially this reason.

`DevicePolicy` ✅ — directly addresses the mixed-device silence at `matmul.rs:137–154` and
`binary.rs:628`, plus the un-warned `binary.rs` GPU fallback. `Strict` as default converts H5's
rank-5 case into an error rather than wrong numbers.

`into_loss` / `ScalarLoss` — the panic it replaces is real: `autograd.rs:~43`,
`assert!(tensor.requires_grad, "Called backward on a tensor that doesn't require grad")`. The
cited line range 43–46 is close.

`PadValue` ✅ — precisely fixes H1, and the generalization ("bare primitive crossing API boundary
carrying semantics wants to be a ZST-constructible newtype") is well-drawn.

`GradFn` closed enum ✅ — `clone_box` is on the trait (`autograd.rs:~30`) and implemented by every
single grad function; `#[derive(Clone)]` would delete all of them. The observation that
`wildcard_enum_match_arm = "deny"` (already in `Cargo.toml`) becomes load-bearing under this
design is the sharpest point in the section: it turns an existing decorative lint into a real
compile-time completeness check.

### Trait design 🔵 sound

Sealing + `Layer`/`IndexedLayer`/`RecurrentLayer` split directly fixes the Embedding panic and
the LSTMCell cell-state discard, both confirmed above. `non_exhaustive` — confirmed absent
everywhere (`rg non_exhaustive src/` finds only `finish_non_exhaustive` in a `Debug` impl), and
the argument that it is free now and expensive later is correct.

"Delete TensorOps / 140 line blob 1009-1208 in tensor should not survive" ✅ — verified extent.

### Arena-based graph 🔵 sound but correctly sequenced last

"cycle safety is prop of every future op author's care" ✅ — nothing today prevents an op from
inserting a node into its own ancestor's `parents`. "top order is free: bw walks id.0..=root /
dfs in autograd and diamond handling goes away" — `autograd.rs` currently does a consumer-count
DFS; an arena with construction-order ids removes it. "clones become copies of (Rc, u32) / 400+
clone economy mostly evaporates" ✅ consistent with my count.

The one cost the section understates: `Storage` clones would remain deep, and per §"clone economy"
those are the expensive ones on the CPU path. The `Arc<[u8]>` + CoW change in §7 is doing more of
the real work than the arena is. Sequencing phase 3 (storage) before phase 4 (arena) already
reflects this.

### Storage: `Arc` + `OnceCell` ✅ correct fix for H3

`OnceCell` makes "never cleared once populated" a type-level fact instead of a false comment, and
`invalidate_cpu_cache` either disappears or becomes `OnceCell::take(&mut self)` — which is
_honest_ because `&mut self` is then structurally required rather than incidentally sufficient.
Deleting the f64/f16/bf16 GPU accessors ✅ — confirmed unreachable and wrong.

Replacing `Deref<Target=[f32]>` with `as_host_f32() -> Result` / `to_host_f32() -> Cow` ✅ — this
is the fix for H8 and for every invisible GPU→CPU sync. "grep-ability is a feature" is the right
argument.

### Loud kernels ✅ — see §"lint config made correctness worse". NaN over 0.0 is the correct choice.

### Cfg hygiene ✅ — "gate once at definition sites" is exactly the fix for all 5 B5 errors, and

"grep for feature outside src/gpu, each hit is a B5 seed, should be 0 after this refactor" is a
checkable acceptance criterion. `cargo hack --each-feature` would have caught B5 on the commit
that introduced it.

### Test architecture ✅

- `assert_trainable!` — verified to catch B1/B2/B3 (see §6).
- Property-based grad checks — "random graph FD check would have caught B4 (unequal rank matmul
  composes triv in random DAG)" ✅, provided the generator is allowed to emit rank-mismatched
  matmul operands, which is exactly the case the hand-written tests never construct.
- `trybuild` UI tests — correct: a compile-fail guarantee that isn't tested isn't a guarantee.
- `cargo-mutants` — the LSTM claim is right in kind. A mutant of `slice_gate` that returns zeros,
  or the wrong gate index, survives `test_lstm_gradient_flow` because that test asserts nothing.
  Note `.gitignore` already contains `mutants.out/`, so this has been attempted before.
- `insta` for state-dict round-trips — would catch the Sequential prefix bug's _symptom_
  (bogus `0.weight` keys in layer 1's sub-state) if the snapshot captured sub-states rather than
  just the top-level dict.

### Sequencing 🔵 reasonable

Phase 0 is correctly scoped as a day of near-zero-risk work, and it retires B5 plus the packaging
findings. One reordering suggestion: **B5 is a shipped-broken build**, and the two-line CI change
plus the `sgd.rs:189` cfg flip fixes it in under an hour. That belongs ahead of everything,
including the rest of phase 0.

Phase 1's `assert_trainable` across all layers is the single highest value-per-hour item in the
whole plan — it is ~19 lines of test and it fences the entire B1–B3 class, and it does not depend
on the opaque-tensor refactor landing first.

### Explicitly declined patterns 🔵 all four declines are correct calls

Const-generic rank in particular: it would break `sum_dim`, `reshape`, and every shape-agnostic
helper in `movement.rs`, and buys nothing the `Shape` newtype doesn't.

---

## 16. Claims that did not hold

Collected for convenience:

| #   | Claim                                                                      | Finding                                                                                                                                                                                                                      |
| --- | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2   | Muon's "frob-norm 'approx spectral' scaling" is a defect                   | ❌ Frobenius normalization is what the reference Muon does. The real deviations are the quadratic (vs quintic) polynomial and the missing `sqrt(max(1, rows/cols))` scaling.                                                 |
| 3   | "GpuSyncError: display w/o impl std::error::Error (gpu/context 40-57)"     | ❌ `impl std::error::Error for GpuSyncError {}` is at `gpu/context.rs:57`, the last line of the cited range.                                                                                                                 |
| 6   | H5 blames `pad_to_u32_4` at kernels:2955                                   | 🟡 That function is real at that exact line but is used only by movement ops. Broadcast-backward pads inline at `kernels.rs:769–800`, and does it worse (left-aligned, breaking the right-alignment its own comment claims). |
| 7   | "eprintln req but failed in mm, bin, tern, conv"                           | 🟡 mm/tern/conv all warn; **`binary.rs` warns nowhere** — the biggest gap, and unnamed.                                                                                                                                      |
| 8   | "col2im tests use varied data" (as the exception)                          | 🟡 Varied input, but weights are still `vec![1.0; …]`.                                                                                                                                                                       |
| 9   | "system_monitor shelling to ps"                                            | 🟡 `gpu/monitor.rs:113` does the shelling. Both modules are `pub`, so the point stands.                                                                                                                                      |
| 10  | Counts: 641 tests / 458 unwraps / 47 prints / 45 clone files / 400+ clones | 🟡 Measured 671+13 / 456 (incl. in-file tests) / 37 real sites / 34 files in `src/` (49 repo-wide) / 417. All the right magnitude; the print count is the furthest off (~40% high).                                          |
| 11  | B4 "29% grad error", control at 4.8e-4                                     | 🟡 Measured 36% mean / 38% max wrt A, and **100% max / 85% mean wrt B**; control at 3.1e-3. Data-dependent; the bug is worse than described.                                                                                 |
| 12  | H2 "\|x\| >= 88"                                                           | 🟡 Threshold is x ≥ 89 and one-sided — negative logits are numerically fine.                                                                                                                                                 |
| 13  | check_grads contamination "err 1.02"                                       | 🟡 Measured 0.50 for one prior backward. Same defect.                                                                                                                                                                        |
| 14  | "cfg gated code paths had no compiler looking from any direction"          | 🟡 The `gpu` arm _is_ compiled by CI (because the "no features" steps run with default features). Only the `not(gpu)` arm is unchecked — which is where all 5 errors are.                                                    |

Claims handled so far:
1 (partial): now have missing_docs set, need to doc... pretty much everything, should build tool to check doc coverage
4: number of try_ pairs. was 6, also something to be handled with other shit
5: still need to handle minigrad-react, but it's not in the repo public face

| 1 | "missing docs are denied and honored" | ❌ `missing_docs` is not configured anywhere. `missing_panics_doc`/`missing_errors_doc` are — and are satisfied with placeholder text (`storage.rs:298` documents a panic in a function that returns `Option`). |
| 4 | "try_/panic twin pattern exists for 4 funcs … bc_shape … panic only" | 🟠 Six twins exist, including `try_broadcast_shape` (binary.rs:438) — the one explicitly listed as absent. |
| 5 | "unfinished minigrad thing" is part of the repo's public face | 🟠 `minigrad-react/` is gitignored and untracked. |

## 17. Things the notes missed

Found while verifying, not claimed anywhere in the document:

1. **ConvTranspose2d clobbers its own bias grad_fn.** `conv_transpose.rs:424` overwrites the
   `AddGradFn` produced by the bias addition at line 413. When `x.requires_grad` is true the bias
   receives gradient via `ConvTranspose2dGradFn`'s third parent; when false, via the surviving
   `AddGradFn`. Two different code paths reach the same result by accident, which is why B1 looks
   like "only the weight is frozen".
2. **B4's grad-wrt-B is much worse than grad-wrt-A.** 18 of 48 weight-gradient entries return
   identically 0.0. The `x.shape.get(y_rank - 1)` read falls off the end of the shorter shape and
   `unwrap_or(1)` collapses `k` to 1.
3. **`Embedding::load_state_dict` is the only layer that validates shape** (`embedding.rs:145`).
   H7's "no element count validation" is true of Linear/Conv/ConvTranspose/BatchNorm but not
   Embedding — useful, because it's the template for the fix.
4. **`binary.rs` has zero GPU-fallback diagnostics** while matmul, ternary, and conv all warn.
   The most-executed op family is the quietest.
5. **`monitor.rs:141` hardcodes `const SYSTEM_MEMORY_MB: usize = 24 * 1024`** — the dev machine's
   RAM, compiled into a published library, used as the denominator for every memory-pressure
   decision.
6. **`Muon` is not re-exported** from the crate root (`lib.rs:47` lists `Adam` and `SGD` only), so
   H8's panic is only reachable via the full module path.
7. **`Sequential::load_state_dict` slices `&key[prefix.len()..]`** on a `String`
   (`sequential.rs:67`) — panics on a multi-byte layer name.
8. **Three more top-level LLM artifacts are tracked** beyond `responses/` and `sys-prompts/`:
   `ADDENDUM.md` (41 KB), `DECISIONS.md`, `glm-53-adversarial-audit.md` (35 KB).

---

## 18. Bottom line

The notes are a reliable document. Every blocking defect is real, reproduces on first attempt, and
reproduces by the mechanism claimed. The high-severity list is likewise sound — H8's Frobenius
bullet is the only genuinely incorrect technical claim in it. The line-number citations are
unusually precise: `movement.rs:1102`, `binary.rs:519–523`, `matmul.rs:137–154`,
`kernels.rs:2955`, `storage.rs:627`, `conv.rs:1213`, `tensor.rs:898–902` are all exact hits.

The counted claims run consistently ~5–40% high (641/458/47/29% vs measured 671/456/37/36%), which
reads like a snapshot taken at a different commit rather than fabrication — and where the count is
checkable to the digit (15 `too_many_lines`, 9 `reason =`, 77 `unwrap_or(0.0)`, 17 test files,
83 export lines, 5 build errors), it is exact.

The central thesis holds under my own evidence, with one sharpening. The notes say the test suite
"can't tell the difference." The ConvTranspose2d case is more specific than that: the suite
contains two tests that assert _exactly_ the property B1 violates (`conv_transpose.rs:718`,
`:843` — "Weights should have gradients"), and both are neutralized by a single line of setup
(`x.borrow_mut().requires_grad = true`) that puts the layer in the one configuration where the
bug does not fire. The suite is not blind to the property. It is aimed at it and misses by one
boolean — which is a stronger argument for `assert_trainable!` with a non-grad input than the
document makes for itself.

**Sources consulted for the Muon claim:**
[Muon: An optimizer for hidden layers in neural networks — Keller Jordan](https://kellerjordan.github.io/posts/muon/),
[How Much Orthogonalization Does Muon Need?](https://arxiv.org/pdf/2606.00371),
[Accelerating Newton-Schulz Iteration for Orthogonalization via Chebyshev-type Polynomials](https://arxiv.org/pdf/2506.10935).

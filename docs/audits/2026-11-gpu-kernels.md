# 2026-11 GPU kernel and WGSL audit

Dated record. Original text is never edited or removed; corrections are appended under `## Corrections` (`.claude/rules/audits.md`).

**Date:** 2026-11-17. **Commit audited:** `ebbd683` (`improvements`). **Finding IDs:** `G1`–`G17`.

**Scope.** Every shader in `src/gpu/shaders/` (27 files), its host dispatch in `src/gpu/kernels.rs`, the element-type template (`src/gpu/shader_template.rs`), and the parts of `context.rs`, `buffer.rs` and `pool.rs` a kernel's result depends on. Each kernel was compared with the CPU path it stands in for (`src/ops/unary.rs`, `binary.rs`, `movement.rs`, `nn/optim/*.rs`, `nn/layers/conv.rs`). The questions were indexing, bounds, launch sizing, precision and accumulation, and fast-math exposure of the kind PR #284 hit. Silent wrong answers came first.

**Method.** Read every file, then run each suspicion as a CPU-versus-GPU probe through the public API. The probes are in the appendix; they are not committed as tests, because an `#[ignore]` needs an issue ID (invariant 6) and this audit files no issues (triage is the maintainer's).

**What ran.** Lavapipe only (Mesa 25.2.8 `mesa-vulkan-drivers`, software Vulkan, native `shaderFloat16`), wgpu 30.0.1, debug build, Linux. Each finding says "confirmed on lavapipe" (a probe printed the numbers quoted) or "by reading" (not run, with the reason). **Nothing here ran on Metal.** Every finding needs `just test-gpu` on the Mac before it is called confirmed or refuted there, and G14 is a list of things only the Mac can answer.

**Result.** Two reachable silent wrong gradients (G1), one silent wrong forward value (G2), two precision findings (G3, G4), two classes of panic where the CPU path works (G5, G6), four performance traps (G7–G10), one cluster in code that is unreachable today (G11), and six low-severity or Metal-only items (G12–G17).

| ID | Severity | One line | Evidence |
| --- | --- | --- | --- |
| G1 | Silent wrong gradient | `shrink_backward` and `stride_backward` launch too few threads; part of the gradient stays zero | lavapipe |
| G2 | Silent wrong value | GPU `modulo` is `a - b*trunc(a/b)`, not an exact remainder; NaN for an infinite divisor | lavapipe |
| G3 | Precision (both backends) | `expand` backward sums in one f32 chain and stalls at 2^24 | lavapipe + CPU |
| G4 | Precision | Adam's bias correction is `pow(beta, t)` on the device: first update off by up to 2.4e-4 relative | lavapipe |
| G5 | Panic, CPU works | 16x16-grid kernels dispatch more than 65535 workgroups per dimension | lavapipe |
| G6 | Panic, CPU works | A tensor over 2^25 f32 elements exceeds the 128 MiB binding limit the device was opened with | lavapipe |
| G7 | Performance | `igemm.wgsl` computes each output up to 256 times | lavapipe (timing) |
| G8 | Performance | `expand_backward` is O(input x output) | lavapipe (timing) |
| G9 | Performance | `col2im` scans every output position per input element | reading |
| G10 | Performance | Every kernel output is zero-filled from a host `Vec` first | reading |
| G11 | Latent (unreachable, #60) | The broadcast backward kernel has three more defects behind the bind-group panic | reading |
| G12 | Low | `exp2` is `pow(2.0, x)`: 2 ulp at 0.5, subnormal results flushed | lavapipe |
| G13 | Low | Kernels bounded by `arrayLength` also process the pool's slack | reading |
| G14 | Unverified | Fast-math exposure list for Metal | reading |
| G15 | Docs | Comments that describe code that has since changed | reading |
| G16 | Low | Conv kernels rely on `u32` wraparound for padding | reading |
| G17 | Handed on | Items that belong to the numerical-stability audit | reading |

## Silent wrong answers

### G1. `shrink_backward` and `stride_backward` launch over the wrong shape

`movement_op` (`kernels.rs:2137`) sizes the launch from `params.new_shape`:

```rust
let output_size = span(&params.new_shape, 0..params.rank as usize).iter().product::<u32>();
... Groups::Flat(output_size.div_ceil(256))
```

Forward, `new_shape` is the result, so that is right. Backward, the wrappers pass `old_shape` = the original input (the result, `GpuBuffer::zeros_of(.., old_shape.product())`) and `new_shape` = the incoming gradient, and the shaders (`movement_backward.wgsl`) run one thread per **result** element (`idx < output_size(params.old_shape, ..)`). The launch still covers only `ceil(new / 256) * 256` threads.

- `pad_backward`, `expand_backward`: the result is smaller than `new_shape`, so the launch over-covers. Correct.
- `permute_backward`: same size. Correct.
- **`shrink_backward`, `stride_backward`: the result is larger.** Result elements at index `>= ceil(new/256)*256` are never visited and keep the zero from `zeros_of`.

Confirmed on lavapipe, F32, F16 and BF16 alike:

| Probe | CPU | GPU |
| --- | --- | --- |
| `x[1000].shrink(900..1000)`, weighted sum, `x.grad[950]` | 51 | 0 (all 100 window gradients are 0) |
| `x[1024].stride_op(2)`, weighted sum, `x.grad[1000]` | 501 | 0 (256 of 1024 entries differ) |
| `x[40,40].shrink(rows 30..40)`, `sum`, sum of `x.grad` | 400 | 0 |
| `x[40,40].stride_op(2,2)`, `sum`, sum of `x.grad` | 400 | 136 |

No panic, no warning, and a result that passes any test whose input has at most 256 elements (one workgroup covers it). CI is green at this commit, so no existing test has a movement backward result larger than its launch.

**Reach.** `LSTMCell` takes its four gates with `gates.shrink(&[(0, batch), (start, start + hidden)])` (`lstm.rs:67`). By the same arithmetic (not run as an LSTM), with `batch = 4, hidden = 64` each gate's backward launches 256 threads over a 1024-element result, so only batch row 0 gets a gradient. Any slicing or strided subsampling of a GPU tensor of more than 256 elements is affected.

**Fix direction.** Launch over the buffer the kernel writes: `Groups::cover(result.len())` in `movement_op`, which is right for all ten movement kernels and removes the dependence on which shape is "new". Add a parity test per movement backward with a result of more than 256 elements and a window or stride grid that lands past the first workgroup.

### G2. GPU `modulo` is not an exact remainder

`mod_op` (`elementwise.wgsl:74`) computes `ld_a(idx) % b`. WGSL defines float `%` as `e1 - e2 * trunc(e1 / e2)`, evaluated in f32, so the quotient is rounded before it is truncated. The CPU uses Rust's `%`, which is `fmod` and exact. `mod_backward_b` and the CPU agree with each other (`-g * floor(a / b)`); only the forward value differs.

Confirmed on lavapipe:

| `a mod b` | CPU | GPU |
| --- | --- | --- |
| `1e10 mod 3` | 1 | 0 |
| `67108864 mod 7` | 4 | 0 |
| `1e20 mod 3` | 2 | 0 |
| `1000.3 mod 0.1` | 0.09997289 | 0 |
| `123456.79 mod 0.001` | 0.00019861746 | 0 |
| `0.3 mod 0.1` | 7.45e-9 | 0 |
| `5 mod inf` | 5 | NaN |
| `-5 mod inf` | inf | NaN |
| `-7.5 mod 2` | 0.5 | 0.5 |

Two different things are in that table. For `|a / b|` at or above about 2^24 the quotient has no fractional bits left and the GPU result is unrelated to the remainder (the first three rows). Below that the error is about `|a| * 2^-24` absolute, which is small next to `a` but can be the whole remainder (rows four to six: both backends return a value within rounding of a multiple of `b`, and they land on opposite sides of the wrap). The infinite-divisor rows are `inf * 0 = NaN` inside the formula.

The S5 fix (#45) made both backends floored; it did not compare them at large quotients. How Metal lowers `%` was not checked (naga's MSL output and Metal's `fmod` under fast math decide it).

**Fix direction.** An exact `fmod` in WGSL is a loop (binary long division on the exponents), not a formula. The cheap options are to special-case an infinite or NaN divisor and document a quotient bound, or to treat `|a / b| >= 2^23` as a kernel gap and take the device policy's CPU path. Either way a parity test should pin the chosen behaviour.

## Precision

### G3. `expand` backward sums in one f32 chain (both backends)

`expand_backward` (`movement_backward.wgsl:214`) adds every contributing gradient into one `var sum: f32` in sequence. Confirmed on lavapipe: the gradient of `sum(expand([1] -> [2^25]))` is 16,777,216, not 33,554,432. The CPU path returns the same 16,777,216, so parity holds and both are wrong. This is the S6 class; #212 fixed it for the CPU binary-broadcast backward, and `reduce.wgsl` fixed it for `sum` with a tree. The realistic exposure is a bias expanded over a very long batch times spatial extent (`Conv2d` expands its bias to the output shape, `conv.rs:1449`), where the result is not a stall but ordinary accumulated rounding. The CPU half belongs to the numerical-stability audit (G17).

### G4. Adam's bias correction is computed per element with `pow`

`optimizer_step.wgsl:61` evaluates `1.0 / (1.0 - pow(p.beta1, p.t))` and the `beta2` twin in f32, inside the shader, for every element. The CPU uses `beta.powi(t)` (`adam.rs:253`). `1 - beta2^t` is 1e-3 at `t = 1` with `beta2 = 0.999`, so an error of one ulp in `pow` (6e-8) is a relative error of 6e-5 in the scale. WGSL does not bound `pow`'s accuracy tightly (it may be `exp2(y * log2(x))`).

Confirmed on lavapipe (`Adam(lr 1e-3, betas (0.9, 0.999), eps 1e-8)`, loss `sum(p*p)`, `p = [1, -2, 0.5, 3]`), relative difference between the CPU's and the GPU's total parameter update:

| Steps | Worst relative difference |
| --- | --- |
| 1 | 2.4e-4 |
| 2 | 1.2e-4 |
| 10 | 4.8e-5 |

That is far above f32 rounding (6e-8) and it is systematic, not noise: it scales every element's step by the same wrong factor. It is invisible to an absolute tolerance on the parameters (the update is `lr`-sized). Metal's `pow` under fast math was not measured.

**Fix direction.** Compute both scales once on the host (in f64, then narrow) and pass them in the uniform in place of `t`; the padding slot and `t` give the two fields. That also removes a `pow` pair per element.

## Panics where the CPU path works

Both are loud (wgpu's uncaptured-error handler panics), so they are GPU gaps, not wrong answers. They matter under the device policy rule that a GPU path never refuses what the CPU path handles (`.claude/rules/gpu.md`), and because the panic text is wgpu's, not Volta's.

### G5. Grid kernels can dispatch more than 65535 workgroups in a dimension

`dispatch_flat` spreads a 1-D launch over two dimensions past 65535 groups. The `Groups::Grid` kernels have no such handling and no check:

| Kernel | Grid | Fails when |
| --- | --- | --- |
| `igemm` | `(16, 16, ceil(out_ch/16) * ceil(B*H_out*W_out/16))` | tiles > 65535, about `out_ch * B*H_out*W_out > 16.7M` |
| `direct_conv` | `(ceil(out_ch/16), ceil(B*H_out*W_out/16), 1)` | `B*H_out*W_out > 1,048,560` |
| `conv_backward_input`, `igemm_backward_input` | `(ceil(in_ch/16), ceil(B*H*W/16), 1)` | `B*H*W > 1,048,560` |
| `col2im` | `(ceil(C/16), ceil(B*H*W/16), 1)` | `B*H*W > 1,048,560` |
| `matmul`, `matmul_backward_a/b` | `(ceil(cols/16), ceil(rows/16), 1)` | an output dimension > 1,048,560 |

Confirmed on lavapipe for `direct_conv` (`[1,1,1025,1024]`, 1x1 kernel), `igemm` (same input) and `matmul` (`[1,1] @ [1,1048577]`):

```
wgpu error: Validation Error ... In a dispatch command, indirect:false
  Each current dispatch group size dimension ([16, 16, 65600]) must be less or equal to 65535
```

The other rows are by reading (same pattern, not run). The `igemm` row is the reachable one: on a GPU, auto-selection picks `IGEMM` above 1M im2col elements, and a batch of 256 at 64 channels and 32x32 (16.8M outputs) or a batch of 32 at 64 channels and 112x112 crosses the limit. The CPU handles all of these.

**Fix direction.** Either fold the long axis into a flat index the way `dispatch_flat` and `flat_global_id` do, or split the launch into several dispatches with a base offset in the uniform. Until then a `pack`-style check with Volta's own message would at least name the kernel and the limit.

### G6. Tensors over 2^25 f32 elements cannot be bound

`GpuContext::new_async` requests `required_limits: wgpu::Limits::default()` (`context.rs:379`): `max_storage_buffer_binding_size` 128 MiB and `max_buffer_size` 256 MiB, whatever the adapter offers. The comment a few lines above (`apply_limit_buckets: false`, "we want the adapter's real ones") describes adapter selection, not the device, which gets the defaults. Kernels bind whole buffers (`as_entire_binding`), and the pool rounds every allocation up to a power of two (`BufferPool::allocation_size`), so a tensor of 2^25 + 1 elements allocates 256 MiB and binds all of it.

Confirmed on lavapipe, `sum` of 2^25 + 1 ones:

```
wgpu error: Validation Error ... In Device::create_bind_group, label = 'Reduce'
  Buffer binding 0 range 268435456 exceeds `max_*_buffer_binding_size` limit 134217728
```

So the largest GPU tensor is exactly 2^25 f32 elements (2^26 for the 16-bit types), a tensor between 2^24 + 1 and 2^25 elements costs twice its size in GPU memory, and one over 2^26 elements would fail at `create_buffer`. None of this is checked or documented at `to_device`. Related: #247 (limit buckets for WASM).

**Fix direction.** Request the adapter's limits for the two buffer sizes (clamped to what Volta wants), bind `0..gpu_bytes(len)` instead of the whole bucket (which also retires G13), and validate the element count at upload with a `VoltaError`.

## Performance traps

None of these changes a result. They are here because each is quadratic or a constant factor in the hundreds, on a path that is the default.

### G7. `igemm.wgsl` computes each output up to 256 times

The kernel is `@workgroup_size(16, 16, 1)` and the host dispatches `Groups::Grid(oc_tile, s_tile, tiles)` = `(16, 16, tiles)` **workgroups** (`kernels.rs:1917`). So within one `z` layer `global_id.x` and `global_id.y` each run over 0..256, not 0..16, and the shader maps them with `actual_oc = oc_tile_idx * 16 + global_id.x` and `actual_spatial = spatial_tile_idx * 16 + global_id.y`. Each layer therefore covers a 256 x 256 block that overlaps the next 15 tiles in each direction: every output element is computed and stored by up to `min(16, oc tiles) * min(16, spatial tiles)` invocations, 256 for 256 or more output channels.

The values written are identical, so the result is right, but (a) concurrent writes to one location are a data race under WGSL, which makes the stored value formally indeterminate, and (b) the work is up to 256 times what `direct_conv` does for the same output. Measured on lavapipe, `[2,8,34,34]`, 16 output channels (the 16x case), 3x3: direct 5.2 ms, iGEMM 103 ms. This is the kernel auto-selection picks on a GPU above 1M im2col elements. There is no tiling benefit to lose: the shader's "tile" loop over input channels is an ordinary loop split in chunks.

**Fix direction.** Dispatch `(1, 1, tiles)`, or drop the tile indirection and use `direct_conv`'s grid. The second also moves `igemm` off the `z` axis, where G5 bites first.

### G8. `expand_backward` is O(input x output)

Each thread owns one input element and loops over **every** element of the expanded gradient, decoding its coordinates, to find the ones that map to it (`movement_backward.wgsl:232`). Measured on lavapipe, `[r,1] -> [r,c]`, `sum`, backward:

| Shape | Time |
| --- | --- |
| `[256,1] -> [256,64]` | 0.14 s |
| `[1024,1] -> [1024,128]` | 1.2 s |
| `[2048,1] -> [2048,256]` | 9.0 s |
| `[4096,1] -> [4096,256]` | 36 s |

`Conv2d` with a bias runs this on every backward (the bias is expanded to the output shape because GPU binary ops do not broadcast). The loop only needs the broadcast axes: for each input element, iterate the product of the expanded dims and compute the source index directly. #81 records the same shape of cost for the (unreachable) binary broadcast backward.

### G9. `col2im` scans every output position per input element

`col2im.wgsl:45` loops `oh` over `0..h_out` and `ow` over `0..w_out` and tests whether the element falls in that window. Only `ceil(kernel/stride)` positions per axis can match, and their range follows from `h`, `kernel_h` and `stride_h`. By reading; cost is `H_out * W_out` iterations per input element instead of at most `kernel_h * kernel_w`. `conv_backward_input.wgsl` already uses the direct form.

### G10. Every kernel output is zero-filled from the host first

`GpuBuffer::zeros_of` (`buffer.rs:162`) allocates a host `vec![0u8; bytes]` and uploads it with `write_buffer`, and every kernel wrapper allocates its result that way. All kernels except the two in G1 write every element of their result, so the fill is a host allocation plus a full-size upload per op whose bytes are then overwritten. It does not make the buffer clean either: a pooled buffer's slack past `len` keeps its old contents (G13). By reading; not measured. A `GpuBuffer::uninit_of` for fully-written outputs is the direction, once G1 no longer leans on the zeros.

## Latent: the unreachable broadcast backward (#60)

### G11. Three more defects behind the bind-group panic

`binary_backward_broadcast_safe` is dead code (`#[expect(dead_code)]`, `gpu_ops.rs:425`) because its bind group does not match the auto layout (#60). By reading only, since it cannot run: fixing the bind group alone would turn the panic into wrong gradients.

1. **Shape alignment.** The host right-aligns each shape into four slots with leading 1s (`kernels.rs:976`: `[3,4]` becomes `[1,1,3,4]`) and passes the true rank. The shader indexes the first `rank` entries (`idx_to_coords(out_idx, params.out_shape, params.out_rank)` reads `shape[0..rank]`, and `pad_shape` copies `shape[i]` for `i < rank`). For any rank below 4 it decodes against the leading 1s, so every output maps to index 0. The size products come out right, which is why nothing would look wrong at dispatch.
2. **Indices stored as floats in the data dtype.** Pass 1 stores `f32(target_idx_a)` into `temp_buffer`, which is `array<ELEM>` in the gradient's dtype, and pass 2 reads it back with `u32(ld_temp_buffer(..))`. Integers are exact up to 2^24 in f32, 2048 in f16 and 256 in bf16. A half-precision operand with more than 2048 (or 256) elements would have neighbouring targets merged. The indices need a `u32` buffer.
3. **Odd split offset.** The two gradients are packed into one buffer and split with `copy_region(a.len(), b.len())`, which panics for a 16-bit dtype when `a.len()` is odd (`buffer.rs:326`).

Pass 2 is also O(targets x outputs), which #81 already records.

## Low severity and Metal-only

### G12. `exp2` is `pow(2.0, x)`

`unary.wgsl:113` and `unary_backward.wgsl` (`exp2_backward`) use `pow(2.0, x)`; WGSL has `exp2`. On lavapipe `exp2(0.5)` is 2 ulp from the CPU's (1.4142138 vs 1.4142135) and `exp2(-140)` is 0 where the CPU returns the subnormal 7.17e-43; integer exponents from -126 to 127 were exact. `pow` on Metal under fast math was not measured.

### G13. `arrayLength` bounds include the pool's slack

Pooled buffers are power-of-two buckets, so `arrayLength` can be up to twice the tensor. `reduce.wgsl`, `muon.wgsl`, `cast.wgsl`, `compare.wgsl`, `where_select.wgsl` and `amp_unscale.wgsl` bound by a count in the uniform and say why. `elementwise.wgsl`, `unary.wgsl`, `unary_backward.wgsl`, `binary_backward.wgsl` and `optimizer_step.wgsl` bound by `len_*()`. The slack is never read as data, so this is harmless today: the cost is up to 2x wasted arithmetic on stale bytes (including an optimizer "update" of garbage). It becomes a bug the day a kernel reduces over, or flags on, its whole binding. Binding `0..gpu_bytes(len)` (G6) fixes every kernel at once.

### G14. Fast-math exposure to check on Metal

PR #284 and S24 showed Metal reassociating and assuming finite values. These expressions have the same shape and were only run on lavapipe, where they agree with the CPU:

- `sigmoid` and `sigmoid_backward`: `1.0 / (1.0 + exp(-x))`. For `x < -88.7`, `exp(-x)` is inf; S24's `tanh` returned NaN from the same `inf` operand. Lavapipe returns 0 for -89, -104, -1000 and -inf.
- `erf_approx` for `|x| >= 1` and `erf_backward`: `exp(-a * a)` with `a * a` overflowing to inf for `|x| > 1.8e19`.
- `max_backward` (`reduce_backward.wgsl`): `v == m` when the maximum is +inf.
- `compare.wgsl`: NaN is handled from the bits, infinities are left to the float comparison.
- Native-f16 stores (`Elem::F16`): `f16(f16_bits_to_f32(bits))` converts an f32 inf or NaN to f16. WGSL lets an implementation assume neither occurs. `amp_unscale`'s `found` flag for an f16 gradient depends on that inf surviving the store and the `f32(..)` load.
- `f16_bits_to_f32`: `select(mag, -mag, sign != 0u)` for a negative zero or negative subnormal half.
- `over_square`: `frexp` of a subnormal divisor (Metal may flush it to zero first).

A probe per bullet, CPU against GPU at the extreme inputs, on the Mac, settles each.

### G15. Comments that no longer describe the code

- `ops/unary.rs:20-21` ("Same order as the `recip_backward` kernel") and the `Div` doc in `ops/binary.rs:35-38` ("the GPU kernels' order"): since PR #284 the kernels use `over_square` (`frexp`/`ldexp`), not `(1/y) * (1/y)`. The CPU and GPU now use different formulas for the same gradient, which the comments deny.
- `context.rs:360` (see G6): the comment on `apply_limit_buckets` reads as if the device ran with the adapter's limits.
- `kernels.rs` doc comments for `im2col` and `col2im` list eleven arguments that are now one `spec`.
- `binary_backward.wgsl:15` calls the same-shape entry points "Legacy ... kept for backward compatibility"; they are the only reachable binary backward.

### G16. Conv kernels rely on `u32` wraparound for padding

`direct_conv.wgsl:55` and `igemm.wgsl:102` compute `oh * stride_h + kh - pad_h` in `u32` and then test `h_in >= 0u && h_in < height`; the first test is always true and the second rejects the wrapped value. `conv_backward_input.wgsl:45` computes `ih + pad_h - kh` the same way and relies on the wrapped value failing either the stride test or `oh < h_out`. Checked by hand: correct for every size that fits a `u32` (a wrapped value divided by a stride below 2^31 is still far above any `h_out`). `conv_backward_weight.wgsl` does the same arithmetic in `i32` and says so. Hygiene only: write the condition as `kh <= ih + pad_h` so the intent survives a refactor.

### G17. Handed to the numerical-stability audit

Not GPU defects (both backends agree), noticed while comparing:

- SGD with momentum folds the learning rate into the velocity (`v = mu*v - lr*g; p += v`) on both backends. PyTorch keeps `v = mu*v + g; p -= lr*v`. The two differ as soon as the learning rate changes between steps.
- Adam state in f16: `(1 - beta2) * g^2` underflows f16 for `|g|` below about 8e-3 at the default `beta2`, so `v` stays 0 and the step is `lr * m_hat / eps`.
- The CPU half of G3.

## Checked and found correct

So a later reader need not redo them:

- **Template conversions** (`f32_to_f16_bits`, `f16_bits_to_f32`, `f32_to_bf16_bits`): the subnormal range, the tie at 2^-25, the carry into the exponent, and the overflow threshold at 65520 were worked by hand.
- **Packed 16-bit stores**: `atomicAnd` then `atomicOr` per half is safe for the kernels that exist (no kernel reads an element another thread writes); G7's duplicate writers store equal values, and any interleaving of two and/or pairs ends with the value set.
- **`flat_global_id` / `dispatch_flat`**: the 2-D spread reconstructs a unique flat index and trailing invocations fail their bounds check.
- **`reduce.wgsl`**: group indexing under a 2-D launch, the per-pass `base < n` guard, the shared-memory tree (no thread reads a slot written in the same round), the `combine` rule for max (identity, NaN, ties), and the mean divisor in the last pass only.
- **`matmul.wgsl`, `matmul_backward.wgsl`**: tile loads and the transposed reads in both backward entries, zero fill past the edges, uniform barrier control flow.
- **`im2col`, `direct_conv`, `conv_backward_input/weight`** (and the two iGEMM backward files, which differ from them only in comments and entry names): index arithmetic against the CPU formulas.
- **`cast`, `compare`, `where_select`, `amp_unscale`, `muon`, `reduce_backward`, `unary`, `unary_backward`, `binary_backward`**: element-for-element against the CPU table; the only differences are G2, G4, G12 and the documented `over_square`.
- **`pad`**: the `u32` underflow of `new_coords - left` lands out of range and takes the fill value; the 16-bit packing of dims 2 and 3 is guarded on the host.

## Not covered

- Metal, in every respect (precision, fast math, limits, driver). See "What ran".
- The packed-f16 path on a device without `SHADER_F16`: lavapipe has native f16, so F16 probes ran natively and only BF16 ran packed.
- `monitor.rs`, `system_monitor.rs`, `early_warning.rs`, `staging_pool.rs` beyond what `download` uses.
- Buffer aliasing in the in-place kernels (the unsharing write path) and the `unsafe`/`bytemuck` surface, which the soundness audit covers.
- Races between test threads (#244 class): nothing new was looked for.

## Appendix: probes

Run as a scratch integration test (`tests/zz.rs`, `#![cfg(feature = "gpu")]`) with `VOLTA_REQUIRE_GPU=1 cargo test --test zz -- --nocapture --test-threads=1`. Helpers:

```rust
use volta::{Adam, Conv2d, ConvAlgo, Device, RawTensor, Tensor};

fn on_gpu(data: Vec<f32>, shape: &[usize], rg: bool) -> Tensor {
    let t = RawTensor::new(data, shape, false).to_device(Device::gpu().expect("gpu"));
    t.set_requires_grad(rg);
    t
}
fn ramp(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i % 97) as f32 * 0.25 + 1.0).collect()
}
```

G1 (run once with `on_gpu`, once with `RawTensor::new(.., true)`, and compare `x.grad()`):

```rust
let x = on_gpu(ramp(1000), &[1000], true);
let w = on_gpu((0..100).map(|i| i as f32 + 1.0).collect(), &[100], false);
x.shrink(&[(900, 1000)]).elem_mul(&w).sum().into_loss().unwrap().backward();

let x = on_gpu(ramp(1024), &[1024], true);
let w = on_gpu((0..512).map(|i| i as f32 + 1.0).collect(), &[512], false);
x.stride_op(&[2]).elem_mul(&w).sum().into_loss().unwrap().backward();
```

G2:

```rust
let a = vec![1.0e10, 67108864.0, 1000.3, -7.5, 5.0, -5.0, 1.0e20, 123456.79, 0.3];
let b = vec![3.0, 7.0, 0.1, 2.0, f32::INFINITY, f32::INFINITY, 3.0, 0.001, 0.1];
let gpu = on_gpu(a.clone(), &[9], false).modulo(&on_gpu(b.clone(), &[9], false)).data();
let cpu = RawTensor::new(a, &[9], false).modulo(&RawTensor::new(b, &[9], false)).data();
```

G3:

```rust
let x = on_gpu(vec![1.0], &[1], true);
x.expand(&[1 << 25]).sum().into_loss().unwrap().backward(); // x.grad() == [16777216.0]
```

G4 (for `steps` in 1, 2, 10; compare `p.data()` minus the initial values, CPU against GPU):

```rust
let p = on_gpu(vec![1.0, -2.0, 0.5, 3.0], &[4], true);
let mut opt = Adam::new(vec![p.clone()], 1e-3, (0.9, 0.999), 1e-8, 0.0);
for _ in 0..steps {
    opt.zero_grad();
    p.elem_mul(&p).sum().into_loss().unwrap().backward();
    opt.step();
}
```

G5, G7 (`algo` is `ConvAlgo::Direct` or `ConvAlgo::IGEMM`; `[1,1,1025,1024]` with `out_ch = 1, k = 1` panics, `[2,8,34,34]` with `out_ch = 16, k = 3` is the timing case), and the matmul case:

```rust
let x = on_gpu(ramp(n), &shape, false);
let conv = Conv2d::new_on_device(shape[1], out_ch, k, 1, 0, false, Device::gpu().unwrap());
conv.set_algo(algo);
let y = conv.forward(&x).data();

on_gpu(vec![2.0], &[1, 1], false).matmul(&on_gpu(vec![1.0; 1_048_577], &[1, 1_048_577], false));
```

G6:

```rust
let n = (1usize << 25) + 1;
on_gpu(vec![1.0; n], &[n], false).sum().item();
```

G8:

```rust
let x = on_gpu(ramp(r), &[r, 1], true);
x.expand(&[r, c]).sum().into_loss().unwrap().backward();
let _ = x.grad();
```

G12:

```rust
on_gpu(vec![0.5, -140.0], &[2], false).exp2().data();
```

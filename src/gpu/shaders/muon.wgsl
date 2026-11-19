// Muon optimizer kernels: momentum, the normalization, and the Newton-Schulz
// building blocks that `GpuKernels::muon_*` composes with `matmul` and `permute`.
//
// Every entry point reads its element count from `params.n`, never from `arrayLength`:
// pooled buffers are allocated in power-of-2 buckets, so a binding can be longer than
// the tensor it holds.
//
// The four buffer slots are shared by all entry points; each entry binds only the slots
// it uses (the pipelines use an auto layout):
//   momentum:    0 = velocity (read/write), 1 = grad, 2 = update (out)
//   scale:       0 = input, 1 = largest magnitude (one element, from `reduce.wgsl`'s
//                `absmax_first`), 2 = output
//   norm:        0 = largest magnitude, 1 = sum of squares of the scaled input (one
//                element, from `sumsq_first`), 2 = the Frobenius norm (one element)
//   normalize:   0 = input, 1 = the Frobenius norm (one element, from `norm`), 2 = output
//   axpby:       0 = x, 1 = y, 2 = output

struct MuonParams {
    n: u32,        // element count
    flag: u32,     // momentum: 1 = Nesterov
    alpha: f32,    // momentum: coefficient; normalize: epsilon; axpby: scale of x
    beta: f32,     // axpby: scale of y
}

@group(0) @binding(0) var<storage, read_write> buf0: array<ELEM>;
@group(0) @binding(1) var<storage, read_write> buf1: array<ELEM>;
@group(0) @binding(2) var<storage, read_write> buf2: array<ELEM>;
@group(0) @binding(3) var<uniform> params: MuonParams;

// v <- alpha * v + g;  update = Nesterov ? alpha * v + g : v
@compute @workgroup_size(256)
fn momentum(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx >= params.n) {
        return;
    }
    let v = params.alpha * ld_buf0(idx) + ld_buf1(idx);
    st_buf0(idx, v);
    if (params.flag == 1u) {
        st_buf2(idx, params.alpha * v + ld_buf1(idx));
    } else {
        st_buf2(idx, v);
    }
}

// What the Frobenius norm divides by before squaring, as the CPU's `frobenius_norm`
// (#373): the largest magnitude `m`, so no square overflows, or 1 when `m` is 0 or inf
// (then the plain sum of squares is the answer). Tested from the bits, since fast math
// may fold comparisons with infinities away; `m` is a magnitude, so its sign bit is 0.
fn norm_scale(m: f32) -> f32 {
    let bits = bitcast<u32>(m);
    if (bits == 0u || bits == 0x7f800000u) {
        return 1.0;
    }
    return m;
}

// buf2[i] = buf0[i] / norm_scale(buf1[0])
@compute @workgroup_size(256)
fn scale(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx >= params.n) {
        return;
    }
    st_buf2(idx, ld_buf0(idx) / norm_scale(ld_buf1(0)));
}

// buf2[0] = norm_scale(buf0[0]) * sqrt(buf1[0]), on one thread (`params.n` is 1; reading
// it also keeps the uniform in this entry point's auto layout)
@compute @workgroup_size(1)
fn norm(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= params.n) {
        return;
    }
    st_buf2(0u, norm_scale(ld_buf0(0)) * sqrt(ld_buf1(0)));
}

// buf2[i] = buf0[i] * (1 / (buf1[0] + alpha)), as the CPU's `x * (1 / (norm + eps))`
@compute @workgroup_size(256)
fn normalize(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx >= params.n) {
        return;
    }
    st_buf2(idx, ld_buf0(idx) * (1.0 / (ld_buf1(0) + params.alpha)));
}

// buf2[i] = alpha * buf0[i] + beta * buf1[i]
@compute @workgroup_size(256)
fn axpby(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx >= params.n) {
        return;
    }
    st_buf2(idx, params.alpha * ld_buf0(idx) + params.beta * ld_buf1(idx));
}

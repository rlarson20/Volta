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
//   normalize:   0 = input, 1 = sum of squares (one element, from `reduce.wgsl`'s
//                `sumsq_first`), 2 = output
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

// buf2[i] = buf0[i] / (sqrt(buf1[0]) + alpha)
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
    st_buf2(idx, ld_buf0(idx) * (1.0 / (sqrt(ld_buf1(0)) + params.alpha)));
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

// Muon optimizer kernels: momentum, the Frobenius norm, and the Newton-Schulz
// building blocks that `GpuKernels::muon_*` composes with `matmul` and `permute`.
//
// Every entry point reads its element count from `params.n`, never from `arrayLength`:
// pooled buffers are allocated in power-of-2 buckets, so a binding can be longer than
// the tensor it holds.
//
// The four buffer slots are shared by all entry points; each entry binds only the slots
// it uses (the pipelines use an auto layout):
//   momentum:    0 = velocity (read/write), 1 = grad, 2 = update (out)
//   sum_squares: 0 = input, 2 = result (one element)
//   normalize:   0 = input, 1 = sum of squares (one element), 2 = output
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

// result[0] = sum of buf0[i]^2. One workgroup: each thread sums a strided slice, then the
// partial sums are combined by a tree reduction in shared memory.
var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn sum_squares(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let tid = local_id.x;
    var acc: f32 = 0.0;
    for (var i: u32 = tid; i < params.n; i = i + 256u) {
        let v = ld_buf0(i);
        acc = acc + v * v;
    }
    partial[tid] = acc;
    workgroupBarrier();
    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            partial[tid] = partial[tid] + partial[tid + stride];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        st_buf2(0, partial[0]);
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

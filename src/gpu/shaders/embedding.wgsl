// `Embedding` forward (a row gather) and backward (a per-row sum), with the indices held
// on the host and uploaded as `u32` (they are data, not tensors: the GPU holds no integer
// dtype). Backward reads a CSR built on the host: row r's occurrences are
// `positions[offsets[r] .. offsets[r + 1]]`, in ascending order, so each weight-gradient
// element is summed in the CPU's order, with no atomics.
//
// Bindings: 0 = weight / out_grad (ELEM), 1 = out (ELEM), 2 = rows or positions (u32),
// 3 = offsets (u32, backward only), 4 = params.

struct Params {
    n: u32,   // indices
    dim: u32, // embedding dim
    vocab: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> src: array<ELEM>;
@group(0) @binding(1) var<storage, read_write> out: array<ELEM>;
@group(0) @binding(2) var<storage, read> idx: array<u32>;
@group(0) @binding(3) var<storage, read> offsets: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;

// out[i, d] = weight[rows[i], d]
@compute @workgroup_size(256)
fn embedding_forward(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let o = flat_global_id(gid, num_wg).x;
    if (o >= params.n * params.dim) {
        return;
    }
    let i = o / params.dim;
    let d = o % params.dim;
    st_out(o, ld_src(idx[i] * params.dim + d));
}

// grad_weight[r, d] = sum over occurrences p of row r of out_grad[p, d]
@compute @workgroup_size(256)
fn embedding_backward(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let o = flat_global_id(gid, num_wg).x;
    if (o >= params.vocab * params.dim) {
        return;
    }
    let r = o / params.dim;
    let d = o % params.dim;
    var acc = 0.0;
    for (var k = offsets[r]; k < offsets[r + 1u]; k = k + 1u) {
        acc = acc + ld_src(idx[k] * params.dim + d);
    }
    st_out(o, acc);
}

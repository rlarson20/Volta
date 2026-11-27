// A strided gather of up to rank 8: `result[i] = input[offset + sum_d coord_d(i) * stride_d]`,
// where `coord_d(i)` are the row-major coordinates of `i` in `shape`. One kernel covers
// every pure re-indexing: `permute` (permuted strides), `expand` and broadcasting (stride 0
// on a broadcast axis) and `transpose`, at any rank up to 8.
//
// Bindings: 0 = input (ELEM), 1 = result (ELEM), 2 = params.

struct Params {
    shape: array<vec4<u32>, 2>,   // result dims, padded with 1s past `rank`
    strides: array<vec4<u32>, 2>, // input element strides per result axis, 0 past `rank`
    rank: u32,
    offset: u32,
    n: u32,                       // result elements
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<ELEM>;
@group(0) @binding(1) var<storage, read_write> result: array<ELEM>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn gather(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let idx = flat_global_id(gid, num_wg).x;
    if (idx >= params.n) {
        return;
    }
    var rem = idx;
    var src = params.offset;
    var d = params.rank;
    loop {
        if (d == 0u) {
            break;
        }
        d = d - 1u;
        let size = params.shape[d / 4u][d % 4u];
        src = src + (rem % size) * params.strides[d / 4u][d % 4u];
        rem = rem / size;
    }
    st_result(idx, ld_input(src));
}

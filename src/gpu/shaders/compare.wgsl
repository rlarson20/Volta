// Elementwise comparisons of two equal-shape `ELEM` buffers into a Bool buffer (one u32 per
// element, 0 or 1). NaN is detected from the bits, not left to the float comparison, which
// WGSL lets an implementation assume never sees NaN: every comparison with a NaN is false,
// except `ne`, which is true (IEEE, PyTorch).

@group(0) @binding(0) var<storage, read> a: array<ELEM>;
@group(0) @binding(1) var<storage, read> b: array<ELEM>;
@group(0) @binding(2) var<storage, read_write> result: array<u32>;

struct CompareParams {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(3) var<uniform> params: CompareParams;

fn is_nan(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}

// Whether either operand of element `i` is NaN, with both operands loaded.
fn operands(i: u32) -> vec3<f32> {
    let x = ld_a(i);
    let y = ld_b(i);
    return vec3<f32>(x, y, select(0.0, 1.0, is_nan(x) || is_nan(y)));
}

@compute @workgroup_size(256)
fn lt(
    @builtin(global_invocation_id) gid_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let gid = flat_global_id(gid_2d, num_wg);
    let i = gid.x;
    if (i < params.n) { let o = operands(i); result[i] = select(0u, 1u, o.z == 0.0 && o.x < o.y); }
}

@compute @workgroup_size(256)
fn le(
    @builtin(global_invocation_id) gid_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let gid = flat_global_id(gid_2d, num_wg);
    let i = gid.x;
    if (i < params.n) { let o = operands(i); result[i] = select(0u, 1u, o.z == 0.0 && o.x <= o.y); }
}

@compute @workgroup_size(256)
fn gt(
    @builtin(global_invocation_id) gid_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let gid = flat_global_id(gid_2d, num_wg);
    let i = gid.x;
    if (i < params.n) { let o = operands(i); result[i] = select(0u, 1u, o.z == 0.0 && o.x > o.y); }
}

@compute @workgroup_size(256)
fn ge(
    @builtin(global_invocation_id) gid_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let gid = flat_global_id(gid_2d, num_wg);
    let i = gid.x;
    if (i < params.n) { let o = operands(i); result[i] = select(0u, 1u, o.z == 0.0 && o.x >= o.y); }
}

@compute @workgroup_size(256)
fn eq(
    @builtin(global_invocation_id) gid_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let gid = flat_global_id(gid_2d, num_wg);
    let i = gid.x;
    if (i < params.n) { let o = operands(i); result[i] = select(0u, 1u, o.z == 0.0 && o.x == o.y); }
}

@compute @workgroup_size(256)
fn ne(
    @builtin(global_invocation_id) gid_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let gid = flat_global_id(gid_2d, num_wg);
    let i = gid.x;
    if (i < params.n) { let o = operands(i); result[i] = select(0u, 1u, o.z != 0.0 || o.x != o.y); }
}

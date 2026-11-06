// `where(cond, x, y)` for equal shapes: a Bool condition (one u32 per element) selects
// between two `ELEM` branches. The selected value is copied through f32, which is exact,
// and stored back in the branches' type.

@group(0) @binding(0) var<storage, read> cond: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<ELEM>;
@group(0) @binding(2) var<storage, read> y: array<ELEM>;
@group(0) @binding(3) var<storage, read_write> result: array<ELEM>;

struct WhereParams {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(4) var<uniform> params: WhereParams;

@compute @workgroup_size(256)
fn where_select(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.n) {
        st_result(i, select(ld_y(i), ld_x(i), cond[i] != 0u));
    }
}

// Element-wise binary operations
// Each thread processes one element

@group(0) @binding(0) var<storage, read> a: array<ELEM>;
@group(0) @binding(1) var<storage, read> b: array<ELEM>;
@group(0) @binding(2) var<storage, read_write> result: array<ELEM>;

// Workgroup size of 256 is a common choice for compute shaders
// It's a good balance between occupancy and register usage
@compute @workgroup_size(256)
fn add(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_a()) {
        st_result(idx, ld_a(idx) + ld_b(idx));
    }
}

@compute @workgroup_size(256)
fn sub(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_a()) {
        st_result(idx, ld_a(idx) - ld_b(idx));
    }
}

@compute @workgroup_size(256)
fn mul(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_a()) {
        st_result(idx, ld_a(idx) * ld_b(idx));
    }
}

@compute @workgroup_size(256)
fn div(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_a()) {
        st_result(idx, ld_a(idx) / ld_b(idx));
    }
}

@compute @workgroup_size(256)
fn max_elem(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_a()) {
        // torch.maximum: NaN if either operand is NaN (WGSL's max may return the other).
        let a = ld_a(idx);
        let b = ld_b(idx);
        st_result(idx, select(select(max(a, b), b, is_nan(b)), a, is_nan(a)));
    }
}

@compute @workgroup_size(256)
fn mod_op(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_a()) {
        let b = ld_b(idx);
        let r = ld_a(idx) % b;
        // Floored, as torch.remainder: the truncated remainder takes b's sign.
        st_result(idx, select(r, r + b, r != 0.0 && ((r < 0.0) != (b < 0.0))));
    }
}

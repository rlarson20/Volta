// Element-wise binary operations
// Each thread processes one element

@group(0) @binding(0) var<storage, read> a: array<ELEM>;
@group(0) @binding(1) var<storage, read> b: array<ELEM>;
@group(0) @binding(2) var<storage, read_write> result: array<ELEM>;

// Workgroup size of 256 is a common choice for compute shaders
// It's a good balance between occupancy and register usage
@compute @workgroup_size(256)
fn add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < len_a()) {
        st_result(idx, ld_a(idx) + ld_b(idx));
    }
}

@compute @workgroup_size(256)
fn sub(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < len_a()) {
        st_result(idx, ld_a(idx) - ld_b(idx));
    }
}

@compute @workgroup_size(256)
fn mul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < len_a()) {
        st_result(idx, ld_a(idx) * ld_b(idx));
    }
}

@compute @workgroup_size(256)
fn div(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < len_a()) {
        st_result(idx, ld_a(idx) / ld_b(idx));
    }
}

@compute @workgroup_size(256)
fn max_elem(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < len_a()) {
        st_result(idx, max(ld_a(idx), ld_b(idx)));
    }
}

@compute @workgroup_size(256)
fn mod_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < len_a()) {
        st_result(idx, ld_a(idx) - ld_b(idx) * floor(ld_a(idx) / ld_b(idx)));
    }
}

@compute @workgroup_size(256)
fn cmplt(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < len_a()) {
        st_result(idx, select(0.0, 1.0, ld_a(idx) < ld_b(idx)));
    }
}

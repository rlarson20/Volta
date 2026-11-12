// Binary backward operations for same-shape inputs: one thread per element, direct
// assignment (no atomics). Broadcasting backward lives in binary_backward_safe.wgsl.

@group(0) @binding(0) var<storage, read> out_grad: array<ELEM>;
@group(0) @binding(1) var<storage, read> a: array<ELEM>;  // first input
@group(0) @binding(2) var<storage, read> b: array<ELEM>;  // second input
@group(0) @binding(3) var<storage, read_write> result_grad: array<ELEM>;

// Helper to force use of all bindings (prevents wgpu from optimizing them out)
fn force_use_bindings(x: f32, y: f32) {
    _ = x;
    _ = y;
}

// ============ Legacy entry points (no broadcasting support) ============
// These are kept for backward compatibility with existing code
// These use simple direct assignment (no atomics needed for same-shape case)

// For add: d(a+b)/da = 1
@compute @workgroup_size(256)
fn add_backward_a(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_out_grad()) {
        // Force use of a and b to ensure all bindings are included in layout
        force_use_bindings(ld_a(idx), ld_b(idx));
        st_result_grad(idx, ld_out_grad(idx));
    }
}

// For add: d(a+b)/db = 1
@compute @workgroup_size(256)
fn add_backward_b(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_out_grad()) {
        force_use_bindings(ld_a(idx), ld_b(idx));
        st_result_grad(idx, ld_out_grad(idx));
    }
}

// For sub: d(a-b)/da = 1
@compute @workgroup_size(256)
fn sub_backward_a(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_out_grad()) {
        force_use_bindings(ld_a(idx), ld_b(idx));
        st_result_grad(idx, ld_out_grad(idx));
    }
}

// For sub: d(a-b)/db = -1
@compute @workgroup_size(256)
fn sub_backward_b(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_out_grad()) {
        force_use_bindings(ld_a(idx), ld_b(idx));
        st_result_grad(idx, -ld_out_grad(idx));
    }
}

// For mul: d(a*b)/da = b
@compute @workgroup_size(256)
fn mul_backward_a(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_out_grad()) {
        force_use_bindings(ld_a(idx), ld_b(idx));
        st_result_grad(idx, ld_out_grad(idx) * ld_b(idx));
    }
}

// For mul: d(a*b)/db = a
@compute @workgroup_size(256)
fn mul_backward_b(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_out_grad()) {
        force_use_bindings(ld_a(idx), ld_b(idx));
        st_result_grad(idx, ld_out_grad(idx) * ld_a(idx));
    }
}

// For div: d(a/b)/da = 1/b
@compute @workgroup_size(256)
fn div_backward_a(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_out_grad()) {
        force_use_bindings(ld_a(idx), ld_b(idx));
        st_result_grad(idx, ld_out_grad(idx) / ld_b(idx));
    }
}

// For div: d(a/b)/db = -a/b²
@compute @workgroup_size(256)
fn div_backward_b(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_out_grad()) {
        let b_inv = 1.0 / ld_b(idx);
        st_result_grad(idx, -ld_out_grad(idx) * ld_a(idx) * b_inv * b_inv);
    }
}

// For max (`max_grad_a`/`max_grad_b`, torch.maximum): the larger operand gets the gradient,
// a tie splits it in half, and a NaN operand gives both sides all of it.
// d(max(a,b))/da:
@compute @workgroup_size(256)
fn max_backward_a(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_out_grad()) {
        st_result_grad(idx, max_grad_a(ld_out_grad(idx), ld_a(idx), ld_b(idx)));
    }
}

// d(max(a,b))/db:
@compute @workgroup_size(256)
fn max_backward_b(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_out_grad()) {
        st_result_grad(idx, max_grad_b(ld_out_grad(idx), ld_a(idx), ld_b(idx)));
    }
}

// For mod (floored): d(a mod b)/da = 1
@compute @workgroup_size(256)
fn mod_backward_a(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_out_grad()) {
        force_use_bindings(ld_a(idx), ld_b(idx));
        st_result_grad(idx, ld_out_grad(idx));
    }
}

// For mod (floored): d(a mod b)/db = -floor(a/b)
@compute @workgroup_size(256)
fn mod_backward_b(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_out_grad()) {
        st_result_grad(idx, -ld_out_grad(idx) * floor(ld_a(idx) / ld_b(idx)));
    }
}

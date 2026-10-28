// Binary backward operations for same-shape inputs: one thread per element, direct
// assignment (no atomics). Broadcasting backward lives in binary_backward_safe.wgsl.

@group(0) @binding(0) var<storage, read> out_grad: array<f32>;
@group(0) @binding(1) var<storage, read> a: array<f32>;  // first input
@group(0) @binding(2) var<storage, read> b: array<f32>;  // second input
@group(0) @binding(3) var<storage, read_write> result_grad: array<f32>;

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
fn add_backward_a(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&out_grad)) {
        // Force use of a and b to ensure all bindings are included in layout
        force_use_bindings(a[idx], b[idx]);
        result_grad[idx] = out_grad[idx];
    }
}

// For add: d(a+b)/db = 1
@compute @workgroup_size(256)
fn add_backward_b(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&out_grad)) {
        force_use_bindings(a[idx], b[idx]);
        result_grad[idx] = out_grad[idx];
    }
}

// For sub: d(a-b)/da = 1
@compute @workgroup_size(256)
fn sub_backward_a(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&out_grad)) {
        force_use_bindings(a[idx], b[idx]);
        result_grad[idx] = out_grad[idx];
    }
}

// For sub: d(a-b)/db = -1
@compute @workgroup_size(256)
fn sub_backward_b(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&out_grad)) {
        force_use_bindings(a[idx], b[idx]);
        result_grad[idx] = -out_grad[idx];
    }
}

// For mul: d(a*b)/da = b
@compute @workgroup_size(256)
fn mul_backward_a(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&out_grad)) {
        force_use_bindings(a[idx], b[idx]);
        result_grad[idx] = out_grad[idx] * b[idx];
    }
}

// For mul: d(a*b)/db = a
@compute @workgroup_size(256)
fn mul_backward_b(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&out_grad)) {
        force_use_bindings(a[idx], b[idx]);
        result_grad[idx] = out_grad[idx] * a[idx];
    }
}

// For div: d(a/b)/da = 1/b
@compute @workgroup_size(256)
fn div_backward_a(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&out_grad)) {
        force_use_bindings(a[idx], b[idx]);
        result_grad[idx] = out_grad[idx] / b[idx];
    }
}

// For div: d(a/b)/db = -a/b²
@compute @workgroup_size(256)
fn div_backward_b(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&out_grad)) {
        let b_inv = 1.0 / b[idx];
        result_grad[idx] = -out_grad[idx] * a[idx] * b_inv * b_inv;
    }
}

// For max: d(max(a,b))/da = 1 if a >= b else 0
@compute @workgroup_size(256)
fn max_backward_a(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&out_grad)) {
        let grad = select(0.0, out_grad[idx], a[idx] >= b[idx]);
        result_grad[idx] = grad;
    }
}

// For max: d(max(a,b))/db = 1 if b > a else 0
@compute @workgroup_size(256)
fn max_backward_b(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&out_grad)) {
        let grad = select(0.0, out_grad[idx], b[idx] > a[idx]);
        result_grad[idx] = grad;
    }
}

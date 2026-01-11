// Binary backward operations WITH broadcasting support
// Gradients are reduced over dimensions that were broadcast in forward pass

// Note: We use atomic<f32> to support both atomic operations (for broadcasting)
// and regular storage (for legacy same-shape operations)
@group(0) @binding(0) var<storage, read> out_grad: array<f32>;
@group(0) @binding(1) var<storage, read> a: array<f32>;  // first input
@group(0) @binding(2) var<storage, read> b: array<f32>;  // second input
@group(0) @binding(3) var<storage, read_write> result_grad: array<f32>;

// Shape information for broadcasting reduction
struct BinaryBackwardParams {
    out_shape: vec4<u32>,
    a_shape: vec4<u32>,
    b_shape: vec4<u32>,
    out_rank: u32,
    a_rank: u32,
    b_rank: u32,
    _padding: u32,
}

@group(0) @binding(4) var<uniform> params: BinaryBackwardParams;

// Compute strides for row-major layout
fn compute_strides(shape: vec4<u32>, rank: u32) -> vec4<u32> {
    var strides = vec4<u32>(1u, 1u, 1u, 1u);
    if (rank > 1u) {
        var i: u32 = rank - 2u;
        loop {
            strides[i] = strides[i + 1u] * shape[i + 1u];
            if (i == 0u) { break; }
            i = i - 1u;
        }
    }
    return strides;
}

// Convert linear index to multi-dimensional coordinates
fn idx_to_coords(idx: u32, shape: vec4<u32>, rank: u32) -> vec4<u32> {
    var coords = vec4<u32>(0u, 0u, 0u, 0u);
    var remaining = idx;
    var i: u32 = rank;
    loop {
        if (i == 0u) { break; }
        i = i - 1u;
        coords[i] = remaining % shape[i];
        remaining = remaining / shape[i];
    }
    return coords;
}

// Convert multi-dimensional coordinates to linear index
fn coords_to_idx(coords: vec4<u32>, strides: vec4<u32>, rank: u32) -> u32 {
    var idx: u32 = 0u;
    for (var i: u32 = 0u; i < rank; i = i + 1u) {
        idx = idx + coords[i] * strides[i];
    }
    return idx;
}

// Pad shape with leading 1s to match target rank
fn pad_shape(shape: vec4<u32>, rank: u32, target_rank: u32) -> vec4<u32> {
    var padded = vec4<u32>(1u, 1u, 1u, 1u);
    let offset = target_rank - rank;
    for (var i: u32 = 0u; i < rank; i = i + 1u) {
        padded[i + offset] = shape[i];
    }
    return padded;
}

// Broadcast reduction helper - reduces gradient over broadcasted dimensions
// Each thread in output grad contributes to the corresponding input gradient position
fn broadcast_reduce(global_id: vec3<u32>) {
    let out_idx = global_id.x;
    let out_size = params.out_shape[0u] * params.out_shape[1u] * params.out_shape[2u] * params.out_shape[3u];

    if (out_idx >= out_size) {
        return;
    }

    // Get output coordinates
    let out_coords = idx_to_coords(out_idx, params.out_shape, params.out_rank);
    let out_strides = compute_strides(params.out_shape, params.out_rank);

    // Read gradient value
    let grad_val = out_grad[out_idx];

    // ============ Reduce to a_grad ============
    // Pad a_shape to match output rank
    let a_padded = pad_shape(params.a_shape, params.a_rank, params.out_rank);

    // Compute a_coords by replacing broadcasted dimensions with 0
    var a_coords_padded = vec4<u32>(0u, 0u, 0u, 0u);
    for (var i: u32 = 0u; i < params.out_rank; i = i + 1u) {
        // If a was size 1 in this dim (broadcast), use 0
        // Otherwise use the output coordinate
        if (a_padded[i] == 1u) {
            a_coords_padded[i] = 0u;
        } else {
            a_coords_padded[i] = out_coords[i];
        }
    }

    // Convert to linear index in a_grad
    let a_strides = compute_strides(params.a_shape, params.a_rank);
    var a_coords = vec4<u32>(0u, 0u, 0u, 0u);
    let a_offset = params.out_rank - params.a_rank;
    for (var i: u32 = 0u; i < params.a_rank; i = i + 1u) {
        a_coords[i] = a_coords_padded[i + a_offset];
    }
    let a_idx = coords_to_idx(a_coords, a_strides, params.a_rank);

    // Atomic add to accumulate gradient (multiple output positions may map to same input)
    result_grad[a_idx] = result_grad[a_idx] + grad_val;

    // ============ Reduce to b_grad (same logic) ============
    let b_padded = pad_shape(params.b_shape, params.b_rank, params.out_rank);
    var b_coords_padded = vec4<u32>(0u, 0u, 0u, 0u);
    for (var i: u32 = 0u; i < params.out_rank; i = i + 1u) {
        if (b_padded[i] == 1u) {
            b_coords_padded[i] = 0u;
        } else {
            b_coords_padded[i] = out_coords[i];
        }
    }

    // For b_grad, we use the second half of result_grad array
    let b_strides = compute_strides(params.b_shape, params.b_rank);
    var b_coords = vec4<u32>(0u, 0u, 0u, 0u);
    let b_offset = params.out_rank - params.b_rank;
    for (var i: u32 = 0u; i < params.b_rank; i = i + 1u) {
        b_coords[i] = b_coords_padded[i + b_offset];
    }
    let b_idx = coords_to_idx(b_coords, b_strides, params.b_rank) + arrayLength(&out_grad);

    result_grad[b_idx] = result_grad[b_idx] + grad_val;
}

// ============ Operation-specific gradient computation ============
// These compute the per-element gradient before broadcasting reduction

// Helper to force use of all bindings (prevents wgpu from optimizing them out)
fn force_use_bindings(x: f32, y: f32) {
    _ = x;
    _ = y;
}

// For add: gradient is just 1.0, so we only need broadcast reduction
@compute @workgroup_size(256)
fn add_broadcast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Force use of a and b to ensure all bindings are included in layout
    force_use_bindings(a[0], b[0]);

    let out_idx = global_id.x;
    let out_size = params.out_shape[0u] * params.out_shape[1u] * params.out_shape[2u] * params.out_shape[3u];

    if (out_idx >= out_size) {
        return;
    }

    let out_coords = idx_to_coords(out_idx, params.out_shape, params.out_rank);
    let grad_val = out_grad[out_idx];

    // Reduce to a_grad
    let a_padded = pad_shape(params.a_shape, params.a_rank, params.out_rank);
    var a_coords_padded = vec4<u32>(0u, 0u, 0u, 0u);
    for (var i: u32 = 0u; i < params.out_rank; i = i + 1u) {
        a_coords_padded[i] = select(out_coords[i], 0u, a_padded[i] == 1u);
    }

    let a_strides = compute_strides(params.a_shape, params.a_rank);
    var a_coords = vec4<u32>(0u, 0u, 0u, 0u);
    let a_offset = params.out_rank - params.a_rank;
    for (var i: u32 = 0u; i < params.a_rank; i = i + 1u) {
        a_coords[i] = a_coords_padded[i + a_offset];
    }
    let a_idx = coords_to_idx(a_coords, a_strides, params.a_rank);
    result_grad[a_idx] = result_grad[a_idx] + grad_val;

    // Reduce to b_grad
    let b_padded = pad_shape(params.b_shape, params.b_rank, params.out_rank);
    var b_coords_padded = vec4<u32>(0u, 0u, 0u, 0u);
    for (var i: u32 = 0u; i < params.out_rank; i = i + 1u) {
        b_coords_padded[i] = select(out_coords[i], 0u, b_padded[i] == 1u);
    }

    let b_strides = compute_strides(params.b_shape, params.b_rank);
    var b_coords = vec4<u32>(0u, 0u, 0u, 0u);
    let b_offset = params.out_rank - params.b_rank;
    for (var i: u32 = 0u; i < params.b_rank; i = i + 1u) {
        b_coords[i] = b_coords_padded[i + b_offset];
    }
    let b_idx = coords_to_idx(b_coords, b_strides, params.b_rank) + arrayLength(&out_grad);
    result_grad[b_idx] = result_grad[b_idx] + grad_val;
}

// For sub: d(a-b)/da = 1, d(a-b)/db = -1
@compute @workgroup_size(256)
fn sub_broadcast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Force use of a and b
    force_use_bindings(a[0], b[0]);

    let out_idx = global_id.x;
    let out_size = params.out_shape[0u] * params.out_shape[1u] * params.out_shape[2u] * params.out_shape[3u];

    if (out_idx >= out_size) {
        return;
    }

    let out_coords = idx_to_coords(out_idx, params.out_shape, params.out_rank);
    let out_strides = compute_strides(params.out_shape, params.out_rank);
    let grad_val = out_grad[out_idx];

    // a_grad: +gradient
    let a_padded = pad_shape(params.a_shape, params.a_rank, params.out_rank);
    var a_coords_padded = vec4<u32>(0u, 0u, 0u, 0u);
    for (var i: u32 = 0u; i < params.out_rank; i = i + 1u) {
        a_coords_padded[i] = select(out_coords[i], 0u, a_padded[i] == 1u);
    }

    let a_strides = compute_strides(params.a_shape, params.a_rank);
    var a_coords = vec4<u32>(0u, 0u, 0u, 0u);
    let a_offset = params.out_rank - params.a_rank;
    for (var i: u32 = 0u; i < params.a_rank; i = i + 1u) {
        a_coords[i] = a_coords_padded[i + a_offset];
    }
    let a_idx = coords_to_idx(a_coords, a_strides, params.a_rank);
    result_grad[a_idx] = result_grad[a_idx] + grad_val;

    // b_grad: -gradient
    let b_padded = pad_shape(params.b_shape, params.b_rank, params.out_rank);
    var b_coords_padded = vec4<u32>(0u, 0u, 0u, 0u);
    for (var i: u32 = 0u; i < params.out_rank; i = i + 1u) {
        b_coords_padded[i] = select(out_coords[i], 0u, b_padded[i] == 1u);
    }

    let b_strides = compute_strides(params.b_shape, params.b_rank);
    var b_coords = vec4<u32>(0u, 0u, 0u, 0u);
    let b_offset = params.out_rank - params.b_rank;
    for (var i: u32 = 0u; i < params.b_rank; i = i + 1u) {
        b_coords[i] = b_coords_padded[i + b_offset];
    }
    let b_idx = coords_to_idx(b_coords, b_strides, params.b_rank) + arrayLength(&out_grad);
    result_grad[b_idx] = result_grad[b_idx] - grad_val;
}

// For mul: d(a*b)/da = b, d(a*b)/db = a
@compute @workgroup_size(256)
fn mul_broadcast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Force use of a and b
    force_use_bindings(a[0], b[0]);

    let out_idx = global_id.x;
    let out_size = params.out_shape[0u] * params.out_shape[1u] * params.out_shape[2u] * params.out_shape[3u];

    if (out_idx >= out_size) {
        return;
    }

    let out_coords = idx_to_coords(out_idx, params.out_shape, params.out_rank);
    let grad_val = out_grad[out_idx];

    // Get input values at this output position
    let a_padded = pad_shape(params.a_shape, params.a_rank, params.out_rank);
    let b_padded = pad_shape(params.b_shape, params.b_rank, params.out_rank);

    var a_coords_padded = vec4<u32>(0u, 0u, 0u, 0u);
    var b_coords_padded = vec4<u32>(0u, 0u, 0u, 0u);

    for (var i: u32 = 0u; i < params.out_rank; i = i + 1u) {
        a_coords_padded[i] = select(out_coords[i], 0u, a_padded[i] == 1u);
        b_coords_padded[i] = select(out_coords[i], 0u, b_padded[i] == 1u);
    }

    // Read input values
    let a_strides = compute_strides(params.a_shape, params.a_rank);
    let b_strides = compute_strides(params.b_shape, params.b_rank);

    var a_coords = vec4<u32>(0u, 0u, 0u, 0u);
    var b_coords = vec4<u32>(0u, 0u, 0u, 0u);
    let a_offset = params.out_rank - params.a_rank;
    let b_offset = params.out_rank - params.b_rank;

    for (var i: u32 = 0u; i < params.a_rank; i = i + 1u) {
        a_coords[i] = a_coords_padded[i + a_offset];
    }
    for (var i: u32 = 0u; i < params.b_rank; i = i + 1u) {
        b_coords[i] = b_coords_padded[i + b_offset];
    }

    let a_idx = coords_to_idx(a_coords, a_strides, params.a_rank);
    let b_idx = coords_to_idx(b_coords, b_strides, params.b_rank);

    let a_val = a[a_idx];
    let b_val = b[b_idx];

    // a_grad: gradient * b
    let a_result_idx = coords_to_idx(a_coords, a_strides, params.a_rank);
    result_grad[a_result_idx] = result_grad[a_result_idx] + grad_val * b_val;

    // b_grad: gradient * a
    let b_result_idx = coords_to_idx(b_coords, b_strides, params.b_rank) + arrayLength(&out_grad);
    result_grad[b_result_idx] = result_grad[b_result_idx] + grad_val * a_val;
}

// For div: d(a/b)/da = 1/b, d(a/b)/db = -a/b²
@compute @workgroup_size(256)
fn div_broadcast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Force use of a and b
    force_use_bindings(a[0], b[0]);

    let out_idx = global_id.x;
    let out_size = params.out_shape[0u] * params.out_shape[1u] * params.out_shape[2u] * params.out_shape[3u];

    if (out_idx >= out_size) {
        return;
    }

    let out_coords = idx_to_coords(out_idx, params.out_shape, params.out_rank);
    let grad_val = out_grad[out_idx];

    let a_padded = pad_shape(params.a_shape, params.a_rank, params.out_rank);
    let b_padded = pad_shape(params.b_shape, params.b_rank, params.out_rank);

    var a_coords_padded = vec4<u32>(0u, 0u, 0u, 0u);
    var b_coords_padded = vec4<u32>(0u, 0u, 0u, 0u);

    for (var i: u32 = 0u; i < params.out_rank; i = i + 1u) {
        a_coords_padded[i] = select(out_coords[i], 0u, a_padded[i] == 1u);
        b_coords_padded[i] = select(out_coords[i], 0u, b_padded[i] == 1u);
    }

    let a_strides = compute_strides(params.a_shape, params.a_rank);
    let b_strides = compute_strides(params.b_shape, params.b_rank);

    var a_coords = vec4<u32>(0u, 0u, 0u, 0u);
    var b_coords = vec4<u32>(0u, 0u, 0u, 0u);
    let a_offset = params.out_rank - params.a_rank;
    let b_offset = params.out_rank - params.b_rank;

    for (var i: u32 = 0u; i < params.a_rank; i = i + 1u) {
        a_coords[i] = a_coords_padded[i + a_offset];
    }
    for (var i: u32 = 0u; i < params.b_rank; i = i + 1u) {
        b_coords[i] = b_coords_padded[i + b_offset];
    }

    let a_idx = coords_to_idx(a_coords, a_strides, params.a_rank);
    let b_idx = coords_to_idx(b_coords, b_strides, params.b_rank);

    let a_val = a[a_idx];
    let b_val = b[b_idx];
    let b_inv = 1.0 / b_val;

    // a_grad: gradient / b
    let a_result_idx = coords_to_idx(a_coords, a_strides, params.a_rank);
    result_grad[a_result_idx] = result_grad[a_result_idx] + grad_val * b_inv;

    // b_grad: -gradient * a / b²
    let b_result_idx = coords_to_idx(b_coords, b_strides, params.b_rank) + arrayLength(&out_grad);
    result_grad[b_result_idx] = result_grad[b_result_idx] - grad_val * a_val * b_inv * b_inv;
}

// For max: d(max(a,b))/da = 1 if a >= b else 0, d(max(a,b))/db = 1 if b > a else 0
@compute @workgroup_size(256)
fn max_broadcast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Force use of a and b
    force_use_bindings(a[0], b[0]);

    let out_idx = global_id.x;
    let out_size = params.out_shape[0u] * params.out_shape[1u] * params.out_shape[2u] * params.out_shape[3u];

    if (out_idx >= out_size) {
        return;
    }

    let out_coords = idx_to_coords(out_idx, params.out_shape, params.out_rank);
    let grad_val = out_grad[out_idx];

    let a_padded = pad_shape(params.a_shape, params.a_rank, params.out_rank);
    let b_padded = pad_shape(params.b_shape, params.b_rank, params.out_rank);

    var a_coords_padded = vec4<u32>(0u, 0u, 0u, 0u);
    var b_coords_padded = vec4<u32>(0u, 0u, 0u, 0u);

    for (var i: u32 = 0u; i < params.out_rank; i = i + 1u) {
        a_coords_padded[i] = select(out_coords[i], 0u, a_padded[i] == 1u);
        b_coords_padded[i] = select(out_coords[i], 0u, b_padded[i] == 1u);
    }

    let a_strides = compute_strides(params.a_shape, params.a_rank);
    let b_strides = compute_strides(params.b_shape, params.b_rank);

    var a_coords = vec4<u32>(0u, 0u, 0u, 0u);
    var b_coords = vec4<u32>(0u, 0u, 0u, 0u);
    let a_offset = params.out_rank - params.a_rank;
    let b_offset = params.out_rank - params.b_rank;

    for (var i: u32 = 0u; i < params.a_rank; i = i + 1u) {
        a_coords[i] = a_coords_padded[i + a_offset];
    }
    for (var i: u32 = 0u; i < params.b_rank; i = i + 1u) {
        b_coords[i] = b_coords_padded[i + b_offset];
    }

    let a_idx = coords_to_idx(a_coords, a_strides, params.a_rank);
    let b_idx = coords_to_idx(b_coords, b_strides, params.b_rank);

    let a_val = a[a_idx];
    let b_val = b[b_idx];

    // a_grad: gradient if a >= b
    let a_result_idx = coords_to_idx(a_coords, a_strides, params.a_rank);
    let a_grad = select(0.0, grad_val, a_val >= b_val);
    result_grad[a_result_idx] = result_grad[a_result_idx] + a_grad;

    // b_grad: gradient if b > a
    let b_result_idx = coords_to_idx(b_coords, b_strides, params.b_rank) + arrayLength(&out_grad);
    let b_grad = select(0.0, grad_val, b_val > a_val);
    result_grad[b_result_idx] = result_grad[b_result_idx] + b_grad;
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
        result_grad[idx] = out_grad[idx] * b[idx];
    }
}

// For mul: d(a*b)/db = a
@compute @workgroup_size(256)
fn mul_backward_b(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&out_grad)) {
        result_grad[idx] = out_grad[idx] * a[idx];
    }
}

// For div: d(a/b)/da = 1/b
@compute @workgroup_size(256)
fn div_backward_a(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&out_grad)) {
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

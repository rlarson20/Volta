// Binary backward operations with RACE-FREE broadcasting support
//
// PROBLEM: When broadcasting gradients, multiple output positions may map to the same
// input gradient position, causing race conditions with concurrent writes:
//
//   result_grad[target] += grad_value  // RACE if multiple threads write to same target!
//
// SOLUTION: Two-pass algorithm avoids conflicts using workgroup-based reduction
//
// PASS 1 (Scatter):
//   Each thread computes (target_idx_a, target_idx_b, grad_value) triple
//   Writes to temporary buffer indexed by global_invocation_id
//   No conflicts: each thread has unique global_invocation_id
//
// PASS 2 (Reduce):
//   Each thread processes one target gradient position
//   Scans temporary buffer and sums matching contributions
//   Deterministic: only one thread writes to each position
//
// PERFORMANCE:
//   - Temporary buffer: O(out_size * 3) memory (stores idx_a, idx_b, val per output)
//   - Pass 1: O(out_size / 256) workgroups
//   - Pass 2: O(max(a_size, b_size) / 256) workgroups
//   - Total: 2x kernel launches, but race-free and portable

// Binding 0: Output gradient [out_size]
@group(0) @binding(0) var<storage, read> out_grad: array<f32>;

// Binding 1-2: Input tensors (for gradient computation)
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;

// Binding 3: Temporary buffer for pass 1 output [out_size * 3]
// Stores (target_idx_a, target_idx_b, grad_value) for each output position
@group(0) @binding(3) var<storage, read_write> temp_buffer: array<f32>;

// Binding 4: Final gradient buffers [a_size + b_size]
// First half: a_grad, Second half: b_grad
@group(0) @binding(4) var<storage, read_write> result_grad: array<f32>;

// Binding 5: Shape information
struct BinaryBackwardParams {
    out_shape: vec4<u32>,
    a_shape: vec4<u32>,
    b_shape: vec4<u32>,
    out_rank: u32,
    a_rank: u32,
    b_rank: u32,
    _padding: u32,
}

@group(0) @binding(5) var<uniform> params: BinaryBackwardParams;

// ============ Helper Functions ============

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

// ============ PASS 1: Scatter ============
// Each thread computes (target_idx_a, target_idx_b, grad_value) and stores in temp buffer

@compute @workgroup_size(256)
fn add_broadcast_pass1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_idx = global_id.x;
    let out_size = params.out_shape[0u] * params.out_shape[1u] * params.out_shape[2u] * params.out_shape[3u];

    if (out_idx >= out_size) {
        return;
    }

    // Get output coordinates
    let out_coords = idx_to_coords(out_idx, params.out_shape, params.out_rank);

    // For add: gradient is just 1.0 * out_grad
    let grad_val = out_grad[out_idx];

    // ============ Compute target_idx for a_grad ============
    let a_padded = pad_shape(params.a_shape, params.a_rank, params.out_rank);
    var a_coords_padded = vec4<u32>(0u, 0u, 0u, 0u);
    for (var i: u32 = 0u; i < params.out_rank; i = i + 1u) {
        if (a_padded[i] == 1u) {
            a_coords_padded[i] = 0u;
        } else {
            a_coords_padded[i] = out_coords[i];
        }
    }

    let a_strides = compute_strides(params.a_shape, params.a_rank);
    var a_coords = vec4<u32>(0u, 0u, 0u, 0u);
    let a_offset = params.out_rank - params.a_rank;
    for (var i: u32 = 0u; i < params.a_rank; i = i + 1u) {
        a_coords[i] = a_coords_padded[i + a_offset];
    }
    let target_idx_a = coords_to_idx(a_coords, a_strides, params.a_rank);

    // ============ Compute target_idx for b_grad ============
    let b_padded = pad_shape(params.b_shape, params.b_rank, params.out_rank);
    var b_coords_padded = vec4<u32>(0u, 0u, 0u, 0u);
    for (var i: u32 = 0u; i < params.out_rank; i = i + 1u) {
        if (b_padded[i] == 1u) {
            b_coords_padded[i] = 0u;
        } else {
            b_coords_padded[i] = out_coords[i];
        }
    }

    let b_strides = compute_strides(params.b_shape, params.b_rank);
    var b_coords = vec4<u32>(0u, 0u, 0u, 0u);
    let b_offset = params.out_rank - params.b_rank;
    for (var i: u32 = 0u; i < params.b_rank; i = i + 1u) {
        b_coords[i] = b_coords_padded[i + b_offset];
    }
    let target_idx_b = coords_to_idx(b_coords, b_strides, params.b_rank);

    // ============ Store in temporary buffer ============
    // temp_buffer layout: [target_a_0, target_b_0, val_0, target_a_1, target_b_1, val_1, ...]
    let temp_idx = out_idx * 3u;
    temp_buffer[temp_idx] = f32(target_idx_a);      // Store target index for a
    temp_buffer[temp_idx + 1u] = f32(target_idx_b); // Store target index for b
    temp_buffer[temp_idx + 2u] = grad_val;          // Store gradient value
}

// ============ PASS 2: Reduce ============
// Each thread processes one gradient position and sums all contributions

@compute @workgroup_size(256)
fn add_broadcast_pass2(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Process both a_grad and b_grad in one pass
    // First half of threads process a_grad, second half process b_grad
    let thread_idx = global_id.x;
    let a_size = params.a_shape[0u] * params.a_shape[1u] * params.a_shape[2u] * params.a_shape[3u];
    let b_size = params.b_shape[0u] * params.b_shape[1u] * params.b_shape[2u] * params.b_shape[3u];
    let max_size = max(a_size, b_size);
    let out_size = params.out_shape[0u] * params.out_shape[1u] * params.out_shape[2u] * params.out_shape[3u];

    if (thread_idx >= max_size) {
        return;
    }

    // ============ Reduce a_grad ============
    if (thread_idx < a_size) {
        var sum_a: f32 = 0.0;
        for (var i: u32 = 0u; i < out_size; i = i + 1u) {
            let temp_idx = i * 3u;
            let target_a = u32(temp_buffer[temp_idx]);
            if (target_a == thread_idx) {
                sum_a = sum_a + temp_buffer[temp_idx + 2u];
            }
        }
        result_grad[thread_idx] = sum_a;
    }

    // ============ Reduce b_grad ============
    if (thread_idx < b_size) {
        var sum_b: f32 = 0.0;
        for (var i: u32 = 0u; i < out_size; i = i + 1u) {
            let temp_idx = i * 3u;
            let target_b = u32(temp_buffer[temp_idx + 1u]);
            if (target_b == thread_idx) {
                sum_b = sum_b + temp_buffer[temp_idx + 2u];
            }
        }
        // b_grad is stored in second half of result_grad
        result_grad[a_size + thread_idx] = sum_b;
    }
}

// ============ Additional Operations (sub, mul, div, max) ============
// These follow the same pattern but compute grad_val differently

// Subtraction: a - b
// grad_a = +out_grad, grad_b = -out_grad
@compute @workgroup_size(256)
fn sub_broadcast_pass1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_idx = global_id.x;
    let out_size = params.out_shape[0u] * params.out_shape[1u] * params.out_shape[2u] * params.out_shape[3u];

    if (out_idx >= out_size) {
        return;
    }

    let out_coords = idx_to_coords(out_idx, params.out_shape, params.out_rank);
    let out_grad_val = out_grad[out_idx];

    // Compute target indices (same logic as add)
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
    let target_idx_a = coords_to_idx(a_coords, a_strides, params.a_rank);

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
    let target_idx_b = coords_to_idx(b_coords, b_strides, params.b_rank);

    let temp_idx = out_idx * 3u;
    temp_buffer[temp_idx] = f32(target_idx_a);
    temp_buffer[temp_idx + 1u] = f32(target_idx_b);
    temp_buffer[temp_idx + 2u] = out_grad_val;
}

@compute @workgroup_size(256)
fn sub_broadcast_pass2(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_idx = global_id.x;
    let a_size = params.a_shape[0u] * params.a_shape[1u] * params.a_shape[2u] * params.a_shape[3u];
    let b_size = params.b_shape[0u] * params.b_shape[1u] * params.b_shape[2u] * params.b_shape[3u];
    let max_size = max(a_size, b_size);
    let out_size = params.out_shape[0u] * params.out_shape[1u] * params.out_shape[2u] * params.out_shape[3u];

    if (thread_idx >= max_size) {
        return;
    }

    // a_grad gets +out_grad
    if (thread_idx < a_size) {
        var sum_a: f32 = 0.0;
        for (var i: u32 = 0u; i < out_size; i = i + 1u) {
            let temp_idx = i * 3u;
            let target_a = u32(temp_buffer[temp_idx]);
            if (target_a == thread_idx) {
                sum_a = sum_a + temp_buffer[temp_idx + 2u];
            }
        }
        result_grad[thread_idx] = sum_a;
    }

    // b_grad gets -out_grad
    if (thread_idx < b_size) {
        var sum_b: f32 = 0.0;
        for (var i: u32 = 0u; i < out_size; i = i + 1u) {
            let temp_idx = i * 3u;
            let target_b = u32(temp_buffer[temp_idx + 1u]);
            if (target_b == thread_idx) {
                sum_b = sum_b - temp_buffer[temp_idx + 2u];  // Negative for subtraction
            }
        }
        result_grad[a_size + thread_idx] = sum_b;
    }
}

// Multiplication: a * b
// grad_a = out_grad * b, grad_b = out_grad * a
@compute @workgroup_size(256)
fn mul_broadcast_pass1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_idx = global_id.x;
    let out_size = params.out_shape[0u] * params.out_shape[1u] * params.out_shape[2u] * params.out_shape[3u];

    if (out_idx >= out_size) {
        return;
    }

    let out_coords = idx_to_coords(out_idx, params.out_shape, params.out_rank);
    let out_grad_val = out_grad[out_idx];

    // Compute target indices
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
    let target_idx_a = coords_to_idx(a_coords, a_strides, params.a_rank);
    let a_val = a[target_idx_a];  // Read a value for gradient computation

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
    let target_idx_b = coords_to_idx(b_coords, b_strides, params.b_rank);
    let b_val = b[target_idx_b];  // Read b value for gradient computation

    let temp_idx = out_idx * 3u;
    temp_buffer[temp_idx] = f32(target_idx_a);
    temp_buffer[temp_idx + 1u] = f32(target_idx_b);
    temp_buffer[temp_idx + 2u] = out_grad_val;  // Will be multiplied in pass2
}

@compute @workgroup_size(256)
fn mul_broadcast_pass2(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_idx = global_id.x;
    let a_size = params.a_shape[0u] * params.a_shape[1u] * params.a_shape[2u] * params.a_shape[3u];
    let b_size = params.b_shape[0u] * params.b_shape[1u] * params.b_shape[2u] * params.b_shape[3u];
    let max_size = max(a_size, b_size);
    let out_size = params.out_shape[0u] * params.out_shape[1u] * params.out_shape[2u] * params.out_shape[3u];

    if (thread_idx >= max_size) {
        return;
    }

    // grad_a = out_grad * b
    if (thread_idx < a_size) {
        var sum_a: f32 = 0.0;
        for (var i: u32 = 0u; i < out_size; i = i + 1u) {
            let temp_idx = i * 3u;
            let target_a = u32(temp_buffer[temp_idx]);
            if (target_a == thread_idx) {
                let b_val = b[u32(temp_buffer[temp_idx + 1u])];  // Read corresponding b value
                sum_a = sum_a + temp_buffer[temp_idx + 2u] * b_val;
            }
        }
        result_grad[thread_idx] = sum_a;
    }

    // grad_b = out_grad * a
    if (thread_idx < b_size) {
        var sum_b: f32 = 0.0;
        for (var i: u32 = 0u; i < out_size; i = i + 1u) {
            let temp_idx = i * 3u;
            let target_b = u32(temp_buffer[temp_idx + 1u]);
            if (target_b == thread_idx) {
                let a_val = a[u32(temp_buffer[temp_idx])];  // Read corresponding a value
                sum_b = sum_b + temp_buffer[temp_idx + 2u] * a_val;
            }
        }
        result_grad[a_size + thread_idx] = sum_b;
    }
}

// Division: a / b
// grad_a = out_grad / b, grad_b = -out_grad * a / b^2
@compute @workgroup_size(256)
fn div_broadcast_pass1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_idx = global_id.x;
    let out_size = params.out_shape[0u] * params.out_shape[1u] * params.out_shape[2u] * params.out_shape[3u];

    if (out_idx >= out_size) {
        return;
    }

    let out_coords = idx_to_coords(out_idx, params.out_shape, params.out_rank);
    let out_grad_val = out_grad[out_idx];

    // Compute target indices and values
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
    let target_idx_a = coords_to_idx(a_coords, a_strides, params.a_rank);
    let a_val = a[target_idx_a];

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
    let target_idx_b = coords_to_idx(b_coords, b_strides, params.b_rank);
    let b_val = b[target_idx_b];

    let temp_idx = out_idx * 3u;
    temp_buffer[temp_idx] = f32(target_idx_a);
    temp_buffer[temp_idx + 1u] = f32(target_idx_b);
    temp_buffer[temp_idx + 2u] = out_grad_val;
}

@compute @workgroup_size(256)
fn div_broadcast_pass2(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_idx = global_id.x;
    let a_size = params.a_shape[0u] * params.a_shape[1u] * params.a_shape[2u] * params.a_shape[3u];
    let b_size = params.b_shape[0u] * params.b_shape[1u] * params.b_shape[2u] * params.b_shape[3u];
    let max_size = max(a_size, b_size);
    let out_size = params.out_shape[0u] * params.out_shape[1u] * params.out_shape[2u] * params.out_shape[3u];

    if (thread_idx >= max_size) {
        return;
    }

    // grad_a = out_grad / b
    if (thread_idx < a_size) {
        var sum_a: f32 = 0.0;
        for (var i: u32 = 0u; i < out_size; i = i + 1u) {
            let temp_idx = i * 3u;
            let target_a = u32(temp_buffer[temp_idx]);
            if (target_a == thread_idx) {
                let b_val = b[u32(temp_buffer[temp_idx + 1u])];
                sum_a = sum_a + temp_buffer[temp_idx + 2u] / b_val;
            }
        }
        result_grad[thread_idx] = sum_a;
    }

    // grad_b = -out_grad * a / b^2
    if (thread_idx < b_size) {
        var sum_b: f32 = 0.0;
        for (var i: u32 = 0u; i < out_size; i = i + 1u) {
            let temp_idx = i * 3u;
            let target_b = u32(temp_buffer[temp_idx + 1u]);
            if (target_b == thread_idx) {
                let a_val = a[u32(temp_buffer[temp_idx])];
                let b_val = b[target_b];
                sum_b = sum_b - temp_buffer[temp_idx + 2u] * a_val / (b_val * b_val);
            }
        }
        result_grad[a_size + thread_idx] = sum_b;
    }
}

// Maximum: max(a, b)
// grad_a = out_grad if a > b else 0
// grad_b = out_grad if b >= a else 0
@compute @workgroup_size(256)
fn max_broadcast_pass1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_idx = global_id.x;
    let out_size = params.out_shape[0u] * params.out_shape[1u] * params.out_shape[2u] * params.out_shape[3u];

    if (out_idx >= out_size) {
        return;
    }

    let out_coords = idx_to_coords(out_idx, params.out_shape, params.out_rank);
    let out_grad_val = out_grad[out_idx];

    // Compute target indices and values
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
    let target_idx_a = coords_to_idx(a_coords, a_strides, params.a_rank);
    let a_val = a[target_idx_a];

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
    let target_idx_b = coords_to_idx(b_coords, b_strides, params.b_rank);
    let b_val = b[target_idx_b];

    // For max, we need to know which was larger to determine gradient routing
    // Store: target_a, target_b, grad_val, a_val, b_val (5 floats)
    let temp_idx = out_idx * 5u;
    temp_buffer[temp_idx] = f32(target_idx_a);
    temp_buffer[temp_idx + 1u] = f32(target_idx_b);
    temp_buffer[temp_idx + 2u] = out_grad_val;
    temp_buffer[temp_idx + 3u] = a_val;
    temp_buffer[temp_idx + 4u] = b_val;
}

@compute @workgroup_size(256)
fn max_broadcast_pass2(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_idx = global_id.x;
    let a_size = params.a_shape[0u] * params.a_shape[1u] * params.a_shape[2u] * params.a_shape[3u];
    let b_size = params.b_shape[0u] * params.b_shape[1u] * params.b_shape[2u] * params.b_shape[3u];
    let max_size = max(a_size, b_size);
    let out_size = params.out_shape[0u] * params.out_shape[1u] * params.out_shape[2u] * params.out_shape[3u];

    if (thread_idx >= max_size) {
        return;
    }

    // grad_a = out_grad if a > b else 0
    if (thread_idx < a_size) {
        var sum_a: f32 = 0.0;
        for (var i: u32 = 0u; i < out_size; i = i + 1u) {
            let temp_idx = i * 5u;
            let target_a = u32(temp_buffer[temp_idx]);
            if (target_a == thread_idx) {
                let a_val = temp_buffer[temp_idx + 3u];
                let b_val = temp_buffer[temp_idx + 4u];
                if (a_val > b_val) {
                    sum_a = sum_a + temp_buffer[temp_idx + 2u];
                }
            }
        }
        result_grad[thread_idx] = sum_a;
    }

    // grad_b = out_grad if b >= a else 0
    if (thread_idx < b_size) {
        var sum_b: f32 = 0.0;
        for (var i: u32 = 0u; i < out_size; i = i + 1u) {
            let temp_idx = i * 5u;
            let target_b = u32(temp_buffer[temp_idx + 1u]);
            if (target_b == thread_idx) {
                let a_val = temp_buffer[temp_idx + 3u];
                let b_val = temp_buffer[temp_idx + 4u];
                if (b_val >= a_val) {
                    sum_b = sum_b + temp_buffer[temp_idx + 2u];
                }
            }
        }
        result_grad[a_size + thread_idx] = sum_b;
    }
}

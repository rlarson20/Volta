// Movement operations backward pass
// These compute gradients by inverting the forward movement operations

@group(0) @binding(0) var<storage, read> out_grad: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

// Parameters for movement operations (up to 4 dimensions)
struct MovementParams {
    old_shape: vec4<u32>,      // Original input shape (target gradient shape)
    new_shape: vec4<u32>,      // Output shape (incoming gradient shape)
    op_params: vec4<u32>,      // Operation-specific parameters
    rank: u32,
    padding2: u32,
    _padding: vec2<u32>,
}

@group(0) @binding(2) var<uniform> params: MovementParams;

// Compute output size based on shape and rank
fn output_size(shape: vec4<u32>, rank: u32) -> u32 {
    var size: u32 = 1u;
    for (var i: u32 = 0u; i < rank; i = i + 1u) {
        size = size * shape[i];
    }
    return size;
}

// Convert linear index to multi-dimensional coordinates (row-major)
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

// Convert multi-dimensional coordinates to linear index using strides
fn coords_to_idx(coords: vec4<u32>, strides: vec4<u32>, rank: u32) -> u32 {
    var idx: u32 = 0u;
    for (var i: u32 = 0u; i < rank; i = i + 1u) {
        idx = idx + coords[i] * strides[i];
    }
    return idx;
}

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

// Permute Backward: Apply inverse permutation to gradient
// Forward: new_coords[i] = old_coords[axes[i]]
// Backward: old_coords[axes[i]] = new_coords[i]
// This is equivalent to permuting with inverse axes
@compute @workgroup_size(256)
fn permute_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let out_size = output_size(params.old_shape, params.rank);

    if (idx < out_size) {
        // Get coordinates in output (original input) space
        let old_coords = idx_to_coords(idx, params.old_shape, params.rank);
        var new_coords = vec4<u32>(0u, 0u, 0u, 0u);

        // Apply inverse permutation: old_coords[axes[i]] = new_coords[i]
        // So: new_coords[axes[i]] = old_coords[i]
        for (var i: u32 = 0u; i < params.rank; i = i + 1u) {
            let axis = params.op_params[i];
            new_coords[axis] = old_coords[i];
        }

        let new_strides = compute_strides(params.new_shape, params.rank);
        let new_idx = coords_to_idx(new_coords, new_strides, params.rank);
        result[idx] = out_grad[new_idx];
    }
}

// Stride Backward: Upsample gradient (inverse of subsampling)
// Forward: old_coords[i] = new_coords[i] * stride[i]
// Backward: Place gradient at strided positions, zero elsewhere
@compute @workgroup_size(256)
fn stride_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let out_size = output_size(params.old_shape, params.rank);

    if (idx < out_size) {
        let old_coords = idx_to_coords(idx, params.old_shape, params.rank);
        var is_strided = true;
        var new_coords = vec4<u32>(0u, 0u, 0u, 0u);

        // Check if this position is on the stride grid
        for (var i: u32 = 0u; i < params.rank; i = i + 1u) {
            let stride_val = params.op_params[i];
            if (old_coords[i] % stride_val != 0u) {
                is_strided = false;
                break;
            }
            new_coords[i] = old_coords[i] / stride_val;
        }

        if (is_strided) {
            let new_strides = compute_strides(params.new_shape, params.rank);
            let new_idx = coords_to_idx(new_coords, new_strides, params.rank);
            result[idx] = out_grad[new_idx];
        } else {
            result[idx] = 0.0;
        }
    }
}

// Shrink Backward: Pad gradient back to original size (inverse of shrink)
// Forward: old_coords[i] = new_coords[i] + range_start[i]
// Backward: Place gradient in window, zero outside
@compute @workgroup_size(256)
fn shrink_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let out_size = output_size(params.old_shape, params.rank);

    if (idx < out_size) {
        let old_coords = idx_to_coords(idx, params.old_shape, params.rank);
        var in_window = true;
        var new_coords = vec4<u32>(0u, 0u, 0u, 0u);

        // Check if position is within the shrunken window
        for (var i: u32 = 0u; i < params.rank; i = i + 1u) {
            let range_start = params.op_params[i];
            if (old_coords[i] < range_start || old_coords[i] >= range_start + params.new_shape[i]) {
                in_window = false;
                break;
            }
            new_coords[i] = old_coords[i] - range_start;
        }

        if (in_window) {
            let new_strides = compute_strides(params.new_shape, params.rank);
            let new_idx = coords_to_idx(new_coords, new_strides, params.rank);
            result[idx] = out_grad[new_idx];
        } else {
            result[idx] = 0.0;
        }
    }
}

// Pad Backward: Extract center region from gradient (remove padding)
// Forward: Adds padding around edges
// Backward: Remove padding, extract center
@compute @workgroup_size(256)
fn pad_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let out_size = output_size(params.old_shape, params.rank);

    if (idx < out_size) {
        let old_coords = idx_to_coords(idx, params.old_shape, params.rank);
        var new_coords = vec4<u32>(0u, 0u, 0u, 0u);

        // Unpack padding and add to coordinates
        for (var i: u32 = 0u; i < params.rank; i = i + 1u) {
            var left: u32 = 0u;

            if (i == 0u) {
                left = params.op_params[0u];
            } else if (i == 1u) {
                left = params.op_params[2u];
            } else if (i == 2u) {
                left = (params.padding2 & 0xFFFF0000u) >> 16u;
            } else if (i == 3u) {
                left = (params._padding[0] & 0xFFFF0000u) >> 16u;
            }

            new_coords[i] = old_coords[i] + left;
        }

        let new_strides = compute_strides(params.new_shape, params.rank);
        let new_idx = coords_to_idx(new_coords, new_strides, params.rank);
        result[idx] = out_grad[new_idx];
    }
}

// Expand Backward: Sum gradients over broadcast dimensions
// Forward: Broadcasts dimensions from size 1 to larger sizes
// Backward: Reduce gradients back to size 1 dimensions
// This requires accumulation - we iterate over output gradient and accumulate into result
@compute @workgroup_size(256)
fn expand_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_idx = global_id.x;
    let old_size = output_size(params.old_shape, params.rank);

    // Each thread handles one position in the result (original input shape)
    if (thread_idx < old_size) {
        let old_coords = idx_to_coords(thread_idx, params.old_shape, params.rank);
        let old_strides = compute_strides(params.old_shape, params.rank);
        let new_strides = compute_strides(params.new_shape, params.rank);
        let new_size = output_size(params.new_shape, params.rank);

        var sum: f32 = 0.0;

        // Sum over all positions in the expanded gradient that map to this input position
        for (var i: u32 = 0u; i < new_size; i = i + 1u) {
            let new_coords = idx_to_coords(i, params.new_shape, params.rank);
            var maps_to_this = true;

            // Check if this output position maps to our input position
            for (var j: u32 = 0u; j < params.rank; j = j + 1u) {
                if (params.old_shape[j] == 1u) {
                    // Broadcast dimension - any new coordinate maps to old coord 0
                    if (old_coords[j] != 0u) {
                        maps_to_this = false;
                        break;
                    }
                } else {
                    // Non-broadcast dimension - coordinates must match
                    if (new_coords[j] != old_coords[j]) {
                        maps_to_this = false;
                        break;
                    }
                }
            }

            if (maps_to_this) {
                sum = sum + out_grad[i];
            }
        }

        result[thread_idx] = sum;
    }
}

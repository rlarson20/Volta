// Movement operations - permute, expand, pad, shrink, stride
// These operations manipulate how data is indexed rather than computing new values

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

// Parameters for movement operations (up to 4 dimensions)
struct MovementParams {
    old_shape: vec4<u32>,
    new_shape: vec4<u32>,
    op_params: vec4<u32>,  // Operation-specific parameters
    rank: u32,
    padding2: u32,
    _padding: vec2<u32>,
}

@group(0) @binding(2) var<uniform> params: MovementParams;

// Compute output size based on shape and rank (only multiply actual dimensions)
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
// For shape [A, B, C], strides are [B*C, C, 1]
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

// Permute: reorder axes according to op_params as permutation [axis0, axis1, ...]
// Example: transpose 2D: axes=[1,0] swaps dimensions
@compute @workgroup_size(256)
fn permute(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let out_size = output_size(params.new_shape, params.rank);

    if (idx < out_size) {
        let new_coords = idx_to_coords(idx, params.new_shape, params.rank);
        var old_coords = vec4<u32>(0u, 0u, 0u, 0u);

        // Apply permutation: new_coords[i] goes to old_coords[axes[i]]
        for (var i: u32 = 0u; i < params.rank; i = i + 1u) {
            let axis = params.op_params[i];
            old_coords[axis] = new_coords[i];
        }

        let old_strides = compute_strides(params.old_shape, params.rank);
        let old_idx = coords_to_idx(old_coords, old_strides, params.rank);
        result[idx] = input[old_idx];
    }
}

// Expand: broadcast dimensions from size 1 to target size
@compute @workgroup_size(256)
fn expand(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let out_size = output_size(params.new_shape, params.rank);

    if (idx < out_size) {
        let new_coords = idx_to_coords(idx, params.new_shape, params.rank);
        var old_coords = vec4<u32>(0u, 0u, 0u, 0u);

        let old_strides = compute_strides(params.old_shape, params.rank);
        for (var i: u32 = 0u; i < params.rank; i = i + 1u) {
            // If old dim was 1, use index 0; otherwise use new coordinate
            if (params.old_shape[i] == 1u) {
                old_coords[i] = 0u;
            } else {
                old_coords[i] = new_coords[i];
            }
        }

        let old_idx = coords_to_idx(old_coords, old_strides, params.rank);
        result[idx] = input[old_idx];
    }
}

// Pad: add zeros around edges (2D only for simplicity)
@compute @workgroup_size(256)
fn pad(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let out_size = output_size(params.new_shape, params.rank);

    if (idx < out_size) {
        let new_coords = idx_to_coords(idx, params.new_shape, params.rank);
        var in_bounds = true;
        var old_coords = vec4<u32>(0u, 0u, 0u, 0u);

        // Unpack padding from op_params: (left0, right0, left1, right1)
        for (var i: u32 = 0u; i < params.rank; i = i + 1u) {
            let left = params.op_params[i * 2u];
            old_coords[i] = new_coords[i] - left;

            if (old_coords[i] >= params.old_shape[i]) {
                in_bounds = false;
            }
        }

        if (in_bounds) {
            let old_strides = compute_strides(params.old_shape, params.rank);
            let old_idx = coords_to_idx(old_coords, old_strides, params.rank);
            result[idx] = input[old_idx];
        } else {
            result[idx] = 0.0;
        }
    }
}

// Shrink: extract subregion
@compute @workgroup_size(256)
fn shrink(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let out_size = output_size(params.new_shape, params.rank);

    if (idx < out_size) {
        let new_coords = idx_to_coords(idx, params.new_shape, params.rank);
        var old_coords = vec4<u32>(0u, 0u, 0u, 0u);

        // op_params contains range starts
        for (var i: u32 = 0u; i < params.rank; i = i + 1u) {
            old_coords[i] = new_coords[i] + params.op_params[i];
        }

        let old_strides = compute_strides(params.old_shape, params.rank);
        let old_idx = coords_to_idx(old_coords, old_strides, params.rank);
        result[idx] = input[old_idx];
    }
}

// Stride: subsample with stride
@compute @workgroup_size(256)
fn stride(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let out_size = output_size(params.new_shape, params.rank);

    if (idx < out_size) {
        let new_coords = idx_to_coords(idx, params.new_shape, params.rank);
        var old_coords = vec4<u32>(0u, 0u, 0u, 0u);

        // op_params contains stride values
        for (var i: u32 = 0u; i < params.rank; i = i + 1u) {
            old_coords[i] = new_coords[i] * params.op_params[i];
        }

        let old_strides = compute_strides(params.old_shape, params.rank);
        let old_idx = coords_to_idx(old_coords, old_strides, params.rank);
        result[idx] = input[old_idx];
    }
}

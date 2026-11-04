// Reduction backward operations
// Each operation broadcasts a gradient value back to the original tensor shape

struct ReduceBackwardParams {
    input_size: u32,     // Total elements in input
    op: u32,             // 0 = sum, 1 = mean, 2 = max
    grad_value: f32,     // The scalar gradient value
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> out_grad: array<ELEM>;  // Scalar gradient (for compatibility)
@group(0) @binding(1) var<storage, read> max_indices: array<u32>;  // For max: which element was max
@group(0) @binding(2) var<storage, read_write> result: array<ELEM>;
@group(0) @binding(3) var<uniform> params: ReduceBackwardParams;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx >= params.input_size) {
        return;
    }

    // Force all bindings to be used to prevent optimization
    let _grad_unused = ld_out_grad(0);
    let _indices_unused = max_indices[0];

    if params.op == 0u {
        // Sum: gradient broadcasts to all elements
        st_result(idx, params.grad_value);
    } else if params.op == 1u {
        // Mean: gradient broadcasts to all elements, divided by count
        st_result(idx, params.grad_value / f32(params.input_size));
    } else {
        // Max: sparse gradient - only the max element gets the gradient
        let is_max = idx == max_indices[0];
        st_result(idx, select(0.0, params.grad_value, is_max));
    }
}

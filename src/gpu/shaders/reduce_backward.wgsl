// Reduction backward operations
// Each operation broadcasts a gradient value back to the original tensor shape

struct ReduceBackwardParams {
    input_size: u32,     // Total elements in input
    op: u32,             // 0 = sum, 1 = mean (`max_backward` ignores it)
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> out_grad: array<ELEM>;  // The scalar upstream gradient (element 0), read on the device
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

    let grad = ld_out_grad(0);

    if params.op == 0u {
        // Sum: gradient broadcasts to all elements
        st_result(idx, grad);
    } else {
        // Mean: gradient broadcasts to all elements, divided by count
        st_result(idx, grad / f32(params.input_size));
    }
}

// Max, as `torch.max()`: the gradient is split evenly over every element equal to the maximum
// (or over every NaN when the maximum is NaN), and the rest get zero (S4). Reads the saved
// input beside the maximum and the number of elements that equal it, both left on the device
// by `reduce.wgsl`.
@group(0) @binding(1) var<storage, read> max_count: array<u32>;
@group(0) @binding(4) var<storage, read> x: array<ELEM>;
@group(0) @binding(5) var<storage, read> max_value: array<f32>;

@compute @workgroup_size(256)
fn max_backward(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx >= params.input_size) {
        return;
    }

    let m = max_value[0];
    let v = ld_x(idx);
    // With a NaN maximum no float comparison is made (see `is_nan`).
    let hit = select(v == m, is_nan(v), is_nan(m));
    st_result(idx, select(0.0, ld_out_grad(0) / f32(max_count[0]), hit));
}

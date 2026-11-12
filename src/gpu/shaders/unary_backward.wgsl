// Unary backward operations
// Each thread computes gradient for one element
// For y = f(x), we compute grad = out_grad * df/dx

@group(0) @binding(0) var<storage, read> out_grad: array<ELEM>;
@group(0) @binding(1) var<storage, read> x: array<ELEM>;
@group(0) @binding(2) var<storage, read_write> result: array<ELEM>;

@compute @workgroup_size(256)
fn exp_backward(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_out_grad()) {
        // d(e^x)/dx = e^x
        st_result(idx, ld_out_grad(idx) * exp(ld_x(idx)));
    }
}

@compute @workgroup_size(256)
fn log_backward(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_out_grad()) {
        // d(ln(x))/dx = 1/x
        st_result(idx, ld_out_grad(idx) / ld_x(idx));
    }
}

@compute @workgroup_size(256)
fn relu_backward(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_out_grad()) {
        // d(max(0, x))/dx = x > 0 ? 1 : 0, and 0 for NaN
        let x = ld_x(idx);
        st_result(idx, select(0.0, ld_out_grad(idx), !is_nan(x) && x > 0.0));
    }
}

@compute @workgroup_size(256)
fn sigmoid_backward(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_out_grad()) {
        // d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
        let s = 1.0 / (1.0 + exp(-ld_x(idx)));
        st_result(idx, ld_out_grad(idx) * s * (1.0 - s));
    }
}

@compute @workgroup_size(256)
fn tanh_backward(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_out_grad()) {
        // d(tanh(x))/dx = 1 - tanh²(x)
        let t = tanh_sat(ld_x(idx));
        st_result(idx, ld_out_grad(idx) * (1.0 - t * t));
    }
}

@compute @workgroup_size(256)
fn sqrt_backward(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_out_grad()) {
        // d(sqrt(x))/dx = 1 / (2 * sqrt(x))
        st_result(idx, ld_out_grad(idx) / (2.0 * sqrt(ld_x(idx))));
    }
}

@compute @workgroup_size(256)
fn sin_backward(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_out_grad()) {
        // d(sin(x))/dx = cos(x)
        st_result(idx, ld_out_grad(idx) * cos(ld_x(idx)));
    }
}

@compute @workgroup_size(256)
fn cos_backward(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_out_grad()) {
        // d(cos(x))/dx = -sin(x)
        st_result(idx, -ld_out_grad(idx) * sin(ld_x(idx)));
    }
}

@compute @workgroup_size(256)
fn neg_backward(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_out_grad()) {
        // d(-x)/dx = -1
        _ = ld_x(idx); // Force binding 1 into layout
        st_result(idx, -ld_out_grad(idx));
    }
}

@compute @workgroup_size(256)
fn recip_backward(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_out_grad()) {
        // d(1/x)/dx = -1/x²
        let x_inv = 1.0 / ld_x(idx);
        st_result(idx, -ld_out_grad(idx) * x_inv * x_inv);
    }
}

@compute @workgroup_size(256)
fn exp2_backward(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_out_grad()) {
        // d(2^x)/dx = 2^x * ln(2)
        st_result(idx, ld_out_grad(idx) * pow(2.0, ld_x(idx)) * 0.6931471805599453); // ln(2)
    }
}

@compute @workgroup_size(256)
fn log2_backward(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_out_grad()) {
        // d(log2(x))/dx = 1 / (x * ln(2))
        st_result(idx, ld_out_grad(idx) / (ld_x(idx) * 0.6931471805599453)); // ln(2)
    }
}

@compute @workgroup_size(256)
fn erf_backward(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_out_grad()) {
        // d erf(x)/dx = 2/sqrt(pi) * exp(-x^2)
        st_result(idx, ld_out_grad(idx) * 1.1283791670955126 * exp(-ld_x(idx) * ld_x(idx)));
    }
}

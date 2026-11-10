// Unary operations
// Each thread processes one element

@group(0) @binding(0) var<storage, read> input: array<ELEM>;
@group(0) @binding(1) var<storage, read_write> result: array<ELEM>;

@compute @workgroup_size(256)
fn neg(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_input()) {
        st_result(idx, -ld_input(idx));
    }
}

@compute @workgroup_size(256)
fn exp_op(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_input()) {
        st_result(idx, exp(ld_input(idx)));
    }
}

@compute @workgroup_size(256)
fn log_op(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_input()) {
        st_result(idx, log(ld_input(idx)));
    }
}

@compute @workgroup_size(256)
fn relu(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_input()) {
        st_result(idx, max(ld_input(idx), 0.0));
    }
}

@compute @workgroup_size(256)
fn sigmoid(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_input()) {
        st_result(idx, 1.0 / (1.0 + exp(-ld_input(idx))));
    }
}

@compute @workgroup_size(256)
fn tanh_op(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_input()) {
        st_result(idx, tanh_sat(ld_input(idx)));
    }
}

@compute @workgroup_size(256)
fn sqrt_op(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_input()) {
        st_result(idx, sqrt(ld_input(idx)));
    }
}

@compute @workgroup_size(256)
fn recip(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_input()) {
        st_result(idx, 1.0 / ld_input(idx));
    }
}

@compute @workgroup_size(256)
fn exp2_op(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_input()) {
        st_result(idx, pow(2.0, ld_input(idx)));
    }
}

@compute @workgroup_size(256)
fn log2_op(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_input()) {
        st_result(idx, log2(ld_input(idx)));
    }
}

@compute @workgroup_size(256)
fn sin_op(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_input()) {
        st_result(idx, sin(ld_input(idx)));
    }
}

@compute @workgroup_size(256)
fn cos_op(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_input()) {
        st_result(idx, cos(ld_input(idx)));
    }
}

// erf via Abramowitz-Stegun 7.1.26 (max abs error ~1.5e-7), odd-symmetric.
// exp(-x*x) underflows to 0 for large |x|, so the result saturates at +/-1.
fn erf_approx(x: f32) -> f32 {
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    let y = 1.0 - poly * exp(-a * a);
    return select(y, -y, x < 0.0);
}

@compute @workgroup_size(256)
fn erf_op(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_input()) {
        st_result(idx, erf_approx(ld_input(idx)));
    }
}

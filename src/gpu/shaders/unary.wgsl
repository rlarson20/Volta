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

// erf, odd-symmetric, within a few f32 ulps of the CPU's `libm::erf` (S22).
//
// |x| < 1: the Maclaurin series erf(x) = 2/sqrt(pi) * x * sum (-z)^n / (n! (2n+1)), z = x^2,
// whose coefficients are exact, so the result is relatively accurate down to zero (twelve
// terms leave under 1e-10 at |x| = 1; the terms shrink from the first, so nothing cancels).
// Abramowitz-Stegun 7.1.26 has an absolute error of 1.5e-7 whatever x is, which near zero
// is the whole value (a relative error of 1e-4 at 1e-3, and 1 at 1e-8).
//
// |x| >= 1: Abramowitz-Stegun 7.1.26 (absolute error 1.5e-7). There erf(x) >= 0.84, so that
// is a relative error under 1.8e-7, and exp(-x*x) underflows to 0 for large |x|, so the
// result saturates at +/-1. The two pieces meet at 1 with a jump of at most 1.5e-7. A NaN
// fails `a < 1.0` and comes out of the second piece as NaN.
fn erf_approx(x: f32) -> f32 {
    let a = abs(x);
    var y: f32;
    if (a < 1.0) {
        let z = a * a;
        // Horner in z, from the n = 11 coefficient down to n = 0.
        var series: f32 = -1.0 / 918086400.0;
        series = series * z + 1.0 / 76204800.0;
        series = series * z - 1.0 / 6894720.0;
        series = series * z + 1.0 / 685440.0;
        series = series * z - 1.0 / 75600.0;
        series = series * z + 1.0 / 9360.0;
        series = series * z - 1.0 / 1320.0;
        series = series * z + 1.0 / 216.0;
        series = series * z - 1.0 / 42.0;
        series = series * z + 1.0 / 10.0;
        series = series * z - 1.0 / 3.0;
        series = series * z + 1.0;
        y = 1.1283791670955126 * a * series;
    } else {
        let t = 1.0 / (1.0 + 0.3275911 * a);
        let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
        y = 1.0 - poly * exp(-a * a);
    }
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

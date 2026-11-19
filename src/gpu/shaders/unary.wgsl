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
        // torch.relu keeps NaN (WGSL's max may return the 0).
        let x = ld_input(idx);
        st_result(idx, select(max(x, 0.0), x, is_nan(x)));
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
        st_result(idx, exp2(ld_input(idx)));
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

// erf, odd-symmetric, within an f32 ulp of the CPU's `libm::erff` (S22, #323).
//
// |x| < 1: the Maclaurin series erf(x) = 2/sqrt(pi) * x * sum (-z)^n / (n! (2n+1)), z = x^2,
// whose coefficients are exact, so the result is relatively accurate down to zero (twelve
// terms leave under 1e-10 at |x| = 1; the terms shrink from the first, so nothing cancels).
// Abramowitz-Stegun 7.1.26 has an absolute error of 1.5e-7 whatever x is, which near zero
// is the whole value (a relative error of 1e-4 at 1e-3, and 1 at 1e-8).
//
// |x| >= 1: erf = 1 - erfc, with erfc from the Chebyshev fit in Numerical Recipes (`erfcc`,
// 2nd ed. 6.2), whose error is under 1.2e-7 *relative to erfc*. GELU's negative tail is
// x/2 * (1 + erf(x / sqrt 2)), where 1 + erf is erfc of a small number: an absolute error in
// erf there is a large relative error in GELU. Abramowitz-Stegun 7.1.26's absolute 1.5e-7
// was 2 ulps of erf at -3/sqrt(2), 4.4e-5 relative in GELU(-3). Here erfc's error is under
// 2e-8 at |x| = 1 (erfc = 0.157) and shrinks with it, under an ulp of erf, so erf is
// within an ulp of `libm::erff` (tests/gpu_gelu_erf.rs; on lavapipe, at most 1 ulp over
// 1 <= |x| < 6, and GELU(-3) is identical to the CPU's). exp(-a*a) underflows to 0 for large |x|,
// so the result saturates at +/-1. A NaN fails `a < 1.0` and comes out of the second piece as
// NaN.
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
        let t = 1.0 / (1.0 + 0.5 * a);
        var p: f32 = 0.17087277;
        p = p * t - 0.82215223;
        p = p * t + 1.48851587;
        p = p * t - 1.13520398;
        p = p * t + 0.27886807;
        p = p * t - 0.18628806;
        p = p * t + 0.09678418;
        p = p * t + 0.37409196;
        p = p * t + 1.00002368;
        p = p * t - 1.26551223;
        let erfc = t * exp(p - a * a);
        y = 1.0 - erfc;
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

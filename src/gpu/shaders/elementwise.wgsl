// Element-wise binary operations
// Each thread processes one element

@group(0) @binding(0) var<storage, read> a: array<ELEM>;
@group(0) @binding(1) var<storage, read> b: array<ELEM>;
@group(0) @binding(2) var<storage, read_write> result: array<ELEM>;

// Workgroup size of 256 is a common choice for compute shaders
// It's a good balance between occupancy and register usage
@compute @workgroup_size(256)
fn add(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_a()) {
        st_result(idx, ld_a(idx) + ld_b(idx));
    }
}

@compute @workgroup_size(256)
fn sub(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_a()) {
        st_result(idx, ld_a(idx) - ld_b(idx));
    }
}

@compute @workgroup_size(256)
fn mul(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_a()) {
        st_result(idx, ld_a(idx) * ld_b(idx));
    }
}

@compute @workgroup_size(256)
fn div(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_a()) {
        st_result(idx, ld_a(idx) / ld_b(idx));
    }
}

@compute @workgroup_size(256)
fn max_elem(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_a()) {
        // torch.maximum: NaN if either operand is NaN (WGSL's max may return the other).
        let a = ld_a(idx);
        let b = ld_b(idx);
        st_result(idx, select(select(max(a, b), b, is_nan(b)), a, is_nan(a)));
    }
}

// Whether `x` is +/-inf, read from its bits (as `is_nan`: fast math may fold comparisons
// with infinities away).
fn is_inf(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7fffffffu) == 0x7f800000u;
}

// `a` mod `b`, floored as `torch.remainder` and the CPU's `BinaryOp::Mod` (#302).
//
// Not the exact remainder: `q = trunc(a / b)` and `a - q * b` each round, so `q` can be one
// off and the result carries up to about `(|a| + |b|) * 2^-23` of rounding error. WGSL
// does not promise that `fma` is fused (lavapipe does not fuse it); where it is, the error
// is smaller. The corrections below keep the result in the floored range regardless: zero
// or b's sign, and `|r| < |b|`. Past `|a / b| = 2^23` the result is not the remainder at
// all. `BinaryOp::Mod` documents both bounds; an exact kernel is #355.
// - An infinite divisor with a finite `a` gives `fmod`'s `a`, then the floored step: the
//   CPU's `a % inf` exactly (WGSL's `%` gives `inf * 0 = NaN`).
// - A zero divisor, an infinite `a` or a NaN gives NaN, as `fmod` does: `a / 0` is inf, so
//   the product is `inf * 0` or the difference is `inf - inf`.
fn floored_mod(a: f32, b: f32) -> f32 {
    if (is_inf(b) && !is_inf(a) && !is_nan(a)) {
        return select(a, a + b, a != 0.0 && ((a < 0.0) != (b < 0.0)));
    }
    var r = fma(-trunc(a / b), b, a);
    // The floored remainder takes b's sign and is smaller than |b|. With `q` one off, `r`
    // can be up to one `b` past that range on either side; each step adds or removes one.
    if (r != 0.0 && ((r < 0.0) != (b < 0.0))) {
        r = r + b;
    }
    if (r != 0.0 && ((r < 0.0) != (b < 0.0))) {
        r = r + b;
    }
    if (abs(r) >= abs(b)) {
        r = r - b;
    }
    return r;
}

@compute @workgroup_size(256)
fn mod_op(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < len_a()) {
        st_result(idx, floored_mod(ld_a(idx), ld_b(idx)));
    }
}

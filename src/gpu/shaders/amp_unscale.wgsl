// `GradScaler::unscale` on the device (PyTorch's `_amp_foreach_non_finite_check_and_unscale_`):
// `out = grad * inv`, rounded to the buffer's type, and `found[0]` set to 1 if any rounded
// result is inf or NaN (checked from the bits, which WGSL's float comparisons don't
// guarantee for NaN). Only `found` is read back.

@group(0) @binding(0) var<storage, read> grad: array<ELEM>;
@group(0) @binding(1) var<storage, read_write> out: array<ELEM>;
@group(0) @binding(2) var<storage, read_write> found: array<atomic<u32>>;

struct UnscaleParams {
    n: u32,
    inv: f32,
    _pad0: u32,
    _pad1: u32,
}
@group(0) @binding(3) var<uniform> params: UnscaleParams;

@compute @workgroup_size(256)
fn amp_unscale(
    @builtin(global_invocation_id) gid_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let gid = flat_global_id(gid_2d, num_wg);
    let i = gid.x;
    if (i < params.n) {
        st_out(i, ld_grad(i) * params.inv);
        let bits = bitcast<u32>(ld_out(i));
        if ((bits & 0x7f800000u) == 0x7f800000u) {
            atomicOr(&found[0], 1u);
        }
    }
}

// Dtype conversion between GPU float types (`to_dtype`): each element is loaded as f32 from
// the `ELEM` input and stored, rounded to nearest-even, in the `ELEM2` output. Widening
// (f16/bf16 to f32) is exact; f32 to f16/bf16 rounds once, like the CPU cast.

@group(0) @binding(0) var<storage, read> input: array<ELEM>;
@group(0) @binding(1) var<storage, read_write> output: array<ELEM2>;

struct CastParams {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(2) var<uniform> params: CastParams;

@compute @workgroup_size(256)
fn cast_elements(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < params.n) {
        st_output(idx, ld_input(idx));
    }
}

// Bool (one u32 per element) to and from floats: `from_bool` writes 1.0 or 0.0 into the
// `ELEM2` output, `to_bool` writes whether the `ELEM` input is nonzero (NaN is nonzero,
// as in PyTorch), detected from the bits.
@group(0) @binding(0) var<storage, read> bool_in: array<u32>;
@group(0) @binding(1) var<storage, read_write> bool_out: array<u32>;

@compute @workgroup_size(256)
fn from_bool(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < params.n) {
        st_output(idx, select(0.0, 1.0, bool_in[idx] != 0u));
    }
}

@compute @workgroup_size(256)
fn to_bool(
    @builtin(global_invocation_id) global_id_2d: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global_id = flat_global_id(global_id_2d, num_wg);
    let idx = global_id.x;
    if (idx < params.n) {
        let bits = bitcast<u32>(ld_input(idx)) & 0x7fffffffu;
        bool_out[idx] = select(0u, 1u, bits != 0u);
    }
}

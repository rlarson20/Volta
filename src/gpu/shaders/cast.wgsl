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
fn cast_elements(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.n) {
        st_output(idx, ld_input(idx));
    }
}

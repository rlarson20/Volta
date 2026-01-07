// Reduction operations
// Note: This is a simplified version. Production code would use
// parallel reduction with shared memory for better performance.

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

struct Params {
    input_size: u32,
    _padding: vec3<u32>,
}

@group(0) @binding(2) var<uniform> params: Params;

// Sequential sum reduction - simple but correct
// A proper implementation would use parallel tree reduction
@compute @workgroup_size(1)
fn sum_reduce(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < params.input_size; i = i + 1u) {
        sum = sum + input[i];
    }
    result[0] = sum;
}

// Sequential max reduction
@compute @workgroup_size(1)
fn max_reduce(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var max_val: f32 = -1.0e9; // Start with very small number
    for (var i: u32 = 0u; i < params.input_size; i = i + 1u) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    result[0] = max_val;
}

// Mean reduction - sum then divide
@compute @workgroup_size(1)
fn mean_reduce(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < params.input_size; i = i + 1u) {
        sum = sum + input[i];
    }
    let count: f32 = f32(params.input_size);
    result[0] = sum / count;
}

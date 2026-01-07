// Unary operations
// Each thread processes one element

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(256)
fn neg(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&input)) {
        result[idx] = -input[idx];
    }
}

@compute @workgroup_size(256)
fn exp_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&input)) {
        result[idx] = exp(input[idx]);
    }
}

@compute @workgroup_size(256)
fn log_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&input)) {
        result[idx] = log(input[idx]);
    }
}

@compute @workgroup_size(256)
fn relu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&input)) {
        result[idx] = max(input[idx], 0.0);
    }
}

@compute @workgroup_size(256)
fn sigmoid(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&input)) {
        result[idx] = 1.0 / (1.0 + exp(-input[idx]));
    }
}

@compute @workgroup_size(256)
fn tanh_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&input)) {
        result[idx] = tanh(input[idx]);
    }
}

@compute @workgroup_size(256)
fn sqrt_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&input)) {
        result[idx] = sqrt(input[idx]);
    }
}

@compute @workgroup_size(256)
fn recip(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&input)) {
        result[idx] = 1.0 / input[idx];
    }
}

@compute @workgroup_size(256)
fn exp2_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&input)) {
        result[idx] = pow(2.0, input[idx]);
    }
}

@compute @workgroup_size(256)
fn log2_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&input)) {
        result[idx] = log2(input[idx]);
    }
}

@compute @workgroup_size(256)
fn sin_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&input)) {
        result[idx] = sin(input[idx]);
    }
}

@compute @workgroup_size(256)
fn cos_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&input)) {
        result[idx] = cos(input[idx]);
    }
}

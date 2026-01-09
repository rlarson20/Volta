// Unary backward operations
// Each thread computes gradient for one element
// For y = f(x), we compute grad = out_grad * df/dx

@group(0) @binding(0) var<storage, read> out_grad: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(256)
fn exp_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&out_grad)) {
        // d(e^x)/dx = e^x
        result[idx] = out_grad[idx] * exp(x[idx]);
    }
}

@compute @workgroup_size(256)
fn log_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&out_grad)) {
        // d(ln(x))/dx = 1/x
        result[idx] = out_grad[idx] / x[idx];
    }
}

@compute @workgroup_size(256)
fn relu_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&out_grad)) {
        // d(max(0, x))/dx = x > 0 ? 1 : 0
        result[idx] = select(0.0, out_grad[idx], x[idx] > 0.0);
    }
}

@compute @workgroup_size(256)
fn sigmoid_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&out_grad)) {
        // d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
        let s = 1.0 / (1.0 + exp(-x[idx]));
        result[idx] = out_grad[idx] * s * (1.0 - s);
    }
}

@compute @workgroup_size(256)
fn tanh_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&out_grad)) {
        // d(tanh(x))/dx = 1 - tanh²(x)
        let t = tanh(x[idx]);
        result[idx] = out_grad[idx] * (1.0 - t * t);
    }
}

@compute @workgroup_size(256)
fn sqrt_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&out_grad)) {
        // d(sqrt(x))/dx = 1 / (2 * sqrt(x))
        result[idx] = out_grad[idx] / (2.0 * sqrt(x[idx]));
    }
}

@compute @workgroup_size(256)
fn sin_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&out_grad)) {
        // d(sin(x))/dx = cos(x)
        result[idx] = out_grad[idx] * cos(x[idx]);
    }
}

@compute @workgroup_size(256)
fn cos_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&out_grad)) {
        // d(cos(x))/dx = -sin(x)
        result[idx] = -out_grad[idx] * sin(x[idx]);
    }
}

@compute @workgroup_size(256)
fn neg_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&out_grad)) {
        // d(-x)/dx = -1
        result[idx] = -out_grad[idx];
    }
}

@compute @workgroup_size(256)
fn recip_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&out_grad)) {
        // d(1/x)/dx = -1/x²
        let x_inv = 1.0 / x[idx];
        result[idx] = -out_grad[idx] * x_inv * x_inv;
    }
}

@compute @workgroup_size(256)
fn exp2_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&out_grad)) {
        // d(2^x)/dx = 2^x * ln(2)
        result[idx] = out_grad[idx] * pow(2.0, x[idx]) * 0.6931471805599453; // ln(2)
    }
}

@compute @workgroup_size(256)
fn log2_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&out_grad)) {
        // d(log2(x))/dx = 1 / (x * ln(2))
        result[idx] = out_grad[idx] / (x[idx] * 0.6931471805599453); // ln(2)
    }
}

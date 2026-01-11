// Optimizer step kernel - unified SGD and Adam parameter updates
//
// Op codes:
// 0 = SGD: param -= lr * grad
// 1 = SGD with momentum: v = momentum*v - lr*grad; param += v
// 2 = Adam: param -= lr * (m / (1-beta1^t)) / (sqrt(v / (1-beta2^t)) + eps)

struct OptimizerParams {
    op: u32,              // 0=SGD, 1=SGD+momentum, 2=Adam
    lr: f32,              // Learning rate
    beta1: f32,           // Adam beta1 or momentum
    beta2: f32,           // Adam beta2
    t: f32,               // Timestep for bias correction
    eps: f32,             // Adam epsilon
    weight_decay: f32,    // L2 regularization
    _padding: f32,
}

@group(0) @binding(0) var<storage, read_write> params: array<f32>;
@group(0) @binding(1) var<storage, read> grads: array<f32>;
@group(0) @binding(2) var<storage, read_write> state1: array<f32>;  // velocity (SGD) or m (Adam)
@group(0) @binding(3) var<storage, read_write> state2: array<f32>;  // v (Adam only)
@group(0) @binding(4) var<uniform> opt_params: OptimizerParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&params)) {
        return;
    }

    let p = opt_params;
    var grad = grads[idx];

    // Apply weight decay: grad = grad + weight_decay * param
    if (p.weight_decay != 0.0) {
        grad = grad + p.weight_decay * params[idx];
    }

    if (p.op == 0u) {
        // Simple SGD: param = param - lr * grad
        params[idx] = params[idx] - p.lr * grad;
    } else if (p.op == 1u) {
        // SGD with momentum
        // v = momentum * v - lr * grad
        state1[idx] = p.beta1 * state1[idx] - p.lr * grad;
        // param = param + v
        params[idx] = params[idx] + state1[idx];
    } else {
        // Adam optimizer
        // m = beta1 * m + (1 - beta1) * grad
        state1[idx] = p.beta1 * state1[idx] + (1.0 - p.beta1) * grad;
        // v = beta2 * v + (1 - beta2) * grad^2
        state2[idx] = p.beta2 * state2[idx] + (1.0 - p.beta2) * grad * grad;

        // Bias correction
        let m_hat_scale = 1.0 / (1.0 - pow(p.beta1, p.t));
        let v_hat_scale = 1.0 / (1.0 - pow(p.beta2, p.t));

        // param = param - lr * m_hat / (sqrt(v_hat) + eps)
        let m_hat = state1[idx] * m_hat_scale;
        let v_hat = state2[idx] * v_hat_scale;
        params[idx] = params[idx] - p.lr * m_hat / (sqrt(v_hat) + p.eps);
    }
}

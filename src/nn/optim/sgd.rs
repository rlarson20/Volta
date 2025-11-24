use crate::tensor::Tensor;

/// Stochastic Gradient Descent optimizer with optional momentum
///
/// Update rule:
/// - Without momentum: θ ← θ - lr·∇θ
/// - With momentum: v ← β·v - lr·∇θ, θ ← θ + v
///
/// Momentum helps accelerate convergence and dampen oscillations.
pub struct SGD {
    params: Vec<Tensor>,
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    velocity: Vec<Vec<f32>>,
}

impl SGD {
    /// Create a new SGD optimizer
    ///
    /// # Arguments
    /// * `params` - List of parameters to optimize
    /// * `lr` - Learning rate (typical: 0.01 to 0.1)
    /// * `momentum` - Momentum coefficient (typical: 0.9, or 0.0 for no momentum)
    /// * `weight_decay` - L2 penalty (typical: 1e-4, or 0.0 for none)
    pub fn new(params: Vec<Tensor>, lr: f32, momentum: f32, weight_decay: f32) -> Self {
        let velocity = if momentum > 0.0 {
            params
                .iter()
                .map(|p| vec![0.0; p.borrow().data.len()])
                .collect()
        } else {
            vec![]
        };

        SGD {
            params,
            lr,
            momentum,
            weight_decay,
            velocity,
        }
    }

    /// Zero all parameter gradients
    ///
    /// Must be called before each backward pass to avoid gradient accumulation
    /// across multiple batches.
    pub fn zero_grad(&self) {
        for param in &self.params {
            param.borrow_mut().grad = None;
        }
    }
    /// Perform one optimization step
    ///
    /// Updates all parameters using their accumulated gradients.
    pub fn step(&mut self) {
        for (i, param) in self.params.iter().enumerate() {
            let mut p = param.borrow_mut();
            if let Some(grad) = &p.grad.clone() {
                // Apply weight decay: grad = grad + weight_decay * param
                let mut active_grad = grad.clone();
                if self.weight_decay != 0.0 {
                    for (g, theta) in active_grad.iter_mut().zip(p.data.iter()) {
                        *g += self.weight_decay * *theta;
                    }
                }

                if self.momentum > 0.0 {
                    // Update velocity: v = momentum·v - lr·grad
                    for (v, &g) in self.velocity[i].iter_mut().zip(active_grad.iter()) {
                        *v = self.momentum * *v - self.lr * g;
                    }
                    // Update parameters: θ = θ + v
                    for (d, &v) in p.data.iter_mut().zip(&self.velocity[i]) {
                        *d += v;
                    }
                } else {
                    // Simple SGD: θ = θ - lr·grad
                    for (d, &g) in p.data.iter_mut().zip(active_grad.iter()) {
                        *d -= self.lr * g;
                    }
                }
            }
        }
    }
}

use crate::storage::Storage;
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
    velocity: Vec<Storage>, // Can be CPU or GPU storage (empty if momentum=0)
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
        // Create velocity state matching the device of each parameter
        let velocity = if momentum > 0.0 {
            params
                .iter()
                .map(|p| {
                    let param = p.borrow();
                    let len = param.data.len();
                    let device = param.device.clone();
                    drop(param);
                    Storage::new_zeros(len, &device)
                })
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
        // Process each parameter based on its device
        for i in 0..self.params.len() {
            let param = &self.params[i];
            let p = param.borrow();

            // Skip parameters without gradients
            let _grad = match &p.grad {
                Some(g) => g.clone(),
                None => continue,
            };

            let is_gpu = p.device.is_gpu();
            drop(p);

            if is_gpu {
                self.step_gpu_param(i);
            } else {
                self.step_cpu_param(i);
            }
        }
    }

    /// GPU-accelerated step for a single parameter
    #[cfg(feature = "gpu")]
    fn step_gpu_param(&mut self, i: usize) {
        use crate::gpu::OptimizerStepParams;

        // For simple SGD (no momentum), use the simple path
        if self.momentum == 0.0 {
            return self.step_gpu_param_simple(i);
        }

        let op_code = 1; // SGD with momentum

        let param = &self.params[i];
        let p = param.borrow_mut();

        let grad = match &p.grad {
            Some(g) => g.clone(),
            None => return,
        };

        // Get velocity state (already on GPU)
        let vel_state = &self.velocity[i];

        // Dummy second state buffer (not used for SGD)
        let dummy_state2 = Storage::new_zeros(p.data.len(), &p.device);

        // Setup optimizer parameters
        let opt_params = OptimizerStepParams {
            op: op_code, // 0=SGD, 1=SGD+momentum
            lr: self.lr,
            beta1: self.momentum, // Used as momentum coefficient
            beta2: 0.0,           // Not used for SGD
            t: 0.0,               // Not used for SGD
            eps: 0.0,             // Not used for SGD
            weight_decay: self.weight_decay,
            _padding: 0.0,
        };

        // Run GPU optimizer step (updates params and velocity in-place)
        let success = crate::RawTensor::gpu_optimizer_step(
            &p.data,
            &grad,
            vel_state,
            &dummy_state2,
            &opt_params,
        )
        .is_some();

        // Invalidate CPU cache since parameter was updated on GPU
        if success && let Storage::Gpu { ref cpu_cache, .. } = p.data {
            *cpu_cache.borrow_mut() = None;
        }
    }

    /// GPU step for simple SGD (no momentum)
    #[cfg(feature = "gpu")]
    fn step_gpu_param_simple(&mut self, i: usize) {
        use crate::gpu::OptimizerStepParams;

        let param = &self.params[i];
        let p = param.borrow_mut();

        let grad = match &p.grad {
            Some(g) => g.clone(),
            None => return,
        };

        // Create dummy state buffers
        let dummy_state1 = Storage::new_zeros(p.data.len(), &p.device);
        let dummy_state2 = Storage::new_zeros(p.data.len(), &p.device);

        let opt_params = OptimizerStepParams {
            op: 0, // Simple SGD
            lr: self.lr,
            beta1: 0.0,
            beta2: 0.0,
            t: 0.0,
            eps: 0.0,
            weight_decay: self.weight_decay,
            _padding: 0.0,
        };

        let success = crate::RawTensor::gpu_optimizer_step(
            &p.data,
            &grad,
            &dummy_state1,
            &dummy_state2,
            &opt_params,
        )
        .is_some();

        if success && let Storage::Gpu { ref cpu_cache, .. } = p.data {
            *cpu_cache.borrow_mut() = None;
        }
    }

    /// CPU update for a single parameter
    #[cfg(feature = "gpu")]
    fn step_cpu_param(&mut self, i: usize) {
        let param = &self.params[i];
        let mut p = param.borrow_mut();

        let grad = match &p.grad {
            Some(g) => g.clone(),
            None => return,
        };

        // Apply weight decay to gradient
        let mut active_grad = grad.clone();
        if self.weight_decay != 0.0 {
            for (g, theta) in active_grad.iter_mut().zip(p.data.iter()) {
                *g += self.weight_decay * *theta;
            }
        }

        if self.momentum > 0.0 {
            // Get mutable access to CPU velocity state
            let vel_slice = self.velocity[i]
                .as_mut_slice()
                .expect("Velocity should be CPU storage");

            // Update velocity: v = momentum·v - lr·grad
            for (v, &g) in vel_slice.iter_mut().zip(active_grad.iter()) {
                *v = self.momentum * *v - self.lr * g;
            }

            // Update parameters: θ = θ + v
            let p_data = p
                .data
                .as_mut_slice()
                .expect("Parameter should be CPU storage");
            for (d, &v) in p_data.iter_mut().zip(vel_slice.iter()) {
                *d += v;
            }
        } else {
            // Simple SGD: θ = θ - lr·grad
            let p_data = p
                .data
                .as_mut_slice()
                .expect("Parameter should be CPU storage");
            for (d, &g) in p_data.iter_mut().zip(active_grad.iter()) {
                *d -= self.lr * g;
            }
        }
    }
}

// Fallback when GPU feature is not enabled
#[cfg(not(feature = "gpu"))]
impl SGD {
    fn step_gpu_param(&mut self, _i: usize) {
        unreachable!("GPU step should not be called without GPU feature");
    }

    fn step_gpu_param_simple(&mut self, _i: usize) {
        unreachable!("GPU step should not be called without GPU feature");
    }
}

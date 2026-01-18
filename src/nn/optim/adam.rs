use crate::storage::Storage;
use crate::tensor::Tensor;

pub struct Adam {
    params: Vec<Tensor>,
    lr: f32,
    betas: (f32, f32),
    eps: f32,
    weight_decay: f32,
    m: Vec<Storage>, // 1st moment (can be CPU or GPU storage)
    v: Vec<Storage>, // 2nd moment (can be CPU or GPU storage)
    t: usize,        // timestep
}

impl Adam {
    #[must_use]
    pub fn new(
        params: Vec<Tensor>,
        lr: f32,
        betas: (f32, f32),
        eps: f32,
        weight_decay: f32,
    ) -> Self {
        // Create state matching the device of each parameter
        let m: Vec<Storage> = params
            .iter()
            .map(|p| {
                let param = p.borrow();
                let len = param.data.len();
                let device = param.device.clone();
                drop(param);
                // Create zero state on the same device as parameter
                Storage::new_zeros(len, &device)
            })
            .collect();

        let v: Vec<Storage> = params
            .iter()
            .map(|p| {
                let param = p.borrow();
                let len = param.data.len();
                let device = param.device.clone();
                drop(param);
                // Create zero state on the same device as parameter
                Storage::new_zeros(len, &device)
            })
            .collect();

        Adam {
            params,
            lr,
            betas,
            eps,
            weight_decay,
            m,
            v,
            t: 0,
        }
    }

    pub fn zero_grad(&self) {
        for param in &self.params {
            param.borrow_mut().grad = None;
        }
    }

    pub fn step(&mut self) {
        self.t += 1;

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

        let param = &self.params[i];
        let p = param.borrow_mut();

        let grad = match &p.grad {
            Some(g) => g.clone(),
            None => return,
        };

        // State is already on GPU (stored as Storage)
        let m_state = &self.m[i];
        let v_state = &self.v[i];

        // Setup optimizer parameters
        let opt_params = OptimizerStepParams {
            op: 2, // Adam
            lr: self.lr,
            beta1: self.betas.0,
            beta2: self.betas.1,
            t: self.t as f32,
            eps: self.eps,
            weight_decay: self.weight_decay,
            _padding: 0.0,
        };

        // Run GPU optimizer step (updates params, m, v in-place)
        let success =
            crate::RawTensor::gpu_optimizer_step(&p.data, &grad, m_state, v_state, &opt_params)
                .is_some();

        // Invalidate CPU cache since parameter was updated on GPU
        if success && let Storage::Gpu { ref cpu_cache, .. } = p.data {
            *cpu_cache.borrow_mut() = None;
        }
    }

    /// CPU update for a single parameter
    #[allow(clippy::needless_range_loop)]
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

        // Get mutable access to CPU state
        let m_slice = self.m[i]
            .as_mut_slice()
            .expect("State should be CPU storage");
        let v_slice = self.v[i]
            .as_mut_slice()
            .expect("State should be CPU storage");

        // Update biased moments
        for j in 0..active_grad.len() {
            m_slice[j] = self.betas.0 * m_slice[j] + (1.0 - self.betas.0) * active_grad[j];
            v_slice[j] = self.betas.1 * v_slice[j] + (1.0 - self.betas.1) * active_grad[j].powi(2);
        }

        // Bias correction
        let m_hat_scale = 1.0 / (1.0 - self.betas.0.powi(self.t as i32));
        let v_hat_scale = 1.0 / (1.0 - self.betas.1.powi(self.t as i32));

        // Update parameters
        let p_data = p
            .data
            .as_mut_slice()
            .expect("Parameter should be CPU storage");
        for j in 0..p_data.len() {
            let m_hat = m_slice[j] * m_hat_scale;
            let v_hat = v_slice[j] * v_hat_scale;
            p_data[j] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

// Fallback when GPU feature is not enabled
#[cfg(not(feature = "gpu"))]
impl Adam {
    fn step_gpu_param(&mut self, _i: usize) {
        unreachable!("GPU step should not be called without GPU feature");
    }
}

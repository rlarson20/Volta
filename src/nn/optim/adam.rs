use crate::index::{fetch, slot};
use crate::real::{Real, dispatch_real};
use crate::storage::Storage;
use crate::tensor::Tensor;

/// Adam optimizer (Kingma & Ba): per-parameter step sizes from running first and second moments of the gradient.
pub struct Adam {
    /// Parameters being optimized.
    pub(super) params: Vec<Tensor>,
    /// Learning rate.
    lr: f32,
    /// Decay rates `(beta1, beta2)` of the first and second moment estimates.
    betas: (f32, f32),
    /// Added to the denominator for numerical stability.
    eps: f32,
    /// L2 penalty, added to the gradient before the moment updates.
    weight_decay: f32,
    /// First-moment estimate per parameter, on that parameter's device.
    m: Vec<Storage>,
    /// Second-moment estimate per parameter, on that parameter's device.
    v: Vec<Storage>,
    /// Number of steps taken, for bias correction.
    t: usize,
}

impl Adam {
    /// Optimizer over `params` with zeroed moment estimates on each parameter's device.
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
                let dtype = param.data.dtype();
                drop(param);
                // Zero state on the parameter's device and in its dtype (GPU storage is
                // f32-only).
                if device.is_gpu() {
                    Storage::new_zeros(len, &device)
                } else {
                    Storage::new_zeros(len, &device).cast(dtype)
                }
            })
            .collect();

        let v: Vec<Storage> = params
            .iter()
            .map(|p| {
                let param = p.borrow();
                let len = param.data.len();
                let device = param.device.clone();
                let dtype = param.data.dtype();
                drop(param);
                // Zero state on the parameter's device and in its dtype (GPU storage is
                // f32-only).
                if device.is_gpu() {
                    Storage::new_zeros(len, &device)
                } else {
                    Storage::new_zeros(len, &device).cast(dtype)
                }
            })
            .collect();

        Self {
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

    /// Clears the gradient of every parameter.
    pub fn zero_grad(&self) {
        for param in &self.params {
            param.borrow_mut().grad = None;
        }
    }

    /// # Panics
    /// unwrap params
    pub fn step(&mut self) {
        // Each parameter steps in its own dtype (a mixed list is fine, as in PyTorch);
        // each must have a kernel. Checked for all before any is updated.
        for param in &self.params {
            crate::real::compute_dtype(|| "Adam step".to_string(), &[param.dtype()]);
        }
        self.t += 1;

        // Process each parameter based on its device
        for i in 0..self.params.len() {
            // SAFETY: We're iterating over valid indices 0..self.params.len()
            debug_assert!(i < self.params.len(), "Parameter index out of bounds");
            let param = self.params.get(i).unwrap();
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
    ///
    /// # Panics
    /// If the GPU kernel cannot run: a CPU fallback cannot write GPU storage, and skipping
    /// the update would leave the parameter silently untrained.
    #[cfg(feature = "gpu")]
    fn step_gpu_param(&mut self, i: usize) {
        use crate::gpu::OptimizerStepParams;

        let param = self.params.get(i).unwrap();
        let mut p = param.borrow_mut();

        let grad = match &p.grad {
            Some(g) => g.clone(),
            None => return,
        };

        // State is already on GPU (stored as Storage)
        let m_state = self.m.get_mut(i).unwrap();
        let v_state = self.v.get_mut(i).unwrap();

        // Setup optimizer parameters
        let opt_params = OptimizerStepParams {
            op: 2, // Adam
            lr: self.lr,
            beta1: self.betas.0,
            beta2: self.betas.1,
            t: self.t as f32,
            eps: self.eps,
            weight_decay: self.weight_decay,
            padding: 0.0,
        };

        // Updates params, m and v in place. `gpu_optimizer_step` unshares each buffer first
        // and drops its host copy, so snapshots and later reads stay correct.
        crate::RawTensor::gpu_optimizer_step(&mut p.data, &grad, m_state, v_state, &opt_params)
            .expect("GPU Adam step failed: the parameter was not updated");
    }

    /// CPU update for a single parameter
    fn step_cpu_param(&mut self, i: usize) {
        let param = self.params.get(i).unwrap();
        let mut p = param.borrow_mut();

        let grad = match &p.grad {
            Some(g) => g.clone(),
            None => return,
        };

        let dtype = p.data.dtype();
        dispatch_real!(dtype, T => {
        let lr = T::from_f64(f64::from(self.lr));
        let eps = T::from_f64(f64::from(self.eps));
        let weight_decay = T::from_f64(f64::from(self.weight_decay));
        let (beta1, beta2) = (
            T::from_f64(f64::from(self.betas.0)),
            T::from_f64(f64::from(self.betas.1)),
        );
        // Apply weight decay to gradient
        let mut active_grad = grad;
        if self.weight_decay != 0.0 {
            for (g, theta) in active_grad
                .host_mut::<T>()
                .iter_mut()
                .zip(p.data.host::<T>().iter())
            {
                *g += weight_decay * *theta;
            }
        }

        // Get mutable access to CPU state
        // SAFETY: Optimizer state is created as CPU storage for CPU parameters
        debug_assert!(self.m.len() == self.params.len(), "State length mismatch");
        debug_assert!(self.v.len() == self.params.len(), "State length mismatch");
        let (m, v) = (
            self.m.get_mut(i).expect("Adam: state per parameter"),
            self.v.get_mut(i).expect("Adam: state per parameter"),
        );
        for s in [&*m, &*v] {
            assert_eq!(
                s.dtype(),
                dtype,
                "Adam state dtype; build the optimizer after Module::to_dtype"
            );
        }
        let (mut m_guard, mut v_guard) = (m.host_mut::<T>(), v.host_mut::<T>());
        let (m_slice, v_slice) = (&mut *m_guard, &mut *v_guard);

        // Update biased moments
        let g_host = active_grad.host::<T>();
        for j in 0..g_host.len() {
            let m_val = fetch(m_slice, j);
            let g_val = fetch(&g_host, j);
            *slot(m_slice, j) = beta1.mul_add(m_val, (1.0 - beta1) * g_val);
            let v_val = fetch(v_slice, j);
            *slot(v_slice, j) = beta2
                .mul_add(v_val, (1.0 - beta2) * g_val.powi(2));
        }

        // Bias correction
        let t = i32::try_from(self.t).expect("Adam step count fits i32");
        let m_hat_scale = 1.0 / (1.0 - beta1.powi(t));
        let v_hat_scale = 1.0 / (1.0 - beta2.powi(t));

        // Update parameters
        let mut p_guard = p.data.host_mut::<T>();
        let p_data = &mut *p_guard;
        for j in 0..p_data.len() {
            let m_val = fetch(m_slice, j);
            let v_val = fetch(v_slice, j);
            let m_hat = m_val * m_hat_scale;
            let v_hat = v_val * v_hat_scale;
            *slot(p_data, j) -= lr * m_hat / (v_hat.sqrt() + eps);
        }
        });
    }
}

// Fallback when GPU feature is not enabled
#[cfg(not(feature = "gpu"))]
impl Adam {
    /// Stub for builds without the `gpu` feature; a GPU parameter cannot exist there.
    fn step_gpu_param(&mut self, _i: usize) {
        unreachable!("GPU step should not be called without GPU feature");
    }
}

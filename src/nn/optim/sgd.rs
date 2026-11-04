use crate::real::{Real, dispatch_real};
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
    /// Parameters being optimized.
    pub(super) params: Vec<Tensor>,
    /// Learning rate.
    lr: f32,
    /// Momentum coefficient; 0 disables momentum.
    momentum: f32,
    /// L2 penalty.
    weight_decay: f32,
    /// Momentum buffer per parameter, on that parameter's device (empty if `momentum` is 0).
    velocity: Vec<Storage>,
}

impl SGD {
    /// Create a new SGD optimizer
    ///
    /// # Arguments
    /// * `params` - List of parameters to optimize
    /// * `lr` - Learning rate (typical: 0.01 to 0.1)
    /// * `momentum` - Momentum coefficient (typical: 0.9, or 0.0 for no momentum)
    /// * `weight_decay` - L2 penalty (typical: 1e-4, or 0.0 for none)
    ///
    /// # Panics
    /// If a hyperparameter is invalid; see [`Self::try_new`].
    #[must_use]
    #[track_caller]
    pub fn new(params: Vec<Tensor>, lr: f32, momentum: f32, weight_decay: f32) -> Self {
        Self::try_new(params, lr, momentum, weight_decay).unwrap_or_else(|e| panic!("SGD: {e}"))
    }

    /// [`Self::new`], checking the hyperparameters first, as PyTorch does.
    ///
    /// # Errors
    /// `VoltaError::InvalidParameter` if `lr`, `momentum` or `weight_decay` is negative
    /// or NaN. A negative momentum used to be ignored on the CPU and to panic on the GPU
    /// (audit A3).
    pub fn try_new(
        params: Vec<Tensor>,
        lr: f32,
        momentum: f32,
        weight_decay: f32,
    ) -> crate::Result<Self> {
        super::check_non_negative("learning rate", lr)?;
        super::check_non_negative("momentum", momentum)?;
        super::check_non_negative("weight_decay", weight_decay)?;
        // Create velocity state matching the device and dtype of each parameter
        let velocity = if momentum > 0.0 {
            params
                .iter()
                .map(|p| {
                    let param = p.borrow();
                    let len = param.data.len();
                    let device = param.device.clone();
                    let dtype = param.data.dtype();
                    drop(param);
                    // In the parameter's dtype on its device (a no-op cast for f32).
                    Storage::new_zeros(len, &device).cast(dtype)
                })
                .collect()
        } else {
            vec![]
        };

        Ok(Self {
            params,
            lr,
            momentum,
            weight_decay,
            velocity,
        })
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
    /// # Panics
    /// unwrap params
    pub fn step(&mut self) {
        // Each parameter steps in its own dtype (a mixed list is fine, as in PyTorch);
        // each must have a kernel. Checked for all before any is updated.
        for param in &self.params {
            crate::real::compute_dtype(|| "SGD step".to_string(), &[param.dtype()]);
        }
        // Process each parameter based on its device
        for i in 0..self.params.len() {
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

        let len = p.data.len();
        let device = p.device.clone();
        // Plain SGD keeps no state, but the kernel binds both state buffers, in the
        // parameter's dtype (the kernel binds one element type).
        let dtype = p.data.dtype();
        let mut dummy_state1 = Storage::new_zeros(len, &device).cast(dtype);
        let mut dummy_state2 = Storage::new_zeros(len, &device).cast(dtype);
        let (op, beta1, state1) = if self.momentum == 0.0 {
            (0, 0.0, &mut dummy_state1)
        } else {
            // Velocity is already on the GPU, with the parameter.
            (1, self.momentum, self.velocity.get_mut(i).unwrap())
        };

        let opt_params = OptimizerStepParams {
            op, // 0=SGD, 1=SGD+momentum
            lr: self.lr,
            beta1, // Momentum coefficient
            beta2: 0.0,
            t: 0.0,
            eps: 0.0,
            weight_decay: self.weight_decay,
            padding: 0.0,
        };

        // Updates the parameter (and velocity) in place. `gpu_optimizer_step` unshares each
        // buffer first and drops its host copy, so snapshots and later reads stay correct.
        crate::RawTensor::gpu_optimizer_step(
            &mut p.data,
            &grad,
            state1,
            &mut dummy_state2,
            &opt_params,
        )
        .expect("GPU SGD step failed: the parameter was not updated");
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
        let momentum = T::from_f64(f64::from(self.momentum));
        let weight_decay = T::from_f64(f64::from(self.weight_decay));
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

        if self.momentum > 0.0 {
            // Get mutable access to CPU velocity state
            let vel = self.velocity.get_mut(i).expect("SGD: a velocity per parameter");
            assert_eq!(
                vel.dtype(),
                dtype,
                "SGD velocity dtype; build the optimizer after Module::to_dtype"
            );
            let mut vel_slice = vel.host_mut::<T>();

            // Update velocity: v = momentum·v - lr·grad
            for (v, &g) in vel_slice.iter_mut().zip(active_grad.host::<T>().iter()) {
                *v = momentum.mul_add(*v, -(lr * g));
            }

            // Update parameters: θ = θ + v
            let mut p_data = p.data.host_mut::<T>();
            for (d, &v) in p_data.iter_mut().zip(vel_slice.iter()) {
                *d += v;
            }
        } else {
            // Simple SGD: θ = θ - lr·grad
            let mut p_data = p.data.host_mut::<T>();
            for (d, &g) in p_data.iter_mut().zip(active_grad.host::<T>().iter()) {
                *d -= lr * g;
            }
        }
        });
    }
}

// Fallback when GPU feature is not enabled
#[cfg(not(feature = "gpu"))]
impl SGD {
    /// Stub for builds without the `gpu` feature; a GPU parameter cannot exist there.
    fn step_gpu_param(&mut self, _i: usize) {
        unreachable!("GPU step should not be called without GPU feature");
    }
}

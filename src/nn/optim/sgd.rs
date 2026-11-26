use crate::error::OrPanic;
use crate::real::{Real, dispatch_real};
use crate::storage::Storage;
use crate::tensor::Tensor;

/// Stochastic Gradient Descent optimizer with optional momentum
///
/// Update rule:
/// - Without momentum: θ ← θ - lr·∇θ
/// - With momentum: v ← β·v + ∇θ, θ ← θ - lr·v
///
/// Momentum helps accelerate convergence and dampen oscillations. The velocity holds
/// gradients only, with `lr` applied at each step, as in PyTorch (`momentum_buffer`, no
/// dampening or Nesterov). So a learning rate that changes between steps applies to the
/// whole velocity at once, and the buffer means the same thing as torch's (#492).
#[derive(Debug)]
pub struct SGD {
    /// Parameters being optimized.
    pub(super) params: Vec<Tensor>,
    /// Learning rate.
    lr: f32,
    /// Momentum coefficient; 0 disables momentum.
    momentum: f32,
    /// L2 penalty.
    weight_decay: f32,
    /// Momentum buffer per parameter, allocated on the parameter's first step (`None` until
    /// then, and always when `momentum` is 0), so a model converted with `to_device` or
    /// `to_dtype` before that step is stepped in its new device and dtype. A conversion
    /// after the first step panics instead.
    velocity: Vec<Option<Storage>>,
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
        Self::try_new(params, lr, momentum, weight_decay).or_panic("SGD")
    }

    /// [`Self::new`], checking the hyperparameters first, as PyTorch does.
    ///
    /// # Errors
    /// `VoltaError::InvalidParameter` if `lr`, `momentum` or `weight_decay` is negative
    /// or NaN, or if the same tensor is listed twice in `params` (#101). A negative
    /// momentum used to be ignored on the CPU and to panic on the GPU (audit A3).
    /// A non-leaf parameter (one with a `grad_fn`) is rejected too (#102); a frozen leaf
    /// (`requires_grad == false`) is accepted and skipped by `step`, as in PyTorch.
    pub fn try_new(
        params: Vec<Tensor>,
        lr: f32,
        momentum: f32,
        weight_decay: f32,
    ) -> crate::Result<Self> {
        /// The op named in errors.
        const OP: &str = "SGD::new";
        super::check_unique(OP, &params)?;
        super::check_leaves(OP, &params)?;
        super::check_non_negative(OP, "lr", "learning rate", lr)?;
        super::check_non_negative(OP, "momentum", "momentum", momentum)?;
        super::check_non_negative(OP, "weight_decay", "weight_decay", weight_decay)?;
        let velocity = params.iter().map(|_| None).collect();

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
            param.zero_grad();
        }
    }

    /// Perform one optimization step
    ///
    /// Updates all parameters using their accumulated gradients.
    ///
    /// # Panics
    /// If a parameter's dtype is not a float. This is checked for every parameter before
    /// any is updated.
    #[track_caller]
    pub fn step(&mut self) {
        // Each parameter steps in its own dtype (a mixed list is fine, as in PyTorch);
        // each must have a kernel. Checked for all before any is updated.
        for param in &self.params {
            crate::real::compute_dtype(|| "SGD step".to_string(), &[param.dtype()]);
        }
        super::assert_grad_lens("SGD", &self.params);
        // Process each parameter based on its device
        for i in 0..self.params.len() {
            let param = self.params.get(i).expect("SGD: a parameter per index");
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

        let param = self.params.get(i).expect("SGD: a parameter per index");
        let Some(grad) = param.grad_storage() else {
            return;
        };

        // Plain SGD keeps no state: the kernel binds a cached placeholder for each state it
        // doesn't use, so a step allocates nothing (#207).
        let (op, beta1, state1) = if self.momentum == 0.0 {
            (0, 0.0, None)
        } else {
            // Allocated on the parameter's first step, on its current device.
            (
                1,
                self.momentum,
                Some(
                    self.velocity
                        .get_mut(i)
                        .expect("SGD: velocity slot per parameter")
                        .get_or_insert_with(|| super::zeros_like(param)),
                ),
            )
        };

        let opt_params = OptimizerStepParams {
            op, // 0=SGD, 1=SGD+momentum
            lr: self.lr,
            beta1, // Momentum coefficient
            beta2: 0.0,
            m_hat_scale: 0.0,
            eps: 0.0,
            weight_decay: self.weight_decay,
            v_hat_scale: 0.0,
        };

        // Updates the parameter (and velocity) in place, bumping its version (S9).
        // `gpu_optimizer_step` unshares each buffer first and drops its host copy, so
        // snapshots and later reads stay correct.
        param
            .update_data(|data| {
                crate::RawTensor::gpu_optimizer_step(data, &grad, state1, None, &opt_params)
            })
            .expect("GPU SGD step failed: the parameter was not updated");
    }

    /// CPU update for a single parameter
    fn step_cpu_param(&mut self, i: usize) {
        let param = self.params.get(i).expect("SGD: a parameter per index");
        let Some(grad) = param.grad_storage() else {
            return;
        };

        let dtype = param.dtype();
        if self.momentum > 0.0 {
            // Allocated on the parameter's first step, in its current dtype.
            self.velocity
                .get_mut(i)
                .expect("SGD: velocity slot per parameter")
                .get_or_insert_with(|| super::zeros_like(param));
        }
        // The step rewrites the parameter in place (S9; `update_data` bumps the version).
        param.update_data(|data| {
            dispatch_real!(dtype, T => {
            let lr = T::from_f64(f64::from(self.lr));
            let momentum = T::from_f64(f64::from(self.momentum));
            let weight_decay = T::from_f64(f64::from(self.weight_decay));
            // Apply weight decay to gradient
            let mut active_grad = grad;
            if self.weight_decay != 0.0 {
                active_grad.with_host_mut::<T, _>(|g_host| {
                    for (g, theta) in g_host.iter_mut().zip(data.host::<T>().iter()) {
                        *g += weight_decay * *theta;
                    }
                });
            }

            if self.momentum > 0.0 {
                let vel = self
                    .velocity
                    .get_mut(i)
                    .and_then(Option::as_mut)
                    .expect("SGD: velocity allocated above");
                assert_eq!(
                    vel.dtype(),
                    dtype,
                    "SGD velocity dtype; convert the model before the optimizer's first step"
                );
                vel.with_host_mut::<T, _>(|vel_slice| {
                    // Update velocity: v = momentum·v + grad (torch's form, #492)
                    for (v, &g) in vel_slice.iter_mut().zip(active_grad.host::<T>().iter()) {
                        *v = momentum.mul_add(*v, g);
                    }

                    // Update parameters: θ = θ - lr·v
                    data.with_host_mut::<T, _>(|p_data| {
                        for (d, &v) in p_data.iter_mut().zip(vel_slice.iter()) {
                            *d -= lr * v;
                        }
                    });
                });
            } else {
                // Simple SGD: θ = θ - lr·grad
                data.with_host_mut::<T, _>(|p_data| {
                    for (d, &g) in p_data.iter_mut().zip(active_grad.host::<T>().iter()) {
                        *d -= lr * g;
                    }
                });
            }
            });
        });
    }
}

// Fallback when GPU feature is not enabled
#[cfg(not(feature = "gpu"))]
impl SGD {
    /// Stub for builds without the `gpu` feature; a GPU parameter cannot exist there.
    #[expect(
        clippy::unused_self,
        reason = "keeps the GPU variant's signature so the call site is the same with and without the feature"
    )]
    fn step_gpu_param(&mut self, _i: usize) {
        unreachable!("GPU step should not be called without GPU feature");
    }
}

#[cfg(test)]
mod tests {
    use super::SGD;
    use crate::index::fetch;
    use crate::storage::Storage;
    use crate::tensor::{RawTensor, Tensor};

    /// Starting parameter.
    const P0: [f32; 4] = [0.5, -1.25, 2.0, 0.75];
    /// Learning rate at each of the five steps: it drops after step 2 and rises at step 4,
    /// as a scheduler would.
    const LRS: [f32; 5] = [0.1, 0.1, 0.02, 0.05, 0.05];
    const MOMENTUM: f32 = 0.9;
    const WEIGHT_DECAY: f32 = 0.01;

    /// The gradient fed at step `k`.
    fn grad(k: usize) -> Vec<f32> {
        (0..P0.len())
            .map(|i| {
                #[expect(clippy::cast_precision_loss, reason = "small test indices")]
                let (i, k) = (i as f32, k as f32);
                (0.7 * i + 1.3 * k).sin() + 0.25
            })
            .collect()
    }

    /// The parameter after each step under PyTorch's `SGD` (no dampening, not Nesterov):
    /// `g += wd·θ; v = β·v + g (v = g on the first step); θ -= lr·v`, in f64 with the f32
    /// hyperparameters widened.
    fn torch_reference() -> Vec<Vec<f64>> {
        let (beta, wd) = (f64::from(MOMENTUM), f64::from(WEIGHT_DECAY));
        let mut p: Vec<f64> = P0.iter().copied().map(f64::from).collect();
        let mut v = vec![0.0f64; P0.len()];
        LRS.iter()
            .enumerate()
            .map(|(k, &lr)| {
                for ((p, v), g) in p.iter_mut().zip(&mut v).zip(grad(k)) {
                    let g = f64::from(g) + wd * *p;
                    *v = beta * *v + g;
                    *p -= f64::from(lr) * *v;
                }
                p.clone()
            })
            .collect()
    }

    /// Steps `param` (on any device) with the learning rate changed between steps, and
    /// compares it against [`torch_reference`] after each step.
    fn check_lr_schedule(param: &Tensor, rel_tol: f64) {
        let mut opt = SGD::new(vec![param.clone()], fetch(&LRS, 0), MOMENTUM, WEIGHT_DECAY);
        let want = torch_reference();
        for (k, (&lr, want)) in LRS.iter().zip(&want).enumerate() {
            opt.lr = lr;
            param.zero_grad();
            param.set_grad(Storage::cpu(grad(k)));
            opt.step();
            let got = param.data();
            assert_eq!(got.len(), want.len());
            for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
                let err = (f64::from(g) - w).abs() / w.abs().max(1.0);
                assert!(
                    err <= rel_tol,
                    "step {k} (lr {lr}), element {i}: got {g}, torch {w}, rel err {err:.3e}"
                );
            }
        }
    }

    /// #492: the velocity holds gradients and `lr` scales it at each step, as in torch, so
    /// a learning rate that changes mid-run still matches torch. The old form
    /// (`v = β·v - lr·g; θ += v`) kept earlier steps' velocity at their old `lr`.
    #[test]
    fn momentum_matches_torch_when_lr_changes() {
        let p = RawTensor::new(P0.to_vec(), &[P0.len()], true);
        check_lr_schedule(&p, 8.0 * f64::from(f32::EPSILON));
    }

    /// #492 on the GPU kernel. Returns early without an adapter, unless
    /// `VOLTA_REQUIRE_GPU` is set.
    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_momentum_matches_torch_when_lr_changes() {
        use crate::device::Device;
        let Some(gpu) = Device::gpu() else {
            assert!(
                std::env::var_os("VOLTA_REQUIRE_GPU").is_none(),
                "VOLTA_REQUIRE_GPU is set but no GPU is available"
            );
            return;
        };
        let p = RawTensor::new(P0.to_vec(), &[P0.len()], false).to_device(gpu);
        p.set_requires_grad(true);
        assert_ne!(p.device(), Device::CPU);
        check_lr_schedule(&p, 32.0 * f64::from(f32::EPSILON));
    }
}

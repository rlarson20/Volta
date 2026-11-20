use crate::error::OrPanic;
use crate::index::{fetch, slot};
use crate::real::{Real, dispatch_real};
use crate::storage::Storage;
use crate::tensor::Tensor;

/// Adam optimizer (Kingma & Ba): per-parameter step sizes from running first and second moments of the gradient.
#[derive(Debug)]
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
    /// Per-parameter state, allocated on the parameter's first step (`None` until then), so
    /// a model converted with `to_device` or `to_dtype` before that step is stepped in its
    /// new device and dtype. A conversion after the first step panics instead.
    state: Vec<Option<AdamState>>,
}

/// One parameter's moment estimates and step count.
#[derive(Debug)]
struct AdamState {
    /// First-moment estimate, on the parameter's device.
    m: Storage,
    /// Second-moment estimate, on the parameter's device.
    v: Storage,
    /// Steps taken with a gradient, for bias correction (torch counts per parameter).
    step: usize,
}

impl AdamState {
    /// Zeroed state in `p`'s current dtype and device.
    fn new(p: &Tensor) -> Self {
        Self {
            m: super::zeros_like(p),
            v: super::zeros_like(p),
            step: 0,
        }
    }
}

/// The refusal for the first half (f16 or bf16) parameter in `params`, if any (#317).
fn half_parameter(params: &[Tensor]) -> Option<String> {
    params.iter().enumerate().find_map(|(i, p)| {
        let dtype = p.dtype();
        matches!(dtype, crate::DType::F16 | crate::DType::BF16).then(|| {
            format!(
                "parameter {i} is {dtype}; Adam keeps its moment estimates in the \
                 parameter's dtype, where the second moment underflows (#317). Keep the \
                 parameters f32 and run the forward under amp::autocast"
            )
        })
    })
}

impl Adam {
    /// Optimizer over `params`. Moment estimates are allocated on each parameter's first step.
    ///
    /// # Panics
    /// If a hyperparameter is invalid; see [`Self::try_new`].
    #[must_use]
    #[track_caller]
    pub fn new(
        params: Vec<Tensor>,
        lr: f32,
        betas: (f32, f32),
        eps: f32,
        weight_decay: f32,
    ) -> Self {
        Self::try_new(params, lr, betas, eps, weight_decay).or_panic("Adam")
    }

    /// [`Self::new`], checking the hyperparameters first, as PyTorch does.
    ///
    /// # Errors
    /// `VoltaError::InvalidParameter` if `lr`, `eps` or `weight_decay` is negative or NaN,
    /// or a beta is outside `[0, 1)` (a beta of 1 makes the bias correction divide by 0),
    /// or if the same tensor is listed twice in `params` (#101).
    /// A non-leaf parameter (one with a `grad_fn`) is rejected too (#102); a frozen leaf
    /// (`requires_grad == false`) is accepted and skipped by `step`, as in PyTorch.
    /// So is a half (f16 or bf16) parameter (#317): Adam keeps its moment estimates in the
    /// parameter's dtype, and in f16 the second moment `(1 - beta2) * g^2` underflows to 0
    /// for `|g|` below about 5.5e-3, which gives a finite but wrong step (and a different
    /// one on the GPU). Keep the parameters f32 and run the forward under
    /// [`crate::amp::autocast`] instead.
    pub fn try_new(
        params: Vec<Tensor>,
        lr: f32,
        betas: (f32, f32),
        eps: f32,
        weight_decay: f32,
    ) -> crate::Result<Self> {
        /// The op named in errors.
        const OP: &str = "Adam::new";
        super::check_unique(OP, &params)?;
        super::check_leaves(OP, &params)?;
        if let Some(message) = half_parameter(&params) {
            return Err(crate::VoltaError::invalid(OP, "params", message));
        }
        super::check_non_negative(OP, "lr", "learning rate", lr)?;
        super::check_non_negative(OP, "eps", "epsilon", eps)?;
        super::check_beta(OP, 0, betas.0)?;
        super::check_beta(OP, 1, betas.1)?;
        super::check_non_negative(OP, "weight_decay", "weight_decay", weight_decay)?;
        let state = params.iter().map(|_| None).collect();

        Ok(Self {
            params,
            lr,
            betas,
            eps,
            weight_decay,
            state,
        })
    }

    /// Clears the gradient of every parameter.
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
    /// If a parameter's dtype is not f32 or f64: an integer dtype has no kernel, and a half
    /// parameter (cast with `Module::to_dtype` after the optimizer was built) is refused as
    /// [`Self::try_new`] refuses it. This is checked for every parameter before any is
    /// updated.
    #[track_caller]
    pub fn step(&mut self) {
        // Each parameter steps in its own dtype (a mixed list is fine, as in PyTorch);
        // each must have a kernel. Checked for all before any is updated.
        for param in &self.params {
            crate::real::compute_dtype(|| "Adam step".to_string(), &[param.dtype()]);
        }
        if let Some(message) = half_parameter(&self.params) {
            panic!("Adam: {message}");
        }
        super::assert_grad_lens("Adam", &self.params);

        // The parameters that have a gradient, and where each lives. Collected first because
        // the per-parameter steps take `&mut self`.
        let targets: Vec<(usize, bool)> = self
            .params
            .iter()
            .enumerate()
            .filter_map(|(i, param)| {
                let p = param.borrow();
                p.grad.is_some().then(|| (i, p.device.is_gpu()))
            })
            .collect();
        for (i, is_gpu) in targets {
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

        let param = self.params.get(i).expect("Adam: a parameter per index");
        let Some(grad) = param.grad_storage() else {
            return;
        };

        // Allocated here on the parameter's first step, on its current device.
        let state = self
            .state
            .get_mut(i)
            .expect("Adam: state slot per parameter")
            .get_or_insert_with(|| AdamState::new(param));
        state.step += 1;
        let (m_hat_scale, v_hat_scale) = bias_correction_scales(self.betas, state.step);
        let (m_state, v_state) = (&mut state.m, &mut state.v);

        // Setup optimizer parameters. The scales are narrowed once here, so every element
        // gets the correctly rounded factor rather than an f32 `pow` of its own (#304).
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the GPU kernel works in f32; narrowing a correctly computed f64 scale rounds once"
        )]
        let opt_params = OptimizerStepParams {
            op: 2, // Adam
            lr: self.lr,
            beta1: self.betas.0,
            beta2: self.betas.1,
            m_hat_scale: m_hat_scale as f32,
            eps: self.eps,
            weight_decay: self.weight_decay,
            v_hat_scale: v_hat_scale as f32,
        };

        // Updates params, m and v in place, bumping the parameter's version (S9).
        // `gpu_optimizer_step` unshares each buffer first and drops its host copy, so
        // snapshots and later reads stay correct.
        param
            .update_data(|data| {
                crate::RawTensor::gpu_optimizer_step(
                    data,
                    &grad,
                    Some(m_state),
                    Some(v_state),
                    &opt_params,
                )
            })
            .expect("GPU Adam step failed: the parameter was not updated");
    }

    /// CPU update for a single parameter
    fn step_cpu_param(&mut self, i: usize) {
        let param = self.params.get(i).expect("Adam: a parameter per index");
        let Some(grad) = param.grad_storage() else {
            return;
        };

        let dtype = param.dtype();
        // Allocated here on the parameter's first step, in its current dtype; before the
        // update, which holds the parameter mutably.
        let state = self
            .state
            .get_mut(i)
            .expect("Adam: state slot per parameter")
            .get_or_insert_with(|| AdamState::new(param));
        // The step rewrites the parameter in place (S9; `update_data` bumps the version).
        param.update_data(|data| {
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
                active_grad.with_host_mut::<T, _>(|g_host| {
                    for (g, theta) in g_host.iter_mut().zip(data.host::<T>().iter()) {
                        *g += weight_decay * *theta;
                    }
                });
            }

            state.step += 1;
            let step = state.step;
            for s in [&state.m, &state.v] {
                assert_eq!(
                    s.dtype(),
                    dtype,
                    "Adam state dtype; convert the model before the optimizer's first step"
                );
            }
            // Bias correction, the same f64 scales the GPU kernel gets (#304)
            let (m_hat_scale, v_hat_scale) = bias_correction_scales(self.betas, step);
            let (m_hat_scale, v_hat_scale) = (T::from_f64(m_hat_scale), T::from_f64(v_hat_scale));

            state.m.with_host_mut::<T, _>(|m_slice| {
            state.v.with_host_mut::<T, _>(|v_slice| {
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

            // Update parameters
            data.with_host_mut::<T, _>(|p_data| {
            for j in 0..p_data.len() {
                let m_val = fetch(m_slice, j);
                let v_val = fetch(v_slice, j);
                let m_hat = m_val * m_hat_scale;
                let v_hat = v_val * v_hat_scale;
                *slot(p_data, j) -= lr * m_hat / (v_hat.sqrt() + eps);
            }
            });
            });
            });
            });
        });
    }
}

/// Adam's bias-correction scales `(1 / (1 - beta1^t), 1 / (1 - beta2^t))` for step `t`,
/// computed in `f64` so that both backends round each scale once, to their own dtype (#304).
///
/// At `t = 1` with `beta2 = 0.999`, `1 - beta2^t` is `1e-3`, so one ulp of error in an f32
/// power becomes about `6e-5` relative in the scale, and it scales every element alike.
///
/// # Panics
/// If `step` does not fit an `i32`.
fn bias_correction_scales(betas: (f32, f32), step: usize) -> (f64, f64) {
    let t = i32::try_from(step).expect("Adam step count fits i32");
    let scale = |beta: f32| 1.0 / (1.0 - f64::from(beta).powi(t));
    (scale(betas.0), scale(betas.1))
}

// Fallback when GPU feature is not enabled
#[cfg(not(feature = "gpu"))]
impl Adam {
    /// Stub for builds without the `gpu` feature; a GPU parameter cannot exist there.
    #[expect(
        clippy::unused_self,
        reason = "keeps the GPU variant's signature so the call site is the same with and without the feature"
    )]
    fn step_gpu_param(&mut self, _i: usize) {
        unreachable!("GPU step should not be called without GPU feature");
    }
}

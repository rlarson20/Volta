use crate::error::OrPanic;
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
    /// Per-parameter state, allocated on the parameter's first step (`None` until then), so
    /// a model converted with `to_device` or `to_dtype` before that step is stepped in its
    /// new device and dtype. A conversion after the first step panics instead.
    state: Vec<Option<AdamState>>,
}

/// One parameter's moment estimates and step count.
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
    fn new(p: &crate::RawTensor) -> Self {
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
                "Adam: parameter {i} is {dtype}; Adam keeps its moment estimates in the \
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
        super::check_unique(&params)?;
        super::check_leaves(&params)?;
        if let Some(message) = half_parameter(&params) {
            return Err(crate::VoltaError::InvalidParameter(message));
        }
        super::check_non_negative("learning rate", lr)?;
        super::check_non_negative("epsilon", eps)?;
        super::check_beta(0, betas.0)?;
        super::check_beta(1, betas.1)?;
        super::check_non_negative("weight_decay", weight_decay)?;
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
            panic!("{message}");
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
        let mut p = param.borrow_mut();

        let grad = match &p.grad {
            Some(g) => g.clone(),
            None => return,
        };
        // The step rewrites the parameter in place (S9).
        p.bump_version();

        // Allocated here on the parameter's first step, on its current device.
        let state = self
            .state
            .get_mut(i)
            .expect("Adam: state slot per parameter")
            .get_or_insert_with(|| AdamState::new(&p));
        state.step += 1;
        let (m_state, v_state) = (&mut state.m, &mut state.v);

        // Setup optimizer parameters
        let opt_params = OptimizerStepParams {
            op: 2, // Adam
            lr: self.lr,
            beta1: self.betas.0,
            beta2: self.betas.1,
            t: state.step as f32,
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
        let param = self.params.get(i).expect("Adam: a parameter per index");
        let mut p = param.borrow_mut();

        let grad = match &p.grad {
            Some(g) => g.clone(),
            None => return,
        };
        // The step rewrites the parameter in place (S9).
        p.bump_version();

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

        // Allocated here on the parameter's first step, in its current dtype.
        let state = self
            .state
            .get_mut(i)
            .expect("Adam: state slot per parameter")
            .get_or_insert_with(|| AdamState::new(&p));
        state.step += 1;
        let step = state.step;
        for s in [&state.m, &state.v] {
            assert_eq!(
                s.dtype(),
                dtype,
                "Adam state dtype; convert the model before the optimizer's first step"
            );
        }
        let (mut m_guard, mut v_guard) = (state.m.host_mut::<T>(), state.v.host_mut::<T>());
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
        let t = i32::try_from(step).expect("Adam step count fits i32");
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
    #[expect(
        clippy::unused_self,
        reason = "keeps the GPU variant's signature so the call site is the same with and without the feature"
    )]
    fn step_gpu_param(&mut self, _i: usize) {
        unreachable!("GPU step should not be called without GPU feature");
    }
}

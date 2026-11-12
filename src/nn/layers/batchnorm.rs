use crate::index::{fetch, slot};
use crate::io::{StateDict, TensorData};
use crate::nn::{Layer, Module};
use crate::real::{Real, dispatch_real};
use crate::shape::Axis;
use crate::tensor::{RawTensor, Tensor};

/// Batch normalization over `[B, C, H, W]` input, with per-channel statistics.
///
/// # Panics
///
/// `forward` panics if the input is not 4D, its channel count is not `num_features`, or, in
/// training mode, it has one value per channel (`B * H * W == 1`), as PyTorch raises. The
/// running statistics are untouched by a rejected batch.
pub struct BatchNorm2d {
    /// Number of channels `C`.
    num_features: usize,
    /// Added to the variance for numerical stability.
    eps: f32,
    /// Weight of the new batch statistics in the running averages.
    momentum: f32,
    /// Training mode uses batch statistics; evaluation mode uses the running ones.
    training: bool,
    // Parameters (Learnable)
    /// Learnable scale, shape `[C]`.
    gamma: Tensor,
    /// Learnable shift, shape `[C]`.
    beta: Tensor,
    // Buffers (Non-learnable)
    /// Running mean used in evaluation mode (a buffer, not learnable).
    running_mean: Tensor,
    /// Running variance used in evaluation mode (a buffer, not learnable).
    running_var: Tensor,
}

impl BatchNorm2d {
    /// Layer over `num_features` channels with `eps = 1e-5` and `momentum = 0.1`.
    #[must_use]
    pub fn new(num_features: usize) -> Self {
        Self::new_with_params(num_features, 1e-5, 0.1)
    }

    /// Layer over `num_features` channels with the given `eps` and `momentum`.
    #[must_use]
    pub fn new_with_params(num_features: usize, eps: f32, momentum: f32) -> Self {
        let gamma = RawTensor::ones(&[num_features]);
        gamma.set_requires_grad(true);

        let beta = RawTensor::zeros(&[num_features]);
        beta.set_requires_grad(true);

        // Running stats are buffers, so they don't require grad
        let running_mean = RawTensor::zeros(&[num_features]);
        let running_var = RawTensor::ones(&[num_features]);

        Self {
            num_features,
            eps,
            momentum,
            training: true,
            gamma,
            beta,
            running_mean,
            running_var,
        }
    }
}

impl Module for BatchNorm2d {
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    /// Converts the running statistics too: they are buffers, not parameters.
    fn to_dtype(&mut self, dtype: crate::DType) {
        crate::nn::cast_parameters(self, dtype);
        self.running_mean.cast_in_place(dtype);
        self.running_var.cast_in_place(dtype);
    }

    fn state_dict(&self) -> StateDict {
        let mut state = StateDict::new();
        state.insert("gamma".to_string(), TensorData::from_tensor(&self.gamma));
        state.insert("beta".to_string(), TensorData::from_tensor(&self.beta));
        state.insert(
            "running_mean".to_string(),
            TensorData::from_tensor(&self.running_mean),
        );
        state.insert(
            "running_var".to_string(),
            TensorData::from_tensor(&self.running_var),
        );
        state
    }

    fn load_state_dict(&mut self, state: &StateDict) -> crate::Result<()> {
        // Create a list of tuples mapping the string key to the specific tensor
        let params = [
            ("gamma", &self.gamma),
            ("beta", &self.beta),
            ("running_mean", &self.running_mean),
            ("running_var", &self.running_var),
        ];

        for (key, tensor) in params {
            if let Some(t) = state.get(key) {
                tensor.load_weight(key, t)?;
            }
        }
        Ok(())
    }
}

impl Layer for BatchNorm2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        // A half input with f32 parameters normalizes in f32 and returns the input's
        // dtype, as PyTorch's norm kernels do (audit A5).
        crate::nn::norm_mixed(x, self.gamma.dtype(), |x| self.forward_in(x))
    }
}

impl BatchNorm2d {
    /// The forward pass with `x` already in the parameters' dtype (or another dtype pair
    /// [`crate::nn::norm_mixed`] leaves alone).
    fn forward_in(&self, x: &Tensor) -> Tensor {
        let x_borrow = x.borrow();
        assert_eq!(
            x_borrow.shape.rank(),
            4,
            "BatchNorm2d expected 4D input (B,C,H,W)"
        );
        assert_eq!(
            x_borrow.shape.dim_at(1),
            self.num_features,
            "Channel mismatch"
        );
        let [b, _, h, w] = x_borrow.shape.fixed::<4>();
        let num_pixels = b * h * w;
        assert!(
            !self.training || num_pixels > 1,
            "BatchNorm2d expected more than 1 value per channel when training, got input shape {:?}",
            x_borrow.shape
        );
        drop(x_borrow);

        // 1. Reshape to normalize over (B, H, W) for each channel
        // We want stats per channel.
        // Input: (B, C, H, W)
        // Permute -> (C, B, H, W)
        let x_perm = x.permute(&[Axis(1), Axis(0), Axis(2), Axis(3)]);

        // We need reshaping to utilize current Tensor ops for mean/var
        // (C, B, H, W) -> (C, B*H*W)
        let x_flat = x_perm.reshape(&[self.num_features, b * h * w]);

        let (mean, var) = if self.training {
            // Calculate batch stats
            let batch_mean = x_flat.mean_dim(Axis(1), true); // (C, 1)

            // Unbiased var: sum((x - mean)^2) / (N - 1)
            // We'll use biased for simplicity/PyTorch-parity in train mode freq: sum / N
            let diff = x_flat.sub(&batch_mean);
            let sq_diff = diff.elem_mul(&diff);
            let batch_var = sq_diff.mean_dim(Axis(1), true); // (C, 1)

            // Update running stats.
            // running_mean = (1 - m) * running_mean + m * batch_mean
            {
                let _no_grad = crate::autograd::no_grad();
                let mut rm = self.running_mean.borrow_mut();
                let mut rv = self.running_var.borrow_mut();
                rm.bump_version();
                rv.bump_version();
                let dtype = rm.data.dtype();
                dispatch_real!(dtype, T => {
                let bm_ref = batch_mean.borrow();
                let bm_data = bm_ref.data.host::<T>();
                let bv_ref = batch_var.borrow();
                let bv_data = bv_ref.data.host::<T>();

                // Note: PyTorch uses unbiased (N-1) for running_var update,
                // but biased for batch norm calculation. The guard above ensures N > 1.
                let m = T::from_f64(f64::from(self.momentum));
                let num_pixels = T::from_usize(num_pixels);

                let mut rm_guard = rm.data.host_mut::<T>();
                let rm_host = &mut *rm_guard;
                let mut rv_guard = rv.data.host_mut::<T>();
                let rv_host = &mut *rv_guard;
                for i in 0..self.num_features {
                    let rm_val = fetch(rm_host, i);
                    let bm_val = fetch(&bm_data, i);
                    *slot(rm_host, i) = (1.0 - m).mul_add(rm_val, m * bm_val);
                    // Bessel correction for running_var update usually applied
                    let bv_val = fetch(&bv_data, i);
                    let unbiased_var = bv_val * num_pixels / (num_pixels - 1.0);
                    let rv_val = fetch(rv_host, i);
                    *slot(rv_host, i) = (1.0 - m).mul_add(rv_val, m * unbiased_var);
                }
                });
            }

            (batch_mean, batch_var)
        } else {
            // Use running stats
            // Must reshape to (C, 1) to broadcast correctly against (C, N)
            // Running stats are host-side layer state (updated in place above), so copy
            // them onto the input's device.
            // In their own dtype: `constant_like(.., data())` would round f64 stats to f32.
            let rm_reshaped = self
                .running_mean
                .detach()
                .reshape(&[self.num_features, 1])
                .to_device(x.device());
            let rv_reshaped = self
                .running_var
                .detach()
                .reshape(&[self.num_features, 1])
                .to_device(x.device());
            (rm_reshaped, rv_reshaped)
        };

        // Normalize: (x - mean) / sqrt(var + eps)
        // Broadcast happens automatically on the last dim (B*H*W)
        let diff = x_flat.sub(&mean);
        let eps_tensor = x.constant_like(vec![self.eps], &[1]);
        let denom = var.add(&eps_tensor).sqrt();
        let x_norm = diff.div(&denom);

        // Scale and shift: gamma * x_norm + beta
        let gamma_r = self.gamma.reshape(&[self.num_features, 1]);
        let beta_r = self.beta.reshape(&[self.num_features, 1]);

        let out_flat = x_norm.elem_mul(&gamma_r).add(&beta_r);

        // Reshape back: (C, B*H*W) -> (C, B, H, W) -> (B, C, H, W)
        let out_perm = out_flat.reshape(&[self.num_features, b, h, w]);
        out_perm.permute(&[Axis(1), Axis(0), Axis(2), Axis(3)])
    }
}

/// Batch Normalization for 2D inputs (B, C).
///
/// Normalizes over the batch dimension (dim=0), computing per-feature statistics.
/// Commonly used after fully-connected layers or LSTM outputs.
///
/// # Panics
///
/// `forward` panics if the input is not 2D, its feature count is not `num_features`, or, in
/// training mode, the batch has one sample (as PyTorch raises). The running statistics are
/// untouched by a rejected batch.
pub struct BatchNorm1d {
    /// Number of features `C`.
    num_features: usize,
    /// Added to the variance for numerical stability.
    eps: f32,
    /// Weight of the new batch statistics in the running averages.
    momentum: f32,
    /// Training mode uses batch statistics; evaluation mode uses the running ones.
    training: bool,
    // Parameters (Learnable)
    /// Learnable scale, shape `[C]`.
    gamma: Tensor,
    /// Learnable shift, shape `[C]`.
    beta: Tensor,
    // Buffers (Non-learnable)
    /// Running mean used in evaluation mode (a buffer, not learnable).
    running_mean: Tensor,
    /// Running variance used in evaluation mode (a buffer, not learnable).
    running_var: Tensor,
}

impl BatchNorm1d {
    /// Layer over `num_features` features with `eps = 1e-5` and `momentum = 0.1`.
    #[must_use]
    pub fn new(num_features: usize) -> Self {
        Self::new_with_params(num_features, 1e-5, 0.1)
    }

    /// Layer over `num_features` features with the given `eps` and `momentum`.
    #[must_use]
    pub fn new_with_params(num_features: usize, eps: f32, momentum: f32) -> Self {
        let gamma = RawTensor::ones(&[num_features]);
        gamma.set_requires_grad(true);

        let beta = RawTensor::zeros(&[num_features]);
        beta.set_requires_grad(true);

        let running_mean = RawTensor::zeros(&[num_features]);
        let running_var = RawTensor::ones(&[num_features]);

        Self {
            num_features,
            eps,
            momentum,
            training: true,
            gamma,
            beta,
            running_mean,
            running_var,
        }
    }
}

impl Module for BatchNorm1d {
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    /// Converts the running statistics too: they are buffers, not parameters.
    fn to_dtype(&mut self, dtype: crate::DType) {
        crate::nn::cast_parameters(self, dtype);
        self.running_mean.cast_in_place(dtype);
        self.running_var.cast_in_place(dtype);
    }

    fn state_dict(&self) -> StateDict {
        let mut state = StateDict::new();
        state.insert("gamma".to_string(), TensorData::from_tensor(&self.gamma));
        state.insert("beta".to_string(), TensorData::from_tensor(&self.beta));
        state.insert(
            "running_mean".to_string(),
            TensorData::from_tensor(&self.running_mean),
        );
        state.insert(
            "running_var".to_string(),
            TensorData::from_tensor(&self.running_var),
        );
        state
    }

    fn load_state_dict(&mut self, state: &StateDict) -> crate::Result<()> {
        let params = [
            ("gamma", &self.gamma),
            ("beta", &self.beta),
            ("running_mean", &self.running_mean),
            ("running_var", &self.running_var),
        ];

        for (key, tensor) in params {
            if let Some(t) = state.get(key) {
                tensor.load_weight(key, t)?;
            }
        }
        Ok(())
    }
}

impl Layer for BatchNorm1d {
    fn forward(&self, x: &Tensor) -> Tensor {
        // A half input with f32 parameters normalizes in f32 and returns the input's
        // dtype, as PyTorch's norm kernels do (audit A5).
        crate::nn::norm_mixed(x, self.gamma.dtype(), |x| self.forward_in(x))
    }
}

impl BatchNorm1d {
    /// The forward pass with `x` already in the parameters' dtype (or another dtype pair
    /// [`crate::nn::norm_mixed`] leaves alone).
    fn forward_in(&self, x: &Tensor) -> Tensor {
        let x_borrow = x.borrow();
        assert_eq!(
            x_borrow.shape.rank(),
            2,
            "BatchNorm1d expected 2D input (B, C)"
        );
        assert_eq!(
            x_borrow.shape.dim_at(1),
            self.num_features,
            "Feature mismatch"
        );
        let batch_size = x_borrow.shape.dim_at(0);
        assert!(
            !self.training || batch_size > 1,
            "BatchNorm1d expected more than 1 value per channel when training, got input shape {:?}",
            x_borrow.shape
        );
        drop(x_borrow);

        let (mean, var) = if self.training {
            // Calculate batch stats: mean over dim=0
            let batch_mean = x.mean_dim(Axis(0), true); // [1, C]

            // Variance: mean((x - mean)^2)
            let diff = x.sub(&batch_mean);
            let sq_diff = diff.elem_mul(&diff);
            let batch_var = sq_diff.mean_dim(Axis(0), true); // [1, C]

            // Update running stats.
            {
                let _no_grad = crate::autograd::no_grad();
                let mut rm = self.running_mean.borrow_mut();
                let mut rv = self.running_var.borrow_mut();
                rm.bump_version();
                rv.bump_version();
                let dtype = rm.data.dtype();
                dispatch_real!(dtype, T => {
                let bm_ref = batch_mean.borrow();
                let bm_data = bm_ref.data.host::<T>();
                let bv_ref = batch_var.borrow();
                let bv_data = bv_ref.data.host::<T>();

                let m = T::from_f64(f64::from(self.momentum));
                let batch_size = T::from_usize(batch_size);
                let mut rm_guard = rm.data.host_mut::<T>();
                let rm_host = &mut *rm_guard;
                let mut rv_guard = rv.data.host_mut::<T>();
                let rv_host = &mut *rv_guard;
                for i in 0..self.num_features {
                    let rm_val = fetch(rm_host, i);
                    let bm_val = fetch(&bm_data, i);
                    *slot(rm_host, i) = (1.0 - m).mul_add(rm_val, m * bm_val);
                    // Bessel correction for running_var update (the guard above ensures N > 1)
                    let bv_val = fetch(&bv_data, i);
                    let unbiased_var = bv_val * batch_size / (batch_size - 1.0);
                    let rv_val = fetch(rv_host, i);
                    *slot(rv_host, i) = (1.0 - m).mul_add(rv_val, m * unbiased_var);
                }
                });
            }

            (batch_mean, batch_var)
        } else {
            // Use running stats, reshaped to [1, C] for broadcasting
            // Running stats are host-side layer state (updated in place above), so copy
            // them onto the input's device.
            // In their own dtype: `constant_like(.., data())` would round f64 stats to f32.
            let rm_reshaped = self
                .running_mean
                .detach()
                .reshape(&[1, self.num_features])
                .to_device(x.device());
            let rv_reshaped = self
                .running_var
                .detach()
                .reshape(&[1, self.num_features])
                .to_device(x.device());
            (rm_reshaped, rv_reshaped)
        };

        // Normalize: (x - mean) / sqrt(var + eps)
        let diff = x.sub(&mean);
        let eps_tensor = x.constant_like(vec![self.eps], &[1]);
        let denom = var.add(&eps_tensor).sqrt();
        let x_norm = diff.div(&denom);

        // Scale and shift: gamma * x_norm + beta
        let gamma_r = self.gamma.reshape(&[1, self.num_features]);
        let beta_r = self.beta.reshape(&[1, self.num_features]);

        x_norm.elem_mul(&gamma_r).add(&beta_r)
    }
}

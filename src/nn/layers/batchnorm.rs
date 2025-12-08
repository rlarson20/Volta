use crate::io::{StateDict, TensorData};
use crate::nn::Module;
use crate::storage::Storage;
use crate::tensor::{RawTensor, Tensor, TensorOps};

pub struct BatchNorm2d {
    num_features: usize,
    eps: f32,
    momentum: f32,
    training: bool,
    // Parameters (Learnable)
    gamma: Tensor,
    beta: Tensor,
    // Buffers (Non-learnable)
    running_mean: Tensor,
    running_var: Tensor,
}

impl BatchNorm2d {
    pub fn new(num_features: usize) -> Self {
        Self::new_with_params(num_features, 1e-5, 0.1)
    }

    pub fn new_with_params(num_features: usize, eps: f32, momentum: f32) -> Self {
        let gamma = RawTensor::ones(&[num_features]);
        gamma.borrow_mut().requires_grad = true;

        let beta = RawTensor::zeros(&[num_features]);
        beta.borrow_mut().requires_grad = true;

        // Running stats are buffers, so they don't require grad
        let running_mean = RawTensor::zeros(&[num_features]);
        let running_var = RawTensor::ones(&[num_features]);

        BatchNorm2d {
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
    fn forward(&self, x: &Tensor) -> Tensor {
        let x_borrow = x.borrow();
        assert_eq!(
            x_borrow.shape.len(),
            4,
            "BatchNorm2d expected 4D input (B,C,H,W)"
        );
        assert_eq!(x_borrow.shape[1], self.num_features, "Channel mismatch");
        drop(x_borrow);

        // 1. Reshape to normalize over (B, H, W) for each channel
        // We want stats per channel.
        // Input: (B, C, H, W)
        // Permute -> (C, B, H, W)
        let x_perm = x.permute(&[1, 0, 2, 3]);

        // We need reshaping to utilize current Tensor ops for mean/var
        // (C, B, H, W) -> (C, B*H*W)
        let b = x.borrow().shape[0];
        let h = x.borrow().shape[2];
        let w = x.borrow().shape[3];
        let num_pixels = (b * h * w) as f32;

        let x_flat = x_perm.reshape(&[self.num_features, b * h * w]);

        let (mean, var) = if self.training {
            // Calculate batch stats
            let batch_mean = x_flat.mean_dim(1, true); // (C, 1)

            // Unbiased var: sum((x - mean)^2) / (N - 1)
            // We'll use biased for simplicity/PyTorch-parity in train mode freq: sum / N
            let diff = x_flat.sub(&batch_mean);
            let sq_diff = diff.elem_mul(&diff);
            let batch_var = sq_diff.mean_dim(1, true); // (C, 1)

            // Update running stats (no_grad logic)
            // running_mean = (1 - m) * running_mean + m * batch_mean
            {
                let mut rm = self.running_mean.borrow_mut();
                let mut rv = self.running_var.borrow_mut();
                let bm_data = &batch_mean.borrow().data;
                let bv_data = &batch_var.borrow().data;

                // Note: PyTorch uses unbiased (N-1) for running_var update,
                // but biased for batch norm calculation.
                // We stick to standard implementation:
                let m = self.momentum;

                for i in 0..self.num_features {
                    rm.data[i] = (1.0 - m) * rm.data[i] + m * bm_data[i];
                    // Bessel correction for running_var update usually applied
                    let unbiased_var = bv_data[i] * num_pixels / (num_pixels - 1.0);
                    rv.data[i] = (1.0 - m) * rv.data[i] + m * unbiased_var;
                }
            }

            (batch_mean, batch_var)
        } else {
            // Use running stats
            // Must reshape to (C, 1) to broadcast correctly against (C, N)
            let rm_reshaped = self.running_mean.reshape(&[self.num_features, 1]);
            let rv_reshaped = self.running_var.reshape(&[self.num_features, 1]);
            (rm_reshaped, rv_reshaped)
        };

        // Normalize: (x - mean) / sqrt(var + eps)
        // Broadcast happens automatically on the last dim (B*H*W)
        let diff = x_flat.sub(&mean);
        let eps_tensor = RawTensor::constant(self.eps, &[1]);
        let denom = var.add(&eps_tensor).sqrt();
        let x_norm = diff.div(&denom);

        // Scale and shift: gamma * x_norm + beta
        let gamma_r = self.gamma.reshape(&[self.num_features, 1]);
        let beta_r = self.beta.reshape(&[self.num_features, 1]);

        let out_flat = x_norm.elem_mul(&gamma_r).add(&beta_r);

        // Reshape back: (C, B*H*W) -> (C, B, H, W) -> (B, C, H, W)
        let out_perm = out_flat.reshape(&[self.num_features, b, h, w]);
        out_perm.permute(&[1, 0, 2, 3])
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
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

    fn load_state_dict(&mut self, state: &StateDict) {
        //TODO: refactor to use match-case since that'd pretty obviously work here
        if let Some(t) = state.get("gamma") {
            let mut b = self.gamma.borrow_mut();
            b.data = Storage::cpu(t.data.clone());
            b.shape = t.shape.clone();
        }
        if let Some(t) = state.get("beta") {
            let mut b = self.beta.borrow_mut();
            b.data = Storage::cpu(t.data.clone());
            b.shape = t.shape.clone();
        }
        if let Some(t) = state.get("running_mean") {
            let mut b = self.running_mean.borrow_mut();
            b.data = Storage::cpu(t.data.clone());
            b.shape = t.shape.clone();
        }
        if let Some(t) = state.get("running_var") {
            let mut b = self.running_var.borrow_mut();
            b.data = Storage::cpu(t.data.clone());
            b.shape = t.shape.clone();
        }
    }
}

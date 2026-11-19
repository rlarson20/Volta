use crate::index::{fetch, slot};
use crate::io::{StateDict, TensorData};
use crate::nn::{Layer, Module};
use crate::real::{Real, dispatch_real};
use crate::shape::Axis;
use crate::tensor::{RawTensor, Tensor};

/// The state and arithmetic `BatchNorm1d` and `BatchNorm2d` share (#119). Each wrapper
/// checks its input's rank and lays it out as a 2-D tensor with the channels on one axis
/// and every value of a channel on the other, then calls [`BatchNorm::normalize`].
struct BatchNorm {
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

impl BatchNorm {
    /// `gamma = 1`, `beta = 0`, running mean 0 and running variance 1, in training mode.
    fn new(num_features: usize, eps: f32, momentum: f32) -> Self {
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

    /// Panics unless `x` has `dtype`, the dtype [`crate::nn::norm_mixed`] computes in,
    /// naming `layer`. That is the parameters' dtype except for a half input, which
    /// `norm_mixed` casts to f32 first. One rule in training and evaluation mode, checked
    /// before any computation (#104), as PyTorch refuses an f64 input to an f32 `BatchNorm`.
    #[track_caller]
    fn check_dtype(&self, layer: &str, x: &Tensor, dtype: crate::DType) {
        let (input, param) = (x.dtype(), self.gamma.dtype());
        assert!(
            input == dtype,
            "{layer}: input is {input}, parameters are {param}; this layer needs one dtype, as \
             in PyTorch: cast the input with to_dtype, or the layer with Module::to_dtype"
        );
    }

    /// `gamma` and `beta`.
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    /// Casts the parameters and the running statistics (buffers, not parameters).
    fn to_dtype<M: Module + ?Sized>(&self, layer: &M, dtype: crate::DType) {
        crate::nn::cast_parameters(layer, dtype);
        self.running_mean.cast_in_place(dtype);
        self.running_var.cast_in_place(dtype);
    }

    /// The parameters and the running statistics.
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

    /// Loads each of the four entries that is present.
    fn load_state_dict(&self, state: &StateDict) -> crate::Result<()> {
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

    /// Normalizes `x`, a 2-D tensor whose `channel_axis` has the `C` channels and whose
    /// other axis has every value of a channel (`n` of them), then scales and shifts it.
    /// In training mode this uses the batch statistics and updates the running ones; the
    /// caller has checked `n > 1` there. `x` is in `dtype`, which the parameters and the
    /// running statistics are cast to ([`crate::nn::norm_mixed`]).
    fn normalize(&self, x: &Tensor, channel_axis: Axis, n: usize, dtype: crate::DType) -> Tensor {
        let c = self.num_features;
        // Per-channel statistics keep the reduced axis, so they broadcast against `x`.
        let (reduce, stat_dims) = if channel_axis == Axis(0) {
            (Axis(1), [c, 1])
        } else {
            (Axis(0), [1, c])
        };

        let (mean, var) = if self.training {
            let batch_mean = x.mean_dim(reduce, true);
            // Biased variance for normalizing, as PyTorch does in training mode.
            let diff = x.sub(&batch_mean);
            let batch_var = diff.elem_mul(&diff).mean_dim(reduce, true);
            self.update_running(&batch_mean, &batch_var, n);
            (batch_mean, batch_var)
        } else {
            // Running stats are host-side layer state (updated in place in training), so
            // copy them onto the input's device, in their own dtype: `constant_like(..,
            // data())` would round f64 stats to f32.
            let stat = |t: &Tensor| {
                t.detach()
                    .reshape(&stat_dims)
                    .to_device(x.device())
                    .to_dtype(dtype)
            };
            (stat(&self.running_mean), stat(&self.running_var))
        };

        // Normalize: (x - mean) / sqrt(var + eps), then gamma * x_norm + beta.
        let diff = x.sub(&mean);
        let eps_tensor = x.constant_like(vec![self.eps], &[1]);
        let x_norm = diff.div(&var.add(&eps_tensor).sqrt());
        x_norm
            .elem_mul(&self.gamma.to_dtype(dtype).reshape(&stat_dims))
            .add(&self.beta.to_dtype(dtype).reshape(&stat_dims))
    }

    /// `running = (1 - momentum) * running + momentum * batch`, with Bessel's correction
    /// (`n / (n - 1)`) on the variance, as PyTorch does. `n > 1`.
    fn update_running(&self, batch_mean: &Tensor, batch_var: &Tensor, n: usize) {
        let _no_grad = crate::autograd::no_grad();
        let dtype = self.running_mean.dtype();
        // Both statistics are rewritten in place (S9; `update_data` bumps each version).
        self.running_mean.update_data(|rm| {
            self.running_var.update_data(|rv| {
                dispatch_real!(dtype, T => {
                let bm_ref = batch_mean.borrow();
                let bm_data = bm_ref.data.host::<T>();
                let bv_ref = batch_var.borrow();
                let bv_data = bv_ref.data.host::<T>();

                let m = T::from_f64(f64::from(self.momentum));
                let n = T::from_usize(n);

                rm.with_host_mut::<T, _>(|rm_host| {
                    rv.with_host_mut::<T, _>(|rv_host| {
                        for i in 0..self.num_features {
                            let rm_val = fetch(rm_host, i);
                            let bm_val = fetch(&bm_data, i);
                            *slot(rm_host, i) = (1.0 - m).mul_add(rm_val, m * bm_val);
                            let unbiased_var = fetch(&bv_data, i) * n / (n - 1.0);
                            let rv_val = fetch(rv_host, i);
                            *slot(rv_host, i) = (1.0 - m).mul_add(rv_val, m * unbiased_var);
                        }
                    });
                });
                });
            });
        });
    }
}

/// Batch normalization over `[B, C, H, W]` input, with per-channel statistics.
///
/// # Panics
///
/// `forward` panics if the input is not 4D, its channel count is not `num_features`, its
/// dtype is not the parameters' (in either mode; a half input with f32 parameters is the
/// exception), or, in training mode, it has one value per channel
/// (`B * H * W == 1`), as PyTorch raises. The running statistics are untouched by a
/// rejected batch.
///
/// A half (f16/bf16) input is normalized in f32 and returned in its own dtype, with f32
/// or half parameters, as PyTorch's half kernels accumulate in f32.
pub struct BatchNorm2d {
    /// Parameters, running statistics and mode.
    core: BatchNorm,
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
        Self {
            core: BatchNorm::new(num_features, eps, momentum),
        }
    }
}

/// Batch Normalization for 2D inputs (B, C).
///
/// Normalizes over the batch dimension (dim=0), computing per-feature statistics.
/// Commonly used after fully-connected layers or LSTM outputs.
///
/// # Panics
///
/// `forward` panics if the input is not 2D, its feature count is not `num_features`, its
/// dtype is not the parameters' (in either mode; a half input with f32 parameters is the
/// exception), or, in training mode, the batch has one sample (as
/// PyTorch raises). The running statistics are untouched by a rejected batch.
///
/// A half (f16/bf16) input is normalized in f32 and returned in its own dtype, with f32
/// or half parameters, as PyTorch's half kernels accumulate in f32.
pub struct BatchNorm1d {
    /// Parameters, running statistics and mode.
    core: BatchNorm,
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
        Self {
            core: BatchNorm::new(num_features, eps, momentum),
        }
    }
}

/// `Module` for a wrapper around [`BatchNorm`]: every method forwards to the core.
macro_rules! batchnorm_module {
    ($layer:ty) => {
        impl Module for $layer {
            fn parameters(&self) -> Vec<Tensor> {
                self.core.parameters()
            }

            fn train(&mut self, mode: bool) {
                self.core.training = mode;
            }

            /// Converts the running statistics too: they are buffers, not parameters.
            fn to_dtype(&mut self, dtype: crate::DType) {
                self.core.to_dtype(self, dtype);
            }

            fn state_dict(&self) -> StateDict {
                self.core.state_dict()
            }

            fn load_state_dict(&mut self, state: &StateDict) -> crate::Result<()> {
                self.core.load_state_dict(state)
            }
        }

        impl Layer for $layer {
            fn forward(&self, x: &Tensor) -> Tensor {
                // A half input normalizes in f32 and returns the input's dtype, as
                // PyTorch's norm kernels do (audit A5, #316).
                crate::nn::norm_mixed(x, self.core.gamma.dtype(), |x, dtype| {
                    self.forward_in(x, dtype)
                })
            }
        }
    };
}

batchnorm_module!(BatchNorm2d);
batchnorm_module!(BatchNorm1d);

impl BatchNorm2d {
    /// The forward pass with `x` already in `dtype`, the dtype to compute in
    /// ([`crate::nn::norm_mixed`]; any other input dtype panics here).
    fn forward_in(&self, x: &Tensor, dtype: crate::DType) -> Tensor {
        self.core.check_dtype("BatchNorm2d", x, dtype);
        let c = self.core.num_features;
        let x_borrow = x.borrow();
        assert_eq!(
            x_borrow.shape.rank(),
            4,
            "BatchNorm2d expected 4D input (B,C,H,W)"
        );
        assert_eq!(x_borrow.shape.dim_at(1), c, "Channel mismatch");
        let [b, _, h, w] = x_borrow.shape.fixed::<4>();
        let num_pixels = b * h * w;
        assert!(
            !self.core.training || num_pixels > 1,
            "BatchNorm2d expected more than 1 value per channel when training, got input shape {:?}",
            x_borrow.shape
        );
        drop(x_borrow);

        // Statistics per channel over (B, H, W): (B, C, H, W) -> (C, B, H, W) -> (C, B*H*W).
        let x_flat = x
            .permute(&[Axis(1), Axis(0), Axis(2), Axis(3)])
            .reshape(&[c, num_pixels]);
        let out_flat = self.core.normalize(&x_flat, Axis(0), num_pixels, dtype);

        // Back: (C, B*H*W) -> (C, B, H, W) -> (B, C, H, W)
        out_flat
            .reshape(&[c, b, h, w])
            .permute(&[Axis(1), Axis(0), Axis(2), Axis(3)])
    }
}

impl BatchNorm1d {
    /// The forward pass with `x` already in `dtype`, the dtype to compute in
    /// ([`crate::nn::norm_mixed`]; any other input dtype panics here).
    fn forward_in(&self, x: &Tensor, dtype: crate::DType) -> Tensor {
        self.core.check_dtype("BatchNorm1d", x, dtype);
        let x_borrow = x.borrow();
        assert_eq!(
            x_borrow.shape.rank(),
            2,
            "BatchNorm1d expected 2D input (B, C)"
        );
        assert_eq!(
            x_borrow.shape.dim_at(1),
            self.core.num_features,
            "Feature mismatch"
        );
        let batch_size = x_borrow.shape.dim_at(0);
        assert!(
            !self.core.training || batch_size > 1,
            "BatchNorm1d expected more than 1 value per channel when training, got input shape {:?}",
            x_borrow.shape
        );
        drop(x_borrow);

        // Already (B, C): statistics per feature over the batch axis.
        self.core.normalize(x, Axis(1), batch_size, dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::{BatchNorm1d, BatchNorm2d};
    use crate::DType;
    use crate::nn::{Layer, Module};
    use crate::tensor::{RawTensor, Tensor};

    fn f64_input(shape: &[usize]) -> Tensor {
        let n = shape.iter().product::<usize>();
        let data = (0..n).map(|i| i as f32 * 0.5 - 1.0).collect::<Vec<_>>();
        RawTensor::new(data, shape, false).to_dtype(DType::F64)
    }

    // #104: an f64 input to an f32 layer panicked in training mode with a bare "DType
    // mismatch" from the running-stat update, and was promoted in evaluation mode.

    #[test]
    #[should_panic(expected = "BatchNorm1d: input is f64, parameters are f32")]
    fn batchnorm1d_refuses_f64_input_in_train_mode() {
        let _ = BatchNorm1d::new(2).forward(&f64_input(&[2, 2]));
    }

    #[test]
    #[should_panic(expected = "BatchNorm1d: input is f64, parameters are f32")]
    fn batchnorm1d_refuses_f64_input_in_eval_mode() {
        let mut bn = BatchNorm1d::new(2);
        bn.eval();
        let _ = bn.forward(&f64_input(&[2, 2]));
    }

    #[test]
    #[should_panic(expected = "BatchNorm2d: input is f64, parameters are f32")]
    fn batchnorm2d_refuses_f64_input_in_train_mode() {
        let _ = BatchNorm2d::new(1).forward(&f64_input(&[1, 1, 2, 2]));
    }

    #[test]
    #[should_panic(expected = "BatchNorm2d: input is f64, parameters are f32")]
    fn batchnorm2d_refuses_f64_input_in_eval_mode() {
        let mut bn = BatchNorm2d::new(1);
        bn.eval();
        let _ = bn.forward(&f64_input(&[1, 1, 2, 2]));
    }

    /// The fix the message names works: the layer in f64 takes the f64 input in both
    /// modes, and its running statistics stay f64.
    #[test]
    fn batchnorm_cast_to_input_dtype_accepts_it_in_both_modes() {
        let x = f64_input(&[2, 2]);
        let mut bn = BatchNorm1d::new(2);
        bn.to_dtype(DType::F64);
        assert_eq!(bn.forward(&x).dtype(), DType::F64);
        assert_eq!(bn.core.running_mean.dtype(), DType::F64);
        bn.eval();
        assert_eq!(bn.forward(&x).dtype(), DType::F64);
    }
}

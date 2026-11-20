use crate::device::Device;
use crate::io::{StateDict, TensorData};
use crate::nn::{Layer, Module};
use crate::tensor::{RawTensor, Tensor};

/// Fully-connected (dense/linear) layer
///
/// Computes: y = xW + b
/// where x is (batch, `in_features`), W is (`in_features`, `out_features`), b is (`out_features`)
pub struct Linear {
    /// Weight, shape `[in_features, out_features]`.
    pub weight: Tensor,
    /// Optional bias, shape `[out_features]`.
    pub bias: Option<Tensor>,
}
impl Module for Linear {
    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }

    fn state_dict(&self) -> StateDict {
        let mut state = StateDict::new();
        state.insert("weight".to_string(), TensorData::from_tensor(&self.weight));
        if let Some(ref b) = self.bias {
            state.insert("bias".to_string(), TensorData::from_tensor(b));
        }
        state
    }

    fn plan_load<'a>(&self, plan: &mut crate::nn::LoadPlan<'a>) {
        plan.param("weight", &self.weight);
        if let Some(bias) = &self.bias {
            plan.param("bias", bias);
        }
    }
}

impl Layer for Linear {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.forward(x)
    }
}
impl Linear {
    /// Create a new linear layer with random initialization
    ///
    /// Uses Xavier initialization.
    ///
    /// # Panics
    /// If `in_features` or `out_features` is zero.
    #[must_use]
    #[track_caller]
    pub fn new(in_features: usize, out_features: usize, use_bias: bool) -> Self {
        let w = RawTensor::xavier_uniform(&[in_features, out_features]);
        w.set_requires_grad(true);
        let b = if use_bias {
            let b = RawTensor::zeros(&[out_features]);
            b.set_requires_grad(true);
            Some(b)
        } else {
            None
        };
        Self { weight: w, bias: b }
    }

    /// Create a new linear layer on a specific device
    ///
    /// Uses Xavier initialization and places tensors on the specified device.
    ///
    /// # Arguments
    /// * `in_features` - Size of input features
    /// * `out_features` - Size of output features
    /// * `use_bias` - Whether to include a bias term
    /// * `device` - Device to place parameters on (CPU or GPU)
    ///
    /// # Example
    /// ```no_run
    /// # use volta::nn::Linear;
    /// # use volta::Device;
    /// # #[cfg(feature = "gpu")]
    /// # {
    /// let device = Device::gpu().expect("GPU required");
    /// let layer = Linear::new_on_device(784, 128, true, device);
    /// // Parameters are now on GPU
    /// # }
    /// ```
    ///
    /// # Panics
    /// If `in_features` or `out_features` is zero, or `device` is a GPU that cannot hold
    /// the parameters.
    #[must_use]
    #[track_caller]
    pub fn new_on_device(
        in_features: usize,
        out_features: usize,
        use_bias: bool,
        device: Device,
    ) -> Self {
        // Move, then mark trainable: `to_device` of a tensor that requires grad is a
        // differentiable copy, not a leaf, so the parameters must be leaves on `device`.
        let w = RawTensor::xavier_uniform(&[in_features, out_features]).to_device(device.clone());
        w.set_requires_grad(true);

        let b = if use_bias {
            let b = RawTensor::zeros(&[out_features]).to_device(device);
            b.set_requires_grad(true);
            Some(b)
        } else {
            None
        };
        Self { weight: w, bias: b }
    }

    /// Forward pass through the layer
    ///
    /// # Panics
    /// If `x`'s last dimension is not `in_features`, or `x` is on another device than the
    /// parameters.
    #[must_use]
    #[track_caller]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // One low-precision op under `amp::autocast`: input, weight and bias are cast
        // together, so the bias add stays in the low dtype (PyTorch's `linear`).
        let (x, weight, bias) = crate::amp::lower_layer(x, &self.weight, self.bias.as_ref());
        let out = x.matmul(&weight);
        if let Some(b) = &bias { out.add(b) } else { out }
    }
}

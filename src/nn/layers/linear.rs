use crate::Storage;
use crate::device::Device;
use crate::io::{StateDict, TensorData};
use crate::nn::Module;
use crate::tensor::{RawTensor, Tensor, TensorOps};

/// Fully-connected (dense/linear) layer
///
/// Computes: y = xW + b
/// where x is (batch, `in_features`), W is (`in_features`, `out_features`), b is (`out_features`)
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}
impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.forward(x)
    }
    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone())
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

    fn load_state_dict(&mut self, state: &StateDict) {
        if let Some(w) = state.get("weight") {
            let mut t = self.weight.borrow_mut();
            t.data = Storage::cpu(w.data.clone());
            t.shape = w.shape.clone();
        }
        if let Some(b) = state.get("bias")
            && self.bias.is_some()
        {
            let bias_tensor = self.bias.as_ref().unwrap();
            let mut t = bias_tensor.borrow_mut();
            t.data = Storage::cpu(b.data.clone());
            t.shape = b.shape.clone();
        }
    }
}
impl Linear {
    /// Create a new linear layer with random initialization
    ///
    /// Uses Xavier initialization.
    pub fn new(in_features: usize, out_features: usize, use_bias: bool) -> Self {
        let w = RawTensor::xavier_uniform(&[in_features, out_features]);
        w.borrow_mut().requires_grad = true;
        let b = if use_bias {
            let b = RawTensor::zeros(&[out_features]);
            b.borrow_mut().requires_grad = true;
            Some(b)
        } else {
            None
        };
        Linear { weight: w, bias: b }
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
    /// # use volta::{Linear, Device};
    /// # #[cfg(feature = "gpu")]
    /// # {
    /// let device = Device::gpu().expect("GPU required");
    /// let layer = Linear::new_on_device(784, 128, true, device);
    /// // Parameters are now on GPU
    /// # }
    /// ```
    pub fn new_on_device(
        in_features: usize,
        out_features: usize,
        use_bias: bool,
        device: Device,
    ) -> Self {
        let w = RawTensor::xavier_uniform(&[in_features, out_features]);
        w.borrow_mut().requires_grad = true;
        let w = w.to_device(device.clone());

        let b = if use_bias {
            let b = RawTensor::zeros(&[out_features]);
            b.borrow_mut().requires_grad = true;
            Some(b.to_device(device))
        } else {
            None
        };
        Linear { weight: w, bias: b }
    }

    /// Forward pass through the layer
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let out = x.matmul(&self.weight);
        if let Some(b) = &self.bias {
            out.add(b)
        } else {
            out
        }
    }
}

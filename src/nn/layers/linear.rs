use crate::Result;
use crate::device::Device;
use crate::error::OrPanic;
use crate::io::{StateDict, TensorData};
use crate::nn::hyper;
use crate::nn::{Layer, Module};
use crate::tensor::{RawTensor, Tensor};

/// Fully-connected (dense/linear) layer
///
/// Computes: y = xW + b
/// where x is (batch, `in_features`), W is (`in_features`, `out_features`), b is (`out_features`)
#[derive(Debug)]
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
    /// If `in_features` or `out_features` is zero (see [`Self::try_new`]).
    #[must_use]
    #[track_caller]
    pub fn new(in_features: usize, out_features: usize, use_bias: bool) -> Self {
        Self::try_new(in_features, out_features, use_bias).or_panic(Self::OP)
    }

    /// [`Self::new`], checking the sizes first.
    ///
    /// # Errors
    /// `VoltaError::InvalidParameter` naming the argument if `in_features` or
    /// `out_features` is zero, or naming both if the weight would have more than
    /// [`crate::shape::MAX_NUMEL`] elements.
    pub fn try_new(in_features: usize, out_features: usize, use_bias: bool) -> Result<Self> {
        Self::check_sizes(in_features, out_features)?;
        let w = RawTensor::xavier_uniform(&[in_features, out_features]);
        w.set_requires_grad(true);
        let b = if use_bias {
            let b = RawTensor::zeros(&[out_features]);
            b.set_requires_grad(true);
            Some(b)
        } else {
            None
        };
        Ok(Self { weight: w, bias: b })
    }

    /// The layer's name in errors.
    const OP: &'static str = "Linear";

    /// `Ok` if both sizes are at least 1 and the weight fits in a tensor.
    fn check_sizes(in_features: usize, out_features: usize) -> Result<()> {
        hyper::at_least_one(Self::OP, "in_features", in_features)?;
        hyper::at_least_one(Self::OP, "out_features", out_features)?;
        hyper::param_numel(
            Self::OP,
            "in_features, out_features",
            &[in_features, out_features],
        )
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
        Self::check_sizes(in_features, out_features).or_panic(Self::OP);
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
        let [in_f, out_f] = self.weight.borrow().shape.fixed::<2>();
        let x_shape = x.borrow().shape.clone();
        assert!(
            x_shape.rank() >= 1 && x_shape.dim_at(x_shape.rank() - 1) == in_f,
            "Linear({in_f} -> {out_f}): expected input [.., {in_f}], got {:?}",
            x_shape.truncate_rank()
        );
        // One low-precision op under `amp::autocast`: input, weight and bias are cast
        // together, so the bias add stays in the low dtype (PyTorch's `linear`).
        let (x, weight, bias) = crate::amp::lower_layer(x, &self.weight, self.bias.as_ref());
        let out = x.matmul(&weight);
        if let Some(b) = &bias { out.add(b) } else { out }
    }
}

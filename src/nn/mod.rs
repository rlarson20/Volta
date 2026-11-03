use crate::device::Device;
use crate::dtype::DType;
use crate::io::StateDict;
use crate::tensor::Tensor;

/// Layers and containers.
pub mod layers;
/// Optimizers.
pub mod optim;
pub mod transformer;

pub use layers::{
    BatchNorm1d, BatchNorm2d, Conv2d, ConvAlgo, ConvTranspose2d, Dropout, Embedding, Flatten, GELU,
    LSTMCell, LayerNorm, Linear, MaxPool2d, PixelShuffle, ReLU, Sequential, SequentialBuilder,
    Sigmoid, Tanh,
};
pub use optim::{Adam, Muon, SGD};
pub use transformer::{
    FeedForward, MultiHeadAttention, PositionalEncoding, PositionalEncodingType,
    ScaledDotProductAttention, TransformerBlock, causal_mask,
};

/// Private supertrait that keeps [`IndexedLayer`] and [`RecurrentLayer`] implementable only inside
/// this crate. `Module` and [`Layer`] stay open so users can write custom layers.
mod sealed {
    /// Marker for the sealed layer traits.
    pub trait SealedLayer {}
}

pub(crate) use sealed::SealedLayer;

/// Anything that owns learnable parameters and can be saved, loaded, and moved between devices.
///
/// This is the base of the layer traits. It has no `forward`, because the input contract differs
/// by kind of layer: see [`Layer`] (one tensor in, one out), [`IndexedLayer`] (integer indices in),
/// and [`RecurrentLayer`] (input plus carried state).
pub trait Module {
    /// Every learnable tensor of the module, in a stable order.
    fn parameters(&self) -> Vec<Tensor>;

    // State dict methods
    /// The parameters as a name-to-values map, ready to save with [`crate::io`].
    fn state_dict(&self) -> StateDict;

    /// Loads the entries of `state` that match this module's parameters.
    ///
    /// Keys the module does not have are ignored, and parameters with no entry keep
    /// their values (use [`crate::io::load_state_dict_checked`] to see which). An entry
    /// that is present must fit: it keeps the parameter's shape, device, and
    /// `requires_grad`, and only the values change.
    ///
    /// # Errors
    ///
    /// `VoltaError::StateShapeMismatch` if an entry's shape differs from its
    /// parameter's, `VoltaError::StateDataLength` if an entry has the wrong number of
    /// values for its own shape. The `key` is the entry's name in `state`. A checkpoint
    /// is data from outside the program, so this is an `Err`, not a panic.
    ///
    /// On `Err` the module may be partly loaded: entries before the bad one have
    /// already been written. Discard the module, or load into a fresh one.
    fn load_state_dict(&mut self, state: &StateDict) -> crate::Result<()>;

    /// Clears the gradient of every parameter.
    fn zero_grad(&mut self) {
        for p in self.parameters() {
            p.borrow_mut().grad = None;
        }
    }

    /// Move all module parameters to a specific device
    ///
    /// This method transfers all parameters (weights, biases, etc.) to the specified device.
    /// The default implementation moves each parameter individually by updating their
    /// storage and device in place.
    ///
    /// # Arguments
    /// * `device` - Target device (CPU or GPU)
    ///
    /// # Example
    /// ```no_run
    /// # use volta::{Linear, Module, Device};
    /// # #[cfg(feature = "gpu")]
    /// # {
    /// let mut layer = Linear::new(784, 128, true);
    /// let device = Device::gpu().expect("GPU required");
    /// layer.to_device(device);
    /// // All parameters are now on GPU
    /// # }
    /// ```
    fn to_device(&mut self, device: Device) {
        for param in self.parameters() {
            {
                let mut p = param.borrow_mut();
                // `Storage::to_device` panics when the GPU is unavailable rather than
                // leaving the data on the CPU under a GPU label.
                p.data = p.data.to_device(&device);
                p.device = device.clone();
            }
            // `set_grad` moves an existing gradient onto the parameter's new device.
            if let Some(g) = param.grad_storage() {
                param.set_grad(g);
            }
        }
    }

    /// Converts every parameter, and its gradient, to `dtype` in place: PyTorch's
    /// `module.to(dtype)` (`.double()`, `.half()`). Values follow [`crate::Storage::cast`],
    /// so f32 to f64 is exact. A layer holding a non-parameter tensor (`BatchNorm`'s running
    /// statistics, `PositionalEncoding`'s table) overrides this to convert it too, and a
    /// container (`Sequential`) forwards it to each child.
    ///
    /// Of the layers, those whose kernels are dtype-generic run in f64 after this; see
    /// ADDENDUM §11 for which.
    ///
    /// # Panics
    /// If `dtype` is not a float dtype (parameters need gradients), or a parameter is on
    /// the GPU and `dtype` is not F32.
    fn to_dtype(&mut self, dtype: DType) {
        cast_parameters(self, dtype);
    }

    /// Switch between training and evaluation modes.
    /// Important for layers like `BatchNorm` and Dropout.
    fn train(&mut self, _mode: bool) {}
    /// Switches to evaluation mode; same as `train(false)`.
    fn eval(&mut self) {
        self.train(false);
    }
}

/// The default of [`Module::to_dtype`]: casts every parameter of `module` to `dtype`.
/// Overrides call it and then convert their buffers.
///
/// # Panics
/// As [`Module::to_dtype`].
#[track_caller]
pub(crate) fn cast_parameters<M: Module + ?Sized>(module: &M, dtype: DType) {
    assert!(
        dtype.is_float(),
        "to_dtype({dtype}): a module's parameters need a float dtype"
    );
    for param in module.parameters() {
        param.cast_in_place(dtype);
    }
}

/// A layer that maps one tensor to one tensor. Only `Layer`s can go in a [`Sequential`].
pub trait Layer: Module {
    /// Run the layer on `x`.
    fn forward(&self, x: &Tensor) -> Tensor;
}

/// A layer whose input is integer indices, such as [`Embedding`]. It is not a [`Layer`], so it
/// cannot be placed in a [`Sequential`]. Sealed: only this crate implements it.
pub trait IndexedLayer: Module + SealedLayer {
    /// Look up the output rows for `indices`.
    fn forward_indices(&self, indices: &[usize]) -> Tensor;
}

/// A layer that carries state between steps, such as [`LSTMCell`]. Sealed: only this crate
/// implements it.
pub trait RecurrentLayer: Module + SealedLayer {
    /// State carried from one step to the next.
    type State;
    /// Zero state for a batch of `batch` sequences.
    fn initial_state(&self, batch: usize) -> Self::State;
    /// Advance one step: update `state` in place and return the step output.
    fn step(&self, x: &Tensor, state: &mut Self::State) -> Tensor;
}

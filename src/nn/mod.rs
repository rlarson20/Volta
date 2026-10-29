use crate::device::Device;
use crate::io::StateDict;
use crate::tensor::Tensor;

pub mod layers;
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
    fn parameters(&self) -> Vec<Tensor>;

    // State dict methods
    fn state_dict(&self) -> StateDict;
    fn load_state_dict(&mut self, state: &StateDict);

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
        use crate::storage::Storage;

        for param in self.parameters() {
            let mut p = param.borrow_mut();

            // Transfer storage to new device
            match (&p.data, &device) {
                (Storage::Cpu { .. }, Device::CPU) => {
                    // Already on CPU, no-op
                }
                #[cfg(feature = "gpu")]
                (Storage::Cpu { .. }, Device::GPU(_)) => {
                    // CPU -> GPU
                    let data = p.data.to_vec();
                    p.data = Storage::gpu(data);
                    p.device = device.clone();
                }
                #[cfg(feature = "gpu")]
                (Storage::Gpu { .. }, Device::CPU) => {
                    // GPU -> CPU
                    let cpu_data = p.data.to_vec();
                    p.data = Storage::cpu(cpu_data);
                    p.device = device.clone();
                }
                #[cfg(feature = "gpu")]
                (Storage::Gpu { .. }, Device::GPU(_)) => {
                    // GPU -> GPU (same or different device)
                    // For now, go through CPU as intermediary
                    // TODO: implement direct GPU-to-GPU transfer
                    let cpu_data = p.data.to_vec();
                    p.data = Storage::gpu(cpu_data);
                    p.device = device.clone();
                }
                #[allow(unreachable_patterns)]
                _ => {
                    // Any other combination (e.g., GPU types when feature disabled)
                    // Just update device marker
                    p.device = device.clone();
                }
            }
        }
    }

    /// Switch between training and evaluation modes.
    /// Important for layers like `BatchNorm` and Dropout.
    fn train(&mut self, _mode: bool) {}
    fn eval(&mut self) {
        self.train(false);
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

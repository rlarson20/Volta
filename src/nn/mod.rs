use crate::device::Device;
use crate::io::StateDict;
use crate::tensor::Tensor;

pub mod layers;
pub mod optim;

pub use layers::{
    BatchNorm1d, BatchNorm2d, Conv2d, ConvAlgo, ConvTranspose2d, Dropout, Embedding, Flatten,
    LSTMCell, Linear, MaxPool2d, PixelShuffle, ReLU, Sequential, SequentialBuilder, Sigmoid, Tanh,
};
pub use optim::{Adam, Muon, SGD};

pub trait Module {
    fn forward(&self, x: &Tensor) -> Tensor;
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

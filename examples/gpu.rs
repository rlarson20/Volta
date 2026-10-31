//! Moves two 1000x1000 random tensors to the GPU, multiplies them there, and copies the product back to the CPU.
//!
//! Without a GPU (or without the `gpu` feature) it says so and exits.
use volta::{Device, RawTensor};

fn main() {
    // `Device::gpu()` is `None` without a GPU (or without the `gpu` feature), and moving a
    // tensor to a GPU that isn't there panics, so stop here instead.
    let Some(gpu) = Device::gpu() else {
        println!("No GPU available; nothing to run.");
        return;
    };
    println!("GPU available: {gpu}");

    // Create tensors (default is CPU)
    let a = RawTensor::randn(&[1000, 1000]);
    let b = RawTensor::randn(&[1000, 1000]);

    // Move to GPU
    let a_gpu = a.to_device(gpu.clone());
    let b_gpu = b.to_device(gpu);

    // Operations on GPU tensors automatically use GPU kernels
    let c_gpu = a_gpu.matmul(&b_gpu);

    // Move result back to CPU if needed
    let c_cpu = c_gpu.to_device(Device::CPU);

    println!("Result shape: {:?}", c_cpu.shape());
}

use volta::{Device, RawTensor, TensorOps};

fn main() {
    // Check if GPU is available
    #[cfg(feature = "gpu")]
    {
        if volta::is_gpu_available() {
            println!("GPU available!");
        } else {
            println!("No GPU, using CPU");
        }
    }

    // Create tensors (default is CPU)
    let a = RawTensor::randn(&[1000, 1000]);
    let b = RawTensor::randn(&[1000, 1000]);

    // Move to GPU
    let a_gpu = a.to_device(Device::GPU("default".to_string()));
    let b_gpu = b.to_device(Device::GPU("default".to_string()));

    // Operations on GPU tensors automatically use GPU kernels
    let c_gpu = a_gpu.matmul(&b_gpu);

    // Move result back to CPU if needed
    let c_cpu = c_gpu.to_device(Device::CPU);

    println!("Result shape: {:?}", c_cpu.borrow().shape);
}

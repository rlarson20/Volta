#[test]
#[cfg(feature = "gpu")]
fn test_tensor_gpu_unary_add_binary_smoke() {
    use volta::gpu::is_gpu_available;
    use volta::{Device, RawTensor, TensorOps};

    if !is_gpu_available() {
        return;
    }

    let dev = Device::GPU("TestDevice".to_string());
    let x = RawTensor::new(vec![1.0, -2.0], &[2], true).to_device(dev.clone());
    let y = RawTensor::new(vec![3.0, 4.0], &[2], true).to_device(dev.clone());

    let z = x.add(&y); // may take GPU path
    let w = z.relu(); // may take GPU path
    let loss = w.sum(); // CPU reduce, GPU-aware backward
    loss.backward();

    // Gradients should exist and live on GPU
    let xb = x.borrow();
    let grad = xb.grad.as_ref().unwrap();
    assert!(grad.is_gpu());
    assert_eq!(grad.to_vec(), vec![1.0, 1.0]);
}

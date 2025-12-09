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

#[test]
#[cfg(feature = "gpu")]
fn test_tensor_gpu_matmul_matches_cpu() {
    use volta::gpu::is_gpu_available;
    use volta::{Device, RawTensor, TensorOps};

    if !is_gpu_available() {
        return;
    }

    let dev = Device::GPU("TestDevice".to_string());

    // 2x3 @ 3x2 -> 2x2
    let a_cpu = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    let b_cpu = RawTensor::new(vec![0.5, -1.0, 2.0, 3.0, 1.5, 0.0], &[3, 2], false);

    let c_cpu = a_cpu.matmul(&b_cpu);

    let a_gpu = a_cpu.to_device(dev.clone());
    let b_gpu = b_cpu.to_device(dev.clone());
    let c_gpu = a_gpu.matmul(&b_gpu);

    // Move GPU result back to CPU for comparison.
    let c_gpu_cpu = c_gpu.to_device(Device::CPU);

    let cpu_vals = c_cpu.borrow().data.to_vec();
    let gpu_vals = c_gpu_cpu.borrow().data.to_vec();

    assert_eq!(cpu_vals.len(), gpu_vals.len());
    for (i, (c, g)) in cpu_vals.iter().zip(gpu_vals.iter()).enumerate() {
        assert!(
            (c - g).abs() < 1e-4,
            "Matmul CPU/GPU mismatch at index {i}: cpu={c}, gpu={g}"
        );
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_tensor_gpu_matmul_backward_consistent_with_cpu() {
    use volta::gpu::is_gpu_available;
    use volta::{Device, RawTensor, TensorOps};

    if !is_gpu_available() {
        return;
    }

    let dev = Device::GPU("TestDevice".to_string());

    // Simple 2x2 matmul with gradients on both operands.
    let x_cpu = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true);
    let w_cpu = RawTensor::new(vec![0.5, -1.0, 1.0, 2.0], &[2, 2], true);

    // GPU copies
    let x_gpu = x_cpu.to_device(dev.clone());
    let w_gpu = w_cpu.to_device(dev.clone());

    // GPU forward/backward
    let y_gpu = x_gpu.matmul(&w_gpu);
    let loss_gpu = y_gpu.sum();
    loss_gpu.backward();

    let x_gpu_grad = x_gpu.borrow().grad.as_ref().unwrap().to_vec();
    let w_gpu_grad = w_gpu.borrow().grad.as_ref().unwrap().to_vec();

    // CPU forward/backward
    let y_cpu = x_cpu.matmul(&w_cpu);
    let loss_cpu = y_cpu.sum();
    loss_cpu.backward();

    let x_cpu_grad = x_cpu.grad().unwrap();
    let w_cpu_grad = w_cpu.grad().unwrap();

    assert_eq!(x_cpu_grad.len(), x_gpu_grad.len());
    assert_eq!(w_cpu_grad.len(), w_gpu_grad.len());

    for (i, (c, g)) in x_cpu_grad.iter().zip(x_gpu_grad.iter()).enumerate() {
        assert!(
            (c - g).abs() < 1e-3,
            "x grad mismatch at {i}: cpu={c}, gpu={g}"
        );
    }
    for (i, (c, g)) in w_cpu_grad.iter().zip(w_gpu_grad.iter()).enumerate() {
        assert!(
            (c - g).abs() < 1e-3,
            "w grad mismatch at {i}: cpu={c}, gpu={g}"
        );
    }
}

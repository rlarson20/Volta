//! GPU smoke tests: unary, binary, `matmul` (forward and backward), `mulacc`, and `sum_dim` / `max_dim` results are compared with the CPU, and outputs and gradients are checked to stay on the GPU. Also checks that a mixed-device binary op panics. Ops with no GPU kernel (`mulacc` backward, `sum_dim`, `max_dim`) run under `DevicePolicy::PreferGpu` and assert their fallbacks.
//!
//! Each test returns early when no GPU is available.
#[cfg(feature = "gpu")]
#[path = "common/mod.rs"]
mod common;

#[cfg(feature = "gpu")]
use common::expect_gaps;
#[cfg(feature = "gpu")]
use volta::Axis;
#[test]
#[cfg(feature = "gpu")]
fn test_tensor_gpu_unary_add_binary_smoke() {
    use volta::gpu::is_gpu_available;
    use volta::{Device, RawTensor};

    if !is_gpu_available() {
        return;
    }

    let dev = Device::gpu().expect("GPU availability checked above");
    let x = RawTensor::new(vec![1.0, -2.0], &[2], false).to_device(dev.clone());
    x.set_requires_grad(true);
    let y = RawTensor::new(vec![3.0, 4.0], &[2], false).to_device(dev.clone());
    y.set_requires_grad(true);

    let z = x.add(&y); // may take GPU path
    let w = z.relu(); // may take GPU path
    let loss = w.sum(); // CPU reduce, GPU-aware backward
    loss.into_loss().expect("scalar loss").backward();

    // Gradients should exist and live on GPU
    let grad = x.grad_storage().unwrap();
    assert!(grad.is_gpu());
    assert_eq!(grad.to_vec(), vec![1.0, 1.0]);
}

#[test]
#[cfg(feature = "gpu")]
fn test_tensor_gpu_sum_output_device_and_grad_on_gpu() {
    use volta::gpu::is_gpu_available;
    use volta::{Device, RawTensor};

    if !is_gpu_available() {
        return;
    }

    let dev = Device::gpu().expect("GPU availability checked above");
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4], false).to_device(dev.clone());
    x.set_requires_grad(true);

    let loss = x.sum();
    assert!(
        loss.device().is_gpu(),
        "sum() output should remain on the same GPU device as the input"
    );

    loss.into_loss().expect("scalar loss").backward();

    let grad_storage = x.grad_storage().expect("Gradient for input missing");
    assert!(
        grad_storage.is_gpu(),
        "Expected input gradient to live on GPU"
    );
    assert_eq!(grad_storage.to_vec(), vec![1.0, 1.0, 1.0, 1.0]);
}

#[test]
#[cfg(feature = "gpu")]
fn test_tensor_gpu_matmul_matches_cpu() {
    use volta::gpu::is_gpu_available;
    use volta::{Device, RawTensor};

    if !is_gpu_available() {
        return;
    }

    let dev = Device::gpu().expect("GPU availability checked above");

    // 2x3 @ 3x2 -> 2x2
    let a_cpu = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    let b_cpu = RawTensor::new(vec![0.5, -1.0, 2.0, 3.0, 1.5, 0.0], &[3, 2], false);

    let c_cpu = a_cpu.matmul(&b_cpu);

    let a_gpu = a_cpu.to_device(dev.clone());
    let b_gpu = b_cpu.to_device(dev.clone());
    let c_gpu = a_gpu.matmul(&b_gpu);

    // Move GPU result back to CPU for comparison.
    let c_gpu_cpu = c_gpu.to_device(Device::CPU);

    let cpu_vals = c_cpu.data();
    let gpu_vals = c_gpu_cpu.data();

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
    use volta::{Device, RawTensor};

    if !is_gpu_available() {
        return;
    }

    let dev = Device::gpu().expect("GPU availability checked above");

    // Simple 2x2 matmul with gradients on both operands.
    let x_cpu = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true);
    let w_cpu = RawTensor::new(vec![0.5, -1.0, 1.0, 2.0], &[2, 2], true);

    // GPU copies
    let x_gpu = x_cpu.detach().to_device(dev.clone());
    x_gpu.set_requires_grad(x_cpu.requires_grad());
    let w_gpu = w_cpu.detach().to_device(dev.clone());
    w_gpu.set_requires_grad(w_cpu.requires_grad());

    // GPU forward/backward
    let y_gpu = x_gpu.matmul(&w_gpu);
    let loss_gpu = y_gpu.sum();
    loss_gpu.into_loss().expect("scalar loss").backward();

    let x_gpu_grad = x_gpu.grad_storage().unwrap().to_vec();
    let w_gpu_grad = w_gpu.grad_storage().unwrap().to_vec();

    // CPU forward/backward
    let y_cpu = x_cpu.matmul(&w_cpu);
    let loss_cpu = y_cpu.sum();
    loss_cpu.into_loss().expect("scalar loss").backward();

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

#[test]
#[cfg(feature = "gpu")]
fn test_tensor_gpu_mulacc_matches_cpu_and_grad() {
    expect_gaps(&["MulAcc"], || {
        use volta::gpu::is_gpu_available;
        use volta::{Device, RawTensor};

        if !is_gpu_available() {
            return;
        }

        let dev = Device::gpu().expect("GPU availability checked above");

        // Simple 1D tensors with requires_grad = true so we can compare gradients.
        let x_cpu = RawTensor::new(vec![1.0, 2.0, -3.0], &[3], true);
        let y_cpu = RawTensor::new(vec![0.5, -1.0, 4.0], &[3], true);
        let z_cpu = RawTensor::new(vec![0.1, 0.2, 0.3], &[3], true);

        let x_gpu = x_cpu.detach().to_device(dev.clone());
        x_gpu.set_requires_grad(x_cpu.requires_grad());
        let y_gpu = y_cpu.detach().to_device(dev.clone());
        y_gpu.set_requires_grad(y_cpu.requires_grad());
        let z_gpu = z_cpu.detach().to_device(dev.clone());
        z_gpu.set_requires_grad(z_cpu.requires_grad());

        // Forward: out = x * y + z
        let out_cpu = x_cpu.mulacc(&y_cpu, &z_cpu);
        let out_gpu = x_gpu.mulacc(&y_gpu, &z_gpu);

        // Compare forward values.
        let out_gpu_cpu = out_gpu.to_device(Device::CPU);
        let f_cpu = out_cpu.data();
        let f_gpu = out_gpu_cpu.data();
        assert_eq!(f_cpu.len(), f_gpu.len());
        for (i, (c, g)) in f_cpu.iter().zip(f_gpu.iter()).enumerate() {
            assert!(
                (c - g).abs() < 1e-5,
                "MulAcc forward CPU/GPU mismatch at {i}: cpu={c}, gpu={g}"
            );
        }

        // Backward: compare gradients w.r.t x.
        out_cpu.sum().into_loss().expect("scalar loss").backward();
        out_gpu.sum().into_loss().expect("scalar loss").backward();

        let gx_cpu = x_cpu.grad().unwrap();
        let gx_gpu = x_gpu.grad_storage().unwrap().to_vec();
        assert_eq!(gx_cpu.len(), gx_gpu.len());
        for (i, (c, g)) in gx_cpu.iter().zip(gx_gpu.iter()).enumerate() {
            assert!(
                (c - g).abs() < 1e-4,
                "MulAcc grad dL/dx CPU/GPU mismatch at {i}: cpu={c}, gpu={g}"
            );
        }
    });
}

#[test]
#[cfg(feature = "gpu")]
fn test_tensor_gpu_unary_relu_matches_cpu() {
    use volta::gpu::is_gpu_available;
    use volta::{Device, RawTensor};

    if !is_gpu_available() {
        return;
    }

    let dev = Device::gpu().expect("GPU availability checked above");

    let x_cpu = RawTensor::randn(&[16]);
    let x_gpu = x_cpu.to_device(dev.clone());

    let y_cpu = x_cpu.relu();
    let y_gpu = x_gpu.relu().to_device(Device::CPU);

    let cpu_vals = y_cpu.data();
    let gpu_vals = y_gpu.data();

    assert_eq!(cpu_vals.len(), gpu_vals.len());
    for (i, (c, g)) in cpu_vals.iter().zip(gpu_vals.iter()).enumerate() {
        assert!(
            (c - g).abs() < 1e-5,
            "ReLU CPU/GPU mismatch at index {i}: cpu={c}, gpu={g}"
        );
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_tensor_gpu_mixed_device_binary_panics() {
    use volta::RawTensor;
    use volta::{Device, is_gpu_available};

    if !is_gpu_available() {
        return;
    }

    let dev = Device::gpu().expect("GPU availability checked above");
    let a = RawTensor::new(vec![1.0, 2.0], &[2], false).to_device(dev);
    let b = RawTensor::new(vec![3.0, 4.0], &[2], false);

    let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| a.add(&b)))
        .expect_err("mixed-device binary op must panic, not fall back to the CPU");
    let msg = err.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(msg.contains("expected all inputs on one device"), "{msg}");
}

#[test]
#[cfg(feature = "gpu")]
fn test_tensor_gpu_binary_basic_matches_cpu() {
    use volta::gpu::is_gpu_available;
    use volta::{Device, RawTensor};

    if !is_gpu_available() {
        return;
    }

    let dev = Device::gpu().expect("GPU availability checked above");

    let a_cpu = RawTensor::new(vec![1.0, -2.0, 3.5, 4.0], &[4], false);
    let b_cpu = RawTensor::new(vec![0.5, 2.0, -1.0, 8.0], &[4], false);

    let a_gpu = a_cpu.to_device(dev.clone());
    let b_gpu = b_cpu.to_device(dev.clone());

    // add
    let add_cpu = a_cpu.add(&b_cpu);
    let add_gpu = a_gpu.add(&b_gpu).to_device(Device::CPU);
    let add_c = add_cpu.data();
    let add_g = add_gpu.data();
    for (i, (c, g)) in add_c.iter().zip(add_g.iter()).enumerate() {
        assert!(
            (c - g).abs() < 1e-5,
            "add CPU/GPU mismatch at index {i}: cpu={c}, gpu={g}"
        );
    }

    // sub
    let sub_cpu = a_cpu.sub(&b_cpu);
    let sub_gpu = a_gpu.sub(&b_gpu).to_device(Device::CPU);
    let sub_c = sub_cpu.data();
    let sub_g = sub_gpu.data();
    for (i, (c, g)) in sub_c.iter().zip(sub_g.iter()).enumerate() {
        assert!(
            (c - g).abs() < 1e-5,
            "sub CPU/GPU mismatch at index {i}: cpu={c}, gpu={g}"
        );
    }

    // mul
    let mul_cpu = a_cpu.elem_mul(&b_cpu);
    let mul_gpu = a_gpu.elem_mul(&b_gpu).to_device(Device::CPU);
    let mul_c = mul_cpu.data();
    let mul_g = mul_gpu.data();
    for (i, (c, g)) in mul_c.iter().zip(mul_g.iter()).enumerate() {
        assert!(
            (c - g).abs() < 1e-5,
            "mul CPU/GPU mismatch at index {i}: cpu={c}, gpu={g}"
        );
    }

    // div
    let div_cpu = a_cpu.div(&b_cpu);
    let div_gpu = a_gpu.div(&b_gpu).to_device(Device::CPU);
    let div_c = div_cpu.data();
    let div_g = div_gpu.data();
    for (i, (c, g)) in div_c.iter().zip(div_g.iter()).enumerate() {
        assert!(
            (c - g).abs() < 1e-5,
            "div CPU/GPU mismatch at index {i}: cpu={c}, gpu={g}"
        );
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_tensor_gpu_sum_dim_device_and_grad_on_gpu() {
    expect_gaps(&["SumDim"], || {
        use volta::gpu::is_gpu_available;
        use volta::{Device, RawTensor};

        if !is_gpu_available() {
            return;
        }

        let dev = Device::gpu().expect("GPU availability checked above");
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false).to_device(dev.clone());
        x.set_requires_grad(true);

        // Sum over last dim: [[1,2],[3,4]] -> [3,7]
        let y = x.sum_dim(Axis(1), false);
        assert!(
            y.device().is_gpu(),
            "sum_dim output should remain on the same GPU device as the input"
        );
        assert_eq!(y.shape(), vec![2]);
        assert_eq!(y.data(), vec![3.0, 7.0]);

        y.sum().into_loss().expect("scalar loss").backward();

        let grad_storage = x.grad_storage().expect("Gradient for input missing");
        assert!(
            grad_storage.is_gpu(),
            "Expected input gradient for sum_dim to live on GPU"
        );
        // Each input element contributes once to the corresponding row sum.
        assert_eq!(grad_storage.to_vec(), vec![1.0, 1.0, 1.0, 1.0]);
    });
}

#[test]
#[cfg(feature = "gpu")]
fn test_tensor_gpu_max_dim_device_and_grad_on_gpu() {
    expect_gaps(&["MaxDim"], || {
        use volta::gpu::is_gpu_available;
        use volta::{Device, RawTensor};

        if !is_gpu_available() {
            return;
        }

        let dev = Device::gpu().expect("GPU availability checked above");
        // Shape [2,3]
        let x = RawTensor::new(vec![1.0, 5.0, 3.0, 2.0, 8.0, 4.0], &[2, 3], false)
            .to_device(dev.clone());
        x.set_requires_grad(true);

        let y = x.max_dim(Axis(1), false);
        assert!(
            y.device().is_gpu(),
            "max_dim output should remain on the same GPU device as the input"
        );
        assert_eq!(y.shape(), vec![2]);
        assert_eq!(y.data(), vec![5.0, 8.0]);

        y.sum().into_loss().expect("scalar loss").backward();

        let grad_storage = x.grad_storage().expect("Gradient for input missing");
        assert!(
            grad_storage.is_gpu(),
            "Expected input gradient for max_dim to live on GPU"
        );
        // Only max elements get gradient: indices (0,1) and (1,1).
        assert_eq!(grad_storage.to_vec(), vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0]);
    });
}

#[test]
#[cfg(feature = "gpu")]
fn test_binary_add_backward_simple() {
    use volta::gpu::is_gpu_available;
    use volta::{Device, RawTensor};

    if !is_gpu_available() {
        return;
    }

    let dev = Device::gpu().expect("GPU availability checked above");
    let x = RawTensor::new(vec![1.0, 2.0], &[2], false).to_device(dev.clone());
    x.set_requires_grad(true);
    let y = RawTensor::new(vec![3.0, 4.0], &[2], false).to_device(dev.clone());
    y.set_requires_grad(true);

    let z = x.add(&y);
    let loss = z.sum();
    loss.into_loss().expect("scalar loss").backward();

    let x_grad = x.grad().unwrap();
    println!("x_grad: {x_grad:?}");
    assert_eq!(x_grad, vec![1.0, 1.0]);

    let y_grad = y.grad().unwrap();
    println!("y_grad: {y_grad:?}");
    assert_eq!(y_grad, vec![1.0, 1.0]);
}

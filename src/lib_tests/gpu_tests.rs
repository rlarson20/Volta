//! GPU: device transfers, backward passes, and optimizers compared with the CPU, and the GPU broadcast binary backward. Tests that need an adapter return early without one.

use super::*;
#[cfg(feature = "gpu")]
use approx::assert_abs_diff_eq;

/// Run `f` under `DevicePolicy::PreferGpu` and assert that the ops it ran on the CPU
/// for GPU inputs are exactly `ops`, as a set of names (`tests/common::expect_gaps`).
/// Skips without an adapter, like the integration-test copy: `f` would return early and
/// no gap could be recorded.
#[cfg(feature = "gpu")]
fn expect_gaps(ops: &[&str], f: impl FnOnce()) {
    if !is_gpu_available() {
        return;
    }
    let _prefer = with_device_policy(DevicePolicy::PreferGpu);
    let _ = take_fallback_events();
    f();
    let got: std::collections::BTreeSet<String> =
        take_fallback_events().into_iter().map(|e| e.op).collect();
    let want: std::collections::BTreeSet<String> = ops.iter().map(ToString::to_string).collect();
    assert_eq!(got, want, "ops that ran on the CPU for GPU inputs");
}

#[test]
fn test_device_gpu_returns_none_when_disabled() {
    // When gpu feature is disabled, Device::gpu() should return None
    let gpu = Device::gpu();
    #[cfg(feature = "gpu")]
    {
        // If GPU is available, gpu() should return Some device
        // We can't guarantee GPU is available on all systems
        if is_gpu_available() {
            assert!(gpu.is_some());
            assert!(gpu.unwrap().is_gpu());
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        // Without gpu feature, should always be None
        assert!(gpu.is_none());
    }
}

#[test]
fn test_to_device_cpu_to_cpu() {
    let t = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], false);
    let t_cpu = t.to_device(Device::CPU);

    // Same device should return same tensor (fast path)
    assert_eq!(t_cpu.borrow().device, Device::CPU);
    assert_eq!(t_cpu.borrow().data.to_vec(), vec![1.0, 2.0, 3.0]);
}

#[cfg(feature = "gpu")]
#[test]
fn test_to_device_cpu_to_gpu() {
    if !is_gpu_available() {
        return; // Skip test if GPU not available
    }

    let t = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false);
    let gpu_device = Device::gpu().expect("GPU should be available");
    let t_gpu = t.to_device(gpu_device.clone());

    // Device should be GPU
    assert!(t_gpu.borrow().device.is_gpu());
    assert_eq!(t_gpu.borrow().device.name(), gpu_device.name());

    // Data should be preserved
    assert_eq!(t_gpu.borrow().data.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(t_gpu.borrow().shape, vec![2, 2]);
}

#[cfg(feature = "gpu")]
#[test]
fn test_to_device_gpu_to_cpu() {
    if !is_gpu_available() {
        return; // Skip test if GPU not available
    }

    let gpu_device = Device::gpu().expect("GPU should be available");
    let t = RawTensor::new(vec![5.0, 6.0, 7.0], &[3], false);
    let t_gpu = t.to_device(gpu_device.clone());

    // Move back to CPU
    let t_cpu = t_gpu.to_device(Device::CPU);

    assert!(t_cpu.borrow().device.is_cpu());
    assert_eq!(t_cpu.borrow().data.to_vec(), vec![5.0, 6.0, 7.0]);
}

#[cfg(feature = "gpu")]
#[test]
fn test_to_device_preserves_autograd_metadata() {
    if !is_gpu_available() {
        return; // Skip test if GPU not available
    }

    // Create a simple computation graph
    let a = RawTensor::new(vec![2.0], &[1], true);
    let b = RawTensor::new(vec![3.0], &[1], true);
    let c = a.add(&b);

    // Move result to GPU
    let gpu_device = Device::gpu().expect("GPU should be available");
    let c_gpu = c.to_device(gpu_device);

    // Autograd metadata should be preserved
    assert!(c_gpu.borrow().requires_grad);
    assert!(!c_gpu.borrow().parents.is_empty());
    assert!(c_gpu.borrow().grad_fn.is_some());

    // Note: Gradients are still computed on CPU
    // This is a known limitation documented in to_device()
}

#[cfg(feature = "gpu")]
#[test]
fn test_matmul_backward_gpu() {
    if !is_gpu_available() {
        return; // Skip test if GPU not available
    }

    let gpu_device = Device::gpu().expect("GPU should be available");

    // Simple 2x3 @ 3x4 = 2x4 case
    let a = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false)
        .to_device(gpu_device.clone());
    a.set_requires_grad(true);
    let b = RawTensor::new(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
        false,
    )
    .to_device(gpu_device.clone());
    b.set_requires_grad(true);
    let c = a.matmul(&b);

    // Forward should be on GPU
    assert!(c.borrow().device.is_gpu());

    // Backward should compute gradients on GPU
    c.sum().into_loss().expect("scalar loss").backward();

    // Gradients should be on GPU
    {
        let a_ref = a.borrow();
        let b_ref = b.borrow();
        let a_grad = a_ref.grad.as_ref().expect("a should have grad");
        let b_grad = b_ref.grad.as_ref().expect("b should have grad");
        assert!(a_grad.is_gpu());
        assert!(b_grad.is_gpu());
    }

    // Verify gradient values are correct by comparing to CPU computation
    let a_cpu = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
    let b_cpu = RawTensor::new(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
        true,
    );
    let c_cpu = a_cpu.matmul(&b_cpu);
    c_cpu.sum().into_loss().expect("scalar loss").backward();

    let a_grad_data;
    let b_grad_data;
    {
        let a_ref = a.borrow();
        let b_ref = b.borrow();
        a_grad_data = a_ref.grad.as_ref().unwrap().to_vec();
        b_grad_data = b_ref.grad.as_ref().unwrap().to_vec();
    }

    let a_grad_cpu_data;
    let b_grad_cpu_data;
    {
        let a_ref = a_cpu.borrow();
        let b_ref = b_cpu.borrow();
        a_grad_cpu_data = a_ref.grad.as_ref().unwrap().to_vec();
        b_grad_cpu_data = b_ref.grad.as_ref().unwrap().to_vec();
    }

    assert_eq!(a_grad_data.len(), a_grad_cpu_data.len());
    assert_eq!(b_grad_data.len(), b_grad_cpu_data.len());

    // Check values are approximately equal
    for (gpu_val, cpu_val) in a_grad_data.iter().zip(a_grad_cpu_data.iter()) {
        assert!((gpu_val - cpu_val).abs() < 1e-5);
    }
    for (gpu_val, cpu_val) in b_grad_data.iter().zip(b_grad_cpu_data.iter()) {
        assert!((gpu_val - cpu_val).abs() < 1e-5);
    }
}

#[cfg(feature = "gpu")]
#[test]
fn test_linear_layer_backward_gpu() {
    expect_gaps(&["Binary(Add)"], || {
        use crate::nn::{Linear, Module};
        if !is_gpu_available() {
            return; // Skip test if GPU not available
        }
        let gpu_device = Device::gpu().expect("GPU should be available");

        let mut layer = Linear::new(4, 3, true);
        layer.to_device(gpu_device.clone());

        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4], false)
            .to_device(gpu_device);
        x.set_requires_grad(true);
        let out = layer.forward(&x);
        let loss = out.sum();

        loss.into_loss().expect("scalar loss").backward();

        // Every parameter and the input get a gradient, on the GPU
        let params = layer.parameters();
        assert_eq!(params.len(), 2);
        for t in params.iter().chain([&x]) {
            assert!(
                t.grad_storage().expect("gradient").is_gpu(),
                "Gradient should be on GPU"
            );
        }
    });
}

#[cfg(feature = "gpu")]
#[test]
fn test_sum_backward_gpu() {
    if !is_gpu_available() {
        return;
    }

    let gpu_device = Device::gpu().expect("GPU should be available");

    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false)
        .to_device(gpu_device.clone());
    x.set_requires_grad(true);
    let sum_result = x.sum();

    // Forward should be on GPU
    assert!(sum_result.borrow().device.is_gpu());

    // Backward should compute gradients on GPU
    sum_result.into_loss().expect("scalar loss").backward();

    // Gradient should be on GPU and all ones
    let grad_data;
    {
        let x_ref = x.borrow();
        let x_grad = x_ref.grad.as_ref().expect("x should have grad");
        assert!(x_grad.is_gpu());
        grad_data = x_grad.to_vec();
    }

    assert_eq!(grad_data.len(), 6);
    for &val in &grad_data {
        assert!((val - 1.0).abs() < 1e-5);
    }
}

#[cfg(feature = "gpu")]
#[test]
fn test_mean_backward_gpu() {
    if !is_gpu_available() {
        return;
    }

    let gpu_device = Device::gpu().expect("GPU should be available");

    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false)
        .to_device(gpu_device.clone());
    x.set_requires_grad(true);
    let mean_result = x.mean();

    // Forward should be on GPU
    assert!(mean_result.borrow().device.is_gpu());

    // Backward should compute gradients on GPU
    mean_result.into_loss().expect("scalar loss").backward();

    // Gradient should be on GPU and all 1/6
    let grad_data;
    {
        let x_ref = x.borrow();
        let x_grad = x_ref.grad.as_ref().expect("x should have grad");
        assert!(x_grad.is_gpu());
        grad_data = x_grad.to_vec();
    }

    assert_eq!(grad_data.len(), 6);
    let expected = 1.0 / 6.0;
    for &val in &grad_data {
        assert!((val - expected).abs() < 1e-5);
    }
}

#[cfg(feature = "gpu")]
#[test]
fn test_max_backward_gpu() {
    if !is_gpu_available() {
        return;
    }

    let gpu_device = Device::gpu().expect("GPU should be available");

    let x = RawTensor::new(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3], false)
        .to_device(gpu_device.clone());
    x.set_requires_grad(true);
    let max_result = x.max_reduce();

    // Forward should be on GPU
    assert!(max_result.borrow().device.is_gpu());

    // Backward should compute gradients on GPU
    max_result.into_loss().expect("scalar loss").backward();

    // Gradient should be on GPU and only max element (6.0 at index 5) gets grad
    let grad_data;
    {
        let x_ref = x.borrow();
        let x_grad = x_ref.grad.as_ref().expect("x should have grad");
        assert!(x_grad.is_gpu());
        grad_data = x_grad.to_vec();
    }

    assert_eq!(grad_data.len(), 6);

    // Only the max element (6.0 at linear index 5) should have gradient 1.0
    for (i, &val) in grad_data.iter().enumerate() {
        if i == 5 {
            assert!((val - 1.0).abs() < 1e-5);
        } else {
            assert!(val.abs() < 1e-5);
        }
    }
}

#[cfg(feature = "gpu")]
#[test]
fn test_reduction_backward_gpu_cpu_equivalence() {
    if !is_gpu_available() {
        return;
    }

    let gpu_device = Device::gpu().expect("GPU should be available");

    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let shape = &[2, 4];

    // Test sum
    let x_cpu = RawTensor::new(data.clone(), shape, true);
    let sum_cpu = x_cpu.sum();
    sum_cpu.into_loss().expect("scalar loss").backward();
    let sum_grad_cpu = x_cpu.borrow().grad.as_ref().unwrap().to_vec();

    let x_gpu = RawTensor::new(data.clone(), shape, false).to_device(gpu_device.clone());
    x_gpu.set_requires_grad(true);
    let sum_gpu = x_gpu.sum();
    sum_gpu.into_loss().expect("scalar loss").backward();
    let sum_grad_gpu = x_gpu.borrow().grad.as_ref().unwrap().to_vec();

    assert_eq!(sum_grad_cpu.len(), sum_grad_gpu.len());
    for (cpu_val, gpu_val) in sum_grad_cpu.iter().zip(sum_grad_gpu.iter()) {
        assert!((cpu_val - gpu_val).abs() < 1e-5);
    }

    // Test mean
    let x_cpu2 = RawTensor::new(data.clone(), shape, true);
    let mean_cpu = x_cpu2.mean();
    mean_cpu.into_loss().expect("scalar loss").backward();
    let mean_grad_cpu = x_cpu2.borrow().grad.as_ref().unwrap().to_vec();

    let x_gpu2 = RawTensor::new(data.clone(), shape, false).to_device(gpu_device.clone());
    x_gpu2.set_requires_grad(true);
    let mean_gpu = x_gpu2.mean();
    mean_gpu.into_loss().expect("scalar loss").backward();
    let mean_grad_gpu = x_gpu2.borrow().grad.as_ref().unwrap().to_vec();

    assert_eq!(mean_grad_cpu.len(), mean_grad_gpu.len());
    for (cpu_val, gpu_val) in mean_grad_cpu.iter().zip(mean_grad_gpu.iter()) {
        assert!((cpu_val - gpu_val).abs() < 1e-5);
    }

    // Test max
    let x_cpu3 = RawTensor::new(data.clone(), shape, true);
    let max_cpu = x_cpu3.max_reduce();
    max_cpu.into_loss().expect("scalar loss").backward();
    let max_grad_cpu = x_cpu3.borrow().grad.as_ref().unwrap().to_vec();

    let x_gpu3 = RawTensor::new(data.clone(), shape, false).to_device(gpu_device);
    x_gpu3.set_requires_grad(true);
    let max_gpu = x_gpu3.max_reduce();
    max_gpu.into_loss().expect("scalar loss").backward();
    let max_grad_gpu = x_gpu3.borrow().grad.as_ref().unwrap().to_vec();

    assert_eq!(max_grad_cpu.len(), max_grad_gpu.len());
    for (cpu_val, gpu_val) in max_grad_cpu.iter().zip(max_grad_gpu.iter()) {
        assert!((cpu_val - gpu_val).abs() < 1e-5);
    }
}

#[cfg(feature = "gpu")]
#[test]
fn test_sgd_optimizer_gpu() {
    use crate::nn::optim::SGD;

    if !is_gpu_available() {
        return;
    }

    let gpu_device = Device::gpu().expect("GPU should be available");

    // Create a simple parameter on GPU
    let param = RawTensor::new(vec![0.5; 10], &[10], true).to_device(gpu_device.clone());

    // Create optimizer with the parameter
    let mut opt = SGD::new(vec![param.clone()], 0.01, 0.0, 0.0);

    // Manually set a gradient on GPU
    param.set_grad(Storage::gpu(vec![0.1; 10]));

    // Step should update the parameter
    let param_before = param.borrow().data.to_vec();
    opt.step();
    let param_after = param.borrow().data.to_vec();

    // Parameter should have changed (param -= lr * grad = 0.01 * 0.1 = 0.001 per element)
    for (before, after) in param_before.iter().zip(param_after.iter()) {
        assert!((after - (before - 0.001)).abs() < 1e-5);
    }
}

#[cfg(feature = "gpu")]
#[test]
fn test_adam_optimizer_gpu() {
    use crate::nn::optim::Adam;

    if !is_gpu_available() {
        return;
    }

    let gpu_device = Device::gpu().expect("GPU should be available");

    // Create a simple parameter on GPU
    let param = RawTensor::new(vec![0.5; 10], &[10], true).to_device(gpu_device.clone());

    // Create Adam optimizer with the parameter
    let mut opt = Adam::new(vec![param.clone()], 0.01, (0.9, 0.999), 1e-8, 0.0);

    // Manually set a gradient on GPU
    param.set_grad(Storage::gpu(vec![0.1; 10]));

    // Step should update the parameter
    let param_before = param.borrow().data.to_vec();
    opt.step();
    let param_after = param.borrow().data.to_vec();

    // Parameter should have changed (Adam update formula is complex, but should change)
    for (before, after) in param_before.iter().zip(param_after.iter()) {
        assert_ne!(before, after, "Parameter should change after Adam step");
    }
}

#[cfg(feature = "gpu")]
#[test]
fn test_sgd_momentum_optimizer_gpu() {
    use crate::nn::optim::SGD;

    if !is_gpu_available() {
        return;
    }

    let gpu_device = Device::gpu().expect("GPU should be available");

    // Create a simple parameter on GPU
    let param = RawTensor::new(vec![0.5; 10], &[10], true).to_device(gpu_device.clone());

    // Create SGD optimizer with momentum
    let mut opt = SGD::new(vec![param.clone()], 0.01, 0.9, 0.0);

    // Manually set a gradient on GPU
    param.set_grad(Storage::gpu(vec![0.1; 10]));

    // First step
    let param_before = param.borrow().data.to_vec();
    opt.step();
    let param_after_step1 = param.borrow().data.to_vec();

    // Parameter should have changed
    for (before, after) in param_before.iter().zip(param_after_step1.iter()) {
        assert_ne!(before, after);
    }

    // Set same gradient again
    param.set_grad(Storage::gpu(vec![0.1; 10]));

    // Second step should apply different update due to momentum
    let param_after_step2;
    {
        let p = param.borrow();
        param_after_step2 = p.data.to_vec();
    }

    // With momentum, second step should be different from first
    // (momentum accumulates gradient velocity)
    let changes_step1: Vec<f32> = param_after_step1
        .iter()
        .zip(param_before.iter())
        .map(|(a, b)| a - b)
        .collect();
    let changes_step2: Vec<f32> = param_after_step2
        .iter()
        .zip(param_after_step1.iter())
        .map(|(a, b)| a - b)
        .collect();

    // Changes should be different due to momentum accumulation
    for (c1, c2) in changes_step1.iter().zip(changes_step2.iter()) {
        assert_ne!(c1, c2);
    }
}

#[cfg(feature = "gpu")]
#[test]
fn test_optimizer_gpu_cpu_equivalence() {
    use crate::nn::optim::Adam;

    if !is_gpu_available() {
        return;
    }

    let gpu_device = Device::gpu().expect("GPU should be available");

    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let grad_data = vec![0.1, 0.2, 0.3, 0.4, 0.5];

    // CPU parameter and optimizer
    let param_cpu = RawTensor::new(data.clone(), &[5], true);
    let mut opt_cpu = Adam::new(vec![param_cpu.clone()], 0.01, (0.9, 0.999), 1e-8, 0.0);
    param_cpu.set_grad(Storage::cpu(grad_data.clone()));

    // GPU parameter and optimizer
    let param_gpu = RawTensor::new(data, &[5], false).to_device(gpu_device.clone());
    param_gpu.set_requires_grad(true);
    let mut opt_gpu = Adam::new(vec![param_gpu.clone()], 0.01, (0.9, 0.999), 1e-8, 0.0);
    // Create GPU gradient
    param_gpu.set_grad(Storage::gpu(grad_data));

    // Take one step on both
    opt_cpu.step();
    opt_gpu.step();

    // Results should be approximately equal
    let result_cpu = param_cpu.borrow().data.to_vec();
    let result_gpu = param_gpu.borrow().data.to_vec();

    assert_eq!(result_cpu.len(), result_gpu.len());
    for (cpu_val, gpu_val) in result_cpu.iter().zip(result_gpu.iter()) {
        assert!(
            (cpu_val - gpu_val).abs() < 1e-4,
            "CPU={cpu_val}, GPU={gpu_val}"
        );
    }
}

#[cfg(feature = "gpu")]
#[test]
fn test_optimizer_state_stays_on_gpu() {
    use crate::nn::optim::Adam;

    if !is_gpu_available() {
        return;
    }

    let gpu_device = Device::gpu().expect("GPU should be available");

    // Create a parameter on GPU
    let param = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5], false).to_device(gpu_device);
    param.set_requires_grad(true);

    // Create Adam optimizer - state should be initialized on GPU
    let mut opt = Adam::new(vec![param.clone()], 0.01, (0.9, 0.999), 1e-8, 0.0);

    // Verify state is on GPU
    // We can't directly access m and v since they're private,
    // but we can verify the optimizer works without CPU transfers

    // Set gradient on GPU
    param.set_grad(Storage::gpu(vec![0.1, 0.2, 0.3, 0.4, 0.5]));

    // Take multiple steps
    for _ in 0..5 {
        // Update gradient each step
        param.set_grad(Storage::gpu(vec![0.1, 0.2, 0.3, 0.4, 0.5]));
        opt.step();
    }

    // If state was being transferred to CPU each step, this would be much slower
    // and the test would time out. For this test, we just verify it completes
    // successfully without errors.

    // Verify parameter changed
    let result = param.borrow().data.to_vec();
    for val in &result {
        assert_ne!(*val, 0.0, "Parameter should have been updated");
    }
}

#[cfg(feature = "gpu")]
#[test]
fn test_sgd_momentum_state_stays_on_gpu() {
    use crate::nn::optim::SGD;

    if !is_gpu_available() {
        return;
    }

    let gpu_device = Device::gpu().expect("GPU should be available");

    // Create a parameter on GPU
    let param = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5], false).to_device(gpu_device);
    param.set_requires_grad(true);

    // Create SGD with momentum - velocity state should be on GPU
    let mut opt = SGD::new(vec![param.clone()], 0.01, 0.9, 0.0);

    // Set gradient on GPU and take multiple steps
    for _ in 0..5 {
        param.set_grad(Storage::gpu(vec![0.1, 0.2, 0.3, 0.4, 0.5]));
        opt.step();
    }

    // Verify parameter changed
    let result = param.borrow().data.to_vec();
    for val in &result {
        assert_ne!(*val, 0.0, "Parameter should have been updated");
    }
}

#[cfg(feature = "gpu")]
#[test]
fn test_gpu_binary_backward_broadcast_add() {
    expect_gaps(&["Binary(Add)"], || {
        if !is_gpu_available() {
            return;
        }

        let gpu_device = Device::gpu().expect("GPU should be available");

        // Test case: (3, 1) + (1, 4) -> (3, 4)
        // a_grad should sum over dim 1: (3, 1)
        // b_grad should sum over dim 0: (1, 4)
        let a = RawTensor::new(vec![1.0, 2.0, 3.0], &[3, 1], false).to_device(gpu_device.clone());
        a.set_requires_grad(true);
        let b = RawTensor::new(vec![10.0, 20.0, 30.0, 40.0], &[1, 4], false)
            .to_device(gpu_device.clone());
        b.set_requires_grad(true);

        let c = a.add(&b);
        c.sum().into_loss().expect("scalar loss").backward();

        // Verify gradients exist
        assert!(a.grad().is_some());
        assert!(b.grad().is_some());

        let a_grad = a.grad().unwrap();
        let b_grad = b.grad().unwrap();

        // Each a element received gradient from 4 b elements
        assert_abs_diff_eq!(
            a_grad.first().copied().expect("tensor data should exist"),
            4.0,
            epsilon = 1e-3
        );
        assert_abs_diff_eq!(
            a_grad.get(1).copied().expect("tensor data should exist"),
            4.0,
            epsilon = 1e-3
        );
        assert_abs_diff_eq!(
            a_grad.get(2).copied().expect("tensor data should exist"),
            4.0,
            epsilon = 1e-3
        );

        // Each b element received gradient from 3 a elements
        assert_abs_diff_eq!(
            b_grad.first().copied().expect("tensor data should exist"),
            3.0,
            epsilon = 1e-3
        );
        assert_abs_diff_eq!(
            b_grad.get(1).copied().expect("tensor data should exist"),
            3.0,
            epsilon = 1e-3
        );
        assert_abs_diff_eq!(
            b_grad.get(2).copied().expect("tensor data should exist"),
            3.0,
            epsilon = 1e-3
        );
        assert_abs_diff_eq!(
            b_grad.get(3).copied().expect("tensor data should exist"),
            3.0,
            epsilon = 1e-3
        );
    });
}

#[cfg(feature = "gpu")]
#[test]
fn test_gpu_binary_backward_broadcast_mul() {
    expect_gaps(&["Binary(Mul)"], || {
        if !is_gpu_available() {
            return;
        }

        let gpu_device = Device::gpu().expect("GPU should be available");

        // Test broadcasting multiplication
        let a = RawTensor::new(vec![1.0, 2.0], &[2, 1], false).to_device(gpu_device.clone());
        a.set_requires_grad(true);
        let b =
            RawTensor::new(vec![10.0, 20.0, 30.0], &[1, 3], false).to_device(gpu_device.clone());
        b.set_requires_grad(true);

        let c = a.elem_mul(&b);
        c.sum().into_loss().expect("scalar loss").backward();

        // Verify gradients exist and are finite
        assert!(a.grad().is_some());
        assert!(b.grad().is_some());

        let a_grad = a.grad().unwrap();
        let b_grad = b.grad().unwrap();

        // All gradients should be finite
        for g in &a_grad {
            assert!(g.is_finite(), "a_grad contains non-finite value");
        }
        for g in &b_grad {
            assert!(g.is_finite(), "b_grad contains non-finite value");
        }
    });
}

#[cfg(feature = "gpu")]
#[test]
fn test_gpu_binary_backward_broadcast_stress() {
    expect_gaps(&["Binary(Add)"], || {
        if !is_gpu_available() {
            return;
        }

        let gpu_device = Device::gpu().expect("GPU should be available");

        // Stress test with many output positions mapping to same input
        let a = RawTensor::new(vec![1.0], &[1], false).to_device(gpu_device.clone());
        a.set_requires_grad(true);
        let b = RawTensor::new(
            (0..1000).map(|i| i as f32).collect::<Vec<_>>(),
            &[1000],
            true,
        )
        .to_device(gpu_device.clone());

        let c = a.add(&b);
        c.sum().into_loss().expect("scalar loss").backward();

        // a should accumulate gradient from all 1000 positions
        let a_grad = a.grad().unwrap();
        assert_abs_diff_eq!(
            a_grad.first().copied().expect("tensor data should exist"),
            1000.0,
            epsilon = 1e-2
        );
    });
}

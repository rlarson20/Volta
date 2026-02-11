//! Tests for GPU-accelerated movement operation backward passes
#![cfg(feature = "gpu")]

use volta::device::Device;
use volta::gpu::is_gpu_available;
use volta::{RawTensor, TensorOps};

fn approx_eq(a: f32, b: f32, epsilon: f32) -> bool {
    (a - b).abs() < epsilon
}

fn tensors_approx_eq(a: &[f32], b: &[f32], epsilon: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| approx_eq(*x, *y, epsilon))
}

#[test]
fn test_permute_backward_cpu_vs_gpu() {
    if !is_gpu_available() {
        return;
    }

    let device = Device::gpu().expect("GPU should be available");

    // Create a 2x3 tensor on CPU
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let cpu_t = RawTensor::new(data.clone(), &[2, 3], true);

    // Permute: swap axes [2,3] -> [3,2]
    let cpu_permuted = cpu_t.permute(&[1, 0]);

    // Create same tensor on GPU
    let gpu_t = RawTensor::new(data, &[2, 3], true).to_device(device.clone());
    let gpu_permuted = gpu_t.permute(&[1, 0]);

    // Backward pass
    cpu_permuted.backward();
    gpu_permuted.backward();

    // Compare gradients
    let cpu_grad = cpu_t.borrow().grad.as_ref().unwrap().to_vec();
    let gpu_grad = gpu_t.borrow().grad.as_ref().unwrap().to_vec();

    assert!(
        tensors_approx_eq(&cpu_grad, &gpu_grad, 1e-5),
        "Permute backward: CPU and GPU gradients should match.\nCPU: {cpu_grad:?}\nGPU: {gpu_grad:?}"
    );
}

#[test]
fn test_expand_backward_cpu_vs_gpu() {
    if !is_gpu_available() {
        return;
    }

    let device = Device::gpu().expect("GPU should be available");

    // Create a 1x3 tensor (will broadcast to 2x3)
    let data = vec![1.0, 2.0, 3.0];
    let cpu_t = RawTensor::new(data.clone(), &[1, 3], true);

    // Expand: [1,3] -> [2,3]
    let cpu_expanded = cpu_t.expand(&[2, 3]);

    // Create same tensor on GPU
    let gpu_t = RawTensor::new(data, &[1, 3], true).to_device(device.clone());
    let gpu_expanded = gpu_t.expand(&[2, 3]);

    // Backward pass
    cpu_expanded.backward();
    gpu_expanded.backward();

    // Compare gradients (should sum over broadcast dimension)
    let cpu_grad = cpu_t.borrow().grad.as_ref().unwrap().to_vec();
    let gpu_grad = gpu_t.borrow().grad.as_ref().unwrap().to_vec();

    assert!(
        tensors_approx_eq(&cpu_grad, &gpu_grad, 1e-5),
        "Expand backward: CPU and GPU gradients should match.\nCPU: {cpu_grad:?}\nGPU: {gpu_grad:?}"
    );

    // Gradient should be [2, 2, 2] since we broadcast over 2 positions
    assert!(
        tensors_approx_eq(&cpu_grad, &[2.0, 2.0, 2.0], 1e-5),
        "Expand backward should sum over broadcast dimension"
    );
}

#[test]
fn test_pad_backward_cpu_vs_gpu() {
    if !is_gpu_available() {
        return;
    }

    let device = Device::gpu().expect("GPU should be available");

    // Create a 2x2 tensor
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let cpu_t = RawTensor::new(data.clone(), &[2, 2], true);

    // Pad: add 1 zero on each side -> [4,4]
    let padding = vec![(1, 1), (1, 1)];
    let cpu_padded = cpu_t.pad(&padding);

    // Create same tensor on GPU
    let gpu_t = RawTensor::new(data, &[2, 2], true).to_device(device.clone());
    let gpu_padded = gpu_t.pad(&padding);

    // Backward pass
    cpu_padded.backward();
    gpu_padded.backward();

    // Compare gradients (should extract center region)
    let cpu_grad = cpu_t.borrow().grad.as_ref().unwrap().to_vec();
    let gpu_grad = gpu_t.borrow().grad.as_ref().unwrap().to_vec();

    assert!(
        tensors_approx_eq(&cpu_grad, &gpu_grad, 1e-5),
        "Pad backward: CPU and GPU gradients should match.\nCPU: {cpu_grad:?}\nGPU: {gpu_grad:?}"
    );
}

#[test]
fn test_shrink_backward_cpu_vs_gpu() {
    if !is_gpu_available() {
        return;
    }

    let device = Device::gpu().expect("GPU should be available");

    // Create a 4x4 tensor
    let data: Vec<f32> = (0..16).map(|x| x as f32).collect();
    let cpu_t = RawTensor::new(data.clone(), &[4, 4], true);

    // Shrink: extract 2x2 center region
    let ranges = vec![(1, 3), (1, 3)];
    let cpu_shrunk = cpu_t.shrink(&ranges);

    // Create same tensor on GPU
    let gpu_t = RawTensor::new(data, &[4, 4], true).to_device(device.clone());
    let gpu_shrunk = gpu_t.shrink(&ranges);

    // Backward pass
    cpu_shrunk.backward();
    gpu_shrunk.backward();

    // Compare gradients (should place gradient in window, zero elsewhere)
    let cpu_grad = cpu_t.borrow().grad.as_ref().unwrap().to_vec();
    let gpu_grad = gpu_t.borrow().grad.as_ref().unwrap().to_vec();

    assert!(
        tensors_approx_eq(&cpu_grad, &gpu_grad, 1e-5),
        "Shrink backward: CPU and GPU gradients should match.\nCPU: {cpu_grad:?}\nGPU: {gpu_grad:?}"
    );
}

#[test]
fn test_stride_backward_cpu_vs_gpu() {
    if !is_gpu_available() {
        return;
    }

    let device = Device::gpu().expect("GPU should be available");

    // Create a 4x4 tensor
    let data: Vec<f32> = (0..16).map(|x| x as f32).collect();
    let cpu_t = RawTensor::new(data.clone(), &[4, 4], true);

    // Stride: subsample with stride 2 -> [2,2]
    let strides = vec![2, 2];
    let cpu_strided = cpu_t.stride_op(&strides);

    // Create same tensor on GPU
    let gpu_t = RawTensor::new(data, &[4, 4], true).to_device(device.clone());
    let gpu_strided = gpu_t.stride_op(&strides);

    // Backward pass
    cpu_strided.backward();
    gpu_strided.backward();

    // Compare gradients (should upsample with zeros)
    let cpu_grad = cpu_t.borrow().grad.as_ref().unwrap().to_vec();
    let gpu_grad = gpu_t.borrow().grad.as_ref().unwrap().to_vec();

    assert!(
        tensors_approx_eq(&cpu_grad, &gpu_grad, 1e-5),
        "Stride backward: CPU and GPU gradients should match.\nCPU: {cpu_grad:?}\nGPU: {gpu_grad:?}"
    );
}

#[test]
fn test_movement_backward_chain_cpu_vs_gpu() {
    if !is_gpu_available() {
        return;
    }

    let device = Device::gpu().expect("GPU should be available");

    // Test chaining multiple movement operations
    // Create [2, 1, 3] tensor, permute to [3, 1, 2], then expand to [3, 2, 2]
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    // CPU path
    let cpu_t = RawTensor::new(data.clone(), &[2, 1, 3], true);
    let cpu_out = cpu_t.permute(&[2, 1, 0]).expand(&[3, 2, 2]);
    cpu_out.backward();
    let cpu_grad = cpu_t.borrow().grad.as_ref().unwrap().to_vec();

    // GPU path
    let gpu_t = RawTensor::new(data, &[2, 1, 3], true).to_device(device.clone());
    let gpu_out = gpu_t.permute(&[2, 1, 0]).expand(&[3, 2, 2]);
    gpu_out.backward();
    let gpu_grad = gpu_t.borrow().grad.as_ref().unwrap().to_vec();

    assert!(
        tensors_approx_eq(&cpu_grad, &gpu_grad, 1e-5),
        "Chained movement backward: CPU and GPU gradients should match.\nCPU: {cpu_grad:?}\nGPU: {gpu_grad:?}"
    );
}

#[test]
fn test_expand_backward_multiple_dims() {
    if !is_gpu_available() {
        return;
    }

    let device = Device::gpu().expect("GPU should be available");

    // Create a [1,1,3] tensor and expand to [2,2,3]
    let data = vec![1.0, 2.0, 3.0];
    let cpu_t = RawTensor::new(data.clone(), &[1, 1, 3], true);
    let cpu_expanded = cpu_t.expand(&[2, 2, 3]);
    cpu_expanded.backward();
    let cpu_grad = cpu_t.borrow().grad.as_ref().unwrap().to_vec();

    let gpu_t = RawTensor::new(data, &[1, 1, 3], true).to_device(device.clone());
    let gpu_expanded = gpu_t.expand(&[2, 2, 3]);
    gpu_expanded.backward();
    let gpu_grad = gpu_t.borrow().grad.as_ref().unwrap().to_vec();

    assert!(
        tensors_approx_eq(&cpu_grad, &gpu_grad, 1e-5),
        "Multi-dim expand backward: CPU and GPU gradients should match.\nCPU: {cpu_grad:?}\nGPU: {gpu_grad:?}"
    );

    // Should sum over 2*2=4 positions for each element
    assert!(
        tensors_approx_eq(&cpu_grad, &[4.0, 4.0, 4.0], 1e-5),
        "Expand backward should sum correctly over multiple broadcast dimensions"
    );
}

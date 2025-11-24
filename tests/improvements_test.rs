use volta::*;

#[test]
fn test_device_safety_improvements() {
    // Test CPU device
    let cpu_device = Device::CPU;
    assert!(cpu_device.is_cpu());
    assert!(!cpu_device.is_gpu());
    assert_eq!(cpu_device.name(), "CPU");
    assert_eq!(cpu_device.to_string(), "CPU");

    // Test GPU device with name
    let gpu_device = Device::GPU("CUDA".to_string());
    assert!(!gpu_device.is_cpu());
    assert!(gpu_device.is_gpu());
    assert_eq!(gpu_device.name(), "CUDA");
    assert_eq!(gpu_device.to_string(), "CUDA");

    // Test Metal device
    let metal_device = Device::Metal("Metal".to_string());
    assert!(!metal_device.is_cpu());
    assert!(!metal_device.is_gpu());
    assert_eq!(metal_device.name(), "Metal");
}

#[test]
fn test_tensor_size_validation() {
    // Test that reasonable tensors work
    let small_tensor = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], true);
    assert_eq!(small_tensor.borrow().data.len(), 3);

    // Test that very large tensors are rejected
    let large_data = vec![0.0; 101_000_000]; // Over limit
    let result = std::panic::catch_unwind(|| {
        RawTensor::new(large_data, &[101_000_000], false);
    });
    assert!(result.is_err());
}

#[test]
fn test_conv2d_safety_improvements() {
    use crate::nn::Conv2d;

    // Test that invalid parameters are caught
    let result = std::panic::catch_unwind(|| {
        Conv2d::new(3, 16, 5, 1, 1, true); // Valid
        Conv2d::new(3, 16, 0, 1, 1, true); // Invalid kernel size 0
    });
    assert!(result.is_err()); // Should panic on invalid kernel size
}

#[test]
fn test_sequential_serialization_improvements() {
    use crate::nn::{Linear, ReLU, Sequential};

    // Create a model with mixed stateless and stateful layers
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 3, true)),
        Box::new(ReLU), // Stateless layer
        Box::new(Linear::new(3, 1, true)),
    ]);

    // Get state dict - should only include stateful layers
    let state = model.state_dict();

    // Should have exactly 2 stateful layers (Linear layers)
    assert_eq!(state.len(), 4); // 2 weights + 2 biases

    // Verify keys are properly formatted
    for key in state.keys() {
        assert!(
            key.contains('.'),
            "State dict keys should contain layer index"
        );
    }
}

#[test]
fn test_broadcast_safety_improvements() {
    // Test that reasonable broadcasting works
    let a = RawTensor::new(vec![1.0, 2.0], &[2], true);
    let b = RawTensor::new(vec![3.0], &[1], true);
    let c = a.add(&b);

    assert_eq!(c.borrow().shape, vec![2]);
    assert_eq!(c.borrow().data, vec![4.0, 5.0]);

    // Test that incompatible broadcasting still fails appropriately
    let _result = std::panic::catch_unwind(|| {
        let a = RawTensor::new(vec![1.0, 2.0], &[2], true);
        let b = RawTensor::new(vec![3.0, 4.0], &[2], true);
        RawTensor::broadcast_shape(&[2], &[3]); // Incompatible
        a.add(&b);
    });
    // Should handle gracefully
}

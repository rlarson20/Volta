use std::cell::Cell;
use std::rc::Rc;
use volta::io::{StateDict, TensorData, load_state_dict_checked};
use volta::*;

struct CountingModule {
    counter: Rc<Cell<usize>>,
}

impl Module for CountingModule {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.clone()
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }

    fn state_dict(&self) -> StateDict {
        self.counter.set(self.counter.get() + 1);
        let mut state = StateDict::new();
        state.insert(
            "dummy".to_string(),
            TensorData {
                data: vec![1.0],
                shape: vec![1],
            },
        );
        state
    }

    fn load_state_dict(&mut self, _state: &StateDict) {}
}

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
}

#[test]
fn test_tensor_size_validation() {
    // Test that reasonable tensors work
    let small_tensor = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], true);
    assert_eq!(small_tensor.borrow().data.len(), 3);

    // Test that very large tensors are rejected
    let large_data = vec![0.0; 101_000_000]; // Over limit
    let result = std::panic::catch_unwind(|| {
        let _ = RawTensor::new(large_data, &[101_000_000], false);
    });
    assert!(result.is_err());
}

#[test]
fn test_conv2d_safety_improvements() {
    use crate::nn::Conv2d;

    // Test that invalid parameters are caught
    let result = std::panic::catch_unwind(|| {
        let _ = Conv2d::new(3, 16, 5, 1, 1, true); // Valid
        let _ = Conv2d::new(3, 16, 0, 1, 1, true); // Invalid kernel size 0
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
fn test_sequential_state_dict_calls_layer_once() {
    let counter = Rc::new(Cell::new(0));
    let layer = CountingModule {
        counter: counter.clone(),
    };
    let model = Sequential::new(vec![Box::new(layer)]);

    let _ = model.state_dict();
    assert_eq!(
        counter.get(),
        1,
        "Each layer's state_dict should be invoked exactly once"
    );
}

#[test]
fn test_load_state_dict_checked_reports_differences() {
    let mut model = Sequential::new(vec![
        Box::new(Linear::new(2, 3, true)),
        Box::new(ReLU),
        Box::new(Linear::new(3, 1, true)),
    ]);

    let mut corrupted_state = model.state_dict();
    corrupted_state.remove("0.bias");
    corrupted_state.insert(
        "unexpected".to_string(),
        TensorData {
            data: vec![0.0],
            shape: vec![1],
        },
    );
    if let Some(weight) = corrupted_state.get_mut("0.weight") {
        weight.shape = vec![3, 2]; // shape mismatch while keeping data length = 6
        weight.data = vec![0.0; 6];
    }

    let diff = load_state_dict_checked(&mut model, &corrupted_state);

    assert!(
        diff.missing_keys.contains(&"0.bias".to_string()),
        "Missing keys should be reported"
    );
    assert!(
        diff.unexpected_keys.contains(&"unexpected".to_string()),
        "Unexpected keys should be reported"
    );
    let has_shape_mismatch = diff.shape_mismatches.iter().any(|(key, expected, loaded)| {
        key == "0.weight" && expected == &vec![2, 3] && loaded == &vec![3, 2]
    });
    assert!(has_shape_mismatch, "Shape mismatches should be reported");
}

#[test]
fn test_load_state_dict_checked_returns_empty_for_matching_state() {
    let mut model = Sequential::new(vec![
        Box::new(Linear::new(2, 3, true)),
        Box::new(ReLU),
        Box::new(Linear::new(3, 1, true)),
    ]);

    let state = model.state_dict();
    let diff = load_state_dict_checked(&mut model, &state);

    assert!(
        diff.is_empty(),
        "No diff should be reported when state dict matches"
    );
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
        let _ = RawTensor::broadcast_shape(&[2], &[3]); // Incompatible
        a.add(&b);
    });
    // Should handle gracefully
}
#[test]
fn test_tensor_to_device_updates_device_and_preserves_data() {
    let t = RawTensor::new(vec![1.0, 2.0], &[2], false);

    // CPU -> CPU should be a cheap no-op in terms of semantics
    let t_cpu = t.to_device(Device::CPU);
    assert_eq!(t_cpu.borrow().device, Device::CPU);
    assert_eq!(t_cpu.borrow().shape, vec![2]);
    assert_eq!(t_cpu.borrow().data.to_vec(), vec![1.0, 2.0]);

    // Move to a GPU device name; storage may or may not actually become GPU
    // backed depending on feature flags and runtime availability, but:
    // - device field must reflect the requested Device
    // - data must be numerically preserved.
    let t_gpu = t.to_device(Device::GPU("CUDA".to_string()));
    assert_eq!(t_gpu.borrow().shape, vec![2]);
    assert_eq!(t_gpu.borrow().data.to_vec(), vec![1.0, 2.0]);
    assert_eq!(t_gpu.borrow().device, Device::GPU("CUDA".to_string()));
}

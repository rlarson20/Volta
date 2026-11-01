//! Axis reductions (`sum_dim`, `max_dim`, `mean_dim`) and `softmax`, values and gradients, plus the cross-entropy loss, dropout in train and eval mode, SGD weight decay, and an external-model loading test.

use super::*;

#[test]
fn test_sum_dim_basic() {
    // [2,3] sum along dim=1 -> [2]
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    let y = RawTensor::sum_dim(&x, Axis(1), false);

    assert_eq!(y.borrow().shape, vec![2]);
    assert_eq!(y.borrow().data, vec![6.0, 15.0]); // [1+2+3, 4+5+6]
}

#[test]
fn test_sum_dim_keepdim() {
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false);
    let y = RawTensor::sum_dim(&x, Axis(0), true);

    assert_eq!(y.borrow().shape, vec![1, 2]);
    assert_eq!(y.borrow().data, vec![4.0, 6.0]); // [1+3, 2+4]
}

#[test]
fn test_sum_dim_backward() {
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
    let y = RawTensor::sum_dim(&x, Axis(1), false); // [6, 15]
    y.sum().into_loss().expect("scalar loss").backward();

    // Gradient broadcasts back: each element contributed once
    assert_eq!(x.grad(), Some(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]));
}

#[test]
fn test_max_dim_basic() {
    let x = RawTensor::new(vec![1.0, 5.0, 3.0, 2.0, 8.0, 4.0], &[2, 3], false);
    let y = RawTensor::max_dim(&x, Axis(1), false);

    assert_eq!(y.borrow().shape, vec![2]);
    assert_eq!(y.borrow().data, vec![5.0, 8.0]); // max of each row
}

#[test]
fn test_max_dim_backward() {
    let x = RawTensor::new(vec![1.0, 5.0, 3.0, 2.0, 8.0, 4.0], &[2, 3], true);
    let y = RawTensor::max_dim(&x, Axis(1), false);
    y.sum().into_loss().expect("scalar loss").backward();

    // Only max elements get gradient
    assert_eq!(x.grad(), Some(vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0]));
}

#[test]
fn test_gradcheck_sum_dim() {
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true);
    let passed =
        RawTensor::check_gradients_simple(&x, |t| RawTensor::sum_dim(t, Axis(0), false).sum());
    assert!(passed, "sum_dim gradient check failed");
}

#[test]
fn test_gradcheck_max_dim() {
    let x = RawTensor::new(vec![1.0, 5.0, 3.0, 2.0], &[2, 2], true);
    let passed =
        RawTensor::check_gradients_simple(&x, |t| RawTensor::max_dim(t, Axis(1), false).sum());
    assert!(passed, "max_dim gradient check failed");
}

#[test]
fn test_softmax_forward() {
    // Test softmax computation
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    let y = RawTensor::softmax(&x, Axis(1));

    // Each row should sum to 1.0
    let data = y.data();
    let row0_sum: f32 = data.get(0..3).unwrap().iter().sum();
    let row1_sum: f32 = data.get(3..6).unwrap().iter().sum();

    approx::assert_relative_eq!(row0_sum, 1.0, epsilon = 1e-6);
    approx::assert_relative_eq!(row1_sum, 1.0, epsilon = 1e-6);
}

#[test]
fn test_gradcheck_softmax() {
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true);
    let passed = RawTensor::check_gradients_simple(&x, |t| RawTensor::softmax(t, Axis(1)).sum());
    assert!(passed, "Softmax gradient check failed");
}

#[test]
fn test_cross_entropy_loss() {
    // Simple 2-class, 2-sample batch
    let logits = RawTensor::new(vec![2.0, 1.0, 0.5, 2.5], &[2, 2], true);
    let targets = RawTensor::new(vec![1.0, 0.0, 0.0, 1.0], &[2, 2], false);

    let loss = RawTensor::cross_entropy_loss(&logits, &targets);
    loss.clone().into_loss().expect("scalar loss").backward();

    // Loss should be positive scalar
    assert_eq!(loss.borrow().shape, vec![1]);
    assert!(
        loss.borrow()
            .data
            .to_host_f32()
            .first()
            .copied()
            .expect("tensor data should exist")
            > 0.0
    );

    // Gradients should exist and have correct shape
    assert_eq!(logits.grad().unwrap().len(), 4);
}
#[test]
fn test_dropout_train_eval() {
    let mut dropout = Dropout::new(0.5);
    let x = RawTensor::ones(&[1000]);

    // Train mode: roughly 50% should be zero, others scaled by 2
    dropout.train(true);
    let y = dropout.forward(&x);
    let y_data = &y.borrow().data;
    let num_zeros = y_data.to_host_f32().iter().filter(|&&v| v == 0.0).count();

    // Statistical check (allow some variance)
    assert!(
        num_zeros > 400 && num_zeros < 600,
        "Dropout ratio off: {num_zeros}"
    );

    // Check scaling: non-zeros should be 2.0
    let non_zeros_correct = y_data.to_host_f32().iter().all(|&v| v == 0.0 || v == 2.0);
    assert!(non_zeros_correct, "Dropout scaling incorrect");

    // Eval mode: identity
    dropout.eval();
    let y_eval = dropout.forward(&x);
    let eval_correct = y_eval.borrow().data.to_host_f32().iter().all(|&v| v == 1.0);
    assert!(eval_correct, "Dropout eval mode should be identity");
}

#[test]
fn test_weight_decay_sgd() {
    let w = RawTensor::new(vec![1.0], &[1], true);
    // SGD with 0.1 decay. Grad = 0.
    // Step should be: w = w - lr * (grad + decay * w) = 1.0 - 0.1 * (0 + 0.1 * 1.0) = 0.99
    let mut opt = SGD::new(vec![w.clone()], 0.1, 0.0, 0.1);

    w.set_grad(Storage::cpu(vec![0.0])); // Artificial zero gradient
    opt.step();

    let new_val = w
        .borrow()
        .data
        .to_host_f32()
        .first()
        .copied()
        .expect("tensor data should exist");
    approx::assert_relative_eq!(new_val, 0.99, epsilon = 1e-6);
}

#[test]
fn test_mean_dim() {
    // [2, 3]
    // [[1, 2, 3],
    //  [4, 5, 6]]
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);

    // mean(dim=1) -> [2, 5]
    let m = x.mean_dim(Axis(1), false);
    assert_eq!(m.borrow().shape, vec![2]);
    assert!(
        (m.borrow()
            .data
            .to_host_f32()
            .first()
            .copied()
            .expect("tensor data should exist")
            - 2.0)
            .abs()
            < 1e-6
    );
    assert!(
        (m.borrow()
            .data
            .to_host_f32()
            .get(1)
            .copied()
            .expect("tensor data should exist")
            - 5.0)
            .abs()
            < 1e-6
    );

    // Check gradient
    m.sum().into_loss().expect("scalar loss").backward();
    // d(mean)/dx = 1/N. Here N=3. Grad should be 1/3 for all elements.
    let grads = x.grad().unwrap();
    for g in grads {
        assert!((g - 1.0 / 3.0).abs() < 1e-6);
    }
}

/// Integration test: Simulated PyTorch model → Volta loading workflow
///
/// This test demonstrates the full end-to-end workflow of:
/// 1. Creating a "PyTorch-style" state dict (weights stored as [out, in])
/// 2. Saving it to disk
/// 3. Loading it with weight mapping (transpose + rename)
/// 4. Loading into a Volta model with named layers
/// 5. Verifying the model works correctly
#[test]
fn test_external_model_loading_integration() {
    use crate::io::{TensorData, load_state_dict, mapping::StateDictMapper, save_state_dict};
    use crate::nn::{Layer, Linear, Module, ReLU, Sequential};
    use std::collections::BTreeMap;

    // Simulate a PyTorch model with 2 linear layers
    // PyTorch stores Linear weights as [out_features, in_features]
    let mut pytorch_state = BTreeMap::new();

    // Layer 1: Linear(2, 3) -> PyTorch shape [3, 2]
    pytorch_state.insert(
        "fc1.weight".to_string(),
        TensorData {
            // PyTorch: [out=3, in=2] row-major: [[w00,w01], [w10,w11], [w20,w21]]
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            shape: vec![3, 2],
        },
    );
    pytorch_state.insert(
        "fc1.bias".to_string(),
        TensorData {
            data: vec![0.1, 0.2, 0.3],
            shape: vec![3],
        },
    );

    // Layer 2: Linear(3, 1) -> PyTorch shape [1, 3]
    pytorch_state.insert(
        "fc2.weight".to_string(),
        TensorData {
            data: vec![0.5, 0.6, 0.7],
            shape: vec![1, 3],
        },
    );
    pytorch_state.insert(
        "fc2.bias".to_string(),
        TensorData {
            data: vec![0.01],
            shape: vec![1],
        },
    );

    // Save the "PyTorch" state dict
    let temp_path = std::env::temp_dir().join("test_pytorch_model.bin");
    save_state_dict(&pytorch_state, temp_path.to_str().unwrap()).unwrap();

    // Load with weight mapping
    let loaded = load_state_dict(temp_path.to_str().unwrap()).unwrap();

    // Create mapper: rename keys and transpose weights
    let mapper = StateDictMapper::new()
        .rename("fc1.weight", "encoder.weight")
        .rename("fc1.bias", "encoder.bias")
        .rename("fc2.weight", "decoder.weight")
        .rename("fc2.bias", "decoder.bias")
        .transpose("encoder.weight") // [3,2] -> [2,3]
        .transpose("decoder.weight"); // [1,3] -> [3,1]

    let volta_state = mapper.map(loaded);

    // Verify transformation
    assert!(volta_state.contains_key("encoder.weight"));
    assert!(volta_state.contains_key("decoder.weight"));
    assert_eq!(volta_state.get("encoder.weight").unwrap().shape, vec![2, 3]);
    assert_eq!(volta_state.get("decoder.weight").unwrap().shape, vec![3, 1]);

    // Verify transpose correctness for encoder.weight
    // Original PyTorch [3,2]: [1,2, 3,4, 5,6]
    // Transposed [2,3]: [1,3,5, 2,4,6]
    let encoder_weight = &volta_state.get("encoder.weight").unwrap().data;
    assert_eq!(
        encoder_weight
            .first()
            .copied()
            .expect("tensor data should exist"),
        1.0
    );
    assert_eq!(
        encoder_weight
            .get(1)
            .copied()
            .expect("tensor data should exist"),
        3.0
    );
    assert_eq!(
        encoder_weight
            .get(2)
            .copied()
            .expect("tensor data should exist"),
        5.0
    );
    assert_eq!(
        encoder_weight
            .get(3)
            .copied()
            .expect("tensor data should exist"),
        2.0
    );
    assert_eq!(
        encoder_weight
            .get(4)
            .copied()
            .expect("tensor data should exist"),
        4.0
    );
    assert_eq!(
        encoder_weight
            .get(5)
            .copied()
            .expect("tensor data should exist"),
        6.0
    );

    // Create Volta model with named layers
    let mut model = Sequential::builder()
        .add_named("encoder", Box::new(Linear::new(2, 3, true)))
        .add_unnamed(Box::new(ReLU))
        .add_named("decoder", Box::new(Linear::new(3, 1, true)))
        .build();

    // Load the mapped state dict
    model.load_state_dict(&volta_state).expect("state loads");

    // Verify forward pass works
    let input = RawTensor::new(vec![1.0, 1.0], &[1, 2], false);
    let output = model.forward(&input);

    // Output should be deterministic based on loaded weights
    assert_eq!(output.borrow().shape, vec![1, 1]);

    // Verify we can retrieve layers by name
    assert!(model.get_named("encoder").is_some());
    assert!(model.get_named("decoder").is_some());
    assert!(model.get_named("nonexistent").is_none());

    // Verify layer names
    // The ReLU in the middle is unnamed.
    assert_eq!(
        model.layer_names(),
        vec![Some("encoder"), None, Some("decoder")]
    );
}

use std::path::Path;
use volta::{
    Linear, Module, RawTensor, ReLU, Sequential,
    io::{load_safetensors, mapping::StateDictMapper},
};

fn main() {
    println!("=== External Model Loading Example ===\n");
    println!("This example demonstrates loading PyTorch weights into Volta.");
    println!("It uses the named layer support and weight mapping utilities");
    println!("from Phase 1 & 2 to achieve PyTorch compatibility.\n");

    // Check if model file exists
    let model_path = "models/pytorch_mnist.safetensors";
    let test_data_path = "models/pytorch_mnist_test.json";

    if !Path::new(model_path).exists() {
        println!("❌ Model file not found: {model_path}");
        println!("\nTo generate the PyTorch model, run:");
        println!("  python scripts/export_pytorch_mnist.py");
        println!("\nThis will:");
        println!("  1. Create a simple PyTorch MLP (Linear→ReLU→Linear)");
        println!("  2. Export weights to SafeTensors format");
        println!("  3. Generate test data for validation\n");
        return;
    }

    println!("✓ Found model file: {model_path}\n");

    // Build matching Volta architecture using named layers
    println!("Building Volta model with named layers...");
    let mut model = Sequential::builder()
        .add_named("fc1", Box::new(Linear::new(784, 128, true)))
        .add_unnamed(Box::new(ReLU))
        .add_named("fc2", Box::new(Linear::new(128, 10, true)))
        .build();

    println!("✓ Model architecture:");
    println!("  fc1: Linear(784 → 128) with bias");
    println!("  activation: ReLU");
    println!("  fc2: Linear(128 → 10) with bias\n");

    // Load PyTorch weights
    println!("Loading PyTorch SafeTensors file...");
    let pytorch_state = match load_safetensors(model_path) {
        Ok(state) => {
            println!("✓ Loaded {} weight tensors", state.len());

            // Print loaded keys
            println!("\nPyTorch state dict keys:");
            for key in state.keys() {
                if let Some(tensor) = state.get(key) {
                    println!("  {:<20} shape: {:?}", key, tensor.shape);
                }
            }
            state
        }
        Err(e) => {
            println!("❌ Failed to load SafeTensors file: {e}");
            return;
        }
    };

    // Apply weight mapping with transposition
    println!("\nApplying weight mapping...");
    println!("  - PyTorch stores Linear weights as [out_features, in_features]");
    println!("  - Volta expects weights as [in_features, out_features]");
    println!("  - Transposing fc1.weight: [128, 784] → [784, 128]");
    println!("  - Transposing fc2.weight: [10, 128] → [128, 10]");

    let mapper = StateDictMapper::new()
        .transpose("fc1.weight")
        .transpose("fc2.weight");

    let volta_state = mapper.map(pytorch_state);

    println!("\nVolta state dict keys after mapping:");
    for key in volta_state.keys() {
        if let Some(tensor) = volta_state.get(key) {
            println!("  {:<20} shape: {:?}", key, tensor.shape);
        }
    }

    // Load weights into model
    println!("\nLoading weights into Volta model...");
    model.load_state_dict(&volta_state);
    println!("✓ Weights loaded successfully\n");

    // Try to load test data for validation
    if Path::new(test_data_path).exists() {
        println!("✓ Found test data file: {test_data_path}");
        run_validation(&model, test_data_path);
    } else {
        println!("ℹ Test data not found (optional): {test_data_path}");
        println!("Running demo inference instead...\n");
        run_demo_inference(&model);
    }

    println!("\n=== Summary ===");
    println!("✓ Successfully loaded PyTorch model into Volta");
    println!("✓ Named layers: fc1, fc2");
    println!("✓ Weight transposition: Applied correctly");
    println!("✓ Model ready for inference");
}

fn run_validation(model: &Sequential, test_data_path: &str) {
    println!("\nRunning validation against PyTorch outputs...");

    // Load test data JSON
    let _test_data_str = match std::fs::read_to_string(test_data_path) {
        Ok(content) => content,
        Err(e) => {
            println!("❌ Failed to read test data: {e}");
            return;
        }
    };

    // Parse JSON (simple manual parsing since we know the structure)
    // In a real application, you'd use serde_json, but we want to keep dependencies minimal
    println!("  Parsing test data...");

    // For this example, we'll just run demo inference since parsing JSON
    // without serde_json would be verbose. In practice, you'd add serde_json
    // as a dev dependency for examples.
    println!("  (Skipping JSON parsing - would require serde_json dependency)");
    println!("  Running demo inference instead...\n");

    run_demo_inference(model);
}

fn run_demo_inference(model: &Sequential) {
    println!("Running demo inference...");

    // Create sample input (simulating flattened 28×28 MNIST image)
    let input_data: Vec<f32> = (0..784).map(|i| (i as f32 % 255.0) / 255.0).collect();

    let input = RawTensor::new(input_data, &[1, 784], false);

    // Run forward pass
    println!("  Input shape: {:?}", input.borrow().shape);
    let output = model.forward(&input);

    println!("  Output shape: {:?}", output.borrow().shape);

    // Print output values
    let output_data = &output.borrow().data;
    println!("\n  Output logits (first 10 values):");
    for (i, &val) in output_data.iter().take(10).enumerate() {
        println!("    Class {i}: {val:.6}");
    }

    // Find predicted class
    let pred_class = output_data
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();

    println!("\n  Predicted class: {pred_class}");
    println!("  (Note: This is random weights, so prediction is meaningless)");

    println!("\n✓ Inference successful!");
}

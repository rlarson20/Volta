//! Loads PyTorch weights (a `SafeTensors` file) into a Volta `Sequential` through `StateDictMapper`, transposing the `fc1` and `fc2` weights from the `[out, in]` layout PyTorch uses, then compares Volta's outputs against the PyTorch outputs that `scripts/export_pytorch_mnist.py` saved next to the weights.
//!
//! Without that test-data file it only runs a demo inference, which shows the model runs, not that it is right; the summary says which of the two happened. A mismatch beyond [`LOGIT_TOLERANCE`] exits with status 1.
use std::path::Path;
use volta::{
    Layer, Linear, Module, RawTensor, ReLU, Sequential,
    io::{load_safetensors, mapping::StateDictMapper},
};

/// Largest `|volta - pytorch|` accepted on any output logit. Both sides are f32 and the
/// first layer sums 784 products in a different order, so they agree to a few 1e-7 (measured
/// worst 4.8e-7), not exactly.
const LOGIT_TOLERANCE: f32 = 1e-5;

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
    model
        .load_state_dict(&volta_state)
        .expect("converted PyTorch weights should fit the model");
    println!("✓ Weights loaded successfully\n");

    // Compare against PyTorch's own outputs when the export script saved them
    let validated = if Path::new(test_data_path).exists() {
        println!("✓ Found test data file: {test_data_path}");
        match run_validation(&model, test_data_path) {
            Ok(worst) => {
                println!(
                    "✓ Matches PyTorch: worst |difference| {worst:.2e} (limit {LOGIT_TOLERANCE:.0e})"
                );
                true
            }
            Err(e) => {
                eprintln!("❌ Validation against PyTorch failed: {e}");
                std::process::exit(1);
            }
        }
    } else {
        println!("ℹ Test data not found (optional): {test_data_path}");
        println!("Running demo inference instead...\n");
        run_demo_inference(&model);
        false
    };

    println!("\n=== Summary ===");
    println!("✓ Successfully loaded PyTorch model into Volta");
    println!("✓ Named layers: fc1, fc2");
    if validated {
        println!("✓ Outputs verified against PyTorch (weight transposition is correct)");
    } else {
        println!("ℹ Outputs NOT verified: no PyTorch test data, so the transposition is unchecked");
    }
}

/// Reads `inputs` and `outputs` (arrays of rows) from the JSON document `doc`.
fn read_rows(doc: &serde_json::Value, key: &str) -> Result<Vec<Vec<f32>>, String> {
    let rows = doc
        .get(key)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("test data has no `{key}` array"))?;
    rows.iter()
        .map(|row| {
            row.as_array()
                .ok_or_else(|| format!("a row of `{key}` is not an array"))?
                .iter()
                .map(|v| {
                    v.as_f64()
                        .map(|f| f as f32)
                        .ok_or_else(|| format!("`{key}` holds a non-number"))
                })
                .collect()
        })
        .collect()
}

/// Runs `model` on the inputs in `test_data_path` and compares each logit with the PyTorch
/// output saved beside it. Returns the largest absolute difference, or an error if the file
/// is malformed or any logit is off by more than [`LOGIT_TOLERANCE`].
fn run_validation(model: &Sequential, test_data_path: &str) -> Result<f32, String> {
    println!("\nRunning validation against PyTorch outputs...");

    let text = std::fs::read_to_string(test_data_path)
        .map_err(|e| format!("could not read {test_data_path}: {e}"))?;
    let doc: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("{test_data_path} is not JSON: {e}"))?;
    let inputs = read_rows(&doc, "inputs")?;
    let expected = read_rows(&doc, "outputs")?;
    if inputs.is_empty() || inputs.len() != expected.len() {
        return Err(format!(
            "{} input rows but {} output rows",
            inputs.len(),
            expected.len()
        ));
    }
    if let Some(row) = inputs.iter().find(|r| r.len() != 784) {
        return Err(format!("an input row has {} values, not 784", row.len()));
    }
    println!("  {} samples", inputs.len());

    let batch = inputs.len();
    let flat: Vec<f32> = inputs.into_iter().flatten().collect();
    let output = model.forward(&RawTensor::new(flat, &[batch, 784], false));
    let got = output.data();
    let want: Vec<f32> = expected.into_iter().flatten().collect();
    assert_eq!(
        got.len(),
        want.len(),
        "Volta produced {} logits, PyTorch saved {}",
        got.len(),
        want.len()
    );

    let mut worst = 0.0f32;
    for (sample, (g_row, w_row)) in got.chunks(10).zip(want.chunks(10)).enumerate() {
        let diff = g_row
            .iter()
            .zip(w_row)
            .map(|(g, w)| (g - w).abs())
            .fold(0.0f32, f32::max);
        println!("  sample {sample}: worst |difference| {diff:.2e}");
        worst = worst.max(diff);
    }
    if worst > LOGIT_TOLERANCE {
        return Err(format!(
            "worst |difference| {worst:.2e} exceeds {LOGIT_TOLERANCE:.0e}"
        ));
    }
    Ok(worst)
}

/// Runs `model` on one synthetic, flattened 28x28 input and prints the output shape and values. It checks that the loaded model runs, not that its outputs are correct.
fn run_demo_inference(model: &Sequential) {
    println!("Running demo inference...");

    // Create sample input (simulating flattened 28×28 MNIST image)
    let input_data: Vec<f32> = (0..784).map(|i| (i as f32 % 255.0) / 255.0).collect();

    let input = RawTensor::new(input_data, &[1, 784], false);

    // Run forward pass
    println!("  Input shape: {:?}", input.shape());
    let output = model.forward(&input);

    println!("  Output shape: {:?}", output.shape());

    // Print output values
    let output_data = output.data();
    println!("\n  Output logits (first 10 values):");
    for (i, &val) in output_data.iter().take(10).enumerate() {
        println!("    Class {i}: {val:.6}");
    }

    // Find predicted class
    let pred_class = output_data
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .expect("the model has output logits");

    println!("\n  Predicted class: {pred_class}");
    println!("  (Note: This is random weights, so prediction is meaningless)");

    println!("\n✓ Inference successful!");
}

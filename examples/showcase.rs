//! # Volta Showcase: Training an MLP for the XOR Problem
//!
//! This example demonstrates the core features of the Volta ML framework:
//! - **Tensor Creation**: Creating tensors for data and targets.
//! - **Model Building**: Using `Sequential`, `Linear`, and `ReLU` to build a multi-layer perceptron.
//! - **Optimizers**: Using the `Adam` optimizer to manage model parameters.
//! - **Training Loop**: A complete loop including forward pass, loss calculation (MSE),
//!   backward pass (autograd), and optimizer step.
//! - **Inference**: Using the trained model to make predictions.
//! - **Persistence**: Saving and loading the weights of a trained layer.

use std::error::Error;
use volta::{Adam, Linear, Module, RawTensor, ReLU, Sequential, Tensor, TensorOps, mse_loss};

fn main() -> Result<(), Box<dyn Error>> {
    println!("--- Volta Showcase: Solving XOR with an MLP ---");

    // 1. Define the XOR dataset
    // Inputs: [[0,0], [0,1], [1,0], [1,1]]
    let x_data: Tensor =
        RawTensor::new(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], &[4, 2], false);
    // Targets: [[0], [1], [1], [0]]
    let y_data: Tensor = RawTensor::new(vec![0.0, 1.0, 1.0, 0.0], &[4, 1], false);

    // 2. Build the Model: a simple Multi-Layer Perceptron (MLP)
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 8, true)),
        Box::new(ReLU),
        Box::new(Linear::new(8, 8, true)),
        Box::new(ReLU),
        Box::new(Linear::new(8, 1, true)),
    ]);
    println!("\nModel Architecture: 2 -> Linear(8) -> ReLU -> Linear(8) -> ReLU -> Linear(1)");

    // 3. Set up the Optimizer
    // We'll use Adam, a popular and effective optimizer.
    let mut opt = Adam::new(model.parameters(), 0.05, (0.9, 0.999), 1e-8, 0.0);
    let epochs = 201;

    // 4. The Training Loop
    println!("\n--- Starting Training ---");
    for epoch in 0..epochs {
        // Zero out gradients from the previous iteration
        opt.zero_grad();

        // Forward pass: get model predictions
        let predictions = model.forward(&x_data);

        // Calculate loss (Mean Squared Error)
        let loss = mse_loss(&predictions, &y_data);

        // Backward pass: compute gradients for all parameters
        loss.backward();

        // Update model weights using the computed gradients
        opt.step();

        if epoch % 20 == 0 {
            println!("Epoch {:>3}: Loss = {:.6}", epoch, loss.borrow().data[0]);
        }
    }
    println!("--- Training Complete ---");

    // 5. Inference after Training
    // Let's see how well the model learned the XOR function.
    println!("\n--- Inference After Training ---");
    let final_predictions = model.forward(&x_data);
    for i in 0..4 {
        let input_slice = &x_data.borrow().data[i * 2..(i * 2) + 2];
        println!(
            "Input: [{:.0}, {:.0}] -> Target: {:.0}, Predicted: {:.4}",
            input_slice[0],
            input_slice[1],
            y_data.borrow().data[i],
            final_predictions.borrow().data[i]
        );
    }

    // // 6. Showcase Model Persistence
    // println!("\n--- Showcasing Model Persistence (Save/Load) ---");
    //
    // // Get a reference to the first trained linear layer
    // // We need to downcast from the `dyn Module` trait object.
    // let first_layer = model
    //     .parameters()
    //     .into_iter()
    //     .find(|p| p.borrow().shape == vec![2, 8]) // Find the 2x8 weight tensor
    //     .and_then(|t| {
    //         // This is a bit complex as we need to find the Module that owns the tensor.
    //         // A real app might have better ways to access layers by name.
    //         model
    //             .as_any()
    //             .downcast_ref::<Sequential>()
    //             .and_then(|seq| seq.layers[0].as_any().downcast_ref::<Linear>())
    //     })
    //     .ok_or("Could not find first Linear layer")?;
    //
    // // Save its state to a file
    // let save_path = "xor_layer1.bin";
    // first_layer.save(save_path)?;
    // println!(
    //     "Saved trained weights of the first layer to '{}'",
    //     save_path
    // );
    //
    // // Create a NEW, completely untrained model
    // let mut fresh_model = Sequential::new(vec![
    //     Box::new(Linear::new(2, 8, true)),
    //     Box::new(ReLU),
    //     Box::new(Linear::new(8, 1, true)),
    // ]);
    //
    // // Load the saved weights into the first layer of the new model
    // let first_layer_mut = fresh_model
    //     .as_any()
    //     .downcast_ref::<Sequential>()
    //     .and_then(|seq| seq.layers[0].as_any().downcast_ref::<Linear>())
    //     .ok_or("Could not get mutable ref to new layer")?;
    //
    // // Create a mutable copy to load into
    // let mut mutable_layer_instance = Linear::new(2, 8, true);
    // mutable_layer_instance.load(save_path)?;
    //
    // // NOTE: A direct mutable downcast isn't straightforward here.
    // // For this showcase, we'll just show the load works and that the weights are different.
    // println!(
    //     "Successfully loaded weights from '{}' into a new layer instance.",
    //     save_path
    // );
    //
    // // Verify weights are the same.
    // let original_weights = &first_layer.weight.borrow().data;
    // let loaded_weights = &mutable_layer_instance.weight.borrow().data;
    // assert_eq!(original_weights.len(), loaded_weights.len());
    //
    // let difference: f32 = original_weights
    //     .iter()
    //     .zip(loaded_weights.iter())
    //     .map(|(a, b)| (a - b).abs())
    //     .sum();
    //
    // if difference < 1e-6 {
    //     println!("Verification successful: Loaded weights match the saved weights.");
    // } else {
    //     println!("Verification FAILED: Weights do not match.");
    // }

    println!("\nShowcase finished successfully!");
    Ok(())
}

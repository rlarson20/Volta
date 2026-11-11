//! The first example in the README
use volta::{
    Adam, Sequential, io,
    nn::{Layer, Linear, Module, ReLU},
    tensor::{RawTensor, mse_loss},
};

fn main() {
    // 1. Define a simple model with named layers: 2 -> 8 -> 1
    let model = Sequential::builder()
        .add_named("fc1", Box::new(Linear::new(2, 8, true)))
        .add_unnamed(Box::new(ReLU))
        .add_named("fc2", Box::new(Linear::new(8, 1, true)))
        .build();

    // 2. Create synthetic data
    let x_data = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let x = RawTensor::new(x_data, &[4, 2], false); // Batch size 4, 2 features

    let y_data = vec![0.0, 1.0, 1.0, 0.0];
    let y = RawTensor::new(y_data, &[4], false); // Flattened targets

    // 3. Set up the optimizer
    let params = model.parameters();
    let mut optimizer = Adam::new(params, 0.1, (0.9, 0.999), 1e-8, 0.0);

    // 4. Training loop
    println!("Training a simple MLP to learn XOR...");
    let mut first_loss = f32::NAN;
    let mut final_loss = f32::NAN;
    for epoch in 0..=300 {
        optimizer.zero_grad();

        let pred = model.forward(&x).reshape(&[4]); //alignment
        let loss = mse_loss(&pred, &y);
        final_loss = loss.item();
        if epoch == 0 {
            first_loss = final_loss;
        }

        if epoch % 20 == 0 {
            println!("Epoch {epoch}: loss = {final_loss:.6}");
        }

        loss.into_loss().expect("scalar loss").backward();
        optimizer.step();
    }
    // The CI smoke run (`just examples-smoke`) relies on this: a NaN or a loss that never
    // drops exits non-zero.
    assert!(
        final_loss.is_finite() && final_loss < first_loss,
        "training did not reduce the loss: first {first_loss}, final {final_loss}"
    );

    // 5. Save and Load State Dict (human-readable keys: "fc1.weight", "fc1.bias", etc.)
    let state = model.state_dict();
    io::save_state_dict(&state, "model.bin").expect("Failed to save");

    // Verify loading
    let mut new_model = Sequential::builder()
        .add_named("fc1", Box::new(Linear::new(2, 8, true)))
        .add_unnamed(Box::new(ReLU))
        .add_named("fc2", Box::new(Linear::new(8, 1, true)))
        .build();
    let loaded_state = io::load_state_dict("model.bin").expect("Failed to load");
    new_model
        .load_state_dict(&loaded_state)
        .expect("Failed to load state");
}

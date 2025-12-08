use volta::{Adam, Sequential, TensorOps, io, nn::*, tensor::*};

fn main() {
    // 1. Define a simple model: 2 -> 8 -> 1
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 8, true)),
        Box::new(ReLU),
        Box::new(Linear::new(8, 1, true)),
    ]);

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
    for epoch in 0..=300 {
        optimizer.zero_grad();

        let pred = model.forward(&x).reshape(&[4]); //alignment
        let loss = mse_loss(&pred, &y);

        if epoch % 20 == 0 {
            println!("Epoch {}: loss = {:.6}", epoch, loss.borrow().data[0]);
        }

        loss.backward();
        optimizer.step();
    }
    // 5. Save and Load State Dict
    let state = model.state_dict();
    io::save_state_dict(&state, "model.bin").expect("Failed to save");

    // Verify loading
    let mut new_model = Sequential::new(vec![
        Box::new(Linear::new(2, 8, true)),
        Box::new(ReLU),
        Box::new(Linear::new(8, 1, true)),
    ]);
    let loaded_state = io::load_state_dict("model.bin").expect("Failed to load");
    new_model.load_state_dict(&loaded_state);
}

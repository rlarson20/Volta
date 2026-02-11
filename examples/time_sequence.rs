use volta::{LSTMCell, Module, ProgressBar, RawTensor, SGD, TensorOps};

fn main() {
    println!("=== Time Sequence Prediction with LSTM ===\n");

    // Generate sine wave data
    let sequence_length = 50;
    let num_sequences = 200;
    let mut all_inputs = Vec::new();
    let mut all_targets = Vec::new();

    for _ in 0..num_sequences {
        let start = rand::random::<f32>() * 10.0;
        for t in 0..sequence_length {
            let x = start + (t as f32) * 0.1;
            let input = (x).sin();
            let target = (x + 0.1).sin(); // Predict next value

            all_inputs.push(input);
            all_targets.push(target);
        }
    }

    println!("Generated {num_sequences} sequences of length {sequence_length}");
    println!("Task: Predict next value in sine wave\n");

    // Build model: LSTM with 1 input, 32 hidden units, 1 output
    let lstm = LSTMCell::new(1, 32, true);
    let output_layer = volta::Linear::new(32, 1, true);

    let mut lstm_params = lstm.parameters();
    lstm_params.extend(output_layer.parameters());

    let mut optimizer = SGD::new(lstm_params, 0.01, 0.9, 0.0);

    println!("Architecture:");
    println!("  LSTM(1 -> 32 hidden units)");
    println!("  Linear(32 -> 1 output)");
    println!("\nStarting training...\n");

    // Training loop
    let num_epochs = 50;
    let mut progress = ProgressBar::new(num_epochs, "Training");

    for epoch in 0..num_epochs {
        optimizer.zero_grad();
        let mut total_loss = 0.0;

        // Process each sequence
        for seq_idx in 0..num_sequences {
            let mut h_state = None;

            // Process each timestep in the sequence
            for t in 0..sequence_length {
                let idx = seq_idx * sequence_length + t;
                let input_val = all_inputs.get(idx).copied().unwrap_or(f32::NAN);
                let target_val = all_targets.get(idx).copied().unwrap_or(f32::NAN);

                // Create input tensor [1, 1] (batch=1, features=1)
                let input = RawTensor::new(vec![input_val], &[1, 1], false);

                // LSTM forward
                let (h_new, c_new) = if let Some((ref h, ref c)) = h_state {
                    lstm.forward_step(&input, Some((h, c)))
                } else {
                    lstm.forward_step(&input, None)
                };

                // Output layer
                let pred = output_layer.forward(&h_new);

                // Loss
                let target = RawTensor::new(vec![target_val], &[1, 1], false);
                let loss = RawTensor::mse_loss(&pred, &target);

                total_loss += loss.borrow().data.first().copied().unwrap_or(f32::NAN);
                loss.backward();

                // Update hidden state for next timestep
                h_state = Some((h_new, c_new));
            }
        }

        optimizer.step();
        progress.update(epoch + 1);

        if epoch % 10 == 9 {
            let avg_loss = total_loss / (num_sequences * sequence_length) as f32;
            println!(" Loss = {avg_loss:.6}");
        }
    }

    println!("\n=== Training Complete ===");

    // Test prediction
    println!("\nTesting on new sequence:");
    let test_start = 5.0;
    let test_length = 20;

    let mut h_state = None;
    for t in 0..test_length {
        let x = test_start + (t as f32) * 0.1;
        let true_current = x.sin();
        let true_next = (x + 0.1).sin();

        let input = RawTensor::new(vec![true_current], &[1, 1], false);

        let (h_new, c_new) = if let Some((ref h, ref c)) = h_state {
            lstm.forward_step(&input, Some((h, c)))
        } else {
            lstm.forward_step(&input, None)
        };

        let pred = output_layer.forward(&h_new);
        let pred_val = pred.borrow().data.first().copied().unwrap_or(f32::NAN);

        if t % 5 == 0 {
            println!(
                "  t={:2}: current={:6.3}, predicted_next={:6.3}, true_next={:6.3}, error={:6.3}",
                t,
                true_current,
                pred_val,
                true_next,
                (pred_val - true_next).abs()
            );
        }

        h_state = Some((h_new, c_new));
    }
}

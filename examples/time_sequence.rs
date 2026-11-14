//! Time-sequence prediction: an `LSTMCell` (1 input, 32 hidden units) followed by a `Linear` output layer, trained with `SGD` with momentum on sine-wave sequences, then tested on a held-out prediction.
use volta::{LSTMCell, Module, ProgressBar, RawTensor, RecurrentLayer, SGD};

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

    // One optimizer step per sequence, on the mean of its per-timestep losses. Summing
    // the gradients of all 10,000 timestep losses into one step per epoch (what this
    // example used to do) makes the step about 10,000 times too large, and training
    // diverges to inf and then NaN within a few epochs (#88).
    let mut first_loss = None;
    let mut last_loss = f32::NAN;
    for epoch in 0..num_epochs {
        let mut total_loss = 0.0;

        for seq_idx in 0..num_sequences {
            optimizer.zero_grad();
            let mut state = lstm.initial_state(1);
            let mut seq_loss: Option<volta::Tensor> = None;

            // Process each timestep in the sequence
            for t in 0..sequence_length {
                let idx = seq_idx * sequence_length + t;
                let input_val = all_inputs.get(idx).copied().expect("index in range");
                let target_val = all_targets.get(idx).copied().expect("index in range");

                // Create input tensor [1, 1] (batch=1, features=1)
                let input = RawTensor::new(vec![input_val], &[1, 1], false);

                // LSTM forward, then the output layer
                let h_new = lstm.step(&input, &mut state);
                let pred = output_layer.forward(&h_new);

                let target = RawTensor::new(vec![target_val], &[1, 1], false);
                let loss = RawTensor::mse_loss(&pred, &target);
                seq_loss = Some(match seq_loss {
                    Some(acc) => acc.add(&loss),
                    None => loss,
                });
            }

            // Backpropagate through the whole sequence once, on the mean step loss.
            let mean = RawTensor::new(vec![1.0 / sequence_length as f32], &[1], false);
            let seq_loss = seq_loss.expect("sequence_length > 0").elem_mul(&mean);
            total_loss += seq_loss.item();
            seq_loss.into_loss().expect("scalar loss").backward();
            optimizer.step();
        }

        progress.update(epoch + 1);

        let avg_loss = total_loss / num_sequences as f32;
        assert!(
            avg_loss.is_finite(),
            "epoch {epoch}: loss diverged to {avg_loss}"
        );
        first_loss.get_or_insert(avg_loss);
        last_loss = avg_loss;
        if epoch % 10 == 9 {
            println!(" Loss = {avg_loss:.6}");
        }
    }
    let first_loss = first_loss.expect("num_epochs > 0");
    assert!(
        last_loss < first_loss,
        "final loss {last_loss} is not below the first epoch's {first_loss}"
    );

    println!("\n=== Training Complete ===");

    // Test prediction
    println!("\nTesting on new sequence:");
    let test_start = 5.0;
    let test_length = 20;

    let mut state = lstm.initial_state(1);
    for t in 0..test_length {
        let x = test_start + (t as f32) * 0.1;
        let true_current = x.sin();
        let true_next = (x + 0.1).sin();

        let input = RawTensor::new(vec![true_current], &[1, 1], false);

        let h_new = lstm.step(&input, &mut state);

        let pred = output_layer.forward(&h_new);
        let pred_val = pred.item();

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
    }
}

use volta::{Linear, Module, ProgressBar, RawTensor, SGD, Sequential, TensorOps};

fn main() {
    println!("=== Polynomial Regression Example ===\n");

    // Generate synthetic data: y = -x^3 + x (polynomial curve)
    // Create 100 training points from -2 to 2
    let num_samples = 100;
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();

    for i in 0..num_samples {
        let x = -2.0 + 4.0 * (i as f32 / num_samples as f32);
        let y = -x.powi(3) + x; // True polynomial
        x_data.push(x);
        y_data.push(y);
    }

    // Create tensors
    let x = RawTensor::new(x_data.clone(), &[num_samples, 1], false);
    let y_true = RawTensor::new(y_data.clone(), &[num_samples, 1], false);

    // Create a simple 2-layer network to approximate the polynomial
    // Input(1) -> Hidden(32) -> Output(1)
    let model = Sequential::new(vec![
        Box::new(Linear::new(1, 32, true)),
        Box::new(volta::ReLU),
        Box::new(Linear::new(32, 32, true)),
        Box::new(volta::ReLU),
        Box::new(Linear::new(32, 1, true)),
    ]);

    let mut optimizer = SGD::new(model.parameters(), 0.01, 0.9, 0.0);

    println!("Training polynomial regression...");
    println!("Architecture: Linear(1->32) -> ReLU -> Linear(32->32) -> ReLU -> Linear(32->1)");
    println!("Target function: y = -x^3 + x\n");

    // Training loop
    let num_epochs = 1000;
    let mut progress = ProgressBar::new(num_epochs, "Training");

    for epoch in 0..num_epochs {
        optimizer.zero_grad();

        let pred = model.forward(&x);
        let loss = RawTensor::mse_loss(&pred, &y_true);

        loss.backward();
        optimizer.step();

        progress.update(epoch + 1);

        if epoch % 100 == 99 {
            let loss_val = loss.borrow().data.first().copied().unwrap_or(f32::NAN);
            println!(" Loss = {loss_val:.6}");
        }
    }

    // Final evaluation
    let final_pred = model.forward(&x);
    let final_loss = RawTensor::mse_loss(&final_pred, &y_true);
    println!("\n=== Training Complete ===");
    println!(
        "Final Loss: {:.6}",
        final_loss
            .borrow()
            .data
            .first()
            .copied()
            .unwrap_or(f32::NAN)
    );

    // Show a few predictions
    println!("\nSample predictions:");
    let pred_data = &final_pred.borrow().data;
    for i in (0..num_samples).step_by(20) {
        println!(
            "  x={:6.2}: pred={:6.3}, true={:6.3}, error={:6.3}",
            x_data.get(i).copied().unwrap_or(f32::NAN),
            pred_data.get(i).copied().unwrap_or(f32::NAN),
            y_data.get(i).copied().unwrap_or(f32::NAN),
            (pred_data.get(i).copied().unwrap_or(f32::NAN)
                - y_data.get(i).copied().unwrap_or(f32::NAN))
            .abs()
        );
    }
}

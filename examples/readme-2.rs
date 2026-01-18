use volta::TensorOps;
use volta::nn::Module;
use volta::{Adam, Conv2d, Flatten, Linear, MaxPool2d, ReLU, Sequential};

fn main() {
    // 1. Define Model
    let model = Sequential::new(vec![
        // Input: 1x28x28
        Box::new(Conv2d::new(1, 6, 5, 1, 2, true)), // Padding 2
        Box::new(ReLU),
        Box::new(MaxPool2d::new(2, 2, 0)),
        // Feature map size here: 6x14x14
        Box::new(Flatten::new()),
        Box::new(Linear::new(6 * 14 * 14, 10, true)),
    ]);

    // 2. Data & Optimizer
    let input = volta::randn(&[4, 1, 28, 28]); // Batch 4
    let target = volta::randn(&[4, 10]); // Dummy targets
    let params = model.parameters();
    let mut optim = Adam::new(params, 1e-3, (0.9, 0.999), 1e-8, 0.0);

    // 3. Training Step
    optim.zero_grad();
    let output = model.forward(&input);
    let loss = volta::mse_loss(&output, &target);
    loss.backward();
    optim.step();

    println!(
        "Loss: {:?}",
        loss.borrow().data.first().copied().unwrap_or(f32::NAN)
    );
}

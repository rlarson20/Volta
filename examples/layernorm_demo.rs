//! `LayerNorm` demonstration: a 2-D `(B, C)` input, a transformer-style 3-D `(B, T, C)` input, and a gradient-flow check that backprop reaches the input, `gamma`, and `beta`.
use volta::RawTensor;
use volta::{LayerNorm, Module, TensorOps};

#[allow(clippy::cast_precision_loss)]
fn main() {
    println!("=== LayerNorm Demonstration ===\n");

    // Example 1: Basic 2D input (B, C)
    println!("Example 1: Basic 2D input (B, C)");
    let ln = LayerNorm::new(vec![4]);
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4], false);
    println!("Input shape: {:?}", x.shape());
    println!("Input data:\n{:?}", x.data());

    let y = ln.forward(&x);
    println!("Output shape: {:?}", y.shape());
    println!("Output data:\n{:?}", y.data());

    // Verify normalization
    let y_data = y.data();
    for i in 0..2 {
        let start = i * 4;
        let end = start + 4;
        let sample = y_data
            .get(start..end)
            .expect("Sample slice should be valid");
        let mean: f32 = sample.iter().sum::<f32>() / 4.0;
        let variance: f32 = sample.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 4.0;
        let std = variance.sqrt();
        println!("Sample {i} - Mean: {mean:.6}, Std: {std:.6}");
    }

    println!("\nExample 2: Transformer-style 3D input (B, T, C)");
    let ln2 = LayerNorm::new(vec![8]);
    let x2 = RawTensor::new(
        (0..160).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
        &[2, 10, 8],
        false,
    );
    println!("Input shape: {:?}", x2.shape());

    let y2 = ln2.forward(&x2);
    println!("Output shape: {:?}", y2.shape());

    // Check parameters
    let params = ln.parameters();
    println!(
        "\nGamma shape: {:?}",
        params.first().map(volta::Tensor::shape)
    );
    println!("Beta shape: {:?}", params.get(1).map(volta::Tensor::shape));
    println!(
        "Gamma requires_grad: {}",
        params.first().is_some_and(volta::Tensor::requires_grad)
    );
    println!(
        "Beta requires_grad: {}",
        params.get(1).is_some_and(volta::Tensor::requires_grad)
    );

    // Test gradient flow
    println!("\nExample 3: Gradient flow");
    let ln3 = LayerNorm::new(vec![4]);
    let x3 = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4], true);
    let y3 = ln3.forward(&x3);
    let loss = y3.sum();
    loss.into_loss().expect("scalar loss").backward();

    println!("Input gradient exists: {}", x3.grad().is_some());
    let params3 = ln3.parameters();
    println!(
        "Gamma gradient exists: {}",
        params3.first().and_then(volta::TensorOps::grad).is_some()
    );
    println!(
        "Beta gradient exists: {}",
        params3.get(1).and_then(volta::TensorOps::grad).is_some()
    );

    println!("\n=== All LayerNorm demonstrations completed successfully! ===");
}

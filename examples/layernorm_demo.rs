//! `LayerNorm` demonstration: a 2-D `(B, C)` input, a transformer-style 3-D `(B, T, C)` input, and a gradient-flow check that backprop through an unevenly weighted loss gives the input, `gamma`, and `beta` nonzero gradients.
use volta::RawTensor;
use volta::{Layer, LayerNorm, Module};

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

    // Test gradient flow. The loss weights the outputs unevenly: the plain sum of a
    // LayerNorm row is a constant, so `y.sum()` would give the input an all-zero gradient
    // and the check below would pass for a broken backward pass.
    println!("\nExample 3: Gradient flow");
    let ln3 = LayerNorm::new(vec![4]);
    let x3 = RawTensor::new(
        vec![1.0, 2.0, 4.0, 8.0, -1.0, 0.5, 3.0, -2.0],
        &[2, 4],
        true,
    );
    let y3 = ln3.forward(&x3);
    let weights = RawTensor::new(
        vec![0.1, -0.4, 0.7, 0.3, -0.2, 0.9, 0.5, -0.6],
        &[2, 4],
        false,
    );
    let loss = y3.elem_mul(&weights).sum();
    loss.into_loss().expect("scalar loss").backward();

    let params3 = ln3.parameters();
    let named = [
        ("Input", x3.grad()),
        ("Gamma", params3.first().and_then(volta::Tensor::grad)),
        ("Beta", params3.get(1).and_then(volta::Tensor::grad)),
    ];
    for (name, grad) in named {
        let grad = grad.unwrap_or_else(|| panic!("{name} has no gradient after backward"));
        println!("{name} gradient: {grad:?}");
        // The CI smoke run (`just examples-smoke`) relies on this: a gradient that is
        // missing or all zero exits non-zero.
        assert!(
            grad.iter().any(|g| g.abs() > 1e-3),
            "{name} gradient is all zero: {grad:?}"
        );
    }

    println!("\n=== All LayerNorm demonstrations completed successfully! ===");
}

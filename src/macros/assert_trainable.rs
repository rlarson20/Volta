/// Assert that every learnable parameter of `$layer` receives a nonzero gradient
/// after a forward + backward pass over `$input`.
///
/// This is the check that would have caught the ConvTranspose2d/LSTM/PositionalEncoding
/// bugs from Part I (ADDENDUM.md §5.3): a layer whose output is silently disconnected
/// from a parameter's autograd graph looks fine at a glance but never trains.
///
/// `$input` must be non-degenerate — use `crate::macros::trainable_input`. Constant
/// inputs make tokens/features identical, which zeroes gradients mathematically for
/// normalization and attention layers (`LayerNorm` gamma, attention Q/K projections).
///
/// One deliberate departure from the originally-sketched version of this macro:
///
/// - `forward`/`parameters` are called via UFCS (`<_ as Module>::method(..)`), not
///   `.method()`. Some layers (`MultiHeadAttention`, `TransformerBlock`) have an
///   inherent `forward` with extra params (causal masking, etc.) alongside
///   `Module::forward`. Inherent methods always win over trait methods in
///   `.method()` resolution regardless of arity, so plain `$layer.forward(&$input)`
///   would resolve to the wrong one and fail to compile for those layers.
#[macro_export]
macro_rules! assert_trainable {
    ($layer:expr, $input:expr) => {{
        #[allow(unused_imports)]
        use $crate::tensor::TensorOps as _;

        let output = <_ as $crate::nn::Module>::forward(&$layer, &$input);
        // Randomly weighted sum, not plain `.sum()`: under a plain sum, normalized
        // activations sum to zero, so e.g. BatchNorm's gamma gets an exactly-zero
        // gradient for ANY input — a false positive, not a severed graph.
        let weights = $crate::tensor::randn_like(&output);
        let loss = output.elem_mul(&weights).sum();
        loss.into_loss()
            .expect("layer output must be differentiable")
            .backward();
        for (i, p) in <_ as $crate::nn::Module>::parameters(&$layer)
            .iter()
            .enumerate()
        {
            let g = p.grad().unwrap_or_else(|| {
                panic!(
                    "parameter {} has no gradient — graph severed or gate bug",
                    i
                )
            });
            assert!(
                g.iter().any(|v| v.abs() > 0.0),
                "parameter {} gradient is identically zero",
                i
            );
        }
    }};
}

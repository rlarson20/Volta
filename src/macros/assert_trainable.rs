/// Assert that every learnable parameter of `$layer` receives a nonzero gradient
/// after a forward + backward pass over `$input`.
///
/// This is the check that would have caught the ConvTranspose2d/LSTM/PositionalEncoding
/// bugs from Part I (ADDENDUM.md §5.3): a layer whose output is silently disconnected
/// from a parameter's autograd graph looks fine at a glance but never trains.
///
/// Two deliberate departures from the originally-sketched version of this macro:
///
/// - It called `.into_loss()` on the summed output (ADDENDUM.md §2.2's `ScalarLoss`
///   typestate). That's Phase 2 work and doesn't exist in this codebase yet, so this
///   calls `.backward()` directly on the `.sum()` result instead — the same pattern
///   the manual test this macro replaces already used.
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
        let loss = output.sum();
        loss.backward();
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

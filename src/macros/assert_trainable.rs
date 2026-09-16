#[macro_export]
macro_rules! assert_trainable {
    ($layer:expr, $input:expr) => {{
        let output = $layer.forward(&$input);
        let loss = output
            .sum()
            .into_loss()
            .expect("layer output must be differentiable");
        loss.backward();
        for (i, p) in $layer.parameters().iter().enumerate() {
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

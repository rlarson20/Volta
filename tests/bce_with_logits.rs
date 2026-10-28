//! `bce_with_logits_loss` stability: the loss stays finite and matches an `f64` reference at saturated logits (+-100), and the gradient is correct at those logits and at `x = 0`.
use volta::{RawTensor, TensorOps, bce_with_logits_loss};

/// Reference: max(x,0) - x*y + ln(1 + exp(-|x|)), evaluated in f64.
fn reference(x: f64, y: f64) -> f64 {
    x.max(0.0) - x * y + (-x.abs()).exp().ln_1p()
}

/// H2: large |logits| must give finite loss and the exact gradient sigmoid(x) - y.
#[test]
fn test_bce_with_logits_saturated_logits_finite() {
    let xs = [-100.0_f32, -89.0, -20.0, 20.0, 89.0, 100.0];
    let ys = [0.0_f32, 1.0, 0.0, 1.0, 0.0, 1.0];
    let n = xs.len() as f64;

    let logits = RawTensor::new(xs.to_vec(), &[6], true);
    let target = RawTensor::new(ys.to_vec(), &[6], false);
    let loss = bce_with_logits_loss(&logits, &target);
    loss.clone().into_loss().expect("scalar loss").backward();

    let expected: f64 = xs
        .iter()
        .zip(&ys)
        .map(|(&x, &y)| reference(f64::from(x), f64::from(y)))
        .sum::<f64>()
        / n;
    let got = f64::from(*loss.data().first().expect("scalar loss"));
    assert!(got.is_finite(), "loss not finite: {got}");
    assert!(
        (got - expected).abs() < 1e-3 * expected.max(1.0),
        "{got} vs {expected}"
    );

    let grad = logits.grad().expect("logits grad missing");
    for ((&x, &y), &g) in xs.iter().zip(&ys).zip(&grad) {
        let want = (1.0 / (1.0 + (-f64::from(x)).exp()) - f64::from(y)) / n;
        assert!(g.is_finite(), "grad NaN/inf at x={x}");
        assert!((f64::from(g) - want).abs() < 1e-6, "x={x}: {g} vs {want}");
    }
}

/// The max(x,0) tie at x=0 must still give sigmoid(0) - y = 0.5 - y.
#[test]
fn test_bce_with_logits_gradient_at_zero() {
    let logits = RawTensor::new(vec![0.0, 0.0], &[2], true);
    let target = RawTensor::new(vec![0.0, 1.0], &[2], false);
    bce_with_logits_loss(&logits, &target)
        .into_loss()
        .expect("scalar loss")
        .backward();
    let grad = logits.grad().expect("grad missing");
    assert!(
        (grad.first().expect("grad[0]") - 0.25).abs() < 1e-6,
        "{grad:?}"
    );
    assert!(
        (grad.get(1).expect("grad[1]") + 0.25).abs() < 1e-6,
        "{grad:?}"
    );
}

//! GPU autograd for `Recip`, `Exp2`, `Log2`, `Sin`, and `Cos`: the gradient of each op on a GPU tensor is checked against its analytic derivative. Only built with the `gpu` feature; skips when no GPU is available.
#[test]
#[cfg(feature = "gpu")]
fn test_new_unary_ops_gpu_autograd() {
    use volta::gpu::is_gpu_available;
    use volta::{Device, RawTensor};

    if !is_gpu_available() {
        return;
    }

    let dev = Device::gpu().expect("GPU availability checked above");

    // Test each new unary operation with gradients
    let x = RawTensor::new(vec![2.0, 4.0], &[2], true).to_device(dev.clone());

    // Recip
    let y_recip = x.recip();
    let loss_recip = y_recip.sum();
    loss_recip.into_loss().expect("scalar loss").backward();
    {
        let expected_grad = [-1.0 / 4.0, -1.0 / 16.0]; // d(1/x)/dx = -1/x^2
        let actual_grad = x.grad_storage().unwrap().to_vec();
        assert_eq!(actual_grad.len(), expected_grad.len());
        for (i, (&actual, &expected)) in actual_grad.iter().zip(&expected_grad).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5,
                "Recip grad failed at index {i}: got {actual}, expected {expected}"
            );
        }
    }

    // Reset gradients for next test
    x.zero_grad();

    // Exp2
    let y_exp2 = x.exp2();
    let loss_exp2 = y_exp2.sum();
    loss_exp2.into_loss().expect("scalar loss").backward();
    {
        let ln2 = std::f32::consts::LN_2;
        let expected_grad = [2_f32.powf(2.0) * ln2, 2_f32.powf(4.0) * ln2]; // d(2^x)/dx = 2^x * ln(2)
        let actual_grad = x.grad_storage().unwrap().to_vec();
        assert_eq!(actual_grad.len(), expected_grad.len());
        for (i, (&actual, &expected)) in actual_grad.iter().zip(&expected_grad).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-4,
                "Exp2 grad failed at index {i}: got {actual}, expected {expected}"
            );
        }
    }

    // Reset gradients
    x.zero_grad();

    // Log2 (using positive inputs)
    let x_log2 = RawTensor::new(vec![1.0, 2.0], &[2], true).to_device(dev.clone());
    let y_log2 = x_log2.log2();
    let loss_log2 = y_log2.sum();
    loss_log2.into_loss().expect("scalar loss").backward();
    {
        let ln2 = std::f32::consts::LN_2;
        let expected_grad = [1.0 / (1.0 * ln2), 1.0 / (2.0 * ln2)]; // d(log2(x))/dx = 1/(x*ln(2))
        let actual_grad = x_log2.grad_storage().unwrap().to_vec();
        assert_eq!(actual_grad.len(), expected_grad.len());
        for (i, (&actual, &expected)) in actual_grad.iter().zip(&expected_grad).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5,
                "Log2 grad failed at index {i}: got {actual}, expected {expected}"
            );
        }
    }

    // Reset gradients
    x_log2.zero_grad();

    // Sin
    let y_sin = x.sin();
    let loss_sin = y_sin.sum();
    loss_sin.into_loss().expect("scalar loss").backward();
    {
        let expected_grad = [2.0_f32.cos(), 4.0_f32.cos()]; // d(sin(x))/dx = cos(x)
        let actual_grad = x.grad_storage().unwrap().to_vec();
        assert_eq!(actual_grad.len(), expected_grad.len());
        for (i, (&actual, &expected)) in actual_grad.iter().zip(&expected_grad).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5,
                "Sin grad failed at index {i}: got {actual}, expected {expected}"
            );
        }
    }

    // Reset gradients
    x.zero_grad();

    // Cos
    let y_cos = x.cos();
    let loss_cos = y_cos.sum();
    loss_cos.into_loss().expect("scalar loss").backward();
    {
        let expected_grad = [-2.0_f32.sin(), -4.0_f32.sin()]; // d(cos(x))/dx = -sin(x)
        let actual_grad = x.grad_storage().unwrap().to_vec();
        assert_eq!(actual_grad.len(), expected_grad.len());
        for (i, (&actual, &expected)) in actual_grad.iter().zip(&expected_grad).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5,
                "Cos grad failed at index {i}: got {actual}, expected {expected}"
            );
        }
    }
}

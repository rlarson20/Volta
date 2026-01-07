#[test]
#[cfg(feature = "gpu")]
fn test_new_unary_ops_gpu_autograd() {
    use volta::gpu::is_gpu_available;
    use volta::{Device, RawTensor, TensorOps};

    if !is_gpu_available() {
        return;
    }

    let dev = Device::GPU("TestDevice".to_string());

    // Test each new unary operation with gradients
    let x = RawTensor::new(vec![2.0, 4.0], &[2], true).to_device(dev.clone());

    // Recip
    let y_recip = x.recip();
    let loss_recip = y_recip.sum();
    loss_recip.backward();
    {
        let xb = x.borrow();
        let grad = xb.grad.as_ref().unwrap();
        let expected_grad = [-1.0 / 4.0, -1.0 / 16.0]; // d(1/x)/dx = -1/x^2
        let actual_grad = grad.to_vec();
        for (i, &actual) in actual_grad.iter().enumerate() {
            assert!(
                (actual - expected_grad[i]).abs() < 1e-5,
                "Recip grad failed at index {}: got {}, expected {}",
                i,
                actual,
                expected_grad[i]
            );
        }
    }

    // Reset gradients for next test
    x.borrow_mut().grad = None;

    // Exp2
    let y_exp2 = x.exp2();
    let loss_exp2 = y_exp2.sum();
    loss_exp2.backward();
    {
        let xb = x.borrow();
        let grad = xb.grad.as_ref().unwrap();
        let ln2 = std::f32::consts::LN_2;
        let expected_grad = [2_f32.powf(2.0) * ln2, 2_f32.powf(4.0) * ln2]; // d(2^x)/dx = 2^x * ln(2)
        let actual_grad = grad.to_vec();
        for (i, &actual) in actual_grad.iter().enumerate() {
            assert!(
                (actual - expected_grad[i]).abs() < 1e-4,
                "Exp2 grad failed at index {}: got {}, expected {}",
                i,
                actual,
                expected_grad[i]
            );
        }
    }

    // Reset gradients
    x.borrow_mut().grad = None;

    // Log2 (using positive inputs)
    let x_log2 = RawTensor::new(vec![1.0, 2.0], &[2], true).to_device(dev.clone());
    let y_log2 = x_log2.log2();
    let loss_log2 = y_log2.sum();
    loss_log2.backward();
    {
        let xb = x_log2.borrow();
        let grad = xb.grad.as_ref().unwrap();
        let ln2 = std::f32::consts::LN_2;
        let expected_grad = [1.0 / (1.0 * ln2), 1.0 / (2.0 * ln2)]; // d(log2(x))/dx = 1/(x*ln(2))
        let actual_grad = grad.to_vec();
        for (i, &actual) in actual_grad.iter().enumerate() {
            assert!(
                (actual - expected_grad[i]).abs() < 1e-5,
                "Log2 grad failed at index {}: got {}, expected {}",
                i,
                actual,
                expected_grad[i]
            );
        }
    }

    // Reset gradients
    x_log2.borrow_mut().grad = None;

    // Sin
    let y_sin = x.sin();
    let loss_sin = y_sin.sum();
    loss_sin.backward();
    {
        let xb = x.borrow();
        let grad = xb.grad.as_ref().unwrap();
        let expected_grad = [2.0_f32.cos(), 4.0_f32.cos()]; // d(sin(x))/dx = cos(x)
        let actual_grad = grad.to_vec();
        for (i, &actual) in actual_grad.iter().enumerate() {
            assert!(
                (actual - expected_grad[i]).abs() < 1e-5,
                "Sin grad failed at index {}: got {}, expected {}",
                i,
                actual,
                expected_grad[i]
            );
        }
    }

    // Reset gradients
    x.borrow_mut().grad = None;

    // Cos
    let y_cos = x.cos();
    let loss_cos = y_cos.sum();
    loss_cos.backward();
    {
        let xb = x.borrow();
        let grad = xb.grad.as_ref().unwrap();
        let expected_grad = [-2.0_f32.sin(), -4.0_f32.sin()]; // d(cos(x))/dx = -sin(x)
        let actual_grad = grad.to_vec();
        for (i, &actual) in actual_grad.iter().enumerate() {
            assert!(
                (actual - expected_grad[i]).abs() < 1e-5,
                "Cos grad failed at index {}: got {}, expected {}",
                i,
                actual,
                expected_grad[i]
            );
        }
    }
}

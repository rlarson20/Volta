use volta::{Dropout, Flatten};
use volta::{
    RawTensor, Storage, TensorOps, check_gradients_simple,
    nn::{BatchNorm2d, Module, SGD},
};

// FIX: doesn't always happen (one in ~8 runs), but still annoying
// failures:
// test_batchnorm_gradient_check
// test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 4 filtered out; finished in 0.06s
// Gradient mismatch at index 0: analytical=-2.980232e-7, numerical=1.311302e-4, error=1.002273e0
// Gradient mismatch at index 8: analytical=-2.980232e-7, numerical=1.192093e-4, error=1.002500e0
// Gradient mismatch at index 25: analytical=2.980232e-7, numerical=-1.043081e-4, error=1.002857e0
// Gradient mismatch at index 26: analytical=2.980232e-7, numerical=1.132488e-4, error=9.973684e-1
// Gradient mismatch at index 34: analytical=-2.980232e-7, numerical=-1.192093e-4, error=9.975000e-1
// Gradient mismatch at index 43: analytical=-2.980232e-7, numerical=1.192093e-4, error=1.002500e0
// Gradient mismatch at index 59: analytical=2.980232e-7, numerical=-1.430511e-4, error=1.002083e0
// Gradient mismatch at index 60: analytical=2.980232e-7, numerical=-1.162291e-4, error=1.002564e0
// Gradient check FAILED: max_error=1.002857e0, mean_error=1.251816e-1
// thread 'test_batchnorm_gradient_check' (83138801) panicked at tests/coverage_fill.rs:18:5:
// BatchNorm2d numerical gradient check failed
#[test]
fn test_batchnorm_gradient_check() {
    // Report indicated BatchNorm gradient paths were uncovered.
    // Use a small batch > 1 to ensure statistics calculation is non-trivial.
    let bn = BatchNorm2d::new(2);
    let x = RawTensor::randn(&[2, 2, 4, 4]); // B=2, C=2, H=4, W=4
    x.borrow_mut().requires_grad = true;

    // BatchNorm is highly sensitive to epsilon/tolerance in numerical checks
    // due to the division by small variances.
    let passed = check_gradients_simple(&x, |t| bn.forward(t).sum());
    assert!(passed, "BatchNorm2d numerical gradient check failed");
}

#[test]
fn test_dropout_edge_cases() {
    // Test coverage gap: dropout mask generation
    let d_zero = Dropout::new(0.0);
    let x = RawTensor::ones(&[2, 2]);
    let y = d_zero.forward(&x);
    // With p=0.0, output should exactly match input
    assert_eq!(y.borrow().data, x.borrow().data);
}

#[test]
fn test_flatten_gradient_check() {
    // Ensure Flatten correctly propagates gradients back to original shape
    let flatten = Flatten::new();
    let x = RawTensor::randn(&[2, 3, 4]); // 2x3x4
    x.borrow_mut().requires_grad = true;

    let passed = check_gradients_simple(&x, |t| {
        let flat = flatten.forward(t);
        flat.sum()
    });
    assert!(passed, "Flatten gradient check failed");
}

#[test]
fn test_sgd_momentum_logic() {
    // Coverage gap: SGD momentum branches
    let t = RawTensor::new(vec![1.0], &[1], true);
    // Manually inject a gradient
    t.borrow_mut().grad = Some(Storage::cpu(vec![0.1]));

    let mut opt = SGD::new(vec![t.clone()], 0.1, 0.9, 0.0);

    // Step 1:
    // v = 0.9*0 - lr*grad = -0.1*0.1 = -0.01
    // w = 1.0 + (-0.01) = 0.99
    opt.step();
    let val1 = t.borrow().data[0];
    assert!((val1 - 0.99).abs() < 1e-6);

    // Step 2:
    // v = 0.9*(-0.01) - 0.1*0.1 = -0.009 - 0.01 = -0.019
    // w = 0.99 + (-0.019) = 0.971
    opt.step();
    let val2 = t.borrow().data[0];
    assert!((val2 - 0.971).abs() < 1e-6);
}

#[test]
fn test_storage_mut_access() {
    // Coverage gap: Storage::as_mut_slice failure/success paths
    let mut s = Storage::cpu(vec![1.0, 2.0, 3.0]);

    // Check successful mutation
    if let Some(slice) = s.as_mut_slice() {
        slice[0] = 10.0;
    }
    assert_eq!(s.as_slice()[0], 10.0);

    // Check iterator access
    for val in &mut s {
        *val += 1.0;
    }
    assert_eq!(s.as_slice(), &[11.0, 3.0, 4.0]);
}

//! `Tensor::try_set_grad` refuses a gradient whose length differs from the tensor's element
//! count (#324). Before, a short gradient made the CPU SGD and Adam steps update only a prefix
//! of the parameter, with no error, and a long one was partly ignored or hit an index panic.
//!
//! Each case seeds a 4-element parameter of ones with a short (2) or long (6) gradient, checks
//! the error, then steps SGD or Adam and checks the parameter is untouched (no gradient was
//! stored). The GPU case runs when an adapter is available.

use volta::nn::{Adam, SGD};
use volta::storage::Storage;
use volta::{Device, RawTensor, Tensor, VoltaError};

/// A 4-element parameter of ones that tracks gradients.
fn param() -> Tensor {
    RawTensor::new(vec![1.0; 4], &[4], true)
}

/// Seeds `p` with a `len`-element gradient of ones and asserts `try_set_grad` refuses it
/// with `ShapeDataMismatch` naming both lengths, leaving no gradient behind.
fn assert_refused(p: &Tensor, len: usize) {
    let err = p
        .try_set_grad(Storage::cpu(vec![1.0; len]))
        .expect_err("a gradient of the wrong length must be refused");
    let VoltaError::ShapeDataMismatch {
        shape,
        elements,
        len: got,
    } = err
    else {
        panic!("expected ShapeDataMismatch, got {err:?}");
    };
    assert_eq!(shape, vec![4]);
    assert_eq!(elements, 4);
    assert_eq!(got, len);
    assert!(p.grad().is_none(), "a refused gradient must not be stored");
}

/// After a refused gradient the step has nothing to apply: every element stays 1.
fn assert_untouched(p: &Tensor) {
    let data = p.to_device(Device::CPU).data();
    assert_eq!(data, vec![1.0; 4], "the parameter must not be updated");
}

#[test]
fn sgd_short_gradient_is_refused() {
    let p = param();
    assert_refused(&p, 2);
    let mut opt = SGD::new(vec![p.clone()], 0.5, 0.0, 0.0);
    opt.step();
    assert_untouched(&p);
}

#[test]
fn sgd_long_gradient_is_refused() {
    let p = param();
    assert_refused(&p, 6);
    let mut opt = SGD::new(vec![p.clone()], 0.5, 0.9, 0.0);
    opt.step();
    assert_untouched(&p);
}

#[test]
fn adam_short_gradient_is_refused() {
    let p = param();
    assert_refused(&p, 2);
    let mut opt = Adam::new(vec![p.clone()], 0.5, (0.9, 0.999), 1e-8, 0.0);
    opt.step();
    assert_untouched(&p);
}

#[test]
fn adam_long_gradient_is_refused() {
    let p = param();
    assert_refused(&p, 6);
    let mut opt = Adam::new(vec![p.clone()], 0.5, (0.9, 0.999), 1e-8, 0.0);
    opt.step();
    assert_untouched(&p);
}

/// The panicking form wraps the fallible one and names both lengths.
#[test]
#[should_panic(expected = "set_grad: Shape mismatch: tensor has 4 elements but data length is 2")]
fn set_grad_panics_on_a_short_gradient() {
    param().set_grad(Storage::cpu(vec![1.0; 2]));
}

/// A gradient of the right length is still stored and stepped: SGD with lr 0.5 on a
/// gradient `[1, 2, 3, 4]` gives `1 - 0.5 * g` in every element, not just a prefix.
#[test]
fn a_matching_gradient_updates_every_element() {
    let p = param();
    p.try_set_grad(Storage::cpu(vec![1.0, 2.0, 3.0, 4.0]))
        .expect("a matching gradient is accepted");
    let mut opt = SGD::new(vec![p.clone()], 0.5, 0.0, 0.0);
    opt.step();
    assert_eq!(p.data(), vec![0.5, 0.0, -0.5, -1.0]);
}

/// The check runs before the gradient moves to the tensor's device, so a GPU parameter
/// refuses a short or long CPU gradient the same way (lavapipe or a real adapter).
#[cfg(feature = "gpu")]
#[test]
fn gpu_parameter_refuses_short_and_long_gradients() {
    let Some(gpu) = Device::gpu() else {
        return;
    };
    for len in [2, 6] {
        let p = param().detach().to_device(gpu.clone());
        p.set_requires_grad(true);
        assert_refused(&p, len);
        let mut sgd = SGD::new(vec![p.clone()], 0.5, 0.0, 0.0);
        sgd.step();
        let mut adam = Adam::new(vec![p.clone()], 0.5, (0.9, 0.999), 1e-8, 0.0);
        adam.step();
        assert_untouched(&p);
    }
}

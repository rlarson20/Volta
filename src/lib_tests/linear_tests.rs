//! `Linear`: shapes, bias, forward value, input gradient, and trainability.

use super::*;

#[test]
fn weight_shape_and_bias_present() {
    let layer = Linear::new(4, 3, true);
    assert_eq!(layer.weight.borrow().shape, vec![4, 3]);
    assert!(layer.bias.is_some());
    let b = layer.bias.as_ref().unwrap();
    assert_eq!(b.borrow().shape, vec![3]);
}

#[test]
fn no_bias_when_use_bias_false() {
    let layer = Linear::new(4, 3, false);
    assert!(layer.bias.is_none());
    assert_eq!(layer.parameters().len(), 1);
}

#[test]
fn forward_shape_with_bias() {
    let layer = Linear::new(5, 2, true);
    let x = RawTensor::new(vec![1.0; 3 * 5], &[3, 5], true);
    let y = layer.forward(&x);
    assert_eq!(y.borrow().shape, vec![3, 2]);
}

#[test]
fn forward_value_with_known_weights() {
    // Build a deterministic Linear: y = xW + b with shape (1,2) input,
    // (2,3) weight, (3,) bias.
    let mut layer = Linear::new(2, 3, true);
    // Override randomly initialized weights with known values.
    {
        let mut w = layer.weight.borrow_mut();
        w.data = Storage::cpu(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
    {
        let bias = layer.bias.as_mut().unwrap();
        let mut b = bias.borrow_mut();
        b.data = Storage::cpu(vec![10.0, 20.0, 30.0]);
    }
    let x = RawTensor::new(vec![1.0, 1.0], &[1, 2], false);
    let y = layer.forward(&x);
    // x @ W = [1*1+1*4, 1*2+1*5, 1*3+1*6] = [5, 7, 9]; + bias = [15, 27, 39]
    assert_eq!(y.borrow().data.to_vec(), vec![15.0, 27.0, 39.0]);
}

#[test]
fn backward_gradcheck_input() {
    // Seeded, with `eps = 0.5`: unseeded, about 1 in 130 weight draws failed at
    // `check_gradients_simple`'s `eps = 1e-2`, and none of the same 400 seeds fail at
    // 0.5. `x @ w + b` is linear in `x`, so the step only affects f32 roundoff.
    crate::tensor::manual_seed(0x11E);
    let layer = Linear::new(3, 2, true);
    // Snapshot the (random) weight so the closure captures stable values.
    let w_snapshot = layer.weight.borrow().data.to_vec();
    let b_snapshot = layer
        .bias
        .as_ref()
        .map(|b| b.borrow().data.to_vec())
        .unwrap_or_default();
    let w_shape = layer.weight.shape();
    let b_shape = layer
        .bias
        .as_ref()
        .expect("Linear::new(.., true) has a bias")
        .shape();
    let x = RawTensor::new(vec![0.5, -1.0, 1.5, 0.25, 0.0, 2.0], &[2, 3], true);

    let (max_err, mean_err, passed) = RawTensor::check_gradients(
        &x,
        |t| {
            let w = RawTensor::from_shape(w_snapshot.clone(), w_shape.clone(), false);
            let b = RawTensor::from_shape(b_snapshot.clone(), b_shape.clone(), false);
            t.matmul(&w).add(&b).sum()
        },
        0.5,
        1e-3,
    );
    assert!(
        passed,
        "Linear input gradient check failed: max_error={max_err:.6e}, mean_error={mean_err:.6e}"
    );
}

#[test]
fn assert_trainable_all_params_get_gradient() {
    let layer = Linear::new(4, 3, true);
    let x = crate::macros::trainable_input(&[2, 4]);
    crate::assert_trainable!(layer, x);
}

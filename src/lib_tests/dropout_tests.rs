//! `Dropout`: `p` of 0 and 1, eval-mode passthrough, shape, and mean preservation.

use super::*;

#[test]
fn p_zero_is_identity() {
    let layer = Dropout::new(0.0);
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4], true);
    let y = <Dropout as Layer>::forward(&layer, &x);
    assert_eq!(y.borrow().data.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(y.borrow().shape, vec![4]);
}

#[test]
fn p_one_zeroes_all_elements() {
    // With keep_prob=0, the mask is all zeros => output is all zeros.
    let layer = Dropout::new(1.0);
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4], true);
    let y = <Dropout as Layer>::forward(&layer, &x);
    for v in y.borrow().data.to_vec() {
        assert_eq!(v, 0.0, "dropout p=1 should zero all elements");
    }
}

#[test]
fn eval_mode_passthrough() {
    // In eval mode, even high-p dropout returns x unchanged.
    let mut layer = Dropout::new(0.9);
    layer.train(false);
    let x = RawTensor::new(vec![5.0; 16], &[16], true);
    let y = <Dropout as Layer>::forward(&layer, &x);
    for v in y.borrow().data.to_vec() {
        assert_eq!(v, 5.0);
    }
}

#[test]
fn shape_preserved_under_training() {
    let layer = Dropout::new(0.5);
    let x = RawTensor::new(vec![1.0; 2 * 3 * 4], &[2, 3, 4], true);
    let y = <Dropout as Layer>::forward(&layer, &x);
    assert_eq!(y.borrow().shape, vec![2, 3, 4]);
}

#[test]
fn statistical_mean_preservation() {
    // For inverted dropout, E[output] == input. With a large enough sample
    // the empirical mean stays within ~5% of the input value.
    crate::tensor::manual_seed(0x00C0_FFEE);
    let layer = Dropout::new(0.5);
    let n = 20_000;
    let x = RawTensor::new(vec![1.0; n], &[n], false);
    let y = <Dropout as Layer>::forward(&layer, &x);
    let sum: f32 = y.borrow().data.to_vec().iter().sum();
    let mean = sum / n as f32;
    // E[mean] = 1.0, std ≈ sqrt(p / ((1-p)*n)) ≈ 0.007 for p=0.5, n=20k.
    // 0.05 is a comfortable bound that keeps this from flaking.
    assert!(
        (mean - 1.0).abs() < 0.05,
        "dropout mean drifted: got {mean}, expected ~1.0"
    );
}

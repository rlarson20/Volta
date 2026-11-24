//! User-supplied movement arguments are validated and returned as `VoltaError` (#96, #97).

use volta::error::VoltaError;
use volta::{PadValue, RawTensor};

fn x23() -> volta::Tensor {
    RawTensor::new(vec![1.0; 6], &[2, 3], false)
}

#[test]
fn try_pad_with_rejects_overflowing_width() {
    let x = x23();
    for padding in [
        [(usize::MAX, 1), (0, 0)],
        [(usize::MAX - 1, 1), (0, 0)],
        [(0, 0), (1, usize::MAX)],
    ] {
        let r = RawTensor::try_pad_with(&x, &padding, PadValue::ZERO);
        assert!(
            matches!(r, Err(VoltaError::InvalidParameter(ref m)) if m.message.contains("overflows")),
            "padding {padding:?}: expected an overflow error, got {:?}",
            r.map(|t| t.shape())
        );
    }
}

#[test]
fn try_pad_with_valid_padding_still_works() {
    let y = RawTensor::try_pad_with(&x23(), &[(1, 0), (0, 2)], PadValue::ZERO).unwrap();
    assert_eq!(y.shape(), vec![3, 5]);
}

#[test]
#[should_panic(expected = "overflows usize")]
fn pad_with_overflow_panics_with_message() {
    let _ = x23().pad_with(&[(usize::MAX, 1), (0, 0)], PadValue::ZERO);
}

#[test]
fn try_stride_op_rejects_zero_stride() {
    let r = RawTensor::try_stride_op(&x23(), &[0, 1]);
    assert!(
        matches!(r, Err(VoltaError::InvalidParameter(ref m)) if m.message.contains("stride 0 for dimension 0")),
        "{:?}",
        r.map(|t| t.shape())
    );
    let r = RawTensor::try_stride_op(&x23(), &[1, 0]);
    assert!(
        matches!(r, Err(VoltaError::InvalidParameter(ref m)) if m.message.contains("dimension 1"))
    );
}

#[test]
fn try_stride_op_rejects_wrong_length() {
    let r = RawTensor::try_stride_op(&x23(), &[1]);
    assert!(matches!(r, Err(VoltaError::InvalidParameter(_))));
}

#[test]
#[should_panic(expected = "stride 0 for dimension 0")]
fn stride_op_zero_stride_panics_with_message() {
    let _ = x23().stride_op(&[0, 1]);
}

#[test]
fn try_stride_op_valid_strides_still_work() {
    let x = RawTensor::new((0..6).map(|i| i as f32).collect(), &[2, 3], false);
    let y = RawTensor::try_stride_op(&x, &[1, 2]).unwrap();
    assert_eq!(y.shape(), vec![2, 2]);
    assert_eq!(y.data(), vec![0.0, 2.0, 3.0, 5.0]);
}

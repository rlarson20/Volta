//! User-facing argument checks on `DataLoader`, `randn_like` and `set_requires_grad`
//! (#93, #99, #134).

use volta::dtype::DType;
use volta::error::VoltaError;
use volta::{DataLoader, RawTensor, randn_like};

/// The `InvalidParameter` message of a `DataLoader::try_new` call that must fail.
fn loader_err(
    data: usize,
    targets: usize,
    data_shape: &[usize],
    target_shape: &[usize],
    batch: usize,
) -> String {
    match DataLoader::try_new(
        vec![0.0; data],
        vec![0.0; targets],
        data_shape,
        target_shape,
        batch,
        true,
    ) {
        Err(VoltaError::InvalidParameter(m)) => m.message,
        Err(e) => panic!("wrong error {e:?}"),
        Ok(_) => panic!(
            "accepted data={data} targets={targets} {data_shape:?} {target_shape:?} batch={batch}"
        ),
    }
}

#[test]
fn dataloader_rejects_zero_batch_size() {
    assert!(loader_err(4, 4, &[1], &[1], 0).contains("batch_size"));
}

#[test]
fn dataloader_rejects_zero_size_shapes() {
    assert!(loader_err(4, 4, &[0], &[1], 2).starts_with("data_shape: "));
    assert!(loader_err(4, 4, &[1], &[0], 2).starts_with("target_shape: "));
}

/// The error names the op and the argument as fields, not only in prose (#377).
#[test]
fn dataloader_errors_name_the_op_and_argument() {
    let err = DataLoader::try_new(vec![0.0; 4], vec![0.0; 4], &[1], &[0], 2, true)
        .err()
        .expect("a zero target dimension is refused");
    let detail = err.detail().expect("InvalidParameter carries a detail");
    assert_eq!(
        (detail.op, detail.arg),
        ("DataLoader::new", Some("target_shape"))
    );
}

#[test]
fn dataloader_rejects_a_ragged_data_length() {
    let m = loader_err(5, 2, &[2], &[1], 2);
    assert!(
        m.contains("5 data values") && m.contains("whole number"),
        "{m}"
    );
}

#[test]
fn dataloader_rejects_a_target_count_mismatch() {
    assert!(loader_err(4, 3, &[2], &[1], 2).contains("target values"));
}

#[test]
#[should_panic(expected = "batch_size must be at least 1")]
fn dataloader_new_panics_naming_the_argument() {
    let _ = DataLoader::new(vec![0.0; 4], vec![0.0; 4], &[1], &[1], 0, false);
}

#[test]
fn dataloader_valid_input_yields_every_sample() {
    let d = DataLoader::try_new(vec![0.0; 6], vec![0.0; 3], &[2], &[1], 2, false).unwrap();
    assert_eq!(d.count(), 2, "3 samples in batches of 2");
}

#[test]
fn randn_like_keeps_float_dtype() {
    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let x = RawTensor::zeros(&[3, 2]).to_dtype(dtype);
        let r = randn_like(&x);
        assert_eq!(r.dtype(), dtype);
        assert_eq!(r.shape(), vec![3, 2]);
        assert!(!r.requires_grad());
    }
}

#[test]
#[should_panic(expected = "not a float dtype")]
fn randn_like_rejects_an_integer_tensor() {
    let _ = randn_like(&RawTensor::from_typed(vec![1i32, 2], &[2]));
}

#[test]
#[should_panic(expected = "set_requires_grad(false) called on a non-leaf tensor")]
fn set_requires_grad_false_on_a_non_leaf_panics() {
    let x = RawTensor::new(vec![1.0, 2.0], &[2], true);
    let y = x.elem_mul(&x);
    y.set_requires_grad(false);
}

#[test]
fn set_requires_grad_false_on_a_leaf_is_allowed() {
    let x = RawTensor::new(vec![1.0, 2.0], &[2], true);
    x.set_requires_grad(false);
    assert!(!x.requires_grad());
}

//! Tests exercising all `VoltaError` variants through public APIs.

use volta::Axis;
use volta::RawTensor;
use volta::dtype::DType;
use volta::error::VoltaError;
use volta::storage::Storage;

// ── ShapeDataMismatch ──────────────────────────────────────────────────

#[test]
fn test_shape_data_mismatch() {
    // 2 elements provided for a 3x3 shape (9 elements expected)
    let result = RawTensor::try_new(vec![1.0, 2.0], &[3, 3], false);
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(matches!(err, VoltaError::ShapeDataMismatch { .. }));
}

// ── DimensionOutOfBounds ───────────────────────────────────────────────

#[test]
fn test_dimension_out_of_bounds() {
    // 2D tensor (shape [2,3]), requesting sum along dim 5
    let x = RawTensor::try_new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false).unwrap();
    let result = RawTensor::try_sum_dim(&x, Axis(5), false);
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(matches!(
        err,
        VoltaError::DimensionOutOfBounds { dim: 5, .. }
    ));
}

// ── BroadcastError ─────────────────────────────────────────────────────

#[test]
fn test_broadcast_error() {
    let result = RawTensor::try_broadcast_shape(&[2, 3], &[4, 5]);
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(matches!(err, VoltaError::BroadcastError(_, _)));
}

// ── InvalidParameter ───────────────────────────────────────────────────

#[test]
fn test_invalid_parameter() {
    // Empty shape should be rejected
    let result = RawTensor::try_he_initialization(&[]);
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(matches!(err, VoltaError::InvalidParameter(_)));
}

// ── InvalidParameter via shrink (start > end) ──────────────────────────

#[test]
fn test_shrink_start_exceeds_end() {
    // The actual code returns InvalidParameter when start > end
    let x = RawTensor::try_new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false).unwrap();
    let result = RawTensor::try_shrink(&x, &[(1, 0), (0, 3)]);
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(matches!(err, VoltaError::InvalidParameter(_)));
}

// ── Io ─────────────────────────────────────────────────────────────────

#[test]
fn test_io_error() {
    // load_safetensors returns std::io::Result, so we check for io::Error
    let result = volta::io::load_safetensors("nonexistent_path_xyz123.safetensors");
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
}

// ── DTypeMismatch ──────────────────────────────────────────────────────

#[test]
fn test_dtype_mismatch() {
    // Create F16 storage (2 bytes per element, 4 bytes = 2 elements)
    let storage = Storage::try_from_bytes(&[0u8; 4], DType::F16).unwrap();
    let result = storage.as_host_f32();
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        VoltaError::DTypeMismatch {
            expected: DType::F32,
            actual: DType::F16,
        }
    ));
}

// ── Display strings for all variants ───────────────────────────────────

#[test]
fn test_error_display() {
    let cases: Vec<(VoltaError, &str)> = vec![
        (
            VoltaError::ShapeDataMismatch {
                shape: vec![3, 3],
                elements: 9,
                len: 2,
            },
            "Shape mismatch",
        ),
        (
            VoltaError::DimensionOutOfBounds {
                dim: 5,
                shape: vec![2, 3],
            },
            "Dimension 5 out of bounds",
        ),
        (
            VoltaError::DTypeMismatch {
                expected: DType::F32,
                actual: DType::F16,
            },
            "DType mismatch",
        ),
        (
            VoltaError::BroadcastError(vec![2, 3], vec![4, 5]),
            "Cannot broadcast",
        ),
        (
            VoltaError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "file missing",
            )),
            "IO error",
        ),
        // Only Volta builds an `ErrorDetail` (#377), so this one comes from a real call;
        // the other two detail variants' `Display` is tested in `src/error.rs`.
        (
            RawTensor::new(vec![1.0, 2.0], &[2], false)
                .try_reshape(&[3])
                .expect_err("2 elements do not reshape to 3"),
            "Invalid parameter: reshape: cannot reshape",
        ),
    ];

    for (err, expected_substr) in &cases {
        let msg = err.to_string();
        assert!(
            msg.contains(expected_substr),
            "Expected display of {err:?} to contain {expected_substr:?}, got {msg:?}"
        );
    }
}

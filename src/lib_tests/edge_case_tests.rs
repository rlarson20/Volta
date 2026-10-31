//! Invalid arguments panic with the right message (reshape, permute, pad, shrink, out-of-range axes), and scalar, infinity, and NaN edge cases of the math ops.

use super::*;

// ===== RESHAPE INVALID SIZE TESTS =====

#[test]
#[should_panic(expected = "cannot reshape [6] (6 elements) to [2, 4] (8 elements)")]
fn test_reshape_invalid_size() {
    // Attempt to reshape 6-element tensor to shape requiring 8 elements
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], false);
    let _ = x.reshape(&[2, 4]); // 2*4 = 8, but we have 6 elements
}

#[test]
#[should_panic(expected = "cannot reshape [3, 4] (12 elements) to [2, 3] (6 elements)")]
fn test_reshape_smaller_target() {
    // Attempt to reshape 12-element tensor to shape requiring 6 elements
    let x = RawTensor::new(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
        false,
    );
    let _ = x.reshape(&[2, 3]); // 2*3 = 6, but we have 12 elements
}

// A zero-sized target shape is rejected as an invalid shape (`Shape::from_dims`),
// before the element counts are compared.
#[test]
#[should_panic(expected = "shape [0, 6] has a zero dimension")]
fn test_reshape_zero_dimension() {
    // Attempt to reshape to shape with zero dimension
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    let _ = x.reshape(&[0, 6]); // zero dimension: rejected as an invalid shape, not as a size mismatch
}

// ===== PERMUTE INVALID AXES TESTS =====

#[test]
#[should_panic(expected = "Axes length must match rank")]
fn test_permute_wrong_length_too_short() {
    // Attempt to permute rank-3 tensor with only 2 axes
    let x = RawTensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2, 2, 2],
        false,
    );
    let _ = x.permute(&[Axis(0), Axis(1)]); // Missing third axis
}

#[test]
#[should_panic(expected = "Axes length must match rank")]
fn test_permute_wrong_length_too_long() {
    // Attempt to permute rank-2 tensor with 3 axes
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false);
    let _ = x.permute(&[Axis(0), Axis(1), Axis(2)]); // Too many axes for rank-2 tensor
}

#[test]
#[should_panic(expected = "Invalid permutation axes")]
fn test_permute_out_of_bounds_axis() {
    // Attempt to permute with axis exceeding tensor rank
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false);
    let _ = x.permute(&[Axis(0), Axis(5)]); // Axis 5 doesn't exist in rank-2 tensor
}

#[test]
#[should_panic(expected = "Invalid permutation axes")]
fn test_permute_duplicate_axes() {
    // Attempt to permute with duplicate axes
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false);
    let _ = x.permute(&[Axis(0), Axis(0)]); // Axis 0 appears twice
}

#[test]
#[should_panic(expected = "Invalid permutation axes")]
fn test_permute_non_contiguous_axes() {
    // Attempt to permute with non-contiguous axes
    let x = RawTensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2, 2, 2],
        false,
    );
    let _ = x.permute(&[Axis(0), Axis(1), Axis(3)]); // Axis 3 doesn't exist (should be 0, 1, 2)
}

// ===== PAD INVALID SIZE TESTS =====

#[test]
#[should_panic(expected = "must match tensor rank")]
fn test_pad_wrong_length_too_short() {
    // Attempt to pad rank-2 tensor with only 1 dimension of padding
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false);
    let _ = x.pad_with(&[(1, 1)], PadValue::ZERO); // Missing padding for second dimension
}

#[test]
#[should_panic(expected = "must match tensor rank")]
fn test_pad_wrong_length_too_long() {
    // Attempt to pad rank-2 tensor with 3 dimensions of padding
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false);
    let _ = x.pad_with(&[(1, 1), (1, 1), (1, 1)], PadValue::ZERO); // Too many padding specifications
}

#[test]
#[should_panic(expected = "check for memory issues")]
fn test_pad_excessive_padding() {
    // Attempt to pad with huge values that would exceed MAX_ALLOC
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false);
    // This would create a tensor with (2 + 5000) * (2 + 5000) = ~25M elements
    // To exceed 100M, we need even larger padding
    let _ = x.pad_with(&[(50000, 50000), (50000, 50000)], PadValue::ZERO); // Way over 100M elements
}

// ===== AXIS BOUNDS TESTS =====

#[test]
fn test_try_sum_dim_out_of_range_axis_errs() {
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false);
    let err = RawTensor::try_sum_dim(&x, Axis(2), false).unwrap_err();
    assert!(
        matches!(err, VoltaError::DimensionOutOfBounds { dim: 2, .. }),
        "{err:?}"
    );
}

#[test]
#[should_panic(expected = "Dimension 2 out of bounds for shape [2, 2]")]
fn test_max_dim_out_of_range_axis_panics() {
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false);
    let _ = x.max_dim(Axis(2), false);
}

#[test]
#[should_panic(expected = "Dimension 2 out of bounds for shape [2, 2]")]
fn test_mean_dim_out_of_range_axis_panics() {
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false);
    let _ = x.mean_dim(Axis(2), false);
}

#[test]
#[should_panic(expected = "Dimension 2 out of bounds for shape [2, 2]")]
fn test_softmax_out_of_range_axis_panics() {
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false);
    let _ = x.softmax(Axis(2));
}

#[test]
fn test_axis_reductions_use_the_named_axis() {
    // [[1, 2, 3], [4, 5, 6]]: Axis(0) collapses rows, Axis(1) collapses columns.
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    assert_eq!(
        x.sum_dim(Axis(0), false).borrow().data.to_vec(),
        vec![5.0, 7.0, 9.0]
    );
    assert_eq!(
        x.sum_dim(Axis(1), false).borrow().data.to_vec(),
        vec![6.0, 15.0]
    );
    assert_eq!(
        x.max_dim(Axis(1), false).borrow().data.to_vec(),
        vec![3.0, 6.0]
    );
    assert_eq!(
        x.mean_dim(Axis(0), false).borrow().data.to_vec(),
        vec![2.5, 3.5, 4.5]
    );
}

// ===== SHRINK VALIDATION TESTS =====

#[test]
fn test_shrink_wrong_length_too_short() {
    // Attempt to shrink rank-2 tensor with only 1 range
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    let result = RawTensor::try_shrink(&x, &[(0, 2)]);

    assert!(result.is_err());
    match result.as_ref().unwrap_err() {
        VoltaError::InvalidParameter(msg) => {
            assert!(
                msg.contains("length"),
                "Error message should mention length"
            );
            assert!(
                msg.contains('2'),
                "Error message should mention expected rank 2"
            );
            assert!(
                msg.contains('1'),
                "Error message should mention actual length 1"
            );
        }
        VoltaError::ShapeDataMismatch { .. }
        | VoltaError::DimensionOutOfBounds { .. }
        | VoltaError::DTypeMismatch { .. }
        | VoltaError::BroadcastError(..)
        | VoltaError::DeviceError(_)
        | VoltaError::Io(_)
        | VoltaError::NotScalar { .. }
        | VoltaError::NoGradRequired
        | VoltaError::OpError(_)
        | VoltaError::StateShapeMismatch { .. }
        | VoltaError::StateDataLength { .. } => {
            panic!("Expected InvalidParameter error, got {result:?}")
        }
    }
}

#[test]
fn test_shrink_wrong_length_too_long() {
    // Attempt to shrink rank-2 tensor with 3 ranges
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    let result = RawTensor::try_shrink(&x, &[(0, 2), (0, 3), (0, 1)]);

    assert!(result.is_err());
    match result.as_ref().unwrap_err() {
        VoltaError::InvalidParameter(msg) => {
            assert!(
                msg.contains("length"),
                "Error message should mention length"
            );
            assert!(
                msg.contains('2'),
                "Error message should mention expected rank 2"
            );
            assert!(
                msg.contains('3'),
                "Error message should mention actual length 3"
            );
        }
        VoltaError::ShapeDataMismatch { .. }
        | VoltaError::DimensionOutOfBounds { .. }
        | VoltaError::DTypeMismatch { .. }
        | VoltaError::BroadcastError(..)
        | VoltaError::DeviceError(_)
        | VoltaError::Io(_)
        | VoltaError::NotScalar { .. }
        | VoltaError::NoGradRequired
        | VoltaError::OpError(_)
        | VoltaError::StateShapeMismatch { .. }
        | VoltaError::StateDataLength { .. } => {
            panic!("Expected InvalidParameter error, got {result:?}")
        }
    }
}

#[test]
fn test_shrink_out_of_bounds_end() {
    // Shrink with end exceeding tensor dimension
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    let result = RawTensor::try_shrink(&x, &[(0, 2), (0, 10)]);

    assert!(result.is_err());
    match result.as_ref().unwrap_err() {
        VoltaError::DimensionOutOfBounds { dim, shape } => {
            assert_eq!(*dim, 1, "Error should be for dimension 1");
            assert_eq!(shape, &vec![2, 3], "Shape should match original tensor");
        }
        VoltaError::ShapeDataMismatch { .. }
        | VoltaError::DTypeMismatch { .. }
        | VoltaError::BroadcastError(..)
        | VoltaError::DeviceError(_)
        | VoltaError::Io(_)
        | VoltaError::NotScalar { .. }
        | VoltaError::NoGradRequired
        | VoltaError::InvalidParameter(_)
        | VoltaError::OpError(_)
        | VoltaError::StateShapeMismatch { .. }
        | VoltaError::StateDataLength { .. } => {
            panic!("Expected DimensionOutOfBounds error, got {result:?}")
        }
    }
}

#[test]
fn test_shrink_start_exceeds_dimension() {
    // Shrink with start position beyond tensor dimension
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    let result = RawTensor::try_shrink(&x, &[(0, 2), (5, 10)]);

    assert!(result.is_err());
    match result.as_ref().unwrap_err() {
        VoltaError::DimensionOutOfBounds { dim, shape } => {
            assert_eq!(*dim, 1, "Error should be for dimension 1");
            assert_eq!(shape, &vec![2, 3], "Shape should match original tensor");
        }
        VoltaError::ShapeDataMismatch { .. }
        | VoltaError::DTypeMismatch { .. }
        | VoltaError::BroadcastError(..)
        | VoltaError::DeviceError(_)
        | VoltaError::Io(_)
        | VoltaError::NotScalar { .. }
        | VoltaError::NoGradRequired
        | VoltaError::InvalidParameter(_)
        | VoltaError::OpError(_)
        | VoltaError::StateShapeMismatch { .. }
        | VoltaError::StateDataLength { .. } => {
            panic!("Expected DimensionOutOfBounds error, got {result:?}")
        }
    }
}

#[test]
fn test_shrink_invalid_range_order() {
    // Shrink with start > end (creates negative size)
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    let result = RawTensor::try_shrink(&x, &[(3, 2), (0, 3)]);

    assert!(result.is_err());
    match result.as_ref().unwrap_err() {
        VoltaError::InvalidParameter(msg) => {
            assert!(
                msg.contains("dimension 0"),
                "Error should mention dimension 0"
            );
            assert!(msg.contains("start"), "Error should mention start");
            assert!(msg.contains("end"), "Error should mention end");
            assert!(msg.contains('3'), "Error should include values");
            assert!(msg.contains('2'), "Error should include values");
        }
        VoltaError::ShapeDataMismatch { .. }
        | VoltaError::DimensionOutOfBounds { .. }
        | VoltaError::DTypeMismatch { .. }
        | VoltaError::BroadcastError(..)
        | VoltaError::DeviceError(_)
        | VoltaError::Io(_)
        | VoltaError::NotScalar { .. }
        | VoltaError::NoGradRequired
        | VoltaError::OpError(_)
        | VoltaError::StateShapeMismatch { .. }
        | VoltaError::StateDataLength { .. } => {
            panic!("Expected InvalidParameter error, got {result:?}")
        }
    }
}

#[test]
fn test_shrink_empty_range() {
    // Shrink with start == end (zero-sized dimension)
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    let result = RawTensor::try_shrink(&x, &[(0, 0), (0, 3)]);

    assert!(result.is_err());
    match result.as_ref().unwrap_err() {
        VoltaError::InvalidParameter(msg) => {
            assert!(
                msg.contains("dimension 0"),
                "Error should mention dimension 0"
            );
            assert!(
                msg.contains("zero-sized"),
                "Error should mention zero-sized dimension"
            );
        }
        VoltaError::ShapeDataMismatch { .. }
        | VoltaError::DimensionOutOfBounds { .. }
        | VoltaError::DTypeMismatch { .. }
        | VoltaError::BroadcastError(..)
        | VoltaError::DeviceError(_)
        | VoltaError::Io(_)
        | VoltaError::NotScalar { .. }
        | VoltaError::NoGradRequired
        | VoltaError::OpError(_)
        | VoltaError::StateShapeMismatch { .. }
        | VoltaError::StateDataLength { .. } => {
            panic!("Expected InvalidParameter error, got {result:?}")
        }
    }
}

#[test]
fn test_shrink_both_empty_ranges() {
    // Shrink with all dimensions having start == end
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    let result = RawTensor::try_shrink(&x, &[(0, 0), (0, 0)]);

    assert!(result.is_err());
    match result.as_ref().unwrap_err() {
        VoltaError::InvalidParameter(msg) => {
            assert!(
                msg.contains("zero-sized"),
                "Error should mention zero-sized dimension"
            );
        }
        VoltaError::ShapeDataMismatch { .. }
        | VoltaError::DimensionOutOfBounds { .. }
        | VoltaError::DTypeMismatch { .. }
        | VoltaError::BroadcastError(..)
        | VoltaError::DeviceError(_)
        | VoltaError::Io(_)
        | VoltaError::NotScalar { .. }
        | VoltaError::NoGradRequired
        | VoltaError::OpError(_)
        | VoltaError::StateShapeMismatch { .. }
        | VoltaError::StateDataLength { .. } => {
            panic!("Expected InvalidParameter error, got {result:?}")
        }
    }
}

#[test]
fn test_shrink_valid_boundary_cases() {
    // Test edge cases that should succeed
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);

    // Shrink to exact dimension (end == dim_size)
    let result = RawTensor::try_shrink(&x, &[(0, 2), (0, 3)]).unwrap();
    assert_eq!(result.borrow().shape, vec![2, 3]);
    assert_eq!(
        result.borrow().data.to_vec(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );

    // Shrink to single element (end - start == 1 for each dimension)
    let result = RawTensor::try_shrink(&x, &[(0, 1), (0, 1)]).unwrap();
    assert_eq!(result.borrow().shape, vec![1, 1]);
    assert_eq!(result.borrow().data.to_vec(), vec![1.0]);

    // Shrink from non-zero start
    let result = RawTensor::try_shrink(&x, &[(1, 2), (1, 3)]).unwrap();
    assert_eq!(result.borrow().shape, vec![1, 2]);
    assert_eq!(result.borrow().data.to_vec(), vec![5.0, 6.0]);
}

#[test]
fn test_shrink_panic_version_still_works() {
    // Test that the panic-based API maintains backward compatibility
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);

    // Valid ranges should work with shrink()
    let result = x.shrink(&[(0, 2), (1, 3)]);
    assert_eq!(result.borrow().shape, vec![2, 2]);
    assert_eq!(result.borrow().data.to_vec(), vec![2.0, 3.0, 5.0, 6.0]);

    // Invalid ranges should panic
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = x.shrink(&[(0, 2), (0, 10)]); // end exceeds dimension
    }));
    assert!(result.is_err(), "shrink() should panic on invalid ranges");
}

// ===== NUMERICAL EDGE CASE TESTS =====

// Division by zero tests
#[test]
fn test_div_by_zero_positive() {
    // Division by zero should produce +inf (positive / positive_zero)
    let x = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], false);
    let y = RawTensor::new(vec![0.0, 0.0, 0.0], &[3], false);
    let z = x.div(&y);

    // All results should be +inf
    for &val in z.borrow().data.to_host_f32().iter() {
        assert!(val.is_infinite(), "Expected infinity, got {val}");
        assert!(
            val.is_sign_positive(),
            "Expected positive infinity, got {val}"
        );
    }
}

#[test]
fn test_div_by_zero_negative() {
    // Division by zero should produce -inf (negative / positive_zero)
    let x = RawTensor::new(vec![-1.0, -2.0, -3.0], &[3], false);
    let y = RawTensor::new(vec![0.0, 0.0, 0.0], &[3], false);
    let z = x.div(&y);

    // All results should be -inf
    for &val in z.borrow().data.to_host_f32().iter() {
        assert!(val.is_infinite(), "Expected infinity, got {val}");
        assert!(
            val.is_sign_negative(),
            "Expected negative infinity, got {val}"
        );
    }
}

#[test]
fn test_div_zero_by_zero() {
    // 0.0 / 0.0 should produce NaN
    let x = RawTensor::new(vec![0.0, 0.0, 0.0], &[3], false);
    let y = RawTensor::new(vec![0.0, 0.0, 0.0], &[3], false);
    let z = x.div(&y);

    // All results should be NaN
    for &val in z.borrow().data.to_host_f32().iter() {
        assert!(val.is_nan(), "Expected NaN, got {val}");
    }
}

#[test]
fn test_div_by_negative_zero() {
    // Division by -0.0 should produce -inf (positive / negative_zero)
    let x = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], false);
    let y = RawTensor::new(vec![-0.0, -0.0, -0.0], &[3], false);
    let z = x.div(&y);

    // All results should be -inf
    for &val in z.borrow().data.to_host_f32().iter() {
        assert!(val.is_infinite(), "Expected infinity, got {val}");
        assert!(
            val.is_sign_negative(),
            "Expected negative infinity, got {val}"
        );
    }
}

#[test]
fn test_div_backward_with_zero_denominator() {
    // Test gradient with zero in denominator
    // ∂(x/y)/∂x = 1/y (undefined when y=0, but should still compute)
    let x = RawTensor::new(vec![6.0], &[1], true);
    let y = RawTensor::new(vec![0.0], &[1], true);
    let z = x.div(&y); // z = inf
    z.into_loss().expect("scalar loss").backward();

    // x.grad() = 1/0 = inf
    assert!(
        x.grad()
            .unwrap()
            .first()
            .copied()
            .expect("gradient should not be empty")
            .is_infinite()
    );
    // y.grad() = -x/y² = -6/0 = -inf
    assert!(
        y.grad()
            .unwrap()
            .first()
            .copied()
            .expect("gradient should not be empty")
            .is_infinite()
    );
}

// Square root edge cases
#[test]
fn test_sqrt_of_negative() {
    // sqrt of negative should produce NaN
    let x = RawTensor::new(vec![-1.0, -4.0, -9.0], &[3], false);
    let y = x.sqrt();

    // All results should be NaN
    for &val in y.borrow().data.to_host_f32().iter() {
        assert!(val.is_nan(), "sqrt of negative should be NaN, got {val}");
    }
}

#[test]
fn test_sqrt_of_zero() {
    // sqrt(0.0) = 0.0
    let x = RawTensor::new(vec![0.0, 0.0, 0.0], &[3], false);
    let y = x.sqrt();

    for &val in y.borrow().data.to_host_f32().iter() {
        assert_eq!(val, 0.0, "sqrt(0) should be 0, got {val}");
    }
}

#[test]
fn test_sqrt_gradient_near_zero() {
    // Test that sqrt gradient doesn't panic at zero (it produces inf)
    // ∂sqrt(x)/∂x = 1/(2*sqrt(x)), which is inf when x=0
    let x = RawTensor::new(vec![0.0], &[1], true);
    let y = x.sqrt();
    y.into_loss().expect("scalar loss").backward();

    // Gradient at x=0 is infinite
    assert!(
        x.grad()
            .unwrap()
            .first()
            .copied()
            .expect("gradient should not be empty")
            .is_infinite()
    );
}

#[test]
fn test_sqrt_gradient_negative() {
    // Test sqrt gradient with negative input
    // Forward pass produces NaN, backward should also handle it
    let x = RawTensor::new(vec![-1.0], &[1], true);
    let y = x.sqrt(); // y = NaN

    // Check that forward produces NaN
    assert!(
        y.borrow()
            .data
            .to_host_f32()
            .first()
            .copied()
            .expect("tensor data should not be empty")
            .is_nan()
    );

    // Backward should compute without panicking
    y.into_loss().expect("scalar loss").backward();

    // The gradient formula 1/(2*sqrt(x)) with x=-1 gives NaN
    assert!(
        x.grad()
            .unwrap()
            .first()
            .copied()
            .expect("gradient should not be empty")
            .is_nan()
    );
}

// Logarithm edge cases
#[test]
fn test_log_of_zero() {
    // log(0) should produce -inf
    let x = RawTensor::new(vec![0.0, 0.0, 0.0], &[3], false);
    let y = x.log();

    // All results should be -inf
    for &val in y.borrow().data.to_host_f32().iter() {
        assert!(val.is_infinite(), "log(0) should be infinite, got {val}");
        assert!(val.is_sign_negative(), "log(0) should be -inf, got {val}");
    }
}

#[test]
fn test_log_of_negative() {
    // log of negative should produce NaN
    let x = RawTensor::new(vec![-1.0, -2.0, -3.0], &[3], false);
    let y = x.log();

    // All results should be NaN
    for &val in y.borrow().data.to_host_f32().iter() {
        assert!(val.is_nan(), "log of negative should be NaN, got {val}");
    }
}

#[test]
fn test_log2_of_zero() {
    // log2(0) should produce -inf
    let x = RawTensor::new(vec![0.0, 0.0, 0.0], &[3], false);
    let y = x.log2();

    // All results should be -inf
    for &val in y.borrow().data.to_host_f32().iter() {
        assert!(val.is_infinite(), "log2(0) should be infinite, got {val}");
        assert!(val.is_sign_negative(), "log2(0) should be -inf, got {val}");
    }
}

#[test]
fn test_log2_of_negative() {
    // log2 of negative should produce NaN
    let x = RawTensor::new(vec![-1.0, -2.0, -4.0], &[3], false);
    let y = x.log2();

    // All results should be NaN
    for &val in y.borrow().data.to_host_f32().iter() {
        assert!(val.is_nan(), "log2 of negative should be NaN, got {val}");
    }
}

#[test]
fn test_log_gradient_at_zero() {
    // Test log gradient at zero: ∂log(x)/∂x = 1/x, which is inf at x=0
    let x = RawTensor::new(vec![0.0], &[1], true);
    let y = x.log(); // y = -inf
    y.into_loss().expect("scalar loss").backward();

    // Gradient at x=0 is infinite
    assert!(
        x.grad()
            .unwrap()
            .first()
            .copied()
            .expect("gradient should not be empty")
            .is_infinite()
    );
    assert!(
        x.grad()
            .unwrap()
            .first()
            .copied()
            .expect("gradient should not be empty")
            .is_sign_positive()
    );
}

#[test]
fn test_log2_gradient_at_zero() {
    // Test log2 gradient at zero: ∂log2(x)/∂x = 1/(x*ln(2)), which is inf at x=0
    let x = RawTensor::new(vec![0.0], &[1], true);
    let y = x.log2(); // y = -inf
    y.into_loss().expect("scalar loss").backward();

    // Gradient at x=0 is infinite
    assert!(
        x.grad()
            .unwrap()
            .first()
            .copied()
            .expect("gradient should not be empty")
            .is_infinite()
    );
    assert!(
        x.grad()
            .unwrap()
            .first()
            .copied()
            .expect("gradient should not be empty")
            .is_sign_positive()
    );
}

// NaN/Infinity propagation tests
#[test]
fn test_nan_propagation_add() {
    // NaN should propagate through addition
    let x = RawTensor::new(vec![1.0, f32::NAN, 3.0], &[3], false);
    let y = RawTensor::new(vec![4.0, 5.0, 6.0], &[3], false);
    let z = x.add(&y);

    // NaN + anything = NaN
    assert_eq!(
        z.borrow()
            .data
            .to_host_f32()
            .first()
            .copied()
            .expect("tensor data should not be empty"),
        5.0
    );
    assert!(
        z.borrow()
            .data
            .to_host_f32()
            .get(1)
            .copied()
            .expect("tensor should have element at index 1")
            .is_nan()
    );
    assert_eq!(
        z.borrow()
            .data
            .to_host_f32()
            .get(2)
            .copied()
            .expect("tensor should have element at index 2"),
        9.0
    );
}

#[test]
fn test_nan_propagation_mul() {
    // NaN should propagate through multiplication
    let x = RawTensor::new(vec![2.0, f32::NAN, 4.0], &[3], false);
    let y = RawTensor::new(vec![3.0, 5.0, 6.0], &[3], false);
    let z = x.elem_mul(&y);

    // NaN * anything = NaN
    assert_eq!(
        z.borrow()
            .data
            .to_host_f32()
            .first()
            .copied()
            .expect("tensor data should not be empty"),
        6.0
    );
    assert!(
        z.borrow()
            .data
            .to_host_f32()
            .get(1)
            .copied()
            .expect("tensor should have element at index 1")
            .is_nan()
    );
    assert_eq!(
        z.borrow()
            .data
            .to_host_f32()
            .get(2)
            .copied()
            .expect("tensor should have element at index 2"),
        24.0
    );
}

#[test]
fn test_inf_propagation_add() {
    // Infinity should propagate correctly through addition
    let x = RawTensor::new(vec![1.0, f32::INFINITY, f32::INFINITY], &[3], false);
    let y = RawTensor::new(vec![4.0, 5.0, f32::NEG_INFINITY], &[3], false);
    let z = x.add(&y);

    assert_eq!(
        z.borrow()
            .data
            .to_host_f32()
            .first()
            .copied()
            .expect("tensor data should exist"),
        5.0
    );
    assert!(
        z.borrow()
            .data
            .to_host_f32()
            .get(1)
            .copied()
            .expect("tensor data should exist")
            .is_infinite()
    ); // inf + 5 = inf
    assert!(
        z.borrow()
            .data
            .to_host_f32()
            .get(2)
            .copied()
            .expect("tensor data should exist")
            .is_nan()
    ); // inf + -inf = NaN
}

#[test]
fn test_inf_propagation_mul() {
    // Infinity should propagate correctly through multiplication
    let x = RawTensor::new(vec![2.0, f32::INFINITY, 0.0], &[3], false);
    let y = RawTensor::new(vec![3.0, 5.0, f32::INFINITY], &[3], false);
    let z = x.elem_mul(&y);

    assert_eq!(
        z.borrow()
            .data
            .to_host_f32()
            .first()
            .copied()
            .expect("tensor data should exist"),
        6.0
    );
    assert!(
        z.borrow()
            .data
            .to_host_f32()
            .get(1)
            .copied()
            .expect("tensor data should exist")
            .is_infinite()
    ); // inf * 5 = inf
    assert!(
        z.borrow()
            .data
            .to_host_f32()
            .get(2)
            .copied()
            .expect("tensor data should exist")
            .is_nan()
    ); // 0 * inf = NaN
}

#[test]
fn test_inf_through_sqrt() {
    // sqrt(inf) = inf
    let x = RawTensor::new(vec![f32::INFINITY], &[1], false);
    let y = x.sqrt();

    assert!(
        y.borrow()
            .data
            .to_host_f32()
            .first()
            .copied()
            .expect("tensor data should exist")
            .is_infinite()
    );
    assert!(
        y.borrow()
            .data
            .to_host_f32()
            .first()
            .copied()
            .expect("tensor data should exist")
            .is_sign_positive()
    );
}

#[test]
fn test_inf_through_log() {
    // log(inf) = inf
    let x = RawTensor::new(vec![f32::INFINITY], &[1], false);
    let y = x.log();

    assert!(
        y.borrow()
            .data
            .to_host_f32()
            .first()
            .copied()
            .expect("tensor data should exist")
            .is_infinite()
    );
    assert!(
        y.borrow()
            .data
            .to_host_f32()
            .first()
            .copied()
            .expect("tensor data should exist")
            .is_sign_positive()
    );
}

#[test]
fn test_inf_through_log2() {
    // log2(inf) = inf
    let x = RawTensor::new(vec![f32::INFINITY], &[1], false);
    let y = x.log2();

    assert!(
        y.borrow()
            .data
            .to_host_f32()
            .first()
            .copied()
            .expect("tensor data should exist")
            .is_infinite()
    );
    assert!(
        y.borrow()
            .data
            .to_host_f32()
            .first()
            .copied()
            .expect("tensor data should exist")
            .is_sign_positive()
    );
}

#[test]
fn test_inf_through_exp() {
    // exp(inf) = inf, exp(-inf) = 0
    let x = RawTensor::new(vec![f32::INFINITY, f32::NEG_INFINITY], &[2], false);
    let y = x.exp();

    assert!(
        y.borrow()
            .data
            .to_host_f32()
            .first()
            .copied()
            .expect("tensor data should exist")
            .is_infinite()
    ); // exp(inf) = inf
    assert_eq!(
        y.borrow()
            .data
            .to_host_f32()
            .get(1)
            .copied()
            .expect("tensor data should exist"),
        0.0
    ); // exp(-inf) = 0
}

#[test]
fn test_nan_sqrt_log_chain() {
    // Test that NaN propagates through operation chain
    let x = RawTensor::new(vec![-1.0], &[1], false);
    let y = x.sqrt(); // NaN
    let z = y.log(); // log(NaN) = NaN

    assert!(
        z.borrow()
            .data
            .to_host_f32()
            .first()
            .copied()
            .expect("tensor data should exist")
            .is_nan()
    );
}

#[test]
fn test_zero_operations_chain() {
    // Test chain with zero divisions
    let x = RawTensor::new(vec![1.0, 0.0, 2.0], &[3], false);
    let y = RawTensor::new(vec![0.0, 1.0, 0.0], &[3], false);
    let z = x.div(&y); // [inf, 0, inf]
    let w = z.elem_mul(&z); // [inf, 0, inf] * [inf, 0, inf] = [inf, 0, inf]

    assert!(
        w.borrow()
            .data
            .to_host_f32()
            .first()
            .copied()
            .expect("tensor data should exist")
            .is_infinite()
    );
    assert_eq!(
        w.borrow()
            .data
            .to_host_f32()
            .get(1)
            .copied()
            .expect("tensor data should exist"),
        0.0
    );
    assert!(
        w.borrow()
            .data
            .to_host_f32()
            .get(2)
            .copied()
            .expect("tensor data should exist")
            .is_infinite()
    );
}

// ===== SCALAR TENSOR EDGE CASES =====

#[test]
fn test_scalar_add() {
    // Scalar + Scalar
    let x = RawTensor::new(vec![5.0], &[1], false);
    let y = RawTensor::new(vec![3.0], &[1], false);
    let z = x.add(&y);

    assert_eq!(z.borrow().shape, vec![1]);
    assert_eq!(
        z.borrow()
            .data
            .to_host_f32()
            .first()
            .copied()
            .expect("tensor data should exist"),
        8.0
    );
}

#[test]
fn test_scalar_mul() {
    // Scalar * Scalar
    let x = RawTensor::new(vec![4.0], &[1], false);
    let y = RawTensor::new(vec![7.0], &[1], false);
    let z = x.elem_mul(&y);

    assert_eq!(z.borrow().shape, vec![1]);
    assert_eq!(
        z.borrow()
            .data
            .to_host_f32()
            .first()
            .copied()
            .expect("tensor data should exist"),
        28.0
    );
}

#[test]
fn test_scalar_sum() {
    // Sum of scalar is itself
    let x = RawTensor::new(vec![42.0], &[1], false);
    let s = x.sum();

    assert_eq!(s.borrow().shape, vec![1]);
    assert_eq!(
        s.borrow()
            .data
            .to_host_f32()
            .first()
            .copied()
            .expect("tensor data should exist"),
        42.0
    );
}

#[test]
fn test_scalar_mean() {
    // Mean of scalar is itself
    let x = RawTensor::new(vec![99.0], &[1], false);
    let m = x.mean();

    assert_eq!(m.borrow().shape, vec![1]);
    assert_eq!(
        m.borrow()
            .data
            .to_host_f32()
            .first()
            .copied()
            .expect("tensor data should exist"),
        99.0
    );
}

#[test]
fn test_scalar_max() {
    // Max of scalar is itself
    let x = RawTensor::new(vec![17.0], &[1], false);
    let m = x.max_reduce();

    assert_eq!(m.borrow().shape, vec![1]);
    assert_eq!(
        m.borrow()
            .data
            .to_host_f32()
            .first()
            .copied()
            .expect("tensor data should exist"),
        17.0
    );
}

#[test]
fn test_scalar_reshape() {
    // Scalar can be reshaped to any shape with product 1
    let x = RawTensor::new(vec![5.0], &[1], false);
    let y = x.reshape(&[1, 1, 1]);

    assert_eq!(y.borrow().shape, vec![1, 1, 1]);
    assert_eq!(
        y.borrow()
            .data
            .to_host_f32()
            .first()
            .copied()
            .expect("tensor data should exist"),
        5.0
    );
}

#[test]
#[should_panic(expected = "Transpose expects 2D tensor")]
fn test_scalar_transpose_panics() {
    // Scalar transpose is not supported - only 2D+ tensors
    let x = RawTensor::new(vec![8.0], &[1], false);
    let _y = x.transpose(); // Should panic
}

#[test]
fn test_scalar_gradient() {
    // Gradient computation on scalar (using multiplication)
    let x = RawTensor::new(vec![3.0], &[1], true);
    let two = RawTensor::new(vec![2.0], &[1], false);
    let y = x.elem_mul(&two); // y = 3 * 2 = 6
    y.into_loss().expect("scalar loss").backward();

    // dy/dx = 2
    assert_eq!(
        x.grad()
            .unwrap()
            .first()
            .copied()
            .expect("gradient should exist"),
        2.0
    );
}

#[test]
fn test_scalar_div_by_scalar() {
    // Scalar / Scalar
    let x = RawTensor::new(vec![10.0], &[1], false);
    let y = RawTensor::new(vec![2.0], &[1], false);
    let z = x.div(&y);

    assert_eq!(z.borrow().shape, vec![1]);
    assert_eq!(
        z.borrow()
            .data
            .to_host_f32()
            .first()
            .copied()
            .expect("tensor data should exist"),
        5.0
    );
}

#[test]
fn test_scalar_sqrt() {
    // Sqrt of scalar
    let x = RawTensor::new(vec![16.0], &[1], false);
    let y = x.sqrt();

    assert_eq!(y.borrow().shape, vec![1]);
    assert_eq!(
        y.borrow()
            .data
            .to_host_f32()
            .first()
            .copied()
            .expect("tensor data should exist"),
        4.0
    );
}

// ===== SCALAR BROADCASTING EDGE CASES =====

#[test]
fn test_scalar_broadcasts_to_vector() {
    // Scalar [1] broadcasts to vector [3]
    let scalar = RawTensor::new(vec![5.0], &[1], false);
    let vector = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], false);
    let result = scalar.add(&vector);

    assert_eq!(result.borrow().shape, vec![3]);
    assert_eq!(result.borrow().data, vec![6.0, 7.0, 8.0]);
}

#[test]
fn test_vector_broadcasts_to_scalar() {
    // Vector [3] broadcasts to scalar [1]
    let vector = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], false);
    let scalar = RawTensor::new(vec![10.0], &[1], false);
    let result = vector.add(&scalar);

    assert_eq!(result.borrow().shape, vec![3]);
    assert_eq!(result.borrow().data, vec![11.0, 12.0, 13.0]);
}

#[test]
fn test_scalar_broadcasts_to_matrix() {
    // Scalar [1] broadcasts to matrix [2, 3]
    let scalar = RawTensor::new(vec![2.0], &[1], false);
    let matrix = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    let result = scalar.elem_mul(&matrix);

    assert_eq!(result.borrow().shape, vec![2, 3]);
    assert_eq!(result.borrow().data, vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn test_scalar_broadcast_with_gradient() {
    // Test gradient flow with scalar broadcasting
    let scalar = RawTensor::new(vec![2.0], &[1], true);
    let vector = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], true);
    let result = scalar.elem_mul(&vector); // [2, 4, 6]
    let sum = result.sum(); // 12
    sum.into_loss().expect("scalar loss").backward();

    // d(loss)/d(scalar) = sum of vector elements = 1+2+3 = 6
    assert_eq!(
        scalar
            .grad()
            .unwrap()
            .first()
            .copied()
            .expect("gradient should exist"),
        6.0
    );

    // d(loss)/d(vector) = scalar = 2 for each element
    assert_eq!(vector.grad().unwrap(), vec![2.0, 2.0, 2.0]);
}

#[test]
fn test_scalar_comparison_ops() {
    // Comparison operations with scalars
    let scalar = RawTensor::new(vec![2.0], &[1], false);
    let vector = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], false);

    let lt = scalar.cmplt(&vector); // [2] < [1,2,3] -> [false, false, true]
    assert_eq!(lt.borrow().shape, vec![3]);
    assert_eq!(lt.borrow().data, vec![0.0, 0.0, 1.0]);
}

#[test]
fn test_scalar_max_elem() {
    // Max operation with scalar broadcasting
    let scalar = RawTensor::new(vec![5.0], &[1], false);
    let vector = RawTensor::new(vec![3.0, 7.0, 2.0], &[3], false);
    let result = scalar.max_elem(&vector);

    assert_eq!(result.borrow().shape, vec![3]);
    assert_eq!(result.borrow().data, vec![5.0, 7.0, 5.0]); // max(5,3), max(5,7), max(5,2)
}

// ===== EMPTY TENSOR CREATION ATTEMPTS (SHOULD PANIC) =====

#[test]
#[should_panic(expected = "has a zero dimension")]
fn test_create_empty_tensor_explicit() {
    // Direct attempt to create tensor with zero elements
    let _empty = RawTensor::new(vec![], &[0], false);
}

#[test]
fn test_try_new_zero_dim_is_invalid_parameter() {
    // The zero check lives in `Shape::from_dims`; assert the variant, not the wording.
    let err = RawTensor::try_new(vec![], &[0], false).err().unwrap();
    assert!(matches!(err, VoltaError::InvalidParameter(_)), "{err:?}");
    let err = RawTensor::try_new(vec![], &[0, 5], false).err().unwrap();
    assert!(matches!(err, VoltaError::InvalidParameter(_)), "{err:?}");
}

#[test]
fn test_try_new_over_max_numel_is_invalid_parameter() {
    // Shape is rejected before data is compared, so no 100M-element allocation is needed.
    let err = RawTensor::try_new(vec![], &[crate::shape::MAX_NUMEL + 1], false)
        .err()
        .unwrap();
    assert!(matches!(err, VoltaError::InvalidParameter(_)), "{err:?}");
}

#[test]
fn test_try_new_data_length_mismatch_is_shape_data_mismatch() {
    let err = RawTensor::try_new(vec![1.0], &[2], false).err().unwrap();
    assert!(
        matches!(err, VoltaError::ShapeDataMismatch { .. }),
        "{err:?}"
    );
}

#[test]
#[should_panic(expected = "Expand: Invalid parameter: shape [100000000, 2] has 200000000 elements")]
fn test_expand_over_max_numel_panics() {
    let x = RawTensor::new(vec![1.0, 2.0], &[1, 2], false);
    let _ = x.expand(&[crate::shape::MAX_NUMEL, 2]);
}

#[test]
fn test_try_broadcast_shape_over_max_numel_is_invalid_parameter() {
    // Each operand is valid, but the broadcast result (400M) exceeds the cap.
    let err = RawTensor::try_broadcast_shape(&[20_000, 1], &[1, 20_000]).unwrap_err();
    assert!(matches!(err, VoltaError::InvalidParameter(_)), "{err:?}");
}

#[test]
#[should_panic(expected = "has a zero dimension")]
fn test_create_empty_tensor_2d() {
    // 2D tensor with zero in first dimension
    let _empty = RawTensor::new(vec![], &[0, 5], false);
}

#[test]
#[should_panic(expected = "shape [0] has a zero dimension")]
fn test_create_empty_tensor_with_data() {
    // Attempt to create tensor with mismatched data (empty data, non-empty shape)
    let _empty = RawTensor::new(vec![1.0], &[0], false);
}

#[test]
#[should_panic(expected = "shape [0] has a zero dimension")]
fn test_reshape_to_empty() {
    // Attempt to reshape non-empty tensor to empty shape
    let x = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], false);
    let _empty = x.reshape(&[0]); // zero dimension: rejected as an invalid shape, not as a size mismatch
}

#[test]
fn test_operation_resulting_in_zero_valid() {
    // Operations that result in value 0 are fine (not empty tensor)
    let x = RawTensor::new(vec![5.0], &[1], false);
    let y = RawTensor::new(vec![5.0], &[1], false);
    let z = x.sub(&y); // Result is 0.0, but still has 1 element

    assert_eq!(z.borrow().shape, vec![1]);
    assert_eq!(
        z.borrow()
            .data
            .to_host_f32()
            .first()
            .copied()
            .expect("tensor data should exist"),
        0.0
    );
}

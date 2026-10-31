use crate::error::{Result, VoltaError};
use crate::index::slot;
use crate::shape::MAX_NUMEL;

/// Normalize data in-place: (x - mean) / std
pub fn normalize(data: &mut [f32], mean: f32, std: f32) {
    for x in data.iter_mut() {
        *x = (*x - mean) / std;
    }
}

/// Convert label indices to one-hot encoding
/// labels: [0, 1, 2, ...] -> one-hot vectors concatenated
///
/// # Errors
///
/// `VoltaError::InvalidParameter` if `num_classes` is zero, if the output would have
/// more than `MAX_NUMEL` elements (checked before it is allocated), or if any label is
/// `>= num_classes`. Labels are data from a file or a user, so a bad one is an error,
/// not a panic; it would otherwise set a slot in the next sample's row, or be dropped
/// on the last one.
pub fn to_one_hot(labels: &[u8], num_classes: usize) -> Result<Vec<f32>> {
    let too_big = || {
        VoltaError::InvalidParameter(format!(
            "one-hot of {} labels x {num_classes} classes is zero-sized or over {MAX_NUMEL} elements",
            labels.len()
        ))
    };
    let total = labels
        .len()
        .checked_mul(num_classes)
        .filter(|&n| n <= MAX_NUMEL)
        .ok_or_else(too_big)?;
    if num_classes == 0 {
        return Err(too_big());
    }
    let mut one_hot = vec![0.0; total];

    for (i, &label) in labels.iter().enumerate() {
        if usize::from(label) >= num_classes {
            return Err(VoltaError::InvalidParameter(format!(
                "label {label} at index {i} is out of range for {num_classes} classes"
            )));
        }
        *slot(&mut one_hot, i * num_classes + usize::from(label)) = 1.0;
    }

    Ok(one_hot)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::fetch;

    #[test]
    fn test_normalize() {
        let mut data = vec![0.0, 0.5, 1.0];
        normalize(&mut data, 0.5, 0.5);

        assert_eq!(data, vec![-1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_to_one_hot() {
        let labels = vec![0, 2, 1];
        let one_hot = to_one_hot(&labels, 3).unwrap();

        // Expected: [1,0,0, 0,0,1, 0,1,0]
        assert_eq!(one_hot.len(), 9);
        assert_eq!(fetch(&one_hot, 0), 1.0); // label 0
        assert_eq!(fetch(&one_hot, 5), 1.0); // label 2
        assert_eq!(fetch(&one_hot, 7), 1.0); // label 1
        assert_eq!(one_hot.iter().sum::<f32>(), 3.0);
    }

    /// A label past `num_classes` is an error, not a slot in the next sample's row.
    #[test]
    fn test_to_one_hot_label_out_of_range_is_an_error() {
        let err = to_one_hot(&[0, 3, 1], 3).unwrap_err();
        assert!(matches!(err, VoltaError::InvalidParameter(_)), "{err:?}");
        assert!(err.to_string().contains("label 3"), "{err}");
        assert!(err.to_string().contains("index 1"), "{err}");
    }

    /// `labels.len() * num_classes` is checked before the output is allocated.
    #[test]
    fn test_to_one_hot_rejects_a_size_that_overflows_or_is_huge() {
        assert!(matches!(
            to_one_hot(&[0, 1], usize::MAX),
            Err(VoltaError::InvalidParameter(_))
        ));
        assert!(matches!(
            to_one_hot(&[0], crate::shape::MAX_NUMEL + 1),
            Err(VoltaError::InvalidParameter(_))
        ));
        assert!(matches!(
            to_one_hot(&[0], 0),
            Err(VoltaError::InvalidParameter(_))
        ));
    }
}

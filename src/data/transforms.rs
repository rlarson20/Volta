use crate::index::slot;

/// Normalize data in-place: (x - mean) / std
pub fn normalize(data: &mut [f32], mean: f32, std: f32) {
    for x in data.iter_mut() {
        *x = (*x - mean) / std;
    }
}

/// Convert label indices to one-hot encoding
/// labels: [0, 1, 2, ...] -> one-hot vectors concatenated
///
/// # Panics
///
/// If any label is `>= num_classes`. Such a label would otherwise set a slot in the
/// next sample's row, or be dropped on the last one.
#[must_use]
pub fn to_one_hot(labels: &[u8], num_classes: usize) -> Vec<f32> {
    let mut one_hot = vec![0.0; labels.len() * num_classes];

    for (i, &label) in labels.iter().enumerate() {
        let offset = i * num_classes;
        assert!(
            (label as usize) < num_classes,
            "label {label} out of range for {num_classes} classes"
        );
        *slot(&mut one_hot, offset + label as usize) = 1.0;
    }

    one_hot
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
        let one_hot = to_one_hot(&labels, 3);

        // Expected: [1,0,0, 0,0,1, 0,1,0]
        assert_eq!(one_hot.len(), 9);
        assert_eq!(fetch(&one_hot, 0), 1.0); // label 0
        assert_eq!(fetch(&one_hot, 5), 1.0); // label 2
        assert_eq!(fetch(&one_hot, 7), 1.0); // label 1
        assert_eq!(one_hot.iter().sum::<f32>(), 3.0);
    }

    /// A label past `num_classes` panics instead of landing in the next sample's row.
    #[test]
    #[should_panic(expected = "out of range")]
    fn test_to_one_hot_label_out_of_range_panics() {
        let _ = to_one_hot(&[0, 3, 1], 3);
    }
}

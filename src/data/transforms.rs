/// Normalize data in-place: (x - mean) / std
pub fn normalize(data: &mut [f32], mean: f32, std: f32) {
    for x in data.iter_mut() {
        *x = (*x - mean) / std;
    }
}

/// Convert label indices to one-hot encoding
/// labels: [0, 1, 2, ...] -> one-hot vectors concatenated
#[must_use]
pub fn to_one_hot(labels: &[u8], num_classes: usize) -> Vec<f32> {
    let mut one_hot = vec![0.0; labels.len() * num_classes];

    for (i, &label) in labels.iter().enumerate() {
        let offset = i * num_classes;
        let idx = offset + label as usize;
        if let Some(slot) = one_hot.get_mut(idx) {
            *slot = 1.0;
        }
    }

    one_hot
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(one_hot.first().copied().unwrap_or(f32::NAN), 1.0); // label 0
        assert_eq!(one_hot.get(5).copied().unwrap_or(f32::NAN), 1.0); // label 2
        assert_eq!(one_hot.get(7).copied().unwrap_or(f32::NAN), 1.0); // label 1
    }
}

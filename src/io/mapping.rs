use crate::io::StateDict;

/// Builder for composable state dict transformations
///
/// StateDictMapper allows you to transform state dicts when loading external models.
/// Transformations are applied in the order they are added.
///
/// # Examples
///
/// ```no_run
/// use volta::io::{load_safetensors, mapping::StateDictMapper};
///
/// // Load PyTorch weights with key renaming and transposition
/// let pytorch_state = load_safetensors("model.safetensors")?;
/// let mapper = StateDictMapper::new()
///     .rename("fc1.weight", "encoder.weight")
///     .transpose("encoder.weight")  // PyTorch stores [out, in], Volta uses [in, out]
///     .strip_prefix("model.");
///
/// let volta_state = mapper.map(pytorch_state);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub type TransformationBox = Box<dyn Fn(&mut StateDict)>;
pub struct StateDictMapper {
    transformations: Vec<TransformationBox>,
}

impl StateDictMapper {
    /// Create a new empty mapper
    #[must_use]
    pub fn new() -> Self {
        Self {
            transformations: Vec::new(),
        }
    }

    /// Rename a single key
    ///
    /// If the key doesn't exist, this operation is a no-op.
    pub fn rename(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        let from = from.into();
        let to = to.into();

        self.transformations
            .push(Box::new(move |state: &mut StateDict| {
                if let Some(value) = state.remove(&from) {
                    state.insert(to.clone(), value);
                }
            }));
        self
    }

    /// Rename all keys with a given prefix
    ///
    /// Example: `rename_prefix("old.", "new.")` changes "old.weight" to "new.weight"
    pub fn rename_prefix(
        mut self,
        old_prefix: impl Into<String>,
        new_prefix: impl Into<String>,
    ) -> Self {
        let old_prefix = old_prefix.into();
        let new_prefix = new_prefix.into();

        self.transformations
            .push(Box::new(move |state: &mut StateDict| {
                let mut updates = Vec::new();
                for (key, value) in state.iter() {
                    if key.starts_with(&old_prefix) {
                        let suffix = &key[old_prefix.len()..];
                        updates.push((
                            key.clone(),
                            format!("{}{}", new_prefix, suffix),
                            value.clone(),
                        ));
                    }
                }
                for (old_key, new_key, value) in updates {
                    state.remove(&old_key);
                    state.insert(new_key, value);
                }
            }));
        self
    }

    /// Strip a prefix from all keys
    ///
    /// Example: `strip_prefix("model.")` changes "model.encoder.weight" to "encoder.weight"
    pub fn strip_prefix(mut self, prefix: impl Into<String>) -> Self {
        let prefix = prefix.into();

        self.transformations
            .push(Box::new(move |state: &mut StateDict| {
                let mut updates = Vec::new();
                for (key, value) in state.iter() {
                    if let Some(stripped) = key.strip_prefix(&prefix) {
                        updates.push((key.clone(), stripped.to_string(), value.clone()));
                    }
                }
                for (old_key, new_key, value) in updates {
                    state.remove(&old_key);
                    state.insert(new_key, value);
                }
            }));
        self
    }

    /// Add a prefix to all keys
    ///
    /// Example: `add_prefix("encoder.")` changes "weight" to "encoder.weight"
    pub fn add_prefix(mut self, prefix: impl Into<String>) -> Self {
        let prefix = prefix.into();

        self.transformations
            .push(Box::new(move |state: &mut StateDict| {
                let mut updates = Vec::new();
                for (key, value) in state.iter() {
                    let new_key = format!("{}{}", prefix, key);
                    updates.push((key.clone(), new_key, value.clone()));
                }
                for (old_key, new_key, value) in updates {
                    state.remove(&old_key);
                    state.insert(new_key, value);
                }
            }));
        self
    }

    /// Transpose a specific 2D tensor (for Linear weight conversion)
    ///
    /// PyTorch Linear layers store weights as [out_features, in_features],
    /// while Volta stores them as [in_features, out_features].
    ///
    /// This operation only affects 2D tensors. Non-2D tensors are left unchanged.
    pub fn transpose(mut self, key: impl Into<String>) -> Self {
        let key = key.into();

        self.transformations
            .push(Box::new(move |state: &mut StateDict| {
                if let Some(tensor_data) = state.get_mut(&key)
                    && tensor_data.shape.len() == 2
                {
                    let [rows, cols] = [tensor_data.shape[0], tensor_data.shape[1]];
                    let mut transposed = vec![0.0; rows * cols];

                    // Transpose row-major layout: [R, C] -> [C, R]
                    for i in 0..rows {
                        for j in 0..cols {
                            transposed[j * rows + i] = tensor_data.data[i * cols + j];
                        }
                    }

                    tensor_data.data = transposed;
                    tensor_data.shape = vec![cols, rows];
                }
            }));
        self
    }

    /// Transpose all tensors matching a pattern
    ///
    /// Example: `transpose_pattern("weight")` transposes all keys containing "weight"
    pub fn transpose_pattern(mut self, pattern: impl Into<String>) -> Self {
        let pattern = pattern.into();

        self.transformations
            .push(Box::new(move |state: &mut StateDict| {
                let matching_keys: Vec<String> = state
                    .keys()
                    .filter(|k| k.contains(&pattern))
                    .cloned()
                    .collect();

                for key in matching_keys {
                    if let Some(tensor_data) = state.get_mut(&key)
                        && tensor_data.shape.len() == 2
                    {
                        let [rows, cols] = [tensor_data.shape[0], tensor_data.shape[1]];
                        let mut transposed = vec![0.0; rows * cols];

                        for i in 0..rows {
                            for j in 0..cols {
                                transposed[j * rows + i] = tensor_data.data[i * cols + j];
                            }
                        }

                        tensor_data.data = transposed;
                        tensor_data.shape = vec![cols, rows];
                    }
                }
            }));
        self
    }

    /// Select only specific keys (for partial loading)
    ///
    /// All other keys are removed from the state dict.
    #[must_use]
    pub fn select_keys(mut self, keys: Vec<String>) -> Self {
        self.transformations
            .push(Box::new(move |state: &mut StateDict| {
                state.retain(|k, _| keys.contains(k));
            }));
        self
    }

    /// Exclude specific keys
    ///
    /// The specified keys are removed from the state dict.
    #[must_use]
    pub fn exclude_keys(mut self, keys: Vec<String>) -> Self {
        self.transformations
            .push(Box::new(move |state: &mut StateDict| {
                state.retain(|k, _| !keys.contains(k));
            }));
        self
    }

    /// Apply a custom transformation function
    ///
    /// This allows arbitrary transformations beyond the built-in methods.
    pub fn transform<F>(mut self, f: F) -> Self
    where
        F: Fn(&mut StateDict) + 'static,
    {
        self.transformations.push(Box::new(f));
        self
    }

    /// Apply all transformations to a state dict (in-place)
    pub fn apply(&self, state: &mut StateDict) {
        for transform in &self.transformations {
            transform(state);
        }
    }

    /// Apply transformations and return new state dict
    #[must_use]
    pub fn map(&self, mut state: StateDict) -> StateDict {
        self.apply(&mut state);
        state
    }
}

impl Default for StateDictMapper {
    fn default() -> Self {
        Self::new()
    }
}

/// Load PyTorch Linear weights (requires transpose)
///
/// PyTorch Linear layers store weights as [out_features, in_features],
/// while Volta stores them as [in_features, out_features].
///
/// This convenience function transposes all keys containing "weight".
#[must_use]
pub fn load_pytorch_linear_weights(state: StateDict) -> StateDict {
    StateDictMapper::new()
        .transpose_pattern("weight")
        .map(state)
}

/// Load HuggingFace transformer weights
///
/// Strips the model prefix and transposes Linear weights.
///
/// # Arguments
/// * `state` - The loaded state dict from HuggingFace
/// * `model_prefix` - The prefix to strip (e.g., "model", "bert", "transformer")
#[must_use]
pub fn load_huggingface_weights(state: StateDict, model_prefix: &str) -> StateDict {
    StateDictMapper::new()
        .strip_prefix(format!("{}.", model_prefix))
        .transpose_pattern("weight")
        .map(state)
}

/// Create a mapper from key-to-key mappings
///
/// # Example
/// ```
/// use volta::io::mapping::create_key_mapping;
///
/// let mapper = create_key_mapping(vec![
///     ("fc1.weight", "encoder.weight"),
///     ("fc1.bias", "encoder.bias"),
///     ("fc2.weight", "decoder.weight"),
///     ("fc2.bias", "decoder.bias"),
/// ]);
/// ```
#[must_use]
pub fn create_key_mapping(mappings: Vec<(&str, &str)>) -> StateDictMapper {
    let mut mapper = StateDictMapper::new();
    for (from, to) in mappings {
        mapper = mapper.rename(from, to);
    }
    mapper
}

/// Load partial state dict (for transfer learning)
///
/// Strips a prefix and keeps only those keys.
#[must_use]
pub fn load_partial(state: StateDict, prefix: &str) -> StateDict {
    StateDictMapper::new()
        .strip_prefix(format!("{}.", prefix))
        .map(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::TensorData;
    use std::collections::BTreeMap;

    fn create_test_state() -> StateDict {
        let mut state = BTreeMap::new();
        state.insert(
            "weight".to_string(),
            TensorData {
                data: vec![1.0, 2.0, 3.0, 4.0],
                shape: vec![2, 2],
            },
        );
        state
    }

    #[test]
    fn test_rename_single_key() {
        let state = create_test_state();
        let mapper = StateDictMapper::new().rename("weight", "new_weight");
        let mapped = mapper.map(state);

        assert!(mapped.contains_key("new_weight"));
        assert!(!mapped.contains_key("weight"));
    }

    #[test]
    fn test_rename_nonexistent_key() {
        let state = create_test_state();
        let mapper = StateDictMapper::new().rename("nonexistent", "new_key");
        let mapped = mapper.map(state);

        // Should not affect the state if key doesn't exist
        assert!(mapped.contains_key("weight"));
        assert!(!mapped.contains_key("new_key"));
    }

    #[test]
    fn test_transpose_2d() {
        let mut state = BTreeMap::new();
        state.insert(
            "fc.weight".to_string(),
            TensorData {
                data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                shape: vec![2, 3], // [2 rows, 3 cols]
            },
        );

        let mapper = StateDictMapper::new().transpose("fc.weight");
        let mapped = mapper.map(state);

        let weight = mapped.get("fc.weight").unwrap();
        assert_eq!(weight.shape, vec![3, 2]); // Transposed
        // Original: [1,2,3, 4,5,6] (row-major)
        // Transposed: [1,4, 2,5, 3,6] (row-major)
        assert_eq!(weight.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_non_2d_is_noop() {
        let mut state = BTreeMap::new();
        state.insert(
            "bias".to_string(),
            TensorData {
                data: vec![1.0, 2.0, 3.0],
                shape: vec![3], // 1D tensor
            },
        );

        let mapper = StateDictMapper::new().transpose("bias");
        let mapped = mapper.map(state);

        let bias = mapped.get("bias").unwrap();
        assert_eq!(bias.shape, vec![3]); // Unchanged
        assert_eq!(bias.data, vec![1.0, 2.0, 3.0]); // Unchanged
    }

    #[test]
    fn test_strip_prefix() {
        let mut state = BTreeMap::new();
        state.insert(
            "model.fc1.weight".to_string(),
            TensorData {
                data: vec![1.0],
                shape: vec![1],
            },
        );
        state.insert(
            "model.fc2.weight".to_string(),
            TensorData {
                data: vec![2.0],
                shape: vec![1],
            },
        );

        let mapper = StateDictMapper::new().strip_prefix("model.");
        let mapped = mapper.map(state);

        assert!(mapped.contains_key("fc1.weight"));
        assert!(mapped.contains_key("fc2.weight"));
        assert!(!mapped.contains_key("model.fc1.weight"));
    }

    #[test]
    fn test_add_prefix() {
        let state = create_test_state();
        let mapper = StateDictMapper::new().add_prefix("encoder.");
        let mapped = mapper.map(state);

        assert!(mapped.contains_key("encoder.weight"));
        assert!(!mapped.contains_key("weight"));
    }

    #[test]
    fn test_select_keys() {
        let mut state = BTreeMap::new();
        state.insert(
            "weight1".to_string(),
            TensorData {
                data: vec![1.0],
                shape: vec![1],
            },
        );
        state.insert(
            "weight2".to_string(),
            TensorData {
                data: vec![2.0],
                shape: vec![1],
            },
        );
        state.insert(
            "weight3".to_string(),
            TensorData {
                data: vec![3.0],
                shape: vec![1],
            },
        );

        let mapper =
            StateDictMapper::new().select_keys(vec!["weight1".to_string(), "weight2".to_string()]);
        let mapped = mapper.map(state);

        assert_eq!(mapped.len(), 2);
        assert!(mapped.contains_key("weight1"));
        assert!(mapped.contains_key("weight2"));
        assert!(!mapped.contains_key("weight3"));
    }

    #[test]
    fn test_exclude_keys() {
        let mut state = BTreeMap::new();
        state.insert(
            "weight1".to_string(),
            TensorData {
                data: vec![1.0],
                shape: vec![1],
            },
        );
        state.insert(
            "weight2".to_string(),
            TensorData {
                data: vec![2.0],
                shape: vec![1],
            },
        );

        let mapper = StateDictMapper::new().exclude_keys(vec!["weight2".to_string()]);
        let mapped = mapper.map(state);

        assert_eq!(mapped.len(), 1);
        assert!(mapped.contains_key("weight1"));
        assert!(!mapped.contains_key("weight2"));
    }

    #[test]
    fn test_chained_transformations() {
        let mut state = BTreeMap::new();
        state.insert(
            "model.encoder.fc.weight".to_string(),
            TensorData {
                data: vec![1.0, 2.0, 3.0, 4.0],
                shape: vec![2, 2],
            },
        );

        let mapper = StateDictMapper::new()
            .strip_prefix("model.")
            .rename_prefix("encoder.", "enc.")
            .transpose("enc.fc.weight");

        let mapped = mapper.map(state);

        assert!(mapped.contains_key("enc.fc.weight"));
        let weight = mapped.get("enc.fc.weight").unwrap();
        assert_eq!(weight.shape, vec![2, 2]); // Transposed (but 2x2 is symmetric)
    }

    #[test]
    fn test_transpose_pattern() {
        let mut state = BTreeMap::new();
        state.insert(
            "fc1.weight".to_string(),
            TensorData {
                data: vec![1.0, 2.0, 3.0, 4.0],
                shape: vec![2, 2],
            },
        );
        state.insert(
            "fc1.bias".to_string(),
            TensorData {
                data: vec![1.0, 2.0],
                shape: vec![2],
            },
        );
        state.insert(
            "fc2.weight".to_string(),
            TensorData {
                data: vec![5.0, 6.0, 7.0, 8.0],
                shape: vec![2, 2],
            },
        );

        let mapper = StateDictMapper::new().transpose_pattern("weight");
        let mapped = mapper.map(state);

        // Both "weight" keys should be transposed, bias should not
        assert_eq!(mapped.get("fc1.weight").unwrap().shape, vec![2, 2]);
        assert_eq!(mapped.get("fc2.weight").unwrap().shape, vec![2, 2]);
        assert_eq!(mapped.get("fc1.bias").unwrap().shape, vec![2]); // Unchanged
    }

    #[test]
    fn test_load_pytorch_linear_weights() {
        let mut state = BTreeMap::new();
        state.insert(
            "linear.weight".to_string(),
            TensorData {
                data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                shape: vec![2, 3], // PyTorch: [out=2, in=3]
            },
        );

        let mapped = load_pytorch_linear_weights(state);

        let weight = mapped.get("linear.weight").unwrap();
        assert_eq!(weight.shape, vec![3, 2]); // Volta: [in=3, out=2]
    }
}

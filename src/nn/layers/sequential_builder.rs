use super::sequential::LayerEntry;
use crate::nn::{Layer, Sequential};

/// Builder for constructing Sequential models with named or unnamed layers
///
/// # Examples
///
/// ```
/// use volta::nn::layers::{Sequential, SequentialBuilder, Linear, ReLU};
///
/// // Create a Sequential with named layers
/// let model = Sequential::builder()
///     .add_named("encoder", Box::new(Linear::new(784, 128, true)))
///     .add_unnamed(Box::new(ReLU))  // Unnamed activation
///     .add_named("decoder", Box::new(Linear::new(128, 10, true)))
///     .build();
/// ```
pub struct SequentialBuilder {
    entries: Vec<LayerEntry>,
}

impl SequentialBuilder {
    /// Create a new empty builder
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add an unnamed layer to the sequence
    ///
    /// The layer will be assigned a numeric index in the state dict
    #[must_use]
    pub fn add_unnamed(mut self, layer: Box<dyn Layer>) -> Self {
        self.entries.push(LayerEntry { name: None, layer });
        self
    }

    /// Add a named layer to the sequence
    ///
    /// The layer will use the provided name in the state dict.
    /// Empty strings are treated as unnamed.
    #[must_use]
    pub fn add_named(mut self, name: impl Into<String>, layer: Box<dyn Layer>) -> Self {
        let name_str = name.into();
        let name_opt = if name_str.is_empty() {
            None
        } else {
            Some(name_str)
        };

        self.entries.push(LayerEntry {
            name: name_opt,
            layer,
        });
        self
    }

    /// Build the Sequential model from the accumulated layers
    ///
    /// # Panics
    /// If a layer name would make two layers share `state_dict` keys, which silently
    /// drops one layer's weights on save: a name used twice, a name that is a decimal
    /// number (unnamed layers use their index as the key prefix), or a name containing
    /// `.` (the key separator, so `"a.b"` collides with layer `"b"` nested in `"a"`).
    #[must_use]
    pub fn build(self) -> Sequential {
        let mut seen = std::collections::HashSet::new();
        for name in self.entries.iter().filter_map(|e| e.name.as_deref()) {
            assert!(
                seen.insert(name),
                "Sequential: layer name {name:?} is used twice; their state_dict keys would collide"
            );
            assert!(
                !name.bytes().all(|b| b.is_ascii_digit()),
                "Sequential: layer name {name:?} is a number, which is reserved for unnamed layer indices"
            );
            assert!(
                !name.contains('.'),
                "Sequential: layer name {name:?} contains '.', the state_dict key separator"
            );
        }
        Sequential {
            layers: self.entries,
        }
    }
}

impl Default for SequentialBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Module;
    use crate::nn::layers::{Linear, ReLU};
    use crate::tensor::Tensor;

    #[test]
    fn test_builder_empty() {
        let model = SequentialBuilder::new().build();
        assert_eq!(model.len(), 0);
    }

    #[test]
    fn test_builder_unnamed() {
        let model = SequentialBuilder::new()
            .add_unnamed(Box::new(Linear::new(2, 3, true)))
            .add_unnamed(Box::new(ReLU))
            .build();

        assert_eq!(model.len(), 2);
        let names = model.layer_names();
        assert_eq!(names.first().copied().unwrap_or(None), None);
        assert_eq!(names.get(1).copied().unwrap_or(None), None);
    }

    #[test]
    fn test_builder_named() {
        let model = SequentialBuilder::new()
            .add_named("encoder", Box::new(Linear::new(2, 3, true)))
            .add_named("decoder", Box::new(Linear::new(3, 1, true)))
            .build();

        assert_eq!(model.len(), 2);
        let names = model.layer_names();
        assert_eq!(names.first().copied().unwrap_or(None), Some("encoder"));
        assert_eq!(names.get(1).copied().unwrap_or(None), Some("decoder"));
    }

    #[test]
    fn test_builder_mixed() {
        let model = SequentialBuilder::new()
            .add_named("fc1", Box::new(Linear::new(2, 3, true)))
            .add_unnamed(Box::new(ReLU))
            .add_named("fc2", Box::new(Linear::new(3, 1, true)))
            .build();

        assert_eq!(model.len(), 3);
        let names = model.layer_names();
        assert_eq!(names.first().copied().unwrap_or(None), Some("fc1"));
        assert_eq!(names.get(1).copied().unwrap_or(None), None);
        assert_eq!(names.get(2).copied().unwrap_or(None), Some("fc2"));
    }

    #[test]
    fn test_builder_empty_string_name() {
        let model = SequentialBuilder::new()
            .add_named("", Box::new(Linear::new(2, 3, true)))
            .build();

        assert_eq!(model.len(), 1);
        let names = model.layer_names();
        assert_eq!(names.first().copied().unwrap_or(None), None); // Empty string treated as unnamed
    }

    /// Every parameter's host data, in order.
    fn param_values(model: &Sequential) -> Vec<Vec<f32>> {
        model.parameters().iter().map(Tensor::data).collect()
    }

    /// Twelve layers puts `"1."` next to `"10."` and `"11."`. The prefix includes the dot,
    /// so each layer loads only its own keys (the greedy-prefix report did not reproduce).
    #[test]
    fn test_state_dict_round_trip_past_ten_layers() {
        let make = || {
            (0..12)
                .fold(SequentialBuilder::new(), |b, i| {
                    if i % 3 == 0 {
                        b.add_named(format!("fc{i}"), Box::new(Linear::new(3, 3, true)))
                    } else {
                        b.add_unnamed(Box::new(Linear::new(3, 3, true)))
                    }
                })
                .build()
        };
        let source = make();
        let mut dest = make();
        assert_ne!(
            param_values(&source),
            param_values(&dest),
            "setup: random init"
        );

        let state = source.state_dict();
        assert_eq!(state.len(), 24, "one weight and one bias per layer");
        dest.load_state_dict(&state).expect("state loads");
        assert_eq!(param_values(&source), param_values(&dest));
    }

    /// The key format: a layer's name, or its index when unnamed, then the layer's own key.
    #[test]
    fn test_state_dict_keys() {
        let model = SequentialBuilder::new()
            .add_named("encoder", Box::new(Linear::new(2, 3, true)))
            .add_unnamed(Box::new(ReLU))
            .add_unnamed(Box::new(Linear::new(3, 1, false)))
            .build();
        let state = model.state_dict();
        let keys: Vec<&str> = state.keys().map(String::as_str).collect();
        assert_eq!(keys, vec!["2.weight", "encoder.bias", "encoder.weight"]);
    }

    #[test]
    #[should_panic(expected = "is used twice")]
    fn test_builder_rejects_duplicate_name() {
        let _ = SequentialBuilder::new()
            .add_named("fc", Box::new(Linear::new(3, 3, true)))
            .add_named("fc", Box::new(Linear::new(3, 3, true)))
            .build();
    }

    #[test]
    #[should_panic(expected = "is a number")]
    fn test_builder_rejects_numeric_name() {
        let _ = SequentialBuilder::new()
            .add_named("1", Box::new(Linear::new(3, 3, true)))
            .add_unnamed(Box::new(Linear::new(3, 3, true)))
            .build();
    }

    #[test]
    #[should_panic(expected = "contains '.'")]
    fn test_builder_rejects_dotted_name() {
        let inner = SequentialBuilder::new()
            .add_named("b", Box::new(Linear::new(3, 3, true)))
            .build();
        let _ = SequentialBuilder::new()
            .add_named("a", Box::new(inner))
            .add_named("a.b", Box::new(Linear::new(3, 3, true)))
            .build();
    }
}

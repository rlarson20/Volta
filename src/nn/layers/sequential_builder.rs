use super::sequential::LayerEntry;
use crate::nn::{Module, Sequential};

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
    pub fn add_unnamed(mut self, layer: Box<dyn Module>) -> Self {
        self.entries.push(LayerEntry { name: None, layer });
        self
    }

    /// Add a named layer to the sequence
    ///
    /// The layer will use the provided name in the state dict.
    /// Empty strings are treated as unnamed.
    #[must_use]
    pub fn add_named(mut self, name: impl Into<String>, layer: Box<dyn Module>) -> Self {
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
    #[must_use]
    pub fn build(self) -> Sequential {
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
    use crate::nn::layers::{Linear, ReLU};

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
}

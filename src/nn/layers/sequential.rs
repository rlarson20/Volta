use crate::io::StateDict;
use crate::nn::{Layer, LoadPlan, Module, insert_prefixed};
use crate::tensor::Tensor;

/// Internal struct to hold a layer with an optional name
pub(crate) struct LayerEntry {
    /// Name used as the state-dict prefix; the index is used if `None`.
    pub(crate) name: Option<String>,
    /// The layer itself.
    pub(crate) layer: Box<dyn Layer>,
}

/// Layers applied in order, each one's output feeding the next.
pub struct Sequential {
    /// The layers, with their optional names.
    pub(crate) layers: Vec<LayerEntry>,
}

/// The layer names (`None` for an unnamed layer) and the parameter count; `dyn Layer` has
/// no `Debug`.
impl std::fmt::Debug for Sequential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sequential")
            .field("layers", &self.layer_names())
            .field("parameters", &self.parameters().len())
            .finish()
    }
}

/// The prefix layer `i` loads from: a named layer reads its name's entries, and its
/// index's if the state dict has none under the name.
fn load_prefix(entry: &LayerEntry, i: usize, plan: &LoadPlan<'_>) -> String {
    match &entry.name {
        Some(name) if plan.has_entries_under(&format!("{name}.")) => format!("{name}."),
        _ => format!("{i}."),
    }
}

impl Module for Sequential {
    /// Forwards to each child, so a child's buffers are converted too (the default would
    /// convert only the parameters).
    fn to_dtype(&mut self, dtype: crate::DType) {
        for entry in &mut self.layers {
            entry.layer.to_dtype(dtype);
        }
    }

    /// Forwards to each child, so a child's buffers move too (`PositionalEncoding`'s
    /// table); the default would move only the parameters (S13).
    fn to_device(&mut self, device: crate::Device) {
        for entry in &mut self.layers {
            entry.layer.to_device(device.clone());
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        crate::nn::unique_parameters(self.layers.iter().flat_map(|e| e.layer.parameters()))
    }

    fn state_dict(&self) -> StateDict {
        let mut state = StateDict::new();
        for (i, entry) in self.layers.iter().enumerate() {
            // Under the layer's name if it has one, else its index.
            let prefix = match &entry.name {
                Some(name) => format!("{name}."),
                None => format!("{i}."),
            };
            insert_prefixed(&mut state, &prefix, entry.layer.state_dict());
        }
        state
    }

    /// Each layer under its prefix: a named layer's name, or its index when the state
    /// dict has no entry under the name. So a checkpoint saved from an unnamed model
    /// loads, strictly too, into a named one.
    fn plan_load<'a>(&self, plan: &mut LoadPlan<'a>) {
        for (i, entry) in self.layers.iter().enumerate() {
            let prefix = load_prefix(entry, i, plan);
            plan.child(&prefix, entry.layer.as_ref());
        }
    }
    fn train(&mut self, mode: bool) {
        for entry in &mut self.layers {
            entry.layer.train(mode);
        }
    }
}

impl Layer for Sequential {
    fn forward(&self, x: &Tensor) -> Tensor {
        let mut current = x.clone();
        for entry in &self.layers {
            current = entry.layer.forward(&current);
        }
        current
    }
}

impl Sequential {
    // Helper constructor for easier testing and building
    /// Builds a container of unnamed layers; state-dict keys are prefixed with each layer's index.
    #[must_use]
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        Self {
            layers: layers
                .into_iter()
                .map(|layer| LayerEntry { name: None, layer })
                .collect(),
        }
    }

    /// Create a new `SequentialBuilder` for building models with named layers
    #[must_use]
    pub const fn builder() -> crate::nn::layers::SequentialBuilder {
        crate::nn::layers::SequentialBuilder::new()
    }

    /// Get a layer by index
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&dyn Layer> {
        self.layers.get(index).map(|e| &*e.layer)
    }

    /// Get a layer by name
    ///
    /// Returns None if no layer with the given name exists
    #[must_use]
    pub fn get_named(&self, name: &str) -> Option<&dyn Layer> {
        self.layers
            .iter()
            .find(|e| e.name.as_ref().is_some_and(|n| n == name))
            .map(|e| &*e.layer)
    }

    /// Get a mutable reference to a layer by name
    ///
    /// Returns None if no layer with the given name exists
    pub fn get_named_mut(&mut self, name: &str) -> Option<&mut (dyn Layer + '_)> {
        for entry in &mut self.layers {
            if entry.name.as_ref().is_some_and(|n| n == name) {
                return Some(&mut *entry.layer);
            }
        }
        None
    }

    /// Get the names of all layers
    ///
    /// Returns a Vec where Some(name) indicates a named layer and None indicates an unnamed layer
    #[must_use]
    pub fn layer_names(&self) -> Vec<Option<&str>> {
        self.layers.iter().map(|e| e.name.as_deref()).collect()
    }

    /// Get the number of layers in the sequential
    #[must_use]
    pub const fn len(&self) -> usize {
        self.layers.len()
    }

    /// Check if the sequential is empty
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

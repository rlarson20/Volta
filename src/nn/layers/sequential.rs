use crate::Linear;
use crate::ReLU;
use crate::io::StateDict;
use crate::nn::Module;
use crate::tensor::Tensor;

/// Internal struct to hold a layer with an optional name
pub(crate) struct LayerEntry {
    pub(crate) name: Option<String>,
    pub(crate) layer: Box<dyn Module>,
}

pub struct Sequential {
    pub(crate) layers: Vec<LayerEntry>,
}

impl Module for Sequential {
    fn forward(&self, x: &Tensor) -> Tensor {
        let mut current = x.clone();
        for entry in &self.layers {
            current = entry.layer.forward(&current);
        }
        current
    }
    fn parameters(&self) -> Vec<Tensor> {
        self.layers
            .iter()
            .flat_map(|e| e.layer.parameters())
            .collect()
    }

    fn state_dict(&self) -> StateDict {
        let mut state = StateDict::new();
        for (i, entry) in self.layers.iter().enumerate() {
            let sub_state = entry.layer.state_dict();
            if sub_state.is_empty() {
                continue;
            }

            // Use name if available, otherwise use numeric index
            let index_str = i.to_string();
            let prefix = entry.name.as_deref().unwrap_or(&index_str);

            for (key, value) in sub_state {
                state.insert(format!("{}.{}", prefix, key), value);
            }
        }
        state
    }

    fn load_state_dict(&mut self, state: &StateDict) {
        for (i, entry) in self.layers.iter_mut().enumerate() {
            // Try name first, fallback to numeric index
            let possible_prefixes: Vec<String> = if let Some(ref name) = entry.name {
                vec![format!("{}.", name), format!("{}.", i)]
            } else {
                vec![format!("{}.", i)]
            };

            // Collect matching keys for this layer
            let mut sub_state = StateDict::new();

            for prefix in possible_prefixes {
                for (key, value) in state.iter() {
                    if key.starts_with(&prefix) {
                        let sub_key = &key[prefix.len()..];
                        if !sub_key.is_empty() {
                            sub_state.insert(sub_key.to_string(), value.clone());
                        }
                    }
                }

                // If we found keys with this prefix, stop trying others
                if !sub_state.is_empty() {
                    break;
                }
            }

            if !sub_state.is_empty() {
                entry.layer.load_state_dict(&sub_state);
            }
        }
    }
    fn train(&mut self, mode: bool) {
        for entry in &mut self.layers {
            entry.layer.train(mode);
        }
    }
}

impl Sequential {
    // Helper constructor for easier testing and building
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Sequential {
            layers: layers
                .into_iter()
                .map(|layer| LayerEntry { name: None, layer })
                .collect(),
        }
    }

    /// Create a new SequentialBuilder for building models with named layers
    pub fn builder() -> crate::nn::layers::SequentialBuilder {
        crate::nn::layers::SequentialBuilder::new()
    }

    /// Get a layer by index
    pub fn get(&self, index: usize) -> Option<&dyn Module> {
        self.layers.get(index).map(|e| &*e.layer)
    }

    /// Get a layer by name
    ///
    /// Returns None if no layer with the given name exists
    pub fn get_named(&self, name: &str) -> Option<&dyn Module> {
        self.layers
            .iter()
            .find(|e| e.name.as_ref().is_some_and(|n| n == name))
            .map(|e| &*e.layer)
    }

    /// Get a mutable reference to a layer by name
    ///
    /// Returns None if no layer with the given name exists
    pub fn get_named_mut(&mut self, name: &str) -> Option<&mut (dyn Module + '_)> {
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
    pub fn layer_names(&self) -> Vec<Option<&str>> {
        self.layers.iter().map(|e| e.name.as_deref()).collect()
    }

    /// Get the number of layers in the sequential
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Check if the sequential is empty
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

// Usage example
#[allow(dead_code)]
impl Sequential {
    fn build_mlp(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Sequential {
        Sequential::new(vec![
            Box::new(Linear::new(input_dim, hidden_dim, true)),
            Box::new(ReLU),
            Box::new(Linear::new(hidden_dim, output_dim, true)),
        ])
    }
}

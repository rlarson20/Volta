use crate::Linear;
use crate::ReLU;
use crate::io::StateDict;
use crate::nn::Module;
use crate::tensor::Tensor;

pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Module for Sequential {
    fn forward(&self, x: &Tensor) -> Tensor {
        let mut current = x.clone();
        for layer in &self.layers {
            current = layer.forward(&current);
        }
        current
    }
    fn parameters(&self) -> Vec<Tensor> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    fn state_dict(&self) -> StateDict {
        let mut state = StateDict::new();
        for (i, layer) in self.layers.iter().enumerate() {
            let sub_state = layer.state_dict();
            for (key, value) in sub_state {
                state.insert(format!("{}.{}", i, key), value);
            }
        }
        state
    }

    fn load_state_dict(&mut self, state: &StateDict) {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let prefix = format!("{}.", i);
            // Filter keys for this layer
            let mut sub_state = StateDict::new();
            for (key, value) in state {
                if key.starts_with(&prefix) {
                    let sub_key = &key[prefix.len()..];
                    if !sub_key.is_empty() {
                        sub_state.insert(sub_key.to_string(), value.clone());
                    }
                }
            }
            if !sub_state.is_empty() {
                layer.load_state_dict(&sub_state);
            }
        }
    }
    fn train(&mut self, mode: bool) {
        for layer in &mut self.layers {
            layer.train(mode);
        }
    }
}

impl Sequential {
    // Helper constructor for easier testing and building
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Sequential { layers }
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

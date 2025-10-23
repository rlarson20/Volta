use crate::Linear;
use crate::ReLU;
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

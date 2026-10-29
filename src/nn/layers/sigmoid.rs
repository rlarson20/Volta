use crate::io::StateDict;
use crate::nn::{Layer, Module};
use crate::tensor::Tensor;

pub struct Sigmoid;

impl Module for Sigmoid {
    fn parameters(&self) -> Vec<Tensor> {
        vec![] // No learnable params
    }

    fn state_dict(&self) -> StateDict {
        StateDict::new()
    }

    fn load_state_dict(&mut self, _state: &StateDict) {
        // Stateless
    }
}

impl Layer for Sigmoid {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.sigmoid()
    }
}

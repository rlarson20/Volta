use crate::io::StateDict;
use crate::nn::Module;
use crate::tensor::Tensor;
use crate::tensor::TensorOps;

pub struct Tanh;

impl Module for Tanh {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.tanh()
    }

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

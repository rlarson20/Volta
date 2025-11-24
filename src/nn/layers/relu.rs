use crate::io::StateDict;
use crate::nn::Module;
use crate::tensor::Tensor;
use crate::tensor::TensorOps;

pub struct ReLU;

impl Module for ReLU {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.relu()
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

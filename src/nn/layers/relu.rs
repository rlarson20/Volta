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
}

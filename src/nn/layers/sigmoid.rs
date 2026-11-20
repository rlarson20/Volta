use crate::io::StateDict;
use crate::nn::{Layer, Module};
use crate::tensor::Tensor;

/// Logistic function `1 / (1 + exp(-x))`.
pub struct Sigmoid;

impl Module for Sigmoid {
    fn parameters(&self) -> Vec<Tensor> {
        vec![] // No learnable params
    }

    fn state_dict(&self) -> StateDict {
        StateDict::new()
    }

    fn plan_load<'a>(&self, _plan: &mut crate::nn::LoadPlan<'a>) {}
}

impl Layer for Sigmoid {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.sigmoid()
    }
}

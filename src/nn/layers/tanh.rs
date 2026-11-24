use crate::io::StateDict;
use crate::nn::{Layer, Module};
use crate::tensor::Tensor;

/// Hyperbolic tangent.
#[derive(Debug, Clone, Copy)]
pub struct Tanh;

impl Module for Tanh {
    fn parameters(&self) -> Vec<Tensor> {
        vec![] // No learnable params
    }

    fn state_dict(&self) -> StateDict {
        StateDict::new()
    }

    fn plan_load<'a>(&self, _plan: &mut crate::nn::LoadPlan<'a>) {}
}

impl Layer for Tanh {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.tanh()
    }
}

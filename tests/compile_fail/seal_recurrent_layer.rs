use volta::io::StateDict;
use volta::{
    Tensor,
    nn::{Module, RecurrentLayer},
};

struct Outsider;

impl Module for Outsider {
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
    fn state_dict(&self) -> StateDict {
        StateDict::new()
    }
    fn plan_load(&self, _plan: &mut volta::nn::LoadPlan<'_>) {
        // Nothing to load.
    }
}

// `RecurrentLayer` is sealed: only the library can implement it.
impl RecurrentLayer for Outsider {
    type State = ();
    fn initial_state(&self, _batch: usize) -> Self::State {}
    fn step(&self, _x: &Tensor, _state: &mut Self::State) -> Tensor {
        unimplemented!()
    }
}

fn main() {}

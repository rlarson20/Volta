use volta::io::StateDict;
use volta::{Module, RecurrentLayer, Tensor};

struct Outsider;

impl Module for Outsider {
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
    fn state_dict(&self) -> StateDict {
        StateDict::new()
    }
    fn load_state_dict(&mut self, _state: &StateDict) {}
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

use volta::io::StateDict;
use volta::{IndexedLayer, Module, Tensor};

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

// `IndexedLayer` is sealed: only the library can implement it.
impl IndexedLayer for Outsider {
    fn forward_indices(&self, _indices: &[usize]) -> Tensor {
        unimplemented!()
    }
}

fn main() {}

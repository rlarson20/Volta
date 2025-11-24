use crate::io::StateDict;
use crate::tensor::Tensor;

pub mod layers;
pub mod optim;

pub use layers::{BatchNorm2d, Conv2d, Linear, MaxPool2d, ReLU, Sequential};
pub use optim::{Adam, Muon, SGD};

pub trait Module {
    fn forward(&self, x: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<Tensor>;

    // State dict methods
    fn state_dict(&self) -> StateDict;
    fn load_state_dict(&mut self, state: &StateDict);

    fn zero_grad(&mut self) {
        for p in self.parameters() {
            p.borrow_mut().grad = None;
        }
    }
    /// Switch between training and evaluation modes.
    /// Important for layers like BatchNorm and Dropout.
    fn train(&mut self, _mode: bool) {}
    fn eval(&mut self) {
        self.train(false);
    }
}

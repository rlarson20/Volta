use crate::tensor::Tensor;

pub mod layers;
pub mod optim;

pub use layers::{Conv2d, Linear, ReLU, Sequential};
pub use optim::{Adam, SGD};

pub trait Module {
    fn forward(&self, x: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<Tensor>;
    fn zero_grad(&mut self) {
        for p in self.parameters() {
            p.borrow_mut().grad = None;
        }
    }
}

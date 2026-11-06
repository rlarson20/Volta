/// Adam optimizer.
pub mod adam;
/// Muon optimizer.
pub mod muon;
/// SGD optimizer.
pub mod sgd;

pub use adam::Adam;
pub use muon::Muon;
pub use sgd::SGD;

use crate::Tensor;

/// The interface every optimizer shares, so code can drive one without knowing which
/// (PyTorch's `torch.optim.Optimizer`): [`crate::amp::GradScaler`] unscales its
/// parameters' gradients and decides whether to call `step`. The inherent methods of the
/// same names do the work; this trait forwards to them.
pub trait Optimizer {
    /// Update every parameter from its gradient.
    fn step(&mut self);
    /// Clear every parameter's gradient.
    fn zero_grad(&self);
    /// The parameters this optimizer updates, in its order.
    fn parameters(&self) -> Vec<Tensor>;
}

/// Implements [`Optimizer`] by forwarding to the inherent `step`/`zero_grad` and the
/// `params` field.
macro_rules! optimizer {
    ($($t:ty),*) => {$(
        impl Optimizer for $t {
            fn step(&mut self) {
                <$t>::step(self);
            }
            fn zero_grad(&self) {
                <$t>::zero_grad(self);
            }
            fn parameters(&self) -> Vec<Tensor> {
                self.params.clone()
            }
        }
    )*};
}

optimizer!(Adam, Muon, SGD);

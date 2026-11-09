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

/// `Ok` if `value` is at least 0 (NaN is not), else PyTorch's message:
/// `Invalid <name> value: <value>`. Optimizer hyperparameters are arguments a user
/// supplies, so they are checked before any state is allocated (invariant 8).
pub(crate) fn check_non_negative(name: &str, value: f32) -> crate::Result<()> {
    if value >= 0.0 {
        Ok(())
    } else {
        Err(crate::VoltaError::InvalidParameter(format!(
            "Invalid {name} value: {value}"
        )))
    }
}

/// `Ok` if `0 <= value < 1`, else PyTorch's `Invalid beta parameter at index <i>`.
pub(crate) fn check_beta(index: usize, value: f32) -> crate::Result<()> {
    if (0.0..1.0).contains(&value) {
        Ok(())
    } else {
        Err(crate::VoltaError::InvalidParameter(format!(
            "Invalid beta parameter at index {index}: {value}"
        )))
    }
}

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

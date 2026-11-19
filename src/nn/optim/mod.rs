/// Adam optimizer.
pub(crate) mod adam;
/// Muon optimizer.
pub(crate) mod muon;
/// SGD optimizer.
pub(crate) mod sgd;

pub use adam::Adam;
pub use muon::Muon;
pub use sgd::SGD;

use crate::Tensor;

/// Zeroed optimizer state shaped like `p`'s data, in its current dtype and on its current
/// device. Called on a parameter's first step, not at construction, so a model converted
/// with `to_device` or `to_dtype` before then is stepped in its new device and dtype.
pub(crate) fn zeros_like(p: &crate::RawTensor) -> crate::storage::Storage {
    // In the parameter's dtype on its device (a no-op cast for f32).
    crate::storage::Storage::new_zeros(p.data.len(), &p.device).cast(p.data.dtype())
}

/// Panics unless every parameter in `params` that has a gradient has exactly one gradient
/// element per parameter element. The CPU steps read the parameter and its gradient at one
/// index, so a mismatch would update only part of the parameter (#324); `set_grad` refuses
/// one, and this catches any other writer. Checked for every parameter before any is updated.
///
/// # Panics
/// If a gradient's length differs from its parameter's.
#[track_caller]
pub(crate) fn assert_grad_lens(optimizer: &str, params: &[Tensor]) {
    for (i, param) in params.iter().enumerate() {
        let p = param.borrow();
        if let Some(grad) = &p.grad {
            assert_eq!(
                grad.len(),
                p.data.len(),
                "{optimizer} step: parameter {i}'s gradient length vs. its element count"
            );
        }
    }
}

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

/// `Ok` if no tensor appears twice in `params` (by identity, [`Tensor::ptr_eq`]). A tied
/// weight listed twice would be stepped twice, moving by `2 * lr * grad`, with its momentum
/// or moment state kept twice (#101). PyTorch only warns; a duplicate is the caller's mistake
/// and a wrong update, so it is an error here. `Module::parameters()` already lists a tied
/// weight once.
pub(crate) fn check_unique(params: &[Tensor]) -> crate::Result<()> {
    for (j, p) in params.iter().enumerate() {
        if let Some(i) = params.iter().take(j).position(|q| q.ptr_eq(p)) {
            return Err(crate::VoltaError::InvalidParameter(format!(
                "parameter {j} is the same tensor as parameter {i} (a tied weight listed twice)"
            )));
        }
    }
    Ok(())
}

/// `Ok` if every tensor in `params` is a leaf (no `grad_fn`). A non-leaf, such as the
/// result of `param.to_dtype(..)` or `param.to_device(..)` on a parameter that requires
/// grad, is a graph intermediate: stepping it overwrites a value in the graph while the real
/// leaf never moves. PyTorch raises `can't optimize a non-leaf Tensor` (#102).
///
/// A leaf with `requires_grad == false` (a frozen parameter) is accepted, as in PyTorch: it
/// never gets a gradient, and `step` skips a parameter without one.
pub(crate) fn check_leaves(params: &[Tensor]) -> crate::Result<()> {
    if let Some(i) = params.iter().position(|p| p.borrow().grad_fn.is_some()) {
        return Err(crate::VoltaError::InvalidParameter(format!(
            "can't optimize a non-leaf tensor (parameter {i} was computed by an op); pass the \
             leaf it came from, or make a leaf on the target device or dtype first"
        )));
    }
    Ok(())
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
///
/// Open for implementation outside the crate: a method added to this trait in a later
/// version comes with a default body, so existing implementations keep compiling.
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

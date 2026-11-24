//! Checks for layer hyperparameters (#133). A layer's sizes, rates and `eps` come from
//! the user (a config file, a UI), so each layer's `try_new` checks them before it
//! allocates anything and names the layer and the argument (invariant 8).

use crate::VoltaError;
use crate::shape::MAX_NUMEL;

/// `Ok` if the size `value` of argument `arg` is at least 1.
pub(crate) fn at_least_one(op: &'static str, arg: &'static str, value: usize) -> crate::Result<()> {
    if value >= 1 {
        Ok(())
    } else {
        Err(VoltaError::invalid(
            op,
            arg,
            format!("{arg} must be at least 1, got {value}"),
        ))
    }
}

/// `Ok` if a parameter whose shape is the product of `dims` fits in a tensor: the element
/// count doesn't overflow `usize` and is at most [`MAX_NUMEL`] (#460). `args` names the
/// arguments the shape comes from. Each size must already be at least 1
/// ([`at_least_one`]); this catches only sizes too large together.
pub(crate) fn param_numel(
    op: &'static str,
    args: &'static str,
    dims: &[usize],
) -> crate::Result<()> {
    match dims.iter().try_fold(1usize, |n, &d| n.checked_mul(d)) {
        Some(n) if n <= MAX_NUMEL => Ok(()),
        Some(n) => Err(VoltaError::invalid(
            op,
            args,
            format!("{args} give a parameter of {n} elements, more than {MAX_NUMEL}"),
        )),
        None => Err(VoltaError::invalid(
            op,
            args,
            format!("{args} give a parameter whose element count overflows usize ({dims:?})"),
        )),
    }
}

/// `Ok` if `eps` is finite and positive: a zero, negative or NaN `eps` lets the
/// normalization divide by zero or take the square root of a negative number.
pub(crate) fn eps(op: &'static str, value: f32) -> crate::Result<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(VoltaError::invalid(
            op,
            "eps",
            format!("eps must be finite and > 0, got {value}"),
        ))
    }
}

/// `Ok` if `value` is in `[0, 1]` (a probability, or a momentum that mixes two values);
/// NaN is not.
pub(crate) fn unit_interval(op: &'static str, arg: &'static str, value: f32) -> crate::Result<()> {
    if (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(VoltaError::invalid(
            op,
            arg,
            format!("{arg} must be in [0, 1], got {value}"),
        ))
    }
}

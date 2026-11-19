/// Test helper that checks a model's parameters receive gradients.
mod assert_trainable;
/// The `define_ops!` macro that generates the unary and binary op families.
mod define_ops;

pub(crate) use define_ops::define_ops;

/// Seeded `N(0, 1)` input with `requires_grad = true`, for [`assert_trainable!`].
///
/// Seeding makes the input and the macro's loss weights (drawn next from the same
/// thread-local RNG) deterministic per test.
#[cfg(test)]
pub(crate) fn trainable_input(shape: &[usize]) -> crate::tensor::Tensor {
    crate::tensor::manual_seed(0x5EED);
    let x = crate::tensor::randn(shape);
    x.set_requires_grad(true);
    x
}

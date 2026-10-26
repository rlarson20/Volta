mod assert_trainable;

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

// Deferred to later phases (ADDENDUM.md §5): assert_close.rs, define_ops.rs,
// gradcheck.rs, gradgradcheck.rs, param_test.rs, tensor.rs are still empty skeleton
// files under src/macros/. Not declared as modules here until they have content.

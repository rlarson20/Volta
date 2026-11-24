/// MNIST IDX file loaders.
pub(crate) mod mnist;
/// Data transforms (normalisation, one-hot encoding).
pub(crate) mod transforms;

pub use mnist::{load_mnist_images, load_mnist_labels};
pub use transforms::{normalize, to_one_hot, try_normalize};

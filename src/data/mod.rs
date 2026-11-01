/// MNIST IDX file loaders.
pub mod mnist;
/// Data transforms (normalisation, one-hot encoding).
pub mod transforms;

pub use mnist::{load_mnist_images, load_mnist_labels};
pub use transforms::{normalize, to_one_hot};

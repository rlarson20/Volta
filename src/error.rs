use crate::dtype::DType;
use thiserror::Error;

#[non_exhaustive]
#[derive(Debug, Error)]
pub enum VoltaError {
    #[error("Shape mismatch: tensor has {elements} elements but data length is {len}")]
    ShapeDataMismatch {
        shape: Vec<usize>,
        elements: usize,
        len: usize,
    },

    #[error("Dimension {dim} out of bounds for shape {shape:?}")]
    DimensionOutOfBounds { dim: usize, shape: Vec<usize> },

    #[error("DType mismatch: expected {expected:?}, got {actual:?}")]
    DTypeMismatch { expected: DType, actual: DType },

    #[error("Cannot broadcast shapes {0:?} and {1:?}")]
    BroadcastError(Vec<usize>, Vec<usize>),

    #[error("Device error: {0}")]
    DeviceError(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error(
        "Cannot start backward from a non-scalar tensor of shape {shape:?}; reduce it to one element first (e.g. `.sum()`)"
    )]
    NotScalar { shape: Vec<usize> },

    #[error("Cannot start backward from a tensor that doesn't require grad")]
    NoGradRequired,

    #[error("Operation failed: {0}")]
    OpError(String),

    /// A state-dict entry's shape differs from the parameter it would load into.
    /// `key` is the entry's name as the caller's state dict spells it (a container
    /// like `Sequential` puts its layer prefix back, e.g. `2.weight`).
    #[error("State dict entry '{key}' has shape {actual:?}, expected {expected:?}")]
    StateShapeMismatch {
        key: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// A state-dict entry holds a different number of values than its own shape
    /// needs (a truncated or corrupt checkpoint). `key` is spelled as in
    /// [`VoltaError::StateShapeMismatch`].
    #[error("State dict entry '{key}' has {len} values but its shape needs {expected}")]
    StateDataLength {
        key: String,
        expected: usize,
        len: usize,
    },
}

impl VoltaError {
    /// Puts `prefix` (e.g. `"2."`) in front of the key of a state-dict error, so an
    /// error from a layer inside a container names the entry in the container's own
    /// state dict. Other variants pass through unchanged.
    #[must_use]
    pub(crate) fn with_key_prefix(mut self, prefix: &str) -> Self {
        if let Self::StateShapeMismatch { key, .. } | Self::StateDataLength { key, .. } = &mut self
        {
            *key = format!("{prefix}{key}");
        }
        self
    }
}

pub type Result<T> = std::result::Result<T, VoltaError>;

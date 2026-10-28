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
}

pub type Result<T> = std::result::Result<T, VoltaError>;

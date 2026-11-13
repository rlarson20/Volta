use crate::dtype::DType;
use thiserror::Error;

/// Every error Volta's fallible (`try_*`) APIs can return.
/// \n Marked `#[non_exhaustive]`, so match with a wildcard arm.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum VoltaError {
    /// The data handed to a constructor does not fill the requested shape.
    #[error("Shape mismatch: tensor has {elements} elements but data length is {len}")]
    ShapeDataMismatch {
        /// The requested shape.
        shape: Vec<usize>,
        /// Element count the shape needs.
        elements: usize,
        /// Length of the data supplied.
        len: usize,
    },

    /// An axis index is outside the tensor's rank.
    #[error("Dimension {dim} out of bounds for shape {shape:?}")]
    DimensionOutOfBounds {
        /// The offending axis.
        dim: usize,
        /// Shape of the tensor it was applied to.
        shape: Vec<usize>,
    },

    /// Two tensors that must share a dtype do not.
    #[error("DType mismatch: expected {expected:?}, got {actual:?}")]
    DTypeMismatch {
        /// The dtype the operation required.
        expected: DType,
        /// The dtype it received.
        actual: DType,
    },

    /// The two shapes (in order) cannot be broadcast together.
    #[error("Cannot broadcast shapes {0:?} and {1:?}")]
    BroadcastError(Vec<usize>, Vec<usize>),

    /// A device operation failed or is unsupported; the message says which.
    #[error("Device error: {0}")]
    DeviceError(String),

    /// An underlying I/O operation failed.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// An argument is out of its valid range; the message names it.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// `backward` was requested on a tensor with more than one element.
    #[error(
        "Cannot start backward from a non-scalar tensor of shape {shape:?}; reduce it to one element first (e.g. `.sum()`)"
    )]
    NotScalar {
        /// Shape of the tensor `backward` was requested on.
        shape: Vec<usize>,
    },

    /// `backward` was requested on a tensor that does not require grad.
    #[error("Cannot start backward from a tensor that doesn't require grad")]
    NoGradRequired,

    /// An operation failed for a reason that has no dedicated variant.
    #[error("Operation failed: {0}")]
    OpError(String),

    /// A state-dict entry's shape differs from the parameter it would load into.
    /// `key` is the entry's name as the caller's state dict spells it (a container
    /// like `Sequential` puts its layer prefix back, e.g. `2.weight`).
    #[error("State dict entry '{key}' has shape {actual:?}, expected {expected:?}")]
    StateShapeMismatch {
        /// Entry name as the caller's state dict spells it.
        key: String,
        /// Shape of the parameter the entry would load into.
        expected: Vec<usize>,
        /// Shape the entry carries.
        actual: Vec<usize>,
    },

    /// A state-dict entry holds a different number of bytes than its own shape and
    /// dtype need (a truncated or corrupt checkpoint). `key` is spelled as in
    /// [`VoltaError::StateShapeMismatch`].
    #[error("State dict entry '{key}' has {len} bytes but its shape and dtype need {expected}")]
    StateDataLength {
        /// Entry name, spelled as in [`VoltaError::StateShapeMismatch`].
        key: String,
        /// Bytes the entry's own shape and dtype need (`usize::MAX` if that overflows).
        expected: usize,
        /// Bytes the entry holds.
        len: usize,
    },

    /// A [`StateDictMapper`](crate::io::mapping::StateDictMapper) renaming step would give
    /// two entries the same name, so one of them would be lost.
    #[error("State dict entries '{first}' and '{second}' both map to '{key}'")]
    StateKeyCollision {
        /// The name both entries would end up under.
        key: String,
        /// The first entry's original name (in key order).
        first: String,
        /// The second entry's original name.
        second: String,
    },

    /// Gradients were requested for a tensor whose dtype has none (integers and bool).
    #[error("Only floating-point tensors can require gradients, got {dtype}")]
    NotDifferentiable {
        /// The tensor's dtype.
        dtype: DType,
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

/// `Result` specialised to [`VoltaError`].
pub type Result<T> = std::result::Result<T, VoltaError>;

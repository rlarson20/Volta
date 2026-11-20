use crate::dtype::DType;
use thiserror::Error;

/// Every error Volta's fallible (`try_*`) APIs can return.
///
/// Marked `#[non_exhaustive]`, so match with a wildcard arm.
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

    /// A device operation failed or is unsupported: the detail names the op and says why.
    #[error("Device error: {0}")]
    DeviceError(ErrorDetail),

    /// An underlying I/O operation failed: a file could not be opened, read or written.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// A file was read but its contents could not be decoded: not a safetensors file, an
    /// entry whose dtype Volta has no equivalent for, or a malformed MNIST IDX header
    /// (#394). Nothing was loaded.
    #[error(
        "Decode error in {}{}: {detail}",
        .path.display(),
        .key.as_ref().map_or_else(String::new, |k| format!(" (entry '{k}')"))
    )]
    Decode {
        /// The file.
        path: std::path::PathBuf,
        /// The entry, when one tensor of a state dict is at fault.
        key: Option<String>,
        /// The loader that failed (`op`, with `arg` `"path"`) and what was wrong.
        detail: ErrorDetail,
    },

    /// An argument is out of its valid range: the detail names the op, the argument, and
    /// what is wrong with it.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(ErrorDetail),

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

    /// An operation failed for a reason that has no dedicated variant: the detail names
    /// the op and says why.
    #[error("Operation failed: {0}")]
    OpError(ErrorDetail),

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

    /// A strict [`Module::load_state_dict`](crate::nn::Module::load_state_dict) got a
    /// state dict whose keys differ from the module's. Nothing was loaded.
    #[error(
        "State dict keys do not match the module: missing {missing:?}, unexpected {unexpected:?}"
    )]
    StateKeyMismatch {
        /// Keys of the module's state dict that `state` lacks, in key order.
        missing: Vec<String>,
        /// Keys of `state` the module does not have, in key order.
        unexpected: Vec<String>,
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

/// The payload of [`VoltaError::InvalidParameter`], [`VoltaError::DeviceError`] and
/// [`VoltaError::OpError`] (#377), and part of [`VoltaError::Decode`]: which operation
/// failed, which argument (if one is at fault), and why, so a caller can match the cause
/// without parsing the message.
///
/// `#[non_exhaustive]`: read the fields, but only Volta builds one, and a new field is
/// not a breaking change.
///
/// ```
/// use volta::{ErrorDetail, RawTensor, VoltaError};
///
/// let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false);
/// let err = x.try_reshape(&[3]).unwrap_err();
/// let VoltaError::InvalidParameter(ErrorDetail { op, arg, .. }) = &err else {
///     panic!("expected InvalidParameter, got {err}");
/// };
/// assert_eq!((*op, *arg), ("reshape", Some("shape")));
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorDetail {
    /// The public operation that rejected the call, as a caller names it (`"reshape"`,
    /// `"to_device"`, `"Conv2d"`, `"Adam::new"`, `"Shape::from_dims"`).
    pub op: &'static str,
    /// The argument at fault, as the operation's signature names it, if one is
    /// (`Some("shape")`); `None` when the failure is not one argument's (no GPU).
    pub arg: Option<&'static str>,
    /// What is wrong, in words. It does not repeat `op`.
    pub message: String,
}

impl std::fmt::Display for ErrorDetail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.op, self.message)
    }
}

impl VoltaError {
    /// [`VoltaError::InvalidParameter`] from its parts.
    #[must_use]
    pub(crate) fn invalid(
        op: &'static str,
        arg: impl Into<Option<&'static str>>,
        message: impl Into<String>,
    ) -> Self {
        Self::InvalidParameter(ErrorDetail::new(op, arg, message))
    }

    /// [`VoltaError::Decode`] of `path` (and entry `key`), reported by loader `op`.
    #[must_use]
    pub(crate) fn decode(
        op: &'static str,
        path: &std::path::Path,
        key: Option<&str>,
        message: impl Into<String>,
    ) -> Self {
        Self::Decode {
            path: path.to_path_buf(),
            key: key.map(str::to_string),
            detail: ErrorDetail::new(op, "path", message),
        }
    }

    /// [`VoltaError::DeviceError`] from its parts.
    #[must_use]
    pub(crate) fn device(
        op: &'static str,
        arg: impl Into<Option<&'static str>>,
        message: impl Into<String>,
    ) -> Self {
        Self::DeviceError(ErrorDetail::new(op, arg, message))
    }

    /// The op, argument and message of an `InvalidParameter`, `DeviceError`, `OpError` or
    /// `Decode`;
    /// `None` for the variants whose fields already say what went wrong.
    #[must_use]
    pub const fn detail(&self) -> Option<&ErrorDetail> {
        match self {
            Self::InvalidParameter(d)
            | Self::DeviceError(d)
            | Self::OpError(d)
            | Self::Decode { detail: d, .. } => Some(d),
            // Listed, not `_`: a new variant is a compile error here.
            Self::ShapeDataMismatch { .. }
            | Self::DimensionOutOfBounds { .. }
            | Self::DTypeMismatch { .. }
            | Self::BroadcastError(..)
            | Self::Io(_)
            | Self::NotScalar { .. }
            | Self::NoGradRequired
            | Self::StateShapeMismatch { .. }
            | Self::StateDataLength { .. }
            | Self::StateKeyMismatch { .. }
            | Self::StateKeyCollision { .. }
            | Self::NotDifferentiable { .. } => None,
        }
    }
}

impl ErrorDetail {
    /// A detail from its parts.
    #[must_use]
    pub(crate) fn new(
        op: &'static str,
        arg: impl Into<Option<&'static str>>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            op,
            arg: arg.into(),
            message: message.into(),
        }
    }
}

/// `Result` specialised to [`VoltaError`].
pub type Result<T> = std::result::Result<T, VoltaError>;

/// Unwrap a `Result`, panicking with `"{what}: {err}"` (or just `"{err}"` when `what` is
/// empty) at the location of the first caller without `#[track_caller]`.
///
/// This is the panicking half of the `try_x` / `x` pair (DECISIONS.md, "Panic inside the
/// program, `Result` where data comes from outside"). Use it instead of
/// `unwrap_or_else(|e| panic!(..))`: `#[track_caller]` does not reach into a closure, so a
/// `panic!` inside one reports the closure's own line in `src/`, not the user's call (#132).
pub(crate) trait OrPanic<T> {
    /// The `Ok` value, or a panic naming `what` and the error.
    ///
    /// # Panics
    /// If `self` is `Err`.
    #[track_caller]
    fn or_panic(self, what: &str) -> T;
}

impl<T> OrPanic<T> for std::result::Result<T, VoltaError> {
    /// A detail error already names its op, so `what` is left out when it is that op
    /// (`reshape: Invalid parameter: reshape: ..` would say it twice).
    #[track_caller]
    fn or_panic(self, what: &str) -> T {
        match self {
            Ok(v) => v,
            Err(e) if what.is_empty() || e.detail().is_some_and(|d| d.op == what) => {
                panic!("{e}")
            }
            Err(e) => panic!("{what}: {e}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each detail variant displays its prefix, then `op: message` (#377).
    #[test]
    fn detail_variants_display_op_then_message() {
        let cases = [
            (
                VoltaError::invalid("reshape", "shape", "bad"),
                "Invalid parameter: reshape: bad",
            ),
            (
                VoltaError::device("to_device", None, "no GPU"),
                "Device error: to_device: no GPU",
            ),
            (
                VoltaError::OpError(ErrorDetail::new("op", None, "failed")),
                "Operation failed: op: failed",
            ),
        ];
        for (err, want) in cases {
            assert_eq!(err.to_string(), want);
        }
    }

    /// `detail` reads the payload of the three detail variants and nothing else.
    #[test]
    fn detail_is_some_only_for_the_detail_variants() {
        let err = VoltaError::invalid("pad", "padding", "too long");
        let d = err.detail().expect("InvalidParameter has a detail");
        assert_eq!(
            (d.op, d.arg, d.message.as_str()),
            ("pad", Some("padding"), "too long")
        );
        assert!(VoltaError::NoGradRequired.detail().is_none());
    }
}

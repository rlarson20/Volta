//! Tensor data types.
//!
//! A tensor of any `DType` can be built (`RawTensor::from_typed`) and read back. Of the
//! ops, the core ops (elementwise, reductions, movement, matmul) compute in f32 or f64; layers and optimizers are f32-only until ADDENDUM §11 P3b, and half types wait for P4; an op panics, naming itself, on a dtype it has no kernel for.
//! `load_safetensors` converts to f32. Mixed precision is planned in ADDENDUM §11.

use std::fmt;

/// Element types a tensor can hold.
///
/// Only `F32` can go through ops today (see the module docs).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum DType {
    /// 16-bit floating point (IEEE 754 half precision)
    F16 = 0,
    /// 16-bit brain floating point (truncated f32 mantissa)
    BF16 = 1,
    /// 32-bit floating point (default)
    #[default]
    F32 = 2,
    /// 64-bit floating point
    F64 = 3,
    /// 32-bit signed integer
    I32 = 4,
    /// 64-bit signed integer
    I64 = 5,
    /// 8-bit unsigned integer
    U8 = 6,
    /// Boolean (stored as u8)
    Bool = 7,
}

impl DType {
    /// Returns the size in bytes of a single element of this dtype
    #[must_use]
    pub const fn size_of(&self) -> usize {
        match self {
            Self::F16 | Self::BF16 => 2,
            Self::F32 | Self::I32 => 4,
            Self::F64 | Self::I64 => 8,
            Self::U8 | Self::Bool => 1,
        }
    }

    /// Returns the name of this dtype as a string
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::U8 => "u8",
            Self::Bool => "bool",
        }
    }

    /// Returns true if this is a floating-point type
    #[must_use]
    pub const fn is_float(&self) -> bool {
        matches!(self, Self::F16 | Self::BF16 | Self::F32 | Self::F64)
    }

    /// Returns true if this is an integer type
    #[must_use]
    pub const fn is_int(&self) -> bool {
        matches!(self, Self::I32 | Self::I64 | Self::U8)
    }

    /// The dtype PyTorch gives an operation on two tensors of these dtypes.
    ///
    /// Matches `torch.result_type` for two dimensioned tensors on all 64 pairs
    /// (`tests/torch_ref_dtypes.rs`): `Bool` yields to anything; a float beats any
    /// integer, whatever its width (`F16` with `I64` is `F16`); `F16` with `BF16` is
    /// `F32`; otherwise the wider type of the same kind wins. Zero-dim tensors and Rust
    /// scalars promote by a different rule, which this does not model.
    ///
    /// Nothing calls this yet: ops require one dtype until implicit promotion lands
    /// (ADDENDUM §11, P6).
    #[must_use]
    pub fn promote(a: Self, b: Self) -> Self {
        use DType::{BF16, Bool, F16, F32, F64, I32, I64, U8};

        match (a, b) {
            _ if a == b => a,
            (Bool, other) | (other, Bool) => other,
            (F64, _) | (_, F64) => F64,
            (F32, _) | (_, F32) | (F16, BF16) | (BF16, F16) => F32,
            (F16 | BF16, _) => a,
            (_, F16 | BF16) => b,
            (I64, _) | (_, I64) => I64,
            (I32, _) | (_, I32) => I32,
            (U8, U8) => U8,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size() {
        assert_eq!(DType::F16.size_of(), 2);
        assert_eq!(DType::BF16.size_of(), 2);
        assert_eq!(DType::F32.size_of(), 4);
        assert_eq!(DType::F64.size_of(), 8);
        assert_eq!(DType::I32.size_of(), 4);
        assert_eq!(DType::I64.size_of(), 8);
        assert_eq!(DType::U8.size_of(), 1);
        assert_eq!(DType::Bool.size_of(), 1);
    }

    #[test]
    fn test_dtype_name() {
        assert_eq!(DType::F32.name(), "f32");
        assert_eq!(DType::F16.name(), "f16");
        assert_eq!(DType::BF16.name(), "bf16");
    }

    #[test]
    fn test_dtype_promote() {
        // Same types
        assert_eq!(DType::promote(DType::F32, DType::F32), DType::F32);

        // Float promotion
        assert_eq!(DType::promote(DType::F32, DType::F64), DType::F64);
        assert_eq!(DType::promote(DType::F16, DType::F32), DType::F32);
        assert_eq!(DType::promote(DType::F16, DType::BF16), DType::F32);

        // Int + Float
        assert_eq!(DType::promote(DType::I32, DType::F32), DType::F32);
        assert_eq!(DType::promote(DType::I64, DType::F64), DType::F64);

        // Int promotion
        assert_eq!(DType::promote(DType::I32, DType::I64), DType::I64);
        assert_eq!(DType::promote(DType::U8, DType::I32), DType::I32);

        // Half floats beat integers and bool (torch), rather than widening to F32
        assert_eq!(DType::promote(DType::F16, DType::I64), DType::F16);
        assert_eq!(DType::promote(DType::I32, DType::BF16), DType::BF16);
        assert_eq!(DType::promote(DType::Bool, DType::F16), DType::F16);
        assert_eq!(DType::promote(DType::Bool, DType::U8), DType::U8);
    }

    #[test]
    fn test_dtype_is_float() {
        assert!(DType::F16.is_float());
        assert!(DType::BF16.is_float());
        assert!(DType::F32.is_float());
        assert!(DType::F64.is_float());
        assert!(!DType::I32.is_float());
        assert!(!DType::I64.is_float());
    }

    #[test]
    fn test_dtype_is_int() {
        assert!(DType::I32.is_int());
        assert!(DType::I64.is_int());
        assert!(DType::U8.is_int());
        assert!(!DType::F32.is_int());
        assert!(!DType::Bool.is_int());
    }
}

//! Data type support for tensors
//!
//! This module provides the DType enum for representing tensor data types,
//! enabling Volta to work with different precisions (f16, bf16, f32, f64)
//! and load models from formats like SafeTensors.

use std::fmt;

/// Supported tensor data types
///
/// Volta internally uses a byte buffer + dtype tag approach, allowing
/// runtime flexibility for mixed-precision operations and easy loading
/// of external model formats.
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
    pub fn size_of(&self) -> usize {
        match self {
            DType::F16 | DType::BF16 => 2,
            DType::F32 | DType::I32 => 4,
            DType::F64 | DType::I64 => 8,
            DType::U8 | DType::Bool => 1,
        }
    }

    /// Returns the name of this dtype as a string
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::I32 => "i32",
            DType::I64 => "i64",
            DType::U8 => "u8",
            DType::Bool => "bool",
        }
    }

    /// Returns true if this is a floating-point type
    #[must_use]
    pub fn is_float(&self) -> bool {
        matches!(self, DType::F16 | DType::BF16 | DType::F32 | DType::F64)
    }

    /// Returns true if this is an integer type
    #[must_use]
    pub fn is_int(&self) -> bool {
        matches!(self, DType::I32 | DType::I64 | DType::U8)
    }

    /// Determines the result dtype when two dtypes are combined in an operation.
    /// Follows type promotion rules similar to PyTorch/NumPy.
    #[must_use]
    pub fn promote(a: DType, b: DType) -> DType {
        use DType::*;

        // Same type -> same type
        if a == b {
            return a;
        }

        // Float types promote to wider float
        match (a, b) {
            // F64 dominates everything
            (F64, _) | (_, F64) => F64,

            // F32 dominates F16/BF16 and integers
            (F32, F16 | BF16 | I32 | I64 | U8 | Bool)
            | (F16 | BF16 | I32 | I64 | U8 | Bool, F32) => F32,

            // F16 and BF16 promote to F32 for safety (including with ints)
            (F16, BF16 | I32 | I64 | U8 | Bool) | (BF16 | I32 | I64 | U8 | Bool, F16) => F32,
            (BF16, I32 | I64 | U8 | Bool) | (I32 | I64 | U8 | Bool, BF16) => F32,

            // Int types promote to wider int
            (I64, I32 | U8 | Bool) | (I32 | U8 | Bool, I64) => I64,
            (I32, U8 | Bool) | (U8 | Bool, I32) => I32,
            (U8, Bool) | (Bool, U8) => U8,

            // Fallback (shouldn't reach here due to if a == b check)
            _ => F32,
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

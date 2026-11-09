//! The float types kernels compute in: `f32` and `f64` (ADDENDUM §11, P3).
//!
//! CPU kernels are generic over [`Real`] and instantiated twice. Elementwise ops come from
//! the `define_ops!` tables, which emit an `f32` and an `f64` function per row, so the
//! table stays one row per op. F16 and BF16 run the `f32` instance: `Storage::host` decodes
//! their inputs and `Storage::from_compute` rounds the output (P4).

use crate::dtype::DType;
use crate::ops::{BinaryOp, UnaryOp};
use crate::storage::Element;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Rem, Sub, SubAssign};

/// A float type CPU kernels compute in. Sealed by [`Element`]: `f32` and `f64` only.
pub(crate) trait Real:
    Element
    + bytemuck::Pod
    + PartialOrd
    + Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Rem<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + Sum
{
    /// Additive identity.
    const ZERO: Self;

    /// Nearest value of this type (exact for `f64`; `f32` rounds).
    fn from_f64(x: f64) -> Self;
    /// A count, exact below 2^24 (`f32`) or 2^53 (`f64`).
    fn from_usize(n: usize) -> Self;

    /// `op` on one element (the `define_ops!` row's forward).
    fn unary(op: UnaryOp, x: Self) -> Self;
    /// `g` times `op`'s derivative at `x` (the row's backward).
    fn unary_backward(op: UnaryOp, g: Self, x: Self) -> Self;
    /// `op` on one pair (the row's forward).
    fn binary(op: BinaryOp, a: Self, b: Self) -> Self;
    /// Per-element `(∂L/∂a, ∂L/∂b)` (the row's backward).
    fn binary_backward(op: BinaryOp, g: Self, a: Self, b: Self) -> (Self, Self);

    /// Square root.
    fn sqrt(self) -> Self;
    /// Dense row-major `(m,k) @ (k,n)`, see `RawTensor::matmul_raw`.
    fn gemm(a: &[Self], b: &[Self], m: usize, k: usize, n: usize) -> Vec<Self>;
}

/// Implements [`Real`] for one float type, naming its `define_ops!` functions.
macro_rules! real {
    ($t:ty, $unary:ident, $unary_bwd:ident, $binary:ident, $binary_bwd:ident, $gemm:ident) => {
        impl Real for $t {
            const ZERO: Self = 0.0;

            #[allow(
                clippy::cast_possible_truncation,
                reason = "f64 to f32 rounds to nearest, which is the point"
            )]
            fn from_f64(x: f64) -> Self {
                x as $t
            }
            #[allow(
                clippy::cast_precision_loss,
                reason = "documented: exact below 2^24 / 2^53"
            )]
            fn from_usize(n: usize) -> Self {
                n as $t
            }
            fn unary(op: UnaryOp, x: Self) -> Self {
                op.$unary(x)
            }
            fn unary_backward(op: UnaryOp, g: Self, x: Self) -> Self {
                op.$unary_bwd(g, x)
            }
            fn binary(op: BinaryOp, a: Self, b: Self) -> Self {
                op.$binary(a, b)
            }
            fn binary_backward(op: BinaryOp, g: Self, a: Self, b: Self) -> (Self, Self) {
                op.$binary_bwd(g, a, b)
            }
            fn sqrt(self) -> Self {
                <$t>::sqrt(self)
            }
            fn gemm(a: &[Self], b: &[Self], m: usize, k: usize, n: usize) -> Vec<Self> {
                crate::RawTensor::$gemm(a, b, m, k, n)
            }
        }
    };
}

real!(f32, forward, backward, forward, backward, matmul_raw);
real!(
    f64,
    forward_f64,
    backward_f64,
    forward_f64,
    backward_f64,
    matmul_raw_f64
);

/// The `f32` names a `define_ops!` unary row can use as `fl::..`.
pub(crate) mod fl32 {
    pub(crate) use std::f32::consts::{LN_2, PI};
    /// The error function.
    pub(crate) fn erf(x: f32) -> f32 {
        libm::erff(x)
    }
}

/// The `f64` names a `define_ops!` unary row can use as `fl::..`.
pub(crate) mod fl64 {
    pub(crate) use std::f64::consts::{LN_2, PI};
    /// The error function.
    pub(crate) fn erf(x: f64) -> f64 {
        libm::erf(x)
    }
}

/// Panics unless every input has one float dtype, naming the op, and returns it. Mixed
/// dtypes wait for promotion (P6); integers and bool have no kernels.
///
/// # Panics
/// If the inputs' dtypes differ, or the common dtype is not a float.
#[track_caller]
pub(crate) fn compute_dtype(op: impl FnOnce() -> String, dtypes: &[DType]) -> DType {
    let first = *dtypes.first().expect("an op has at least one input");
    if let Some((i, &d)) = dtypes.iter().enumerate().find(|&(_, &d)| d != first) {
        panic!(
            "{}: {} (input {i}); mixed dtypes need an explicit to_dtype until ADDENDUM §11 P6",
            op(),
            crate::VoltaError::DTypeMismatch {
                expected: first,
                actual: d,
            }
        );
    }
    match first {
        DType::F32 | DType::F64 | DType::F16 | DType::BF16 => first,
        DType::I32 | DType::I64 | DType::U8 | DType::Bool => panic!(
            "{}: no {first} kernel; ops compute in f32 or f64 (ADDENDUM §11)",
            op()
        ),
    }
}

/// Run `$body` with `$T` bound to the Rust type of `$dtype`, which must be a float
/// (`f32` for F32, F16 and BF16; `f64` for F64;
/// (check it with [`compute_dtype`] first).
macro_rules! dispatch_real {
    ($dtype:expr, $T:ident => $body:expr) => {
        match $dtype {
            $crate::dtype::DType::F32 | $crate::dtype::DType::F16 | $crate::dtype::DType::BF16 => {
                type $T = f32;
                $body
            }
            $crate::dtype::DType::F64 => {
                type $T = f64;
                $body
            }
            other => unreachable!("dispatch_real on {other}: compute_dtype admits floats only"),
        }
    };
}

pub(crate) use dispatch_real;

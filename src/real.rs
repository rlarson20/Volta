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
    + Default
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

/// The operands of an elementwise op, cast to one dtype (ADDENDUM §11 P6): under
/// `DTypePolicy::Promote`, [`DType::promote`] of all of them (PyTorch's `result_type` for
/// dimensioned tensors), reached with a differentiable `to_dtype` on each operand that
/// differs, so its gradient comes back in its own dtype. `float_result` is true division:
/// an integer or bool result dtype becomes F32, as PyTorch's `div` does. Under
/// `DTypePolicy::Strict` mixed operands panic instead. Returns the dtype and the operands.
///
/// # Panics
/// Under `DTypePolicy::Strict`, if the operands' dtypes differ.
#[track_caller]
pub(crate) fn promote_operands(
    op: impl FnOnce() -> String,
    operands: &[&crate::Tensor],
    float_result: bool,
) -> (DType, Vec<crate::Tensor>) {
    let dtypes: Vec<DType> = operands.iter().map(|t| t.dtype()).collect();
    let first = *dtypes.first().expect("an op has at least one input");
    let mixed = dtypes.iter().enumerate().find(|&(_, &d)| d != first);
    if let (Some((i, &d)), crate::DTypePolicy::Strict) = (mixed, crate::dtype_policy()) {
        panic!(
            "{}: {} (input {i}) under DTypePolicy::Strict; cast with to_dtype",
            op(),
            crate::VoltaError::DTypeMismatch {
                expected: first,
                actual: d,
            }
        );
    }
    let common = dtypes.iter().copied().fold(first, DType::promote);
    let target = if float_result && !common.is_float() {
        DType::F32
    } else {
        common
    };
    let cast = operands.iter().map(|t| t.to_dtype(target)).collect();
    (target, cast)
}

/// The mixed-dtype panic of [`compute_dtype`] (ops that need one dtype).
#[track_caller]
fn mixed_dtypes(op: &str, expected: DType, actual: DType, input: usize) -> ! {
    panic!(
        "{op}: {} (input {input}); this op needs one dtype, as in PyTorch: cast with \
         to_dtype (a layer with Module::to_dtype)",
        crate::VoltaError::DTypeMismatch { expected, actual }
    )
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
        mixed_dtypes(&op(), first, d, i);
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

/// A carrier for an element's bits: movement ops copy elements without reading them, so
/// every dtype of one width shares one kernel instance (`u8` for U8 and Bool, `u16` for the
/// half types, `u32` for F32 and I32, `u64` for F64 and I64). Floats, integers and bool
/// all move through these; only arithmetic needs [`Real`].
pub(crate) trait Bits: bytemuck::Pod + Default {}
impl Bits for u8 {}
impl Bits for u16 {}
impl Bits for u32 {}
impl Bits for u64 {}

/// Run `$body` with `$B` bound to the [`Bits`] carrier of `$dtype`'s width. Any dtype.
macro_rules! dispatch_bits {
    ($dtype:expr, $B:ident => $body:expr) => {
        match $crate::dtype::DType::size_of(&$dtype) {
            1 => {
                type $B = u8;
                $body
            }
            2 => {
                type $B = u16;
                $body
            }
            4 => {
                type $B = u32;
                $body
            }
            8 => {
                type $B = u64;
                $body
            }
            n => unreachable!("dispatch_bits: no {n}-byte dtype"),
        }
    };
}

pub(crate) use dispatch_bits;

/// Run `$body` with `$T` bound to `$dtype`'s own element type (any dtype), for a kernel
/// that reads values natively (comparisons). Half types compare as themselves, which is
/// exact: `half`'s ordering is IEEE's.
macro_rules! dispatch_elem {
    ($dtype:expr, $T:ident => $body:expr) => {
        match $dtype {
            $crate::dtype::DType::F16 => {
                type $T = half::f16;
                $body
            }
            $crate::dtype::DType::BF16 => {
                type $T = half::bf16;
                $body
            }
            $crate::dtype::DType::F32 => {
                type $T = f32;
                $body
            }
            $crate::dtype::DType::F64 => {
                type $T = f64;
                $body
            }
            $crate::dtype::DType::I32 => {
                type $T = i32;
                $body
            }
            $crate::dtype::DType::I64 => {
                type $T = i64;
                $body
            }
            $crate::dtype::DType::U8 => {
                type $T = u8;
                $body
            }
            $crate::dtype::DType::Bool => {
                type $T = bool;
                $body
            }
        }
    };
}

pub(crate) use dispatch_elem;

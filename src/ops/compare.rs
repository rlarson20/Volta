//! Elementwise comparisons: Bool results with broadcasting, on any dtype (ADDENDUM §11 P5).
//!
//! A comparison has no gradient. Its output is Bool, which `edge()` never gives a grad fn,
//! so `CompareGradFn` is never attached; `edge()` takes one only to name the op.

use crate::autograd::{GradFn, edge};
use crate::device::Device;
use crate::real::{dispatch_elem, promote_operands};
use crate::shape::Shape;
use crate::storage::Storage;
use crate::{RawTensor, Tensor};

/// An elementwise comparison, as PyTorch's `lt`, `le`, `gt`, `ge`, `eq` and `ne`. NaN
/// compares false except under `Ne` (IEEE, as in PyTorch); `false < true` for Bool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpOp {
    /// `a < b`
    Lt,
    /// `a <= b`
    Le,
    /// `a > b`
    Gt,
    /// `a >= b`
    Ge,
    /// `a == b`
    Eq,
    /// `a != b`
    Ne,
}

impl CmpOp {
    /// The comparison on one pair.
    fn apply<T: PartialOrd>(self, a: T, b: T) -> bool {
        match self {
            Self::Lt => a < b,
            Self::Le => a <= b,
            Self::Gt => a > b,
            Self::Ge => a >= b,
            Self::Eq => a == b,
            Self::Ne => a != b,
        }
    }
}

/// The grad fn `edge()` is handed for a comparison. Never attached (the output is Bool).
#[derive(Clone)]
pub(crate) struct CompareGradFn {
    /// The comparison, for the op name.
    pub(crate) op: CmpOp,
}

impl CompareGradFn {
    /// `"Compare(Lt)"` and so on.
    pub(crate) fn op_name(&self) -> String {
        format!("Compare({:?})", self.op)
    }
}

impl RawTensor {
    /// `a <op> b` elementwise with broadcasting, as a Bool tensor, on any dtype.
    ///
    /// Operands of different dtypes compare in [`crate::DType::promote`] of the two (under
    /// the default [`crate::DTypePolicy`]), as in PyTorch.
    ///
    /// # Panics
    /// If the dtypes differ under `DTypePolicy::Strict`, the shapes don't broadcast, or
    /// an operand is on the GPU: Bool tensors cannot live there until P8.
    #[must_use]
    #[track_caller]
    pub fn compare(a: &Tensor, b: &Tensor, op: CmpOp) -> Tensor {
        let name = || CompareGradFn { op }.op_name();
        // Mixed operands compare in their promoted dtype (`DTypePolicy`), as in PyTorch.
        let (dtype, operands) = promote_operands(name, &[a, b], false);
        let [a, b] = operands.as_slice() else {
            unreachable!("two operands in, two out")
        };
        assert!(
            !a.device().is_gpu() && !b.device().is_gpu(),
            "{}: a comparison returns Bool, and Bool tensors cannot live on the GPU until \
             ADDENDUM §11 P8; compare on the CPU",
            name()
        );
        let (da, sa) = {
            let s = a.borrow();
            (s.data.clone(), s.shape.clone())
        };
        let (db, sb) = {
            let s = b.borrow();
            (s.data.clone(), s.shape.clone())
        };
        let out_shape = Self::broadcast_shape(sa.truncate_rank(), sb.truncate_rank());
        let result: Vec<bool> = dispatch_elem!(dtype, T => {
            let host = |s: &Storage| -> Vec<T> {
                s.as_host::<T>()
                    .unwrap_or_else(|e| panic!("{}: {e}", name()))
                    .to_vec()
            };
            let a = Self::broadcast_to(&host(&da), sa.truncate_rank(), &out_shape);
            let b = Self::broadcast_to(&host(&db), sb.truncate_rank(), &out_shape);
            a.iter().zip(&b).map(|(&x, &y)| op.apply(x, y)).collect()
        });
        edge(
            [a.clone(), b.clone()],
            Storage::from_typed(result),
            Shape::derived(out_shape),
            Device::CPU,
            GradFn::Compare(CompareGradFn { op }),
        )
    }
}

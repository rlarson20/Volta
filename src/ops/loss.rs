//! Fused losses: ops whose PyTorch gradient is not the derivative of any composition of
//! Volta's elementwise ops, so they carry their own `GradFn` variant.

use crate::autograd::{GradFn, edge, is_grad_enabled, no_grad};
use crate::device::Device;
use crate::dtype::DType;
use crate::{RawTensor, Tensor};

/// PyTorch's floor on each `log` in binary cross entropy (`binary_cross_entropy`).
const BCE_LOG_FLOOR: f32 = -100.0;
/// PyTorch's floor on `p * (1 - p)`, the denominator of the BCE gradient (`EPSILON` in
/// `aten/src/ATen/native/Loss.cpp`). Given as f32 and widened, so an f64 loss uses
/// `1e-12` rounded to f32 (a relative difference of about 4e-9).
const BCE_GRAD_EPS: f32 = 1e-12;

/// Backward of [`RawTensor::bce_elementwise`]: PyTorch's
/// `grad * (p - y) / max(p * (1 - p), 1e-12)`. That is the derivative of the unclamped
/// loss wherever `p * (1 - p) >= 1e-12`; at `p = 0` or `p = 1` it stays large and finite
/// where the clamped forward is flat, which is what PyTorch returns. The target gets no
/// gradient (forward rejects one that needs it).
#[derive(Clone)]
pub(crate) struct BceGradFn;

impl BceGradFn {
    /// The gradient for `pred`, computed with tensor ops on the inputs' device; `None`
    /// for `target`.
    #[expect(
        clippy::unused_self,
        reason = "uniform `GradFn` dispatch: every variant's `backward` takes `&self` (DECISIONS.md, #171)"
    )]
    pub(crate) fn backward(&self, out_grad: &RawTensor, parents: &[Tensor]) -> Vec<Option<Tensor>> {
        let [pred, target] = parents else {
            panic!(
                "Bce backward: 2 parents (pred, target), got {}",
                parents.len()
            );
        };
        let _no_grad = no_grad();
        let g = RawTensor::new_with_storage(
            out_grad.data.clone(),
            out_grad.shape.clone(),
            out_grad.device.clone(),
            false,
        );
        let (p, y) = (pred.detach(), target.detach());
        let shape = p.shape();
        let dims = shape.truncate_rank();
        let n = shape.numel();
        let one = p.constant_like(vec![1.0; n], dims);
        let eps = p.constant_like(vec![BCE_GRAD_EPS; n], dims);
        let denom = p.elem_mul(&one.sub(&p)).max_elem(&eps);
        let grad = g.elem_mul(&p.sub(&y)).div(&denom);
        vec![Some(grad), None]
    }
}

impl RawTensor {
    /// Per-element binary cross entropy, PyTorch's
    /// `-(y * max(log p, -100) + (1 - y) * max(log(1 - p), -100))`, with
    /// [`BceGradFn`]'s gradient. Both logs are floored before the multiply, so `p = 0` or
    /// `p = 1` gives a finite loss rather than `0 * -inf = NaN` (S7). The forward runs as
    /// tensor ops, so GPU inputs stay on the GPU.
    ///
    /// # Panics
    /// If `pred` and `target` differ in shape, `target` requires grad while grad is
    /// enabled (PyTorch: "the derivative for 'target' is not implemented"), or a
    /// prediction is outside `[0, 1]` or NaN (PyTorch's range check). The range check
    /// reads `pred` back to the host.
    #[track_caller]
    pub(crate) fn bce_elementwise(pred: &Tensor, target: &Tensor) -> Tensor {
        let shape = pred.shape();
        assert_eq!(
            shape,
            target.shape(),
            "bce_loss: target shape must match pred (no broadcasting, as in PyTorch)"
        );
        assert!(
            !(is_grad_enabled() && target.requires_grad()),
            "bce_loss: the derivative for target is not implemented (as in PyTorch); detach() it"
        );
        let values = pred.borrow().data.to_device(&Device::CPU).cast(DType::F64);
        let values = values.as_host::<f64>().expect("cast to f64");
        if let Some((i, p)) = values
            .iter()
            .enumerate()
            .find(|&(_, p)| !(0.0..=1.0).contains(p))
        {
            panic!("bce_loss: every pred must be in [0, 1], got {p} at flat index {i}");
        }

        let loss = {
            let _no_grad = no_grad();
            let dims = shape.truncate_rank();
            let n = shape.numel();
            let one = pred.constant_like(vec![1.0; n], dims);
            let floor = pred.constant_like(vec![BCE_LOG_FLOOR; n], dims);
            let log_p = pred.log().max_elem(&floor);
            let log_q = one.sub(pred).log().max_elem(&floor);
            // Both logs are finite here, so neither product can be 0 * -inf.
            target
                .elem_mul(&log_p)
                .add(&one.sub(target).elem_mul(&log_q))
                .neg()
        };
        let (data, out_shape, device) = {
            let r = loss.borrow();
            (r.data.clone(), r.shape.clone(), r.device.clone())
        };
        edge(
            [pred.clone(), target.clone()],
            data,
            out_shape,
            device,
            GradFn::Bce(BceGradFn),
        )
    }
}

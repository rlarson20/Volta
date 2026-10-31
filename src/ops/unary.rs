use crate::autograd::{GradFn, edge};
use crate::device::Device;
use crate::device_policy::kernel_failed;
use crate::macros::define_ops;
use crate::storage::Storage;
use crate::{RawTensor, Tensor};

// One row per op: CPU forward, CPU backward, and the GPU pipelines (or `cpu_only`).
// Adding an op is one row here plus, if it has a GPU kernel, a WGSL entry point and a
// `ComputePipelines` field. See `define_ops!` for the row grammar.
define_ops! {
    /// Unary operations: single input, single output
    ///
    /// Each variant's doc gives its derivative.
    unary UnaryOp {
        /// d(-x)/dx = -1
        Neg => |x| -x, |g, _x| -g, gpu(neg, neg_backward);
        /// d(1/x)/dx = -1/x²
        Recip => |x| 1.0 / x, |g, x| -g / (x * x), gpu(recip, recip_backward);
        /// d(√x)/dx = 1/(2√x)
        Sqrt => |x| x.sqrt(), |g, x| g / (2.0 * x.sqrt()), gpu(sqrt, sqrt_backward);
        /// d(2ˣ)/dx = 2ˣ·ln(2)
        Exp2 => |x| x.exp2(),
            |g, x| g * x.exp2() * std::f32::consts::LN_2,
            gpu(exp2, exp2_backward);
        /// d(eˣ)/dx = eˣ
        Exp => |x| x.exp(), |g, x| g * x.exp(), gpu(exp, exp_backward);
        /// d(log₂(x))/dx = 1/(x·ln(2))
        Log2 => |x| x.log2(),
            |g, x| g / (x * std::f32::consts::LN_2),
            gpu(log2, log2_backward);
        /// d(ln(x))/dx = 1/x
        Log => |x| x.ln(), |g, x| g / x, gpu(log, log_backward);
        /// d(sin(x))/dx = cos(x)
        Sin => |x| x.sin(), |g, x| g * x.cos(), gpu(sin, sin_backward);
        /// d(cos(x))/dx = -sin(x)
        Cos => |x| x.cos(), |g, x| -g * x.sin(), gpu(cos, cos_backward);
        /// d(tanh(x))/dx = 1 - tanh²(x)
        Tanh => |x| x.tanh(),
            |g, x| {
                let t = x.tanh();
                g * t.mul_add(-t, 1.0)
            },
            gpu(tanh, tanh_backward);
        /// d(σ(x))/dx = σ(x)·(1-σ(x))
        Sigmoid => |x| 1.0 / (1.0 + (-x).exp()),
            |g, x| {
                let s = 1.0 / (1.0 + (-x).exp());
                g * s * (1.0 - s)
            },
            gpu(sigmoid, sigmoid_backward);
        /// d(max(0,x))/dx = x > 0 ? 1 : 0
        ReLU => |x| x.max(0.0), |g, x| if x > 0.0 { g } else { 0.0 }, gpu(relu, relu_backward);
        /// d(erf(x))/dx = 2/√π · exp(-x²)
        Erf => |x| libm::erff(x),
            |g, x| g * (2.0 / std::f32::consts::PI.sqrt()) * (-x * x).exp(),
            gpu(erf, erf_backward);
    }
}

/// Gradient function for unary operations
///
/// Stores which operation was performed so backward can apply the correct derivative.
#[derive(Clone)]
pub(crate) struct UnaryGradFn {
    op: UnaryOp,
}

impl UnaryGradFn {
    /// The op's name in device errors and fallback events, e.g. `"Unary(Exp)"`.
    pub(crate) fn op_name(&self) -> String {
        format!("Unary({:?})", self.op)
    }

    pub(crate) fn backward(&self, out_grad: &RawTensor, parents: &[Tensor]) -> Vec<Option<Tensor>> {
        let x = parents
            .first()
            .map(|p| p.borrow())
            .expect("unary ops require 1 parent");

        // GPU path when both are on the GPU and the op has a kernel. A `None` from the
        // kernel is then a failure, not a reason to compute on the CPU.
        if out_grad.device.is_gpu() && x.device.is_gpu() && self.op.has_gpu_kernel() {
            let grad_storage = RawTensor::gpu_unary_backward(&out_grad.data, &x.data, self.op)
                .unwrap_or_else(|| kernel_failed(&self.op_name()));
            return vec![Some(RawTensor::new_with_storage(
                grad_storage,
                x.shape.clone(),
                x.device.clone(),
                false,
            ))];
        }

        // CPU path (CPU tensors, or a `cpu_only` op). Chain rule: ∂L/∂x = ∂L/∂y · ∂y/∂x, with the derivative from
        // the `UnaryOp` table.
        let grad_data: Vec<f32> = out_grad
            .data
            .to_host_f32()
            .iter()
            .zip(x.data.to_host_f32().iter())
            .map(|(&g, &x)| self.op.backward(g, x))
            .collect();
        vec![Some(RawTensor::from_shape(
            grad_data,
            x.shape.clone(),
            false,
        ))]
    }
}

// ===== UNARY OPERATIONS =====
impl RawTensor {
    /// Apply a unary operation element-wise
    ///
    /// This is the unified implementation for all unary ops.
    /// Creates a new tensor and sets up gradient tracking if needed.
    #[must_use]
    pub fn unary_op(t: &Tensor, op: UnaryOp) -> Tensor {
        let (data, shape, device) = {
            let s = t.borrow();
            (s.data.clone(), s.shape.clone(), s.device.clone())
        };

        // GPU path when the tensor is on the GPU and the op has a kernel. A `None` from
        // the kernel is then a failure, not a reason to compute on the CPU.
        if device.is_gpu() && op.has_gpu_kernel() {
            let storage = Self::gpu_unary(&data, op)
                .unwrap_or_else(|| kernel_failed(&UnaryGradFn { op }.op_name()));
            return edge(
                [t.clone()],
                storage,
                shape,
                device,
                GradFn::Unary(UnaryGradFn { op }),
            );
        }

        // CPU path (CPU tensor, or a `cpu_only` op on the GPU: `edge()` applies the policy).
        let host_data = data.to_vec();
        let result: Vec<f32> = host_data.iter().map(|&x| op.forward(x)).collect();

        edge(
            [t.clone()],
            Storage::cpu(result),
            shape,
            Device::CPU,
            GradFn::Unary(UnaryGradFn { op }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{check_gradients, manual_seed, randn};

    /// Inputs inside each op's domain, away from kinks (`ReLU` at 0) and poles.
    fn domain_input(op: UnaryOp) -> Tensor {
        manual_seed(0x0123);
        let raw = randn(&[3, 4]);
        let vals: Vec<f32> = raw
            .data()
            .iter()
            .map(|&v| match op {
                // Positive and away from 0: Log/Log2/Sqrt need x > 0, Recip has a pole.
                UnaryOp::Log | UnaryOp::Log2 | UnaryOp::Sqrt | UnaryOp::Recip => 0.5 + v.abs(),
                // |x| >= 0.3 so the finite-difference step never straddles the kink.
                UnaryOp::ReLU => (0.3 + v.abs()).copysign(v),
                UnaryOp::Neg
                | UnaryOp::Exp2
                | UnaryOp::Exp
                | UnaryOp::Sin
                | UnaryOp::Cos
                | UnaryOp::Tanh
                | UnaryOp::Sigmoid
                | UnaryOp::Erf => v,
            })
            .collect();
        RawTensor::new(vals, &[3, 4], true)
    }

    #[test]
    fn unary_gradcheck_every_op() {
        for &op in UnaryOp::ALL {
            let x = domain_input(op);
            let (max_err, _mean, passed) =
                check_gradients(&x, |t| RawTensor::unary_op(t, op).sum(), 1e-2, 1e-2);
            assert!(passed, "{op:?}: gradcheck max_error={max_err:.3e}");
        }
    }

    #[test]
    fn unary_all_lists_each_variant_once() {
        assert_eq!(UnaryOp::ALL.len(), 13);
        for (i, a) in UnaryOp::ALL.iter().enumerate() {
            for b in UnaryOp::ALL.iter().skip(i + 1) {
                assert_ne!(a, b);
            }
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn every_unary_op_has_gpu_pipelines() {
        // Erf used to name kernels that didn't exist (forward silently fell back to CPU,
        // backward panicked). Now every op either has both pipelines or says `cpu_only`.
        let Some(ctx) = crate::gpu::get_gpu_context() else {
            return;
        };
        for &op in UnaryOp::ALL {
            assert!(
                op.gpu_pipelines(ctx.pipelines()).is_some(),
                "{op:?} has no GPU pipelines"
            );
        }
    }
}

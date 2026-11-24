//! `div` and `recip` gradients at very large and very small denominators, where squaring
//! the denominator first overflows or underflows f32 though the true gradient is finite
//! (#92). On the GPU too when an adapter is present (the CPU and GPU used to disagree).

use volta::{RawTensor, Tensor};

#[cfg(feature = "gpu")]
#[path = "common/mod.rs"]
mod common;

/// Relative closeness, for values spanning many orders of magnitude.
fn assert_rel(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (g, w) in got.iter().zip(want) {
        assert!(
            g.is_finite() && ((g - w) / w).abs() < 1e-5,
            "{label}: got {got:?}, want {want:?}"
        );
    }
}

/// `b.grad()` after `sum(a / b)`.
fn div_grad_b(a: &Tensor, b: &Tensor) -> Vec<f32> {
    a.div(b).sum().into_loss().expect("scalar").backward();
    b.grad().expect("b grad")
}

/// `b.grad()` after `sum(recip(b) * w)`.
fn recip_grad(b: &Tensor, w: &Tensor) -> Vec<f32> {
    b.recip()
        .elem_mul(w)
        .sum()
        .into_loss()
        .expect("scalar")
        .backward();
    b.grad().expect("b grad")
}

#[test]
fn div_grad_b_is_finite_at_extreme_denominators() {
    // d(a/b)/db = -a/b²: -1e30/1e50 = -1e-20 (b² overflows), -1e-30/1e-60 = -1e30 (underflows).
    let a = RawTensor::new(vec![1e30, 1e-30], &[2], true);
    let b = RawTensor::new(vec![1e25, 1e-30], &[2], true);
    assert_rel("div grad b", &div_grad_b(&a, &b), &[-1e-20, -1e30]);
    assert_rel("div grad a", &a.grad().expect("a grad"), &[1e-25, 1e30]);
}

#[test]
fn recip_grad_is_finite_at_extreme_inputs() {
    // d(1/b)/db * w = -w/b²: -1e30/1e50 = -1e-20, -1e-30/1e-50 = -1e20.
    let b = RawTensor::new(vec![1e25, 1e-25], &[2], true);
    let w = RawTensor::new(vec![1e30, 1e-30], &[2], false);
    assert_rel("recip grad", &recip_grad(&b, &w), &[-1e-20, -1e20]);
}

/// The broadcast path (`b` of shape `[1]` against `a` of shape `[2]`) sums the b-gradient
/// over the broadcast axis, through the same formula.
#[test]
fn div_grad_b_broadcast_is_finite_at_a_small_denominator() {
    let a = RawTensor::new(vec![1e-30, 2e-30], &[2], true);
    let b = RawTensor::new(vec![1e-30], &[1], true);
    assert_rel("div broadcast grad b", &div_grad_b(&a, &b), &[-3e30]);
}

#[cfg(feature = "gpu")]
mod gpu {
    //! The same cases on the GPU. Each test prints `GPU-RAN` first, because without an
    //! adapter it returns early and still reports `ok`.

    use super::common::{gpu_device, skip_if_no_gpu};
    use super::{assert_rel, div_grad_b, recip_grad};
    use volta::{RawTensor, Tensor};

    /// A GPU leaf with these values.
    fn leaf(v: Vec<f32>, requires_grad: bool) -> Tensor {
        let n = v.len();
        let t = RawTensor::new(v, &[n], false).to_device(gpu_device());
        t.set_requires_grad(requires_grad);
        t
    }

    #[test]
    fn gpu_div_and_recip_grads_are_finite_at_extreme_denominators() {
        if skip_if_no_gpu() {
            return;
        }
        println!("GPU-RAN gpu_div_and_recip_grads_are_finite_at_extreme_denominators");
        let a = leaf(vec![1e30, 1e-30], true);
        let b = leaf(vec![1e25, 1e-30], true);
        assert_rel("gpu div grad b", &div_grad_b(&a, &b), &[-1e-20, -1e30]);
        let b = leaf(vec![1e25, 1e-25], true);
        let w = leaf(vec![1e30, 1e-30], false);
        assert_rel("gpu recip grad", &recip_grad(&b, &w), &[-1e-20, -1e20]);
    }

    /// GPU broadcast backward is a device-policy gap today (the two-pass kernels are
    /// unreachable until #60), so under `PreferGpu` this runs the CPU formula on GPU
    /// inputs. It pins the result for when the `div_broadcast_pass*` kernels (same fix)
    /// are wired in.
    #[test]
    fn gpu_div_grad_b_broadcast_is_finite_at_a_small_denominator() {
        if skip_if_no_gpu() {
            return;
        }
        println!("GPU-RAN gpu_div_grad_b_broadcast_is_finite_at_a_small_denominator");
        let a = leaf(vec![1e-30, 2e-30], true);
        let b = leaf(vec![1e-30], true);
        let _prefer = volta::with_device_policy(volta::DevicePolicy::PreferGpu);
        assert_rel("gpu div broadcast grad b", &div_grad_b(&a, &b), &[-3e30]);
    }
}

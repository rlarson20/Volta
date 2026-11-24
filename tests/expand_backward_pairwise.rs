//! `expand` backward sums each source element's terms pairwise and visits only the
//! broadcast axes (#303, #308).
//!
//! Both backends used to add every incoming gradient into its source element in one f32
//! chain, which loses about `n * 2^-24` relative: 4% low at `n = 2^22` and stuck at 2^24 for
//! a sum of ones. Every conv bias gradient is such a sum over batch x H x W, because
//! `Conv2d` adds its bias through `expand`. The GPU kernel also looped over the whole
//! expanded gradient for each source element (O(input x output)). Each test compares with
//! an f64 reference. GPU tests run on any adapter (lavapipe in a cloud session) and cover
//! both kernels: one thread per source (under 256 terms) and one workgroup per source.
//! Imports use only paths that exist both before and after #341.

#[path = "common/mod.rs"]
mod common;

use volta::nn::{Conv2d, ConvAlgo, Module};
use volta::{Device, RawTensor, Tensor};

/// 2^22 terms: where a single f32 chain is 4% low.
const N: usize = 1 << 22;

/// Relative error allowed against the f64 sum: pairwise over `N` terms is a small multiple
/// of `2^-24 * (128 + 22)`; the old chain was 4e-2.
const REL_TOL: f64 = 1e-5;

/// `0.1f32 * n`, summed exactly (every term is the same f32, so the sum is a product).
fn exact_tenths(n: usize) -> f64 {
    f64::from(0.1f32) * f64::from(u32::try_from(n).expect("n fits u32"))
}

fn assert_rel(label: &str, got: f32, want: f64, tol: f64) {
    let rel = ((f64::from(got) - want) / want).abs();
    assert!(
        rel < tol,
        "{label}: got {got}, f64 reference {want}, relative error {rel:e}"
    );
}

/// `x`'s gradient for the loss `sum(expand(x, new_shape) * w)`, with `x` and `w` on
/// `device`. The gradient of each source element is the sum of `w` over its broadcast
/// positions.
fn expand_grad(x_shape: &[usize], new_shape: &[usize], w: &[f32], device: &Device) -> Vec<f32> {
    let numel: usize = x_shape.iter().product();
    let x = RawTensor::new(vec![1.0; numel], x_shape, false).to_device(device.clone());
    x.set_requires_grad(true);
    let w = RawTensor::new(w.to_vec(), new_shape, false).to_device(device.clone());
    x.expand(new_shape)
        .elem_mul(&w)
        .sum()
        .into_loss()
        .expect("scalar loss")
        .backward();
    x.grad_storage().expect("x has a gradient").to_vec()
}

/// The f64 reference for [`expand_grad`]: `w` summed over the axes where `x_shape` is 1.
fn expand_grad_f64(x_shape: &[usize], new_shape: &[usize], w: &[f32]) -> Vec<f64> {
    let numel: usize = x_shape.iter().product();
    assert_eq!(x_shape.len(), new_shape.len(), "expand keeps the rank");
    let mut out = vec![0.0f64; numel];
    for (i, &wi) in w.iter().enumerate() {
        let mut rem = i;
        let mut src = 0;
        let mut src_stride = 1;
        for (&old, &new) in x_shape.iter().zip(new_shape).rev() {
            let coord = rem % new;
            rem /= new;
            if old != 1 {
                src += coord * src_stride;
            }
            src_stride *= old;
        }
        *out.get_mut(src).expect("source index in range") += f64::from(wi);
    }
    out
}

/// Deterministic weights in [-1, 1) that don't cancel to a small sum by accident: mostly
/// positive (a late-training bias gradient is same-signed), with a varying magnitude.
fn weights(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = ((i * 7919) % 1000) as f32 / 1000.0;
            if i % 5 == 0 { -t } else { t }
        })
        .collect()
}

/// Asserts `got` matches `want` elementwise within `tol` relative to `max(|want|, 1)`.
fn assert_matches_f64(label: &str, got: &[f32], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let err = (f64::from(g) - w).abs() / w.abs().max(1.0);
        assert!(
            err < tol,
            "{label}[{i}]: got {g}, f64 reference {w}, error {err:e}"
        );
    }
}

/// The bias gradient of a 1x1 `Conv2d` over a `[16, 1, 512, 512]` input (2^22 outputs),
/// with upstream gradient 0.1 everywhere.
fn conv_bias_grad(algo: ConvAlgo, device: &Device) -> f32 {
    let mut conv = Conv2d::new_on_device(1, 1, 1, 1, 0, true, device.clone());
    conv.set_algo(algo);
    let x = RawTensor::new(vec![0.5; N], &[16, 1, 512, 512], false).to_device(device.clone());
    let upstream =
        RawTensor::new(vec![0.1; N], &[16, 1, 512, 512], false).to_device(device.clone());
    conv.forward(&x)
        .elem_mul(&upstream)
        .sum()
        .into_loss()
        .expect("scalar loss")
        .backward();
    let params: Vec<Tensor> = conv.parameters();
    let bias = params.get(1).expect("Conv2d with bias has two parameters");
    let grad = bias.grad_storage().expect("bias has a gradient").to_vec();
    assert_eq!(grad.len(), 1);
    *grad.first().expect("one bias element")
}

#[test]
fn cpu_expand_backward_of_scalar_to_2_pow_22() {
    let got = expand_grad(&[1], &[N], &vec![0.1; N], &Device::CPU);
    assert_eq!(got.len(), 1);
    assert_rel(
        "CPU expand [1] -> [2^22] * 0.1",
        got[0],
        exact_tenths(N),
        REL_TOL,
    );
}

#[test]
fn cpu_expand_backward_of_ones_past_2_pow_24_is_exact() {
    // A chain of ones stalls at 2^24; 2^25 is exact in f32 and pairwise.
    let n = 1 << 25;
    let x = RawTensor::new(vec![1.0], &[1], true);
    x.expand(&[n])
        .sum()
        .into_loss()
        .expect("scalar loss")
        .backward();
    let g = x.grad_storage().expect("x has a gradient").to_vec();
    assert_eq!(g, vec![33_554_432.0]);
}

#[test]
fn cpu_expand_backward_mixed_axes_matches_f64() {
    // Broadcast axes 1 and 3, kept axes 0 and 2; inner and outer broadcast both visited.
    for (x_shape, new_shape) in [
        (vec![2, 1, 3, 1], vec![2, 5, 3, 7]),
        (vec![1, 4, 1], vec![300, 4, 9]),
        (vec![3, 1], vec![3, 1]),
    ] {
        let w = weights(new_shape.iter().product());
        let got = expand_grad(&x_shape, &new_shape, &w, &Device::CPU);
        let want = expand_grad_f64(&x_shape, &new_shape, &w);
        assert_matches_f64(
            &format!("CPU {x_shape:?} -> {new_shape:?}"),
            &got,
            &want,
            1e-5,
        );
    }
}

#[test]
fn cpu_conv2d_bias_grad_at_2_pow_22_outputs() {
    for algo in [ConvAlgo::Direct, ConvAlgo::Im2col, ConvAlgo::IGEMM] {
        let got = conv_bias_grad(algo, &Device::CPU);
        assert_rel(
            &format!("CPU Conv2d bias ({algo:?})"),
            got,
            exact_tenths(N),
            REL_TOL,
        );
    }
}

/// The same sums on the GPU, through both `expand_backward` kernels.
#[cfg(feature = "gpu")]
mod gpu {
    use super::common::{gpu_device, skip_if_no_gpu};
    use super::*;

    #[test]
    fn gpu_expand_backward_of_scalar_to_2_pow_22() {
        if skip_if_no_gpu() {
            return;
        }
        let got = expand_grad(&[1], &[N], &vec![0.1; N], &gpu_device());
        assert_eq!(got.len(), 1);
        assert_rel(
            "GPU expand [1] -> [2^22] * 0.1",
            got[0],
            exact_tenths(N),
            REL_TOL,
        );
    }

    #[test]
    fn gpu_expand_backward_both_kernels_match_f64() {
        if skip_if_no_gpu() {
            return;
        }
        for (x_shape, new_shape) in [
            // Under 256 terms per source: one thread each.
            (vec![2, 1, 3, 1], vec![2, 5, 3, 7]),
            (vec![300, 1], vec![300, 4]),
            // 256 or more: one workgroup each, including more sources than one workgroup.
            (vec![2, 1, 3, 1], vec![2, 64, 3, 8]),
            (vec![1, 4, 1], vec![300, 4, 9]),
            (vec![1000, 1], vec![1000, 256]),
        ] {
            let w = weights(new_shape.iter().product());
            let got = expand_grad(&x_shape, &new_shape, &w, &gpu_device());
            let want = expand_grad_f64(&x_shape, &new_shape, &w);
            assert_matches_f64(
                &format!("GPU {x_shape:?} -> {new_shape:?}"),
                &got,
                &want,
                1e-5,
            );
        }
    }

    #[test]
    fn gpu_expand_backward_4096_by_256() {
        // #308's slowest probe (36 s on lavapipe before): now O(output).
        if skip_if_no_gpu() {
            return;
        }
        let (x_shape, new_shape) = ([4096, 1], [4096, 256]);
        let w = weights(4096 * 256);
        let start = std::time::Instant::now();
        let got = expand_grad(&x_shape, &new_shape, &w, &gpu_device());
        eprintln!(
            "GPU expand backward [4096,1] -> [4096,256]: {:?}",
            start.elapsed()
        );
        let want = expand_grad_f64(&x_shape, &new_shape, &w);
        assert_matches_f64("GPU [4096,1] -> [4096,256]", &got, &want, 1e-5);
    }

    #[test]
    fn gpu_conv2d_bias_grad_at_2_pow_22_outputs() {
        if skip_if_no_gpu() {
            return;
        }
        let got = conv_bias_grad(ConvAlgo::Auto, &gpu_device());
        assert_rel("GPU Conv2d bias", got, exact_tenths(N), REL_TOL);
    }
}

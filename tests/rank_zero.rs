//! Rank 0 is a real shape, as in PyTorch (#100): full reductions, a dot product and the
//! losses return shape `[]` (one element), agree with a chain of axis reductions, train
//! through `backward`, and promote as zero-dim operands do in `torch.result_type`.

use volta::{Axis, DType, RawTensor, mse_loss};

/// The rank-0 shape.
fn scalar() -> Vec<usize> {
    Vec::new()
}

/// `sum()` and `sum_dim(.., false)` over every axis give the same shape `[]` and value.
#[test]
fn full_and_chained_axis_sums_agree() {
    let x = RawTensor::new((1..=24).map(|v| v as f32).collect(), &[2, 3, 4], false);
    let full = x.sum();
    let chained = x
        .sum_dim(Axis(2), false)
        .sum_dim(Axis(1), false)
        .sum_dim(Axis(0), false);
    assert_eq!(full.shape(), scalar());
    assert_eq!(chained.shape(), scalar());
    assert_eq!(full.item(), chained.item());
    assert_eq!(full.item(), 300.0);
}

/// `mean()`, `max_reduce()` and a 1-D dot product are rank 0 too.
#[test]
fn mean_max_and_dot_are_rank_zero() {
    let x = RawTensor::new(vec![1.0, 5.0, 3.0], &[3], false);
    assert_eq!(x.mean().shape(), scalar());
    assert_eq!(x.max_reduce().shape(), scalar());
    let dot = x.matmul(&x);
    assert_eq!(dot.shape(), scalar());
    assert_eq!(dot.item(), 35.0);
}

/// A loss is rank 0 and backpropagates; its gradient has the input's shape.
#[test]
fn a_rank_zero_loss_trains() {
    let pred = RawTensor::new(vec![1.0, 2.0, 4.0], &[3], true);
    let target = RawTensor::new(vec![0.0, 2.0, 1.0], &[3], false);
    let loss = mse_loss(&pred, &target);
    assert_eq!(loss.shape(), scalar());
    assert!((loss.item() - 10.0 / 3.0).abs() < 1e-6);
    loss.into_loss().expect("a rank-0 loss").backward();
    let grad = pred.grad().expect("pred has a gradient");
    let want = [2.0 / 3.0, 0.0, 2.0];
    assert_eq!(grad.len(), want.len());
    for (g, w) in grad.iter().zip(want) {
        assert!((g - w).abs() < 1e-6, "{grad:?}");
    }
}

/// A rank-0 operand broadcasts against any shape.
#[test]
fn a_rank_zero_operand_broadcasts() {
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false);
    let s = RawTensor::new(vec![2.0; 3], &[3], false).sum(); // 6.0, shape []
    let y = x.elem_mul(&s);
    assert_eq!(y.shape(), vec![2, 2]);
    assert_eq!(y.data(), vec![6.0, 12.0, 18.0, 24.0]);
}

/// A zero-dim operand promotes only across categories (`torch.result_type`): an f32
/// scalar keeps an f16 tensor in f16, but makes an i32 tensor f32.
#[test]
fn a_rank_zero_operand_promotes_like_torch() {
    let s = RawTensor::new(vec![0.5], &[1], false).sum(); // f32, shape []
    let half = RawTensor::new(vec![1.0, 2.0], &[2], false).to_dtype(DType::F16);
    let prod = half.elem_mul(&s);
    assert_eq!(prod.dtype(), DType::F16);
    assert_eq!(prod.data(), vec![0.5, 1.0]);

    let ints = RawTensor::new(vec![1.0, 2.0], &[2], false).to_dtype(DType::I32);
    assert_eq!(ints.elem_mul(&s).dtype(), DType::F32);

    // Dimensioned against dimensioned is unchanged: f16[2] * f32[1] is f32.
    let one = RawTensor::new(vec![0.5], &[1], false);
    assert_eq!(half.elem_mul(&one).dtype(), DType::F32);
}

/// On the GPU, a rank-0 loss times a `[1]` constant needs no broadcast: both buffers hold
/// one element, so the same-shape kernels run forward and backward under the default
/// `Strict` device policy, with no gap (lavapipe or a real adapter).
#[cfg(feature = "gpu")]
#[test]
fn gpu_rank_zero_times_a_one_element_tensor_runs_on_the_gpu() {
    use volta::Device;
    let Some(gpu) = Device::gpu() else {
        assert!(
            std::env::var_os("VOLTA_REQUIRE_GPU").is_none(),
            "VOLTA_REQUIRE_GPU is set but no GPU is available"
        );
        return;
    };
    let x = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], false).to_device(gpu.clone());
    x.set_requires_grad(true);
    let three = RawTensor::new(vec![3.0], &[1], false).to_device(gpu);
    let y = x.sum().elem_mul(&three);
    assert_eq!(y.shape(), vec![1]);
    assert!(y.device().is_gpu());
    assert_eq!(y.item(), 18.0);
    y.into_loss().expect("one element").backward();
    assert_eq!(x.grad().expect("x has a gradient"), vec![3.0, 3.0, 3.0]);
}

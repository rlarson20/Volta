//! GPU buffer limits and bindings (#306) and outputs left unfilled (#310).
//!
//! The device used to open with wgpu's default limits (a 128 MiB storage binding), and
//! kernels bound a pooled buffer whole, power-of-two slack included, so a `sum` of
//! 2^25 + 1 f32 ones bound a 256 MiB buffer and failed wgpu validation. The device now
//! requests the adapter's limits, each kernel binds only its tensor's bytes, and an upload
//! over the limit is a `VoltaError` from `try_to_device`. Kernel outputs that the shader
//! writes in full are no longer zero-filled from the host, so they start as whatever a
//! pooled buffer held: the stale-pool cases check those ops still compute every element.
//! Runs on any adapter; lavapipe in a cloud session.

#![cfg(feature = "gpu")]

#[path = "common/mod.rs"]
mod common;

use common::{gpu_device, skip_if_no_gpu};
use volta::gpu::max_tensor_bytes;
use volta::{DType, Device, RawTensor, Tensor, VoltaError};

/// WebGPU's guaranteed minimum `max_storage_buffer_binding_size`, and wgpu's default.
const WGPU_DEFAULT_BINDING: u64 = 128 << 20;

/// A GPU tensor's values on the host.
fn host(t: &Tensor) -> Vec<f32> {
    t.to_device(Device::CPU).to_dtype(DType::F32).data()
}

/// The largest f32 tensor a cheap test may build on the host for these checks.
const HOST_BUDGET_BYTES: u64 = 512 << 20;

/// The limit is the adapter's, at least the WebGPU minimum, and it is what the device
/// uses. A tensor of 2^25 + 1 f32 elements (the audit's repro, one element over wgpu's old
/// 128 MiB default) sums when the adapter binds it (Apple GPUs do). Where it does not
/// (lavapipe binds 128 MiB), `try_to_device` refuses it with a `VoltaError` before
/// allocating, and the largest tensor that fits still sums.
#[test]
fn tensors_up_to_the_binding_limit() {
    if skip_if_no_gpu() {
        return;
    }
    let limit = max_tensor_bytes().expect("a GPU is available");
    assert!(
        limit >= WGPU_DEFAULT_BINDING,
        "binding limit {limit} is under WebGPU's minimum"
    );
    let repro = (1usize << 25) + 1;
    let fitting = if (repro as u64) * 4 <= limit {
        repro
    } else {
        // Over the limit: the upload is an error, not a wgpu validation panic.
        let x = RawTensor::new(vec![1.0; repro], &[repro], false);
        match x.try_to_device(gpu_device()) {
            Err(VoltaError::DeviceError(msg)) => {
                assert!(msg.message.contains("storage binding limit"), "{msg}");
            }
            other => panic!("expected a DeviceError, got {:?}", other.map(|t| t.shape())),
        }
        usize::try_from(limit / 4).expect("limit fits usize")
    };
    if (fitting as u64) * 4 > HOST_BUDGET_BYTES {
        eprintln!("skipping the sum: {fitting} elements is over the test's host budget");
        return;
    }
    let x = RawTensor::new(vec![1.0; fitting], &[fitting], false).to_device(gpu_device());
    let total = x.sum().to_device(Device::CPU).item();
    // Above 2^24 not every count is an f32; the spacing below 2^27 is at most 8.
    assert!(
        (f64::from(total) - fitting as f64).abs() <= 8.0,
        "sum of {fitting} ones is {total}"
    );
}

/// `try_to_device` returns the errors `to_device` panics with, before allocating: an F64
/// tensor is refused, and the CPU stays reachable.
#[test]
fn try_to_device_reports_errors() {
    if skip_if_no_gpu() {
        return;
    }
    let x = RawTensor::new(vec![1.0, 2.0], &[2], false).to_dtype(DType::F64);
    match x.try_to_device(gpu_device()) {
        Err(VoltaError::DeviceError(msg)) => assert!(msg.message.contains("not f64"), "{msg}"),
        other => panic!("expected a DeviceError, got {:?}", other.map(|t| t.shape())),
    }
    let back = RawTensor::new(vec![1.0, 2.0], &[2], false)
        .try_to_device(gpu_device())
        .expect("an f32 tensor fits")
        .try_to_device(Device::CPU)
        .expect("the CPU is always reachable");
    assert_eq!(back.data(), vec![1.0, 2.0]);
}

/// Fill the pool's bucket for `len` elements of `dtype` with NaN, then drop it, so the next
/// output of that size is likely a buffer whose old bytes are all NaN.
fn poison_pool(len: usize, dtype: DType) {
    for _ in 0..4 {
        let junk = RawTensor::new(vec![f32::NAN; len], &[len], false)
            .to_dtype(dtype)
            .to_device(gpu_device());
        assert!(host(&junk).iter().all(|v| v.is_nan()));
    }
}

/// The ops whose outputs are no longer zero-filled (#310) still write every element when
/// the pooled buffer they get holds NaN: unary, binary, their backward kernels, compare,
/// where, cast and matmul, at lengths that leave slack in the bucket (and, for f16 and
/// bf16, a padding half-word).
#[test]
fn unfilled_outputs_are_fully_written() {
    if skip_if_no_gpu() {
        return;
    }
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        for len in [5usize, 1001] {
            let a_vals: Vec<f32> = (0..len).map(|i| (i % 7) as f32 * 0.25 + 0.5).collect();
            let b_vals: Vec<f32> = (0..len).map(|i| (i % 5) as f32 * 0.5 + 1.0).collect();
            let cpu_a = RawTensor::new(a_vals.clone(), &[len], false).to_dtype(dtype);
            let cpu_b = RawTensor::new(b_vals.clone(), &[len], false).to_dtype(dtype);
            let a = RawTensor::new(a_vals, &[len], false)
                .to_dtype(dtype)
                .to_device(gpu_device());
            let b = RawTensor::new(b_vals, &[len], false)
                .to_dtype(dtype)
                .to_device(gpu_device());
            for leaf in [&a, &b, &cpu_a, &cpu_b] {
                leaf.set_requires_grad(true);
            }

            let check = |label: &str, g: Vec<f32>, c: Vec<f32>| {
                assert_eq!(g.len(), c.len(), "{label} {dtype} len {len}");
                for (i, (x, y)) in g.iter().zip(&c).enumerate() {
                    assert!(
                        (x - y).abs() <= 1e-2 * y.abs().max(1.0),
                        "{label} {dtype} len {len} [{i}]: GPU {x}, CPU {y}"
                    );
                }
            };

            poison_pool(len, dtype);
            check("exp", host(&a.exp()), host(&cpu_a.exp()));
            poison_pool(len, dtype);
            check("mul", host(&a.elem_mul(&b)), host(&cpu_a.elem_mul(&cpu_b)));
            poison_pool(len, dtype);
            check(
                "cast",
                host(&a.to_dtype(DType::F32)),
                host(&cpu_a.to_dtype(DType::F32)),
            );
            poison_pool(len, dtype);
            check(
                "where",
                host(&a.lt(&b).where_op(&a, &b)),
                host(&cpu_a.lt(&cpu_b).where_op(&cpu_a, &cpu_b)),
            );

            // Backward: unary (exp) and binary (mul) gradient kernels.
            let gpu_loss = a.exp().elem_mul(&b).sum();
            let cpu_loss = cpu_a.exp().elem_mul(&cpu_b).sum();
            poison_pool(len, dtype);
            gpu_loss.into_loss().expect("scalar").backward();
            cpu_loss.into_loss().expect("scalar").backward();
            check(
                "grad a",
                a.grad().expect("a has a gradient"),
                cpu_a.grad().expect("cpu a has a gradient"),
            );
            check(
                "grad b",
                b.grad().expect("b has a gradient"),
                cpu_b.grad().expect("cpu b has a gradient"),
            );
        }
    }

    // Matmul's output: (3 x 4) @ (4 x 5) = 15 elements, a 16-element bucket.
    let av: Vec<f32> = (0..12).map(|i| i as f32 * 0.5).collect();
    let bv: Vec<f32> = (0..20).map(|i| 1.0 - i as f32 * 0.1).collect();
    let cpu = RawTensor::new(av.clone(), &[3, 4], false).matmul(&RawTensor::new(
        bv.clone(),
        &[4, 5],
        false,
    ));
    poison_pool(15, DType::F32);
    let gpu = RawTensor::new(av, &[3, 4], false)
        .to_device(gpu_device())
        .matmul(&RawTensor::new(bv, &[4, 5], false).to_device(gpu_device()));
    let (g, c) = (host(&gpu), cpu.data());
    assert_eq!(g.len(), c.len());
    for (x, y) in g.iter().zip(&c) {
        assert!(
            (x - y).abs() <= 1e-4 * y.abs().max(1.0),
            "matmul: GPU {x}, CPU {y}"
        );
    }
}

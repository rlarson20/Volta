//! Shared helpers for CPU/GPU parity tests.
//!
//! These helpers run the same op on a CPU tensor and a GPU copy of the same
//! tensor and assert that forward (and optionally backward) results match
//! within a tolerance. Tests early-exit cleanly via `skip_if_no_gpu` when
//! no GPU adapter is available, so the file can be compiled and linked
//! everywhere `--features gpu` is enabled.
//!
//! GPU broadcasting is not supported by Volta — every helper takes matched
//! shapes (or single-input variants) only.

#![cfg(feature = "gpu")]
#![allow(dead_code)] // each test file uses only a subset

use volta::gpu::is_gpu_available;
use volta::tensor::TensorOps;
use volta::{Device, RawTensor, Tensor};

pub const TOL_FWD_GPU: f32 = 1e-5;
pub const TOL_BWD_GPU: f32 = 1e-4;
pub const TOL_MATMUL_FWD: f32 = 1e-4;
pub const TOL_MATMUL_BWD: f32 = 1e-3;

/// Returns `true` if the test should bail because no GPU is available.
/// Idiomatic use: `if skip_if_no_gpu() { return; }` at the top of each test.
#[must_use]
pub fn skip_if_no_gpu() -> bool {
    !is_gpu_available()
}

#[must_use]
pub fn gpu_device() -> Device {
    Device::GPU("ParityTest".to_string())
}

/// Build a deterministic CPU tensor with values linearly spaced over `range`.
/// Avoids zero-magnitude inputs that would blow up `log`/`recip`/`sqrt` etc.
#[must_use]
pub fn make_input(shape: &[usize], requires_grad: bool, range: (f32, f32)) -> Tensor {
    let n: usize = shape.iter().product();
    let (lo, hi) = range;
    let denom = (n.max(1)) as f32;
    let data: Vec<f32> = (0..n)
        .map(|i| lo + ((i as f32) / denom) * (hi - lo))
        .collect();
    RawTensor::new(data, shape, requires_grad)
}

/// Element-wise comparison of two flattened tensors with absolute tolerance.
pub fn assert_close(label: &str, cpu: &[f32], gpu: &[f32], tol: f32) {
    assert_eq!(
        cpu.len(),
        gpu.len(),
        "{label}: length mismatch (cpu={}, gpu={})",
        cpu.len(),
        gpu.len()
    );
    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let diff = (c - g).abs();
        assert!(
            diff < tol,
            "{label} mismatch at index {i}: cpu={c}, gpu={g}, diff={diff}, tol={tol}"
        );
    }
}

/// Forward parity for a unary op: same input data on CPU and GPU should
/// produce numerically equivalent outputs.
pub fn assert_unary_parity<F>(op_name: &str, op: F, shape: &[usize], range: (f32, f32), tol: f32)
where
    F: Fn(&Tensor) -> Tensor,
{
    if skip_if_no_gpu() {
        return;
    }
    let x_cpu = make_input(shape, false, range);
    let x_gpu = x_cpu.to_device(gpu_device());

    let y_cpu = op(&x_cpu);
    let y_gpu = op(&x_gpu).to_device(Device::CPU);

    assert_close(
        op_name,
        &y_cpu.borrow().data.to_vec(),
        &y_gpu.borrow().data.to_vec(),
        tol,
    );
}

/// Forward + backward parity for a unary op: also compares ∂L/∂x with L = sum(y).
pub fn assert_unary_parity_backward<F>(
    op_name: &str,
    op: F,
    shape: &[usize],
    range: (f32, f32),
    tol_fwd: f32,
    tol_bwd: f32,
) where
    F: Fn(&Tensor) -> Tensor,
{
    if skip_if_no_gpu() {
        return;
    }
    let x_cpu = make_input(shape, true, range);
    let x_gpu = x_cpu.to_device(gpu_device());

    let y_cpu = op(&x_cpu);
    y_cpu.sum().backward();
    let y_gpu = op(&x_gpu);
    y_gpu.sum().backward();

    assert_close(
        &format!("{op_name} fwd"),
        &y_cpu.borrow().data.to_vec(),
        &y_gpu.to_device(Device::CPU).borrow().data.to_vec(),
        tol_fwd,
    );

    let g_cpu = x_cpu.grad().expect("CPU gradient missing");
    let g_gpu = x_gpu
        .borrow()
        .grad
        .as_ref()
        .expect("GPU gradient missing")
        .to_vec();
    assert_close(&format!("{op_name} bwd"), &g_cpu, &g_gpu, tol_bwd);
}

/// Forward parity for a binary op with matched (non-broadcast) shapes.
pub fn assert_binary_parity<F>(op_name: &str, op: F, shape: &[usize], range: (f32, f32), tol: f32)
where
    F: Fn(&Tensor, &Tensor) -> Tensor,
{
    if skip_if_no_gpu() {
        return;
    }
    let a_cpu = make_input(shape, false, range);
    // Slightly different value space for `b` to avoid degenerate cases.
    let b_cpu = make_input(shape, false, (range.0 + 0.1, range.1 - 0.1));
    let a_gpu = a_cpu.to_device(gpu_device());
    let b_gpu = b_cpu.to_device(gpu_device());

    let c_cpu = op(&a_cpu, &b_cpu);
    let c_gpu = op(&a_gpu, &b_gpu).to_device(Device::CPU);

    assert_close(
        op_name,
        &c_cpu.borrow().data.to_vec(),
        &c_gpu.borrow().data.to_vec(),
        tol,
    );
}

/// Forward + backward parity for a binary op with matched shapes.
pub fn assert_binary_parity_backward<F>(
    op_name: &str,
    op: F,
    shape: &[usize],
    range: (f32, f32),
    tol_fwd: f32,
    tol_bwd: f32,
) where
    F: Fn(&Tensor, &Tensor) -> Tensor,
{
    if skip_if_no_gpu() {
        return;
    }
    let a_cpu = make_input(shape, true, range);
    let b_cpu = make_input(shape, true, (range.0 + 0.1, range.1 - 0.1));
    let a_gpu = a_cpu.to_device(gpu_device());
    let b_gpu = b_cpu.to_device(gpu_device());

    let c_cpu = op(&a_cpu, &b_cpu);
    c_cpu.sum().backward();
    let c_gpu = op(&a_gpu, &b_gpu);
    c_gpu.sum().backward();

    assert_close(
        &format!("{op_name} fwd"),
        &c_cpu.borrow().data.to_vec(),
        &c_gpu.to_device(Device::CPU).borrow().data.to_vec(),
        tol_fwd,
    );

    let ga_cpu = a_cpu.grad().expect("CPU a-grad missing");
    let ga_gpu = a_gpu
        .borrow()
        .grad
        .as_ref()
        .expect("GPU a-grad missing")
        .to_vec();
    assert_close(&format!("{op_name} ∂/∂a"), &ga_cpu, &ga_gpu, tol_bwd);

    let gb_cpu = b_cpu.grad().expect("CPU b-grad missing");
    let gb_gpu = b_gpu
        .borrow()
        .grad
        .as_ref()
        .expect("GPU b-grad missing")
        .to_vec();
    assert_close(&format!("{op_name} ∂/∂b"), &gb_cpu, &gb_gpu, tol_bwd);
}

/// Forward parity for a reduction (sum/mean/max + axis variants).
pub fn assert_reduce_parity<F>(op_name: &str, op: F, shape: &[usize], tol: f32)
where
    F: Fn(&Tensor) -> Tensor,
{
    if skip_if_no_gpu() {
        return;
    }
    let x_cpu = make_input(shape, false, (-2.0, 2.0));
    let x_gpu = x_cpu.to_device(gpu_device());
    let y_cpu = op(&x_cpu);
    let y_gpu = op(&x_gpu).to_device(Device::CPU);
    assert_close(
        op_name,
        &y_cpu.borrow().data.to_vec(),
        &y_gpu.borrow().data.to_vec(),
        tol,
    );
}

/// Forward + backward parity for matmul over a given (`a_shape`, `b_shape`) pair.
/// Useful for sweeping the matmul shape matrix.
pub fn assert_matmul_parity(a_shape: &[usize], b_shape: &[usize], tol_fwd: f32, tol_bwd: f32) {
    if skip_if_no_gpu() {
        return;
    }
    let a_cpu = make_input(a_shape, true, (-1.0, 1.0));
    let b_cpu = make_input(b_shape, true, (-1.0, 1.0));
    let a_gpu = a_cpu.to_device(gpu_device());
    let b_gpu = b_cpu.to_device(gpu_device());

    let c_cpu = a_cpu.matmul(&b_cpu);
    c_cpu.sum().backward();
    let c_gpu = a_gpu.matmul(&b_gpu);
    c_gpu.sum().backward();

    let label = format!("matmul {a_shape:?} @ {b_shape:?}");
    assert_close(
        &format!("{label} fwd"),
        &c_cpu.borrow().data.to_vec(),
        &c_gpu.to_device(Device::CPU).borrow().data.to_vec(),
        tol_fwd,
    );

    let ga_cpu = a_cpu.grad().expect("CPU a-grad missing");
    let ga_gpu = a_gpu
        .borrow()
        .grad
        .as_ref()
        .expect("GPU a-grad missing")
        .to_vec();
    assert_close(&format!("{label} ∂/∂a"), &ga_cpu, &ga_gpu, tol_bwd);

    let gb_cpu = b_cpu.grad().expect("CPU b-grad missing");
    let gb_gpu = b_gpu
        .borrow()
        .grad
        .as_ref()
        .expect("GPU b-grad missing")
        .to_vec();
    assert_close(&format!("{label} ∂/∂b"), &gb_cpu, &gb_gpu, tol_bwd);
}

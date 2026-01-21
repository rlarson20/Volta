use crate::autograd::GradFn;
use crate::{RawTensor, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

// ===== TRANSPOSE GRADIENT =====

/// Gradient function for 2D transpose
///
/// Transpose is its own inverse: (A^T)^T = A
/// So the gradient just transposes back.
pub struct TransposeGradFn;

impl GradFn for TransposeGradFn {
    fn backward(&self, out_grad: &RawTensor, _parents: &[Tensor]) -> Vec<Option<Tensor>> {
        let transposed = RawTensor::transpose_2d(&out_grad.data, &out_grad.shape);
        let dim0 = out_grad.shape.first().copied().unwrap_or(1);
        let dim1 = out_grad.shape.get(1).copied().unwrap_or(1);
        let new_shape = vec![dim1, dim0];
        vec![Some(RawTensor::new(transposed, &new_shape, false))]
    }

    fn clone_box(&self) -> Box<dyn GradFn> {
        Box::new(TransposeGradFn)
    }
}

// ===== MATRIX MULTIPLICATION =====

impl RawTensor {
    /// Transpose a 2D matrix
    ///
    /// For shape [m, n], produces shape [n, m]
    pub(crate) fn transpose_2d(data: &[f32], shape: &[usize]) -> Vec<f32> {
        assert_eq!(shape.len(), 2, "Transpose expects 2D shape");
        let m = shape.first().copied().unwrap_or(1);
        let n = shape.get(1).copied().unwrap_or(1);
        let mut result = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                if let Some(src) = data.get(i * n + j)
                    && let Some(dst) = result.get_mut(j * m + i)
                {
                    *dst = *src;
                }
            }
        }
        result
    }

    // Helper: raw matmul computation
    /// Raw matrix multiplication: (m,k) @ (k,n) -> (m,n)
    /// Uses `cblas_sgemm` on macOS, `matrixmultiply::sgemm` elsewhere
    #[must_use]
    pub fn matmul_raw(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        #[cfg(all(feature = "accelerate", target_os = "macos"))]
        {
            unsafe extern "C" {
                fn cblas_sgemm(
                    layout: i32,
                    trans_a: i32,
                    trans_b: i32,
                    m: i32,
                    n: i32,
                    k: i32,
                    alpha: f32,
                    a: *const f32,
                    lda: i32,
                    b: *const f32,
                    ldb: i32,
                    beta: f32,
                    c: *mut f32,
                    ldc: i32,
                );
            }

            let mut result = vec![0.0; m * n];
            unsafe {
                cblas_sgemm(
                    101, // CblasRowMajor
                    111, // CblasNoTrans
                    111, // CblasNoTrans
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    a.as_ptr(),
                    k as i32,
                    b.as_ptr(),
                    n as i32,
                    0.0,
                    result.as_mut_ptr(),
                    n as i32,
                );
            }
            result
        }
        #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
        {
            // Use matrixmultiply crate for portable performance
            let mut result = vec![0.0; m * n];
            unsafe {
                matrixmultiply::sgemm(
                    m,
                    k,
                    n,
                    1.0,
                    a.as_ptr(),
                    k as isize,
                    1,
                    b.as_ptr(),
                    n as isize,
                    1,
                    0.0,
                    result.as_mut_ptr(),
                    n as isize,
                    1,
                );
            }
            result
        }
    }

    /// Matrix multiplication with multiple cases
    ///
    /// Supports:
    /// - (m,n) @ (n,p) -> (m,p)  [standard matmul]
    /// - (m,n) @ (n,) -> (m,)    [matrix-vector]
    /// - (n,) @ (n,p) -> (p,)    [vector-matrix]
    /// - (n,) @ (n,) -> scalar   [dot product]
    ///   TODO: since i updated the device stuff, fix the unused vars since it'll probably be
    ///   relevant somewhere
    /// # Panics
    /// Dimension mismatch
    pub fn matmul(self_t: &Tensor, other: &Tensor) -> Tensor {
        let (data_a, shape_a, req_a, _dev_a) = {
            let s = self_t.borrow();
            (
                s.data.clone(),
                s.shape.clone(),
                s.requires_grad,
                s.device.clone(),
            )
        };
        let (data_b, shape_b, req_b, _dev_b) = {
            let o = other.borrow();
            (
                o.data.clone(),
                o.shape.clone(),
                o.requires_grad,
                o.device.clone(),
            )
        };

        // Handle different cases
        match (shape_a.len(), shape_b.len()) {
            (2, 2) => {
                // Standard 2D matmul: (m,n) @ (n,p) -> (m,p)
                let m = shape_a.first().copied().unwrap_or(1);
                let n = shape_a.get(1).copied().unwrap_or(1);
                let n2 = shape_b.first().copied().unwrap_or(1);
                let p = shape_b.get(1).copied().unwrap_or(1);
                assert_eq!(n, n2, "Matmul dimension mismatch: ({m},{n}) @ ({n2},{p})");

                // If both inputs live on the same GPU device, try the GPU path first.
                // Fallback to the existing CPU implementation if anything fails.
                #[cfg(feature = "gpu")]
                {
                    if let Some(device) = RawTensor::common_gpu_device(&[self_t, other]) {
                        if let Some(storage) = Self::gpu_matmul(&data_a, &data_b, m, n, p) {
                            let requires_grad = req_a || req_b;
                            let out = Rc::new(RefCell::new(RawTensor {
                                data: storage,
                                shape: vec![m, p],
                                grad: None,
                                requires_grad,
                                grad_fn: None,
                                parents: vec![self_t.clone(), other.clone()],
                                device: device.clone(),
                            }));
                            if requires_grad {
                                out.borrow_mut().grad_fn = Some(Box::new(MatMulGradFn));
                            }
                            return out;
                        }
                        eprintln!("Warning: GPU matmul requested but failed; falling back to CPU");
                    }
                }

                // CPU fallback: BLAS/Accelerate or matrixmultiply on host data.
                let result_data = Self::matmul_raw(&data_a, &data_b, m, n, p);
                let out = Self::new(result_data, &[m, p], req_a || req_b);

                if out.borrow().requires_grad {
                    out.borrow_mut().parents = vec![self_t.clone(), other.clone()];
                    out.borrow_mut().grad_fn = Some(Box::new(MatMulGradFn));
                }
                out
            }
            (2, 1) => {
                // Matrix-vector: (m,n) @ (n,) -> (m,)
                let m = shape_a.first().copied().unwrap_or(1);
                let n = shape_a.get(1).copied().unwrap_or(1);
                let n2 = shape_b.first().copied().unwrap_or(1);
                assert_eq!(n, n2, "Matmul dimension mismatch: ({m},{n}) @ ({n2})");

                let mut result_data = vec![0.0; m];
                for i in 0..m {
                    let mut sum = 0.0;
                    for j in 0..n {
                        if let Some(a_val) = data_a.get(i * n + j)
                            && let Some(b_val) = data_b.get(j)
                        {
                            sum += *a_val * *b_val;
                        }
                    }
                    if let Some(slot) = result_data.get_mut(i) {
                        *slot = sum;
                    }
                }

                let out = Self::new(result_data, &[m], req_a || req_b);

                if out.borrow().requires_grad {
                    out.borrow_mut().parents = vec![self_t.clone(), other.clone()];
                    out.borrow_mut().grad_fn = Some(Box::new(MatMulGradFn));
                }
                out
            }
            (1, 2) => {
                // Vector-matrix: (n,) @ (n,p) -> (p,)
                let n = shape_a.first().copied().unwrap_or(1);
                let n2 = shape_b.first().copied().unwrap_or(1);
                let p = shape_b.get(1).copied().unwrap_or(1);
                assert_eq!(n, n2, "Matmul dimension mismatch: ({n}) @ ({n2},{p})");

                let mut result_data = vec![0.0; p];
                for j in 0..p {
                    let mut sum = 0.0;
                    for i in 0..n {
                        if let Some(a_val) = data_a.get(i)
                            && let Some(b_val) = data_b.get(i * p + j)
                        {
                            sum += *a_val * *b_val;
                        }
                    }

                    if let Some(slot) = result_data.get_mut(j) {
                        *slot = sum;
                    }
                }

                let out = Self::new(result_data, &[p], req_a || req_b);

                if out.borrow().requires_grad {
                    out.borrow_mut().parents = vec![self_t.clone(), other.clone()];
                    out.borrow_mut().grad_fn = Some(Box::new(MatMulGradFn));
                }
                out
            }
            (1, 1) => {
                // Dot product: (n,) @ (n,) -> scalar
                let n = shape_a.first().copied().unwrap_or(1);
                let n2 = shape_b.first().copied().unwrap_or(1);
                assert_eq!(n, n2, "Dot product dimension mismatch: ({n}) @ ({n2})");

                let sum: f32 = data_a.iter().zip(&data_b).map(|(a, b)| a * b).sum();
                let out = Self::new(vec![sum], &[1], req_a || req_b);

                if out.borrow().requires_grad {
                    out.borrow_mut().parents = vec![self_t.clone(), other.clone()];
                    out.borrow_mut().grad_fn = Some(Box::new(MatMulGradFn));
                }
                out
            }
            _ => {
                // Batched MatMul: (B, M, K) @ (B, K, N) -> (B, M, N)
                // We require strict matching of batch dimensions for now.
                let rank_a = shape_a.len();
                let rank_b = shape_b.len();
                assert!(
                    rank_a >= 3 && rank_b >= 3,
                    "Batched matmul requires rank >= 3"
                );

                let m = *shape_a.get(rank_a.wrapping_sub(2)).unwrap_or(&1);
                let k = *shape_a.get(rank_a.wrapping_sub(1)).unwrap_or(&1);
                let k2 = *shape_b.get(rank_b.wrapping_sub(2)).unwrap_or(&1);
                let n = *shape_b.get(rank_b.wrapping_sub(1)).unwrap_or(&1);
                assert_eq!(k, k2, "Matmul dimension mismatch in batch");

                // 1. Broadcast batch dimensions
                let batch_a = shape_a.get(..rank_a.saturating_sub(2)).unwrap_or(&[]);
                let batch_b = shape_b.get(..rank_b.saturating_sub(2)).unwrap_or(&[]);
                let batch_out = Self::broadcast_shape(batch_a, batch_b);

                // 2. Expand inputs to broadcasted batch shape
                // Target shapes for A: [*batch_out, m, k]
                let mut target_a = batch_out.clone();
                target_a.extend_from_slice(&[m, k]);
                let data_a_expanded = Self::broadcast_to(&data_a, &shape_a, &target_a);

                let mut target_b = batch_out.clone();
                target_b.extend_from_slice(&[k, n]);
                let data_b_expanded = Self::broadcast_to(&data_b, &shape_b, &target_b);

                let batch_count: usize = batch_out.iter().product();

                let mut result_data = Vec::with_capacity(batch_count * m * n);

                let stride_a = m * k;
                let stride_b = k * n;

                for b in 0..batch_count {
                    let start_a = b * stride_a;
                    let start_b = b * stride_b;
                    if let Some(slice_a) = data_a_expanded.get(start_a..start_a + stride_a)
                        && let Some(slice_b) = data_b_expanded.get(start_b..start_b + stride_b)
                    {
                        let chunk_result = Self::matmul_raw(slice_a, slice_b, m, k, n);
                        result_data.extend_from_slice(&chunk_result);
                    }
                }

                let mut out_shape = batch_out;
                out_shape.push(m);
                out_shape.push(n);

                let out = Self::new(result_data, &out_shape, req_a || req_b);
                if out.borrow().requires_grad {
                    out.borrow_mut().parents = vec![self_t.clone(), other.clone()];
                    out.borrow_mut().grad_fn = Some(Box::new(MatMulGradFn));
                }
                out
            }
        }
    }

    /// Transpose a 2D tensor
    /// # Panics
    /// Non-2d tensor
    pub fn transpose(self_t: &Tensor) -> Tensor {
        let (data, shape, req_grad) = {
            let s = self_t.borrow();
            (s.data.clone(), s.shape.clone(), s.requires_grad)
        };

        assert_eq!(shape.len(), 2, "Transpose expects 2D tensor");

        let transposed_data = Self::transpose_2d(&data, &shape);
        let dim0 = shape.first().copied().unwrap_or(1);
        let dim1 = shape.get(1).copied().unwrap_or(1);
        let new_shape = vec![dim1, dim0];

        let out = Self::new(transposed_data, &new_shape, req_grad);

        if req_grad {
            out.borrow_mut().parents = vec![self_t.clone()];
            // Use movement grad with inverse permutation [1,0]
            out.borrow_mut().grad_fn = Some(Box::new(TransposeGradFn));
        }
        out
    }
}

/// Gradient function for matrix multiplication
///
/// For z = x @ y:
/// - ∂L/∂x = ∂L/∂z @ y^T
/// - ∂L/∂y = x^T @ ∂L/∂z
///
/// Handles multiple cases: 2D×2D, 2D×1D, 1D×2D, 1D×1D (dot product)
pub struct MatMulGradFn;

impl GradFn for MatMulGradFn {
    fn backward(&self, out_grad: &RawTensor, parents: &[Tensor]) -> Vec<Option<Tensor>> {
        let x = parents
            .first()
            .map(|p| p.borrow())
            .expect("matmul requires 2 parents");
        let y = parents
            .get(1)
            .map(|p| p.borrow())
            .expect("matmul requires 2 parents");

        // Check if we can do GPU backward (same pattern as unary/binary)
        #[cfg(feature = "gpu")]
        {
            // For GPU backward, we need:
            // - out_grad on GPU
            // - The corresponding input (x for grad_x, y for grad_y) on GPU
            // - Same device for all involved tensors
            let gpu_available = out_grad.device.is_gpu()
                && ((x.device.is_gpu() && y.device.is_gpu())
                    || (!x.requires_grad || !y.requires_grad));

            if gpu_available {
                // For MVP, only handle the standard 2D×2D case on GPU
                if x.shape.len() == 2 && y.shape.len() == 2 {
                    let m = out_grad.shape.first().copied().unwrap_or(1);
                    let n = out_grad.shape.get(1).copied().unwrap_or(1);
                    let k = y.shape.first().copied().unwrap_or(1); // y is (k, n)

                    let grad_x = if x.requires_grad {
                        if let Some(storage) =
                            RawTensor::gpu_matmul_backward_a(&out_grad.data, &y.data, m, k, n)
                        {
                            Some(RawTensor::new_with_storage(
                                storage,
                                &x.shape,
                                x.device.clone(),
                                false,
                            ))
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    let grad_y = if y.requires_grad {
                        if let Some(storage) =
                            RawTensor::gpu_matmul_backward_b(&x.data, &out_grad.data, m, k, n)
                        {
                            Some(RawTensor::new_with_storage(
                                storage,
                                &y.shape,
                                y.device.clone(),
                                false,
                            ))
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    return vec![grad_x, grad_y];
                }
                // For other cases (matrix-vector, etc.), fall through to CPU implementation
            }
        }

        // For z = x @ y where x: (m,n), y: (n,p), z: (m,p)
        // ∂L/∂x = ∂L/∂z @ y^T  -> (m,p) @ (p,n) = (m,n)
        // ∂L/∂y = x^T @ ∂L/∂z  -> (n,m) @ (m,p) = (n,p)

        let grad_x = if x.requires_grad {
            match (x.shape.len(), y.shape.len()) {
                (2, 2) => {
                    // Standard 2D: ∂L/∂x = out_grad @ y^T
                    let y_t = RawTensor::transpose_2d(&y.data, &y.shape);
                    let m = out_grad.shape.first().copied().unwrap_or(1);
                    let n = out_grad.shape.get(1).copied().unwrap_or(1);
                    let k = y.shape.first().copied().unwrap_or(1);
                    let grad_data = RawTensor::matmul_raw(&out_grad.data, &y_t, m, n, k);
                    Some(RawTensor::new(grad_data, &x.shape, false))
                }
                (2, 1) => {
                    // Matrix-vector: (m,n) @ (n,) -> (m,)
                    // ∂L/∂x = ∂L/∂z[:,None] @ v[None,:] = outer(out_grad, v)
                    let m = x.shape.first().copied().unwrap_or(1);
                    let n = x.shape.get(1).copied().unwrap_or(1);
                    let mut grad_data = vec![0.0; m * n];
                    for i in 0..m {
                        let gz_i = out_grad.data.get(i).copied().unwrap_or(0.0);
                        for j in 0..n {
                            if let Some(y_val) = y.data.get(j)
                                && let Some(slot) = grad_data.get_mut(i * n + j)
                            {
                                *slot = gz_i * *y_val;
                            }
                        }
                    }
                    Some(RawTensor::new(grad_data, &x.shape, false))
                }
                (1, 2) => {
                    // Vector-matrix: (n,) @ (n,p) -> (p,)
                    // ∂L/∂x: out_grad is (p,), y is (n,p)
                    // grad_x = out_grad @ y^T -> (p,) @ (p,n) -> (n,)
                    let len_x = x.shape.first().copied().unwrap_or(1);
                    let len_y_p = y.shape.get(1).copied().unwrap_or(1);
                    let mut grad_data = vec![0.0; len_x];
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..len_x {
                        for j in 0..len_y_p {
                            if let Some(og) = out_grad.data.get(j)
                                && let Some(y_val) = y.data.get(i * len_y_p + j)
                                && let Some(slot) = grad_data.get_mut(i)
                            {
                                *slot += *og * *y_val;
                            }
                        }
                    }
                    Some(RawTensor::new(grad_data, &x.shape, false))
                }
                (1, 1) => {
                    // Dot: (n,) @ (n,) -> scalar
                    // ∂L/∂x = out_grad * y
                    let og = out_grad.data.first().copied().unwrap_or(0.0);
                    let grad_data: Vec<f32> = y.data.iter().map(|&v| og * v).collect();
                    Some(RawTensor::new(grad_data, &x.shape, false))
                }
                _ => {
                    // Batched case: (B, M, K) @ (B, K, N) -> (B, M, N)
                    // dL/dx = dL/dz @ y^T
                    // Performed batch-wise
                    let rank = x.shape.len();
                    let m = *x.shape.get(rank.wrapping_sub(2)).unwrap_or(&1);
                    let k = *x.shape.get(rank.wrapping_sub(1)).unwrap_or(&1);
                    let n = *y.shape.get(rank.wrapping_sub(1)).unwrap_or(&1); // y is (B, K, N)

                    // Batch dimensions of output
                    let batch_dims_out = out_grad
                        .shape
                        .get(..out_grad.shape.len().saturating_sub(2))
                        .unwrap_or(&[]);
                    let batch_count: usize = batch_dims_out.iter().product();

                    // We need y broadcasted to match out_grad's batch dims + [K, N]
                    let mut target_y_shape = batch_dims_out.to_vec();
                    target_y_shape.extend_from_slice(&[k, n]);
                    let y_data_expanded =
                        RawTensor::broadcast_to(&y.data, &y.shape, &target_y_shape);

                    let stride_out = m * n;
                    let stride_y = k * n;

                    let mut grad_data_expanded = Vec::with_capacity(batch_count * m * k);

                    for b in 0..batch_count {
                        if let Some(out_slice) =
                            out_grad.data.get(b * stride_out..(b + 1) * stride_out)
                            && let Some(y_slice) =
                                y_data_expanded.get(b * stride_y..(b + 1) * stride_y)
                        {
                            // Transpose y_slice (K, N) -> (N, K)
                            // But for matmul_raw we need (M,N) @ (N,K) -> (M,K)
                            // We can use transpose_2d helper on the slice
                            let y_t = RawTensor::transpose_2d(y_slice, &[k, n]);
                            let chunk = RawTensor::matmul_raw(out_slice, &y_t, m, n, k);
                            grad_data_expanded.extend_from_slice(&chunk);
                        }
                    }
                    // Reduce gradients if x was broadcast
                    // grad_data_expanded has shape [*batch_out, m, k]
                    let mut grad_x_full_shape = batch_dims_out.to_vec();
                    grad_x_full_shape.extend_from_slice(&[m, k]);

                    let grad_reduced = RawTensor::sum_over_broadcast_dims(
                        &grad_data_expanded,
                        &grad_x_full_shape,
                        &x.shape,
                    );

                    Some(RawTensor::new(grad_reduced, &x.shape, false))
                }
            }
        } else {
            None
        };

        let grad_y = if y.requires_grad {
            match (x.shape.len(), y.shape.len()) {
                (2, 2) => {
                    let x_t = RawTensor::transpose_2d(&x.data, &x.shape);
                    let n = x.shape.get(1).copied().unwrap_or(1);
                    let m = x.shape.first().copied().unwrap_or(1);
                    let p = out_grad.shape.get(1).copied().unwrap_or(1);
                    let grad_data = RawTensor::matmul_raw(&x_t, &out_grad.data, n, m, p);
                    Some(RawTensor::new(grad_data, &y.shape, false))
                }
                (2, 1) => {
                    // Matrix-vector: (m,n) @ (n,) -> (m,)
                    // ∂L/∂v = X^T @ ∂L/∂z -> (n,)
                    let m = x.shape.first().copied().unwrap_or(1);
                    let n = x.shape.get(1).copied().unwrap_or(1);
                    let mut grad_data = vec![0.0; n];
                    #[allow(clippy::needless_range_loop)]
                    for j in 0..n {
                        let mut sum = 0.0;
                        for i in 0..m {
                            if let Some(x_val) = x.data.get(i * n + j)
                                && let Some(og) = out_grad.data.get(i)
                            {
                                sum += *x_val * *og;
                            }
                        }
                        if let Some(slot) = grad_data.get_mut(j) {
                            *slot = sum;
                        }
                    }
                    Some(RawTensor::new(grad_data, &y.shape, false))
                }
                (1, 2) => {
                    // grad_y = x^T @ out_grad -> (n,1) @ (1,p) -> (n,p)
                    let dim0 = y.shape.first().copied().unwrap_or(1);
                    let dim1 = y.shape.get(1).copied().unwrap_or(1);
                    let mut grad_data = vec![0.0; dim0 * dim1];
                    for i in 0..dim0 {
                        for j in 0..dim1 {
                            if let Some(x_val) = x.data.get(i)
                                && let Some(og) = out_grad.data.get(j)
                                && let Some(slot) = grad_data.get_mut(i * dim1 + j)
                            {
                                *slot = *x_val * *og;
                            }
                        }
                    }
                    Some(RawTensor::new(grad_data, &y.shape, false))
                }
                (1, 1) => {
                    // Dot: (n,) @ (n,) -> scalar
                    let og = out_grad.data.first().copied().unwrap_or(0.0);
                    let grad_data: Vec<f32> = x.data.iter().map(|&u| og * u).collect();
                    Some(RawTensor::new(grad_data, &y.shape, false))
                }
                _ => {
                    // Batched case: (B, M, K) @ (B, K, N) -> (B, M, N)
                    // dL/dy = x^T @ dL/dz
                    let rank = y.shape.len();
                    let m = *x.shape.get(rank.wrapping_sub(2)).unwrap_or(&1);
                    let k = *x.shape.get(rank.wrapping_sub(1)).unwrap_or(&1);
                    let n = *y.shape.get(rank.wrapping_sub(1)).unwrap_or(&1);

                    let batch_dims_out = out_grad
                        .shape
                        .get(..out_grad.shape.len().saturating_sub(2))
                        .unwrap_or(&[]);
                    let batch_count: usize = batch_dims_out.iter().product();

                    // Expand X to match output batch dims
                    let mut target_x_shape = batch_dims_out.to_vec();
                    target_x_shape.extend_from_slice(&[m, k]);
                    let x_data_expanded =
                        RawTensor::broadcast_to(&x.data, &x.shape, &target_x_shape);

                    let stride_x = m * k;
                    let stride_out = m * n;

                    let mut grad_data_expanded = Vec::with_capacity(batch_count * k * n);

                    for b in 0..batch_count {
                        if let Some(x_slice) = x_data_expanded.get(b * stride_x..(b + 1) * stride_x)
                            && let Some(out_slice) =
                                out_grad.data.get(b * stride_out..(b + 1) * stride_out)
                        {
                            let x_t = RawTensor::transpose_2d(x_slice, &[m, k]);
                            // (K, M) @ (M, N) -> (K, N)
                            let chunk = RawTensor::matmul_raw(&x_t, out_slice, k, m, n);
                            grad_data_expanded.extend_from_slice(&chunk);
                        }
                    }

                    // Reduce gradients if y was broadcast
                    let mut grad_y_full_shape = batch_dims_out.to_vec();
                    grad_y_full_shape.extend_from_slice(&[k, n]);

                    let grad_reduced = RawTensor::sum_over_broadcast_dims(
                        &grad_data_expanded,
                        &grad_y_full_shape,
                        &y.shape,
                    );

                    Some(RawTensor::new(grad_reduced, &y.shape, false))
                }
            }
        } else {
            None
        };

        vec![grad_x, grad_y]
    }

    fn clone_box(&self) -> Box<dyn GradFn> {
        Box::new(MatMulGradFn)
    }
}

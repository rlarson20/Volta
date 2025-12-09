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
        let new_shape = vec![out_grad.shape[1], out_grad.shape[0]];
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
        let (m, n) = (shape[0], shape[1]);
        let mut result = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                result[j * m + i] = data[i * n + j];
            }
        }
        result
    }

    // Helper: raw matmul computation
    /// Raw matrix multiplication: (m,k) @ (k,n) -> (m,n)
    /// Uses cblas_sgemm on macOS, matrixmultiply::sgemm elsewhere
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
    pub fn matmul(self_t: &Tensor, other: &Tensor) -> Tensor {
        let (data_a, shape_a, req_a, dev_a) = {
            let s = self_t.borrow();
            (
                s.data.clone(),
                s.shape.clone(),
                s.requires_grad,
                s.device.clone(),
            )
        };
        let (data_b, shape_b, req_b, dev_b) = {
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
                let (m, n) = (shape_a[0], shape_a[1]);
                let (n2, p) = (shape_b[0], shape_b[1]);
                assert_eq!(
                    n, n2,
                    "Matmul dimension mismatch: ({},{}) @ ({},{})",
                    m, n, n2, p
                );

                // If both inputs live on the same GPU device, try the GPU path first.
                // Fallback to the existing CPU implementation if anything fails.
                #[cfg(feature = "gpu")]
                {
                    if dev_a.is_gpu() && dev_b.is_gpu() && dev_a == dev_b {
                        if let Some(storage) = Self::gpu_matmul(&data_a, &data_b, m, n, p) {
                            let requires_grad = req_a || req_b;
                            let out = Rc::new(RefCell::new(RawTensor {
                                data: storage,
                                shape: vec![m, p],
                                grad: None,
                                requires_grad,
                                grad_fn: None,
                                parents: vec![self_t.clone(), other.clone()],
                                device: dev_a.clone(),
                            }));
                            if requires_grad {
                                out.borrow_mut().grad_fn = Some(Box::new(MatMulGradFn));
                            }
                            return out;
                        } else {
                            eprintln!(
                                "Warning: GPU matmul requested but failed; falling back to CPU"
                            );
                        }
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
                let (m, n) = (shape_a[0], shape_a[1]);
                let n2 = shape_b[0];
                assert_eq!(n, n2, "Matmul dimension mismatch: ({},{}) @ ({})", m, n, n2);

                let mut result_data = vec![0.0; m];
                for i in 0..m {
                    let mut sum = 0.0;
                    for j in 0..n {
                        sum += data_a[i * n + j] * data_b[j];
                    }
                    result_data[i] = sum;
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
                let n = shape_a[0];
                let (n2, p) = (shape_b[0], shape_b[1]);
                assert_eq!(n, n2, "Matmul dimension mismatch: ({}) @ ({},{})", n, n2, p);

                let mut result_data = vec![0.0; p];
                for j in 0..p {
                    let mut sum = 0.0;
                    for i in 0..n {
                        sum += data_a[i] * data_b[i * p + j];
                    }
                    result_data[j] = sum;
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
                let n = shape_a[0];
                let n2 = shape_b[0];
                assert_eq!(n, n2, "Dot product dimension mismatch: ({}) @ ({})", n, n2);

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

                let m = shape_a[rank_a - 2];
                let k = shape_a[rank_a - 1];
                let k2 = shape_b[rank_b - 2];
                let n = shape_b[rank_b - 1];
                assert_eq!(k, k2, "Matmul dimension mismatch in batch");

                // 1. Broadcast batch dimensions
                let batch_a = &shape_a[..rank_a - 2];
                let batch_b = &shape_b[..rank_b - 2];
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
                    let slice_a = &data_a_expanded[start_a..start_a + stride_a];
                    let slice_b = &data_b_expanded[start_b..start_b + stride_b];

                    let chunk_result = Self::matmul_raw(slice_a, slice_b, m, k, n);
                    result_data.extend_from_slice(&chunk_result);
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
    pub fn transpose(self_t: &Tensor) -> Tensor {
        let (data, shape, req_grad) = {
            let s = self_t.borrow();
            (s.data.clone(), s.shape.clone(), s.requires_grad)
        };

        assert_eq!(shape.len(), 2, "Transpose expects 2D tensor");

        let transposed_data = Self::transpose_2d(&data, &shape);
        let new_shape = vec![shape[1], shape[0]];

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
        let x = parents[0].borrow();
        let y = parents[1].borrow();

        // For z = x @ y where x: (m,n), y: (n,p), z: (m,p)
        // ∂L/∂x = ∂L/∂z @ y^T  -> (m,p) @ (p,n) = (m,n)
        // ∂L/∂y = x^T @ ∂L/∂z  -> (n,m) @ (m,p) = (n,p)

        let grad_x = if x.requires_grad {
            match (x.shape.len(), y.shape.len()) {
                (2, 2) => {
                    // Standard 2D: ∂L/∂x = out_grad @ y^T
                    let y_t = RawTensor::transpose_2d(&y.data, &y.shape);
                    let grad_data = RawTensor::matmul_raw(
                        &out_grad.data,
                        &y_t,
                        out_grad.shape[0],
                        out_grad.shape[1],
                        y.shape[0],
                    );
                    Some(RawTensor::new(grad_data, &x.shape, false))
                }
                (2, 1) => {
                    // Matrix-vector: (m,n) @ (n,) -> (m,)
                    // ∂L/∂x = ∂L/∂z[:,None] @ v[None,:] = outer(out_grad, v)
                    let m = x.shape[0];
                    let n = x.shape[1];
                    let mut grad_data = vec![0.0; m * n];
                    for i in 0..m {
                        let gz_i = out_grad.data[i];
                        for j in 0..n {
                            grad_data[i * n + j] = gz_i * y.data[j];
                        }
                    }
                    Some(RawTensor::new(grad_data, &x.shape, false))
                }
                (1, 2) => {
                    // Vector-matrix: (n,) @ (n,p) -> (p,)
                    // ∂L/∂x: out_grad is (p,), y is (n,p)
                    // grad_x = out_grad @ y^T -> (p,) @ (p,n) -> (n,)
                    let mut grad_data = vec![0.0; x.shape[0]];
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..x.shape[0] {
                        for j in 0..y.shape[1] {
                            grad_data[i] += out_grad.data[j] * y.data[i * y.shape[1] + j];
                        }
                    }
                    Some(RawTensor::new(grad_data, &x.shape, false))
                }
                (1, 1) => {
                    // Dot: (n,) @ (n,) -> scalar
                    // ∂L/∂x = out_grad * y
                    let og = out_grad.data[0];
                    let grad_data: Vec<f32> = y.data.iter().map(|&v| og * v).collect();
                    Some(RawTensor::new(grad_data, &x.shape, false))
                }
                _ => {
                    // Batched case: (B, M, K) @ (B, K, N) -> (B, M, N)
                    // dL/dx = dL/dz @ y^T
                    // Performed batch-wise
                    let rank = x.shape.len();
                    let m = x.shape[rank - 2];
                    let k = x.shape[rank - 1];
                    let n = y.shape[rank - 1]; // y is (B, K, N)

                    // Batch dimensions of output
                    let batch_dims_out = &out_grad.shape[..out_grad.shape.len() - 2];
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
                        let out_slice = &out_grad.data[b * stride_out..(b + 1) * stride_out];
                        let y_slice = &y_data_expanded[b * stride_y..(b + 1) * stride_y];
                        // Transpose y_slice (K, N) -> (N, K)
                        // But for matmul_raw we need (M,N) @ (N,K) -> (M,K)
                        // We can use transpose_2d helper on the slice
                        let y_t = RawTensor::transpose_2d(y_slice, &[k, n]);
                        let chunk = RawTensor::matmul_raw(out_slice, &y_t, m, n, k);
                        grad_data_expanded.extend_from_slice(&chunk);
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
                    let grad_data = RawTensor::matmul_raw(
                        &x_t,
                        &out_grad.data,
                        x.shape[1],
                        x.shape[0],
                        out_grad.shape[1],
                    );
                    Some(RawTensor::new(grad_data, &y.shape, false))
                }
                (2, 1) => {
                    // Matrix-vector: (m,n) @ (n,) -> (m,)
                    // ∂L/∂v = X^T @ ∂L/∂z -> (n,)
                    let m = x.shape[0];
                    let n = x.shape[1];
                    let mut grad_data = vec![0.0; n];
                    #[allow(clippy::needless_range_loop)]
                    for j in 0..n {
                        let mut sum = 0.0;
                        for i in 0..m {
                            sum += x.data[i * n + j] * out_grad.data[i];
                        }
                        grad_data[j] = sum;
                    }
                    Some(RawTensor::new(grad_data, &y.shape, false))
                }
                (1, 2) => {
                    // grad_y = x^T @ out_grad -> (n,1) @ (1,p) -> (n,p)
                    let mut grad_data = vec![0.0; y.shape[0] * y.shape[1]];
                    for i in 0..y.shape[0] {
                        for j in 0..y.shape[1] {
                            grad_data[i * y.shape[1] + j] = x.data[i] * out_grad.data[j];
                        }
                    }
                    Some(RawTensor::new(grad_data, &y.shape, false))
                }
                (1, 1) => {
                    // Dot: (n,) @ (n,) -> scalar
                    let og = out_grad.data[0];
                    let grad_data: Vec<f32> = x.data.iter().map(|&u| og * u).collect();
                    Some(RawTensor::new(grad_data, &y.shape, false))
                }
                _ => {
                    // Batched case: (B, M, K) @ (B, K, N) -> (B, M, N)
                    // dL/dy = x^T @ dL/dz
                    let rank = y.shape.len();
                    let m = x.shape[rank - 2];
                    let k = x.shape[rank - 1];
                    let n = y.shape[rank - 1];

                    let batch_dims_out = &out_grad.shape[..out_grad.shape.len() - 2];
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
                        let x_slice = &x_data_expanded[b * stride_x..(b + 1) * stride_x];
                        let out_slice = &out_grad.data[b * stride_out..(b + 1) * stride_out];
                        let x_t = RawTensor::transpose_2d(x_slice, &[m, k]);
                        // (K, M) @ (M, N) -> (K, N)
                        let chunk = RawTensor::matmul_raw(&x_t, out_slice, k, m, n);
                        grad_data_expanded.extend_from_slice(&chunk);
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

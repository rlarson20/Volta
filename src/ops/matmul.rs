use crate::autograd::GradFn;
use crate::{RawTensor, Tensor};

// ===== MATRIX MULTIPLICATION =====

impl RawTensor {
    /// Transpose a 2D matrix
    ///
    /// For shape [m, n], produces shape [n, m]
    fn transpose_2d(data: &[f32], shape: &[usize]) -> Vec<f32> {
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
    /// Uses naive O(mnk) algorithm. For production, use optimized BLAS.
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
            let mut result = vec![0.0; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for p in 0..k {
                        sum += a[i * k + p] * b[p * n + j];
                    }
                    result[i * n + j] = sum;
                }
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
        let (data_a, shape_a, req_a) = {
            let s = self_t.borrow();
            (s.data.clone(), s.shape.clone(), s.requires_grad)
        };
        let (data_b, shape_b, req_b) = {
            let o = other.borrow();
            (o.data.clone(), o.shape.clone(), o.requires_grad)
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
            _ => panic!(
                "Matmul not supported for shapes: {:?} @ {:?}",
                shape_a, shape_b
            ),
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
            out.borrow_mut().grad_fn = Some(Box::new(MatMulGradFn));
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
                _ => None,
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
                _ => None,
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

use crate::autograd::{GradFn, edge};
use crate::device::Device;
use crate::index::{fetch, slot};
use crate::shape::{Shape, fixed_dims};
use crate::spec::MatmulSpec;
use crate::storage::Storage;
use crate::{RawTensor, Tensor};

// ===== TRANSPOSE GRADIENT =====

/// Gradient function for 2D transpose
///
/// Transpose is its own inverse: (A^T)^T = A
/// So the gradient just transposes back.
#[derive(Clone)]
pub(crate) struct TransposeGradFn;

impl TransposeGradFn {
    pub(crate) fn backward(
        &self,
        out_grad: &RawTensor,
        _parents: &[Tensor],
    ) -> Vec<Option<Tensor>> {
        let transposed =
            RawTensor::transpose_2d(&out_grad.data.to_host_f32(), out_grad.shape.truncate_rank());
        let [dim0, dim1] = out_grad.shape.fixed::<2>();
        let new_shape = vec![dim1, dim0];
        vec![Some(RawTensor::new(transposed, &new_shape, false))]
    }
}

// ===== MATRIX MULTIPLICATION =====

impl RawTensor {
    /// Transpose a 2D matrix
    ///
    /// For shape [m, n], produces shape [n, m]
    pub(crate) fn transpose_2d(data: &[f32], shape: &[usize]) -> Vec<f32> {
        assert_eq!(shape.len(), 2, "Transpose expects 2D shape");
        let [m, n] = fixed_dims::<2>(shape);
        let mut result = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                *slot(&mut result, j * m + i) = fetch(data, i * n + j);
            }
        }
        result
    }

    /// Raw matrix multiplication: (m,k) @ (k,n) -> (m,n)
    /// Uses `cblas_sgemm` on macOS, `matrixmultiply::sgemm` elsewhere
    ///
    /// Layout contract (DECISIONS.md, "Matmul layout: contiguous-only"): `a` is a dense
    /// row-major m×k buffer and `b` a dense row-major k×n buffer. There are no strides, so
    /// a transposed or sliced view must be made contiguous before it gets here.
    ///
    /// # Panics
    /// `a.len() != m * k`, `b.len() != k * n`, or a dimension that does not fit in `i32`
    /// (cblas path only)
    #[must_use]
    pub(crate) fn matmul_raw(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        assert_eq!(
            a.len(),
            m * k,
            "matmul_raw: A has the wrong length for ({m},{k})"
        );
        assert_eq!(
            b.len(),
            k * n,
            "matmul_raw: B has the wrong length for ({k},{n})"
        );

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

            let dim = |x: usize| i32::try_from(x).expect("matmul dim exceeds i32");
            let (mi, ki, ni) = (dim(m), dim(k), dim(n));
            let mut result = vec![0.0; m * n];
            // SAFETY: the asserts above fix `a.len() == m*k` and `b.len() == k*n`, and `result`
            // holds m*n floats. With row-major leading dimensions k, n, n every access is in
            // bounds, and the three buffers are distinct allocations.
            unsafe {
                cblas_sgemm(
                    101, // CblasRowMajor
                    111, // CblasNoTrans
                    111, // CblasNoTrans
                    mi,
                    ni,
                    ki,
                    1.0,
                    a.as_ptr(),
                    ki,
                    b.as_ptr(),
                    ni,
                    0.0,
                    result.as_mut_ptr(),
                    ni,
                );
            }
            result
        }
        #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
        {
            // Use matrixmultiply crate for portable performance
            let mut result = vec![0.0; m * n];
            // SAFETY: the asserts above fix `a.len() == m*k` and `b.len() == k*n`, and `result`
            // holds m*n floats. The row strides (k, n, n) and unit column strides keep every
            // access in bounds, and the three buffers are distinct allocations.
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
    /// - (..b,m,k) @ (..b,k,n) -> (..b,m,n)  [batched; equal rank, batch dims may broadcast]
    /// - (m,n) @ (n,) -> (m,)    [matrix-vector]
    /// - (n,) @ (n,p) -> (p,)    [vector-matrix]
    /// - (n,) @ (n,) -> scalar   [dot product]
    /// # Panics
    /// Dimension mismatch, or matrix operands of unequal rank (see `MatmulSpec`)
    #[must_use]
    pub fn matmul(self_t: &Tensor, other: &Tensor) -> Tensor {
        let (data_a, shape_a) = {
            let s = self_t.borrow();
            (s.data.clone(), s.shape.clone())
        };
        let (data_b, shape_b) = {
            let o = other.borrow();
            (o.data.clone(), o.shape.clone())
        };

        // Handle different cases
        match (shape_a.rank(), shape_b.rank()) {
            (2, 1) => {
                let (a_host, b_host) = (data_a.to_host_f32(), data_b.to_host_f32());
                // Matrix-vector: (m,n) @ (n,) -> (m,)
                let [m, n] = shape_a.fixed::<2>();
                let [n2] = shape_b.fixed::<1>();
                assert_eq!(n, n2, "Matmul dimension mismatch: ({m},{n}) @ ({n2})");

                let mut result_data = vec![0.0; m];
                for i in 0..m {
                    let mut sum = 0.0;
                    for j in 0..n {
                        sum += fetch(&a_host, i * n + j) * fetch(&b_host, j);
                    }
                    *slot(&mut result_data, i) = sum;
                }

                edge(
                    [self_t.clone(), other.clone()],
                    Storage::cpu(result_data),
                    Shape::derived(vec![m]),
                    Device::CPU,
                    GradFn::MatMul(MatMulGradFn::vector()),
                )
            }
            (1, 2) => {
                let (a_host, b_host) = (data_a.to_host_f32(), data_b.to_host_f32());
                // Vector-matrix: (n,) @ (n,p) -> (p,)
                let [n] = shape_a.fixed::<1>();
                let [n2, p] = shape_b.fixed::<2>();
                assert_eq!(n, n2, "Matmul dimension mismatch: ({n}) @ ({n2},{p})");

                let mut result_data = vec![0.0; p];
                for j in 0..p {
                    let mut sum = 0.0;
                    for i in 0..n {
                        sum += fetch(&a_host, i) * fetch(&b_host, i * p + j);
                    }

                    *slot(&mut result_data, j) = sum;
                }

                edge(
                    [self_t.clone(), other.clone()],
                    Storage::cpu(result_data),
                    Shape::derived(vec![p]),
                    Device::CPU,
                    GradFn::MatMul(MatMulGradFn::vector()),
                )
            }
            (1, 1) => {
                let (a_host, b_host) = (data_a.to_host_f32(), data_b.to_host_f32());
                // Dot product: (n,) @ (n,) -> scalar
                let [n] = shape_a.fixed::<1>();
                let [n2] = shape_b.fixed::<1>();
                assert_eq!(n, n2, "Dot product dimension mismatch: ({n}) @ ({n2})");

                let sum: f32 = a_host.iter().zip(b_host.iter()).map(|(a, b)| a * b).sum();
                edge(
                    [self_t.clone(), other.clone()],
                    Storage::cpu(vec![sum]),
                    Shape::derived(vec![1]),
                    Device::CPU,
                    GradFn::MatMul(MatMulGradFn::vector()),
                )
            }
            _ => {
                // Matrix forms: rank >= 2 on both sides. The spec validates rank, inner dims and
                // batch broadcasting once; everything below reads m/k/n from it.
                let spec = MatmulSpec::new(&shape_a, &shape_b).unwrap_or_else(|e| panic!("{e}"));

                if !spec.is_batched() {
                    // Standard 2D matmul: (m,k) @ (k,n) -> (m,n)
                    //
                    // If both inputs live on the same GPU device, try the GPU path first.
                    // Fallback to the existing CPU implementation if anything fails.
                    if let Some(device) = Self::common_gpu_device(&[self_t, other]) {
                        if let Some(storage) = Self::gpu_matmul(&data_a, &data_b, &spec) {
                            return edge(
                                [self_t.clone(), other.clone()],
                                storage,
                                Shape::derived(spec.out_dims()),
                                device,
                                GradFn::MatMul(MatMulGradFn::matrix(spec)),
                            );
                        }
                        log::warn!("GPU matmul requested but failed; falling back to CPU");
                    }

                    // CPU fallback: BLAS/Accelerate or matrixmultiply on host data.
                    let result_data = Self::matmul_raw(
                        &data_a.to_host_f32(),
                        &data_b.to_host_f32(),
                        spec.m,
                        spec.k,
                        spec.n,
                    );
                    return edge(
                        [self_t.clone(), other.clone()],
                        Storage::cpu(result_data),
                        Shape::derived(spec.out_dims()),
                        Device::CPU,
                        GradFn::MatMul(MatMulGradFn::matrix(spec)),
                    );
                }

                // Batched: (..b, m, k) @ (..b, k, n) -> (..b, m, n)
                let (m, k, n) = (spec.m, spec.k, spec.n);
                let data_a_expanded = Self::broadcast_to(
                    &data_a.to_host_f32(),
                    &spec.a_dims(),
                    &spec.a_expanded_dims(),
                );
                let data_b_expanded = Self::broadcast_to(
                    &data_b.to_host_f32(),
                    &spec.b_dims(),
                    &spec.b_expanded_dims(),
                );

                let batch_count = spec.batch_count();
                let (stride_a, stride_b) = (m * k, k * n);
                assert_eq!(
                    data_a_expanded.len(),
                    batch_count * stride_a,
                    "broadcast A has the wrong length"
                );
                assert_eq!(
                    data_b_expanded.len(),
                    batch_count * stride_b,
                    "broadcast B has the wrong length"
                );

                let mut result_data = Vec::with_capacity(batch_count * m * n);
                for (slice_a, slice_b) in data_a_expanded
                    .chunks_exact(stride_a)
                    .zip(data_b_expanded.chunks_exact(stride_b))
                {
                    result_data.extend_from_slice(&Self::matmul_raw(slice_a, slice_b, m, k, n));
                }

                edge(
                    [self_t.clone(), other.clone()],
                    Storage::cpu(result_data),
                    Shape::derived(spec.out_dims()),
                    Device::CPU,
                    GradFn::MatMul(MatMulGradFn::matrix(spec)),
                )
            }
        }
    }

    /// Transpose a 2D tensor
    /// # Panics
    /// Non-2d tensor
    #[must_use]
    pub fn transpose(self_t: &Tensor) -> Tensor {
        let (data, shape) = {
            let s = self_t.borrow();
            (s.data.clone(), s.shape.clone())
        };

        assert_eq!(shape.rank(), 2, "Transpose expects 2D tensor");

        let transposed_data = Self::transpose_2d(&data.to_host_f32(), shape.truncate_rank());
        let [dim0, dim1] = shape.fixed::<2>();
        let new_shape = vec![dim1, dim0];

        edge(
            [self_t.clone()],
            Storage::cpu(transposed_data),
            Shape::derived(new_shape),
            Device::CPU,
            GradFn::Transpose(TransposeGradFn),
        )
    }
}

/// Gradient function for matrix multiplication
///
/// For z = x @ y:
/// - ∂L/∂x = ∂L/∂z @ y^T
/// - ∂L/∂y = x^T @ ∂L/∂z
///
/// Handles 2D×2D, batched (equal rank, batch dims may broadcast), 2D×1D, 1D×2D, and
/// 1D×1D (dot product).
#[derive(Clone)]
pub(crate) struct MatMulGradFn {
    /// `Some` for the matrix forms (rank >= 2 on both sides). `None` for the vector
    /// forms, whose backward reads the parents' shapes directly.
    spec: Option<MatmulSpec>,
}

impl MatMulGradFn {
    fn matrix(spec: MatmulSpec) -> Self {
        Self { spec: Some(spec) }
    }

    fn vector() -> Self {
        Self { spec: None }
    }

    /// ∂L/∂a = ∂L/∂c @ b^T, summed back to `a`'s shape if its batch dims were broadcast.
    fn matrix_grad_a(spec: &MatmulSpec, out_grad: &[f32], b: &[f32]) -> Vec<f32> {
        let (m, k, n) = (spec.m, spec.k, spec.n);
        if !spec.is_batched() {
            let b_t = RawTensor::transpose_2d(b, &[k, n]);
            return RawTensor::matmul_raw(out_grad, &b_t, m, n, k);
        }
        let batch_count = spec.batch_count();
        assert_eq!(
            out_grad.len(),
            batch_count * m * n,
            "out_grad has the wrong length"
        );
        let b_expanded = RawTensor::broadcast_to(b, &spec.b_dims(), &spec.b_expanded_dims());
        let mut expanded = Vec::with_capacity(batch_count * m * k);
        for (og, b_slice) in out_grad
            .chunks_exact(m * n)
            .zip(b_expanded.chunks_exact(k * n))
        {
            let b_t = RawTensor::transpose_2d(b_slice, &[k, n]);
            expanded.extend_from_slice(&RawTensor::matmul_raw(og, &b_t, m, n, k));
        }
        RawTensor::sum_over_broadcast_dims(&expanded, &spec.a_expanded_dims(), &spec.a_dims())
    }

    /// ∂L/∂b = a^T @ ∂L/∂c, summed back to `b`'s shape if its batch dims were broadcast.
    fn matrix_grad_b(spec: &MatmulSpec, a: &[f32], out_grad: &[f32]) -> Vec<f32> {
        let (m, k, n) = (spec.m, spec.k, spec.n);
        if !spec.is_batched() {
            let a_t = RawTensor::transpose_2d(a, &[m, k]);
            return RawTensor::matmul_raw(&a_t, out_grad, k, m, n);
        }
        let batch_count = spec.batch_count();
        assert_eq!(
            out_grad.len(),
            batch_count * m * n,
            "out_grad has the wrong length"
        );
        let a_expanded = RawTensor::broadcast_to(a, &spec.a_dims(), &spec.a_expanded_dims());
        let mut expanded = Vec::with_capacity(batch_count * k * n);
        for (a_slice, og) in a_expanded
            .chunks_exact(m * k)
            .zip(out_grad.chunks_exact(m * n))
        {
            let a_t = RawTensor::transpose_2d(a_slice, &[m, k]);
            expanded.extend_from_slice(&RawTensor::matmul_raw(&a_t, og, k, m, n));
        }
        RawTensor::sum_over_broadcast_dims(&expanded, &spec.b_expanded_dims(), &spec.b_dims())
    }
}

impl MatMulGradFn {
    pub(crate) fn backward(&self, out_grad: &RawTensor, parents: &[Tensor]) -> Vec<Option<Tensor>> {
        let x = parents
            .first()
            .map(|p| p.borrow())
            .expect("matmul requires 2 parents");
        let y = parents
            .get(1)
            .map(|p| p.borrow())
            .expect("matmul requires 2 parents");

        // Check if we can do GPU backward (same pattern as unary/binary)
        // For GPU backward, we need:
        // - out_grad on GPU
        // - The corresponding input (x for grad_x, y for grad_y) on GPU
        // - Same device for all involved tensors
        let gpu_available = out_grad.device.is_gpu()
            && ((x.device.is_gpu() && y.device.is_gpu()) || (!x.requires_grad || !y.requires_grad));

        // Only the plain 2D×2D case runs on GPU; everything else uses the CPU path below.
        if gpu_available && let Some(spec) = self.spec.as_ref().filter(|s| !s.is_batched()) {
            let grad_x = if x.requires_grad {
                RawTensor::gpu_matmul_backward_a(&out_grad.data, &y.data, spec).map(|storage| {
                    RawTensor::new_with_storage(storage, x.shape.clone(), x.device.clone(), false)
                })
            } else {
                None
            };

            let grad_y = if y.requires_grad {
                RawTensor::gpu_matmul_backward_b(&x.data, &out_grad.data, spec).map(|storage| {
                    RawTensor::new_with_storage(storage, y.shape.clone(), y.device.clone(), false)
                })
            } else {
                None
            };

            // Return early only if *every* required gradient was computed. A `None` for a
            // tensor that needs one would silently freeze that parameter.
            if (!x.requires_grad || grad_x.is_some()) && (!y.requires_grad || grad_y.is_some()) {
                return vec![grad_x, grad_y];
            }
            log::warn!("GPU matmul backward failed for a required gradient; falling back to CPU");
        }

        // For z = x @ y where x: (m,n), y: (n,p), z: (m,p)
        // ∂L/∂x = ∂L/∂z @ y^T  -> (m,p) @ (p,n) = (m,n)
        // ∂L/∂y = x^T @ ∂L/∂z  -> (n,m) @ (m,p) = (n,p)

        // CPU path: one host view per operand (a GPU operand syncs here, once).
        let x_host = x.data.to_host_f32();
        let y_host = y.data.to_host_f32();
        let g_host = out_grad.data.to_host_f32();

        let grad_x = if x.requires_grad {
            match (&self.spec, x.shape.rank(), y.shape.rank()) {
                (Some(spec), _, _) => Some(RawTensor::from_shape(
                    Self::matrix_grad_a(spec, &g_host, &y_host),
                    x.shape.clone(),
                    false,
                )),
                (None, 2, 1) => {
                    // Matrix-vector: (m,n) @ (n,) -> (m,)
                    // ∂L/∂x = ∂L/∂z[:,None] @ v[None,:] = outer(out_grad, v)
                    let [m, n] = x.shape.fixed::<2>();
                    let mut grad_data = vec![0.0; m * n];
                    for i in 0..m {
                        let gz_i = fetch(&g_host, i);
                        for j in 0..n {
                            *slot(&mut grad_data, i * n + j) = gz_i * fetch(&y_host, j);
                        }
                    }
                    Some(RawTensor::from_shape(grad_data, x.shape.clone(), false))
                }
                (None, 1, 2) => {
                    // Vector-matrix: (n,) @ (n,p) -> (p,)
                    // ∂L/∂x: out_grad is (p,), y is (n,p)
                    // grad_x = out_grad @ y^T -> (p,) @ (p,n) -> (n,)
                    let [len_x] = x.shape.fixed::<1>();
                    let [_, len_y_p] = y.shape.fixed::<2>();
                    let mut grad_data = vec![0.0; len_x];
                    for i in 0..len_x {
                        for j in 0..len_y_p {
                            *slot(&mut grad_data, i) +=
                                fetch(&g_host, j) * fetch(&y_host, i * len_y_p + j);
                        }
                    }
                    Some(RawTensor::from_shape(grad_data, x.shape.clone(), false))
                }
                (None, 1, 1) => {
                    // Dot: (n,) @ (n,) -> scalar
                    // ∂L/∂x = out_grad * y
                    let og = fetch(&g_host, 0);
                    let grad_data: Vec<f32> = y_host.iter().map(|&v| og * v).collect();
                    Some(RawTensor::from_shape(grad_data, x.shape.clone(), false))
                }
                (None, rx, ry) => {
                    unreachable!("vector matmul backward with ranks {rx} and {ry}")
                }
            }
        } else {
            None
        };

        let grad_y = if y.requires_grad {
            match (&self.spec, x.shape.rank(), y.shape.rank()) {
                (Some(spec), _, _) => Some(RawTensor::from_shape(
                    Self::matrix_grad_b(spec, &x_host, &g_host),
                    y.shape.clone(),
                    false,
                )),
                (None, 2, 1) => {
                    // Matrix-vector: (m,n) @ (n,) -> (m,)
                    // ∂L/∂v = X^T @ ∂L/∂z -> (n,)
                    let [m, n] = x.shape.fixed::<2>();
                    let mut grad_data = vec![0.0; n];
                    for j in 0..n {
                        let mut sum = 0.0;
                        for i in 0..m {
                            sum += fetch(&x_host, i * n + j) * fetch(&g_host, i);
                        }
                        *slot(&mut grad_data, j) = sum;
                    }
                    Some(RawTensor::from_shape(grad_data, y.shape.clone(), false))
                }
                (None, 1, 2) => {
                    // grad_y = x^T @ out_grad -> (n,1) @ (1,p) -> (n,p)
                    let [dim0, dim1] = y.shape.fixed::<2>();
                    let mut grad_data = vec![0.0; dim0 * dim1];
                    for i in 0..dim0 {
                        for j in 0..dim1 {
                            *slot(&mut grad_data, i * dim1 + j) =
                                fetch(&x_host, i) * fetch(&g_host, j);
                        }
                    }
                    Some(RawTensor::from_shape(grad_data, y.shape.clone(), false))
                }
                (None, 1, 1) => {
                    // Dot: (n,) @ (n,) -> scalar
                    let og = fetch(&g_host, 0);
                    let grad_data: Vec<f32> = x_host.iter().map(|&u| og * u).collect();
                    Some(RawTensor::from_shape(grad_data, y.shape.clone(), false))
                }
                (None, rx, ry) => {
                    unreachable!("vector matmul backward with ranks {rx} and {ry}")
                }
            }
        } else {
            None
        };

        vec![grad_x, grad_y]
    }
}

#[cfg(test)]
mod tests {
    use crate::index::{fetch, slot, span};
    use crate::spec::MatmulSpec;
    use crate::tensor::{RawTensor, Tensor, randn};

    /// Runs `f` and returns the panic message, failing if it does not panic.
    fn panic_message(f: impl FnOnce() -> Tensor) -> String {
        let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
            .expect_err("expected a panic");
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
            .expect("panic payload should be a string")
    }

    fn operand(dims: &[usize]) -> Tensor {
        let t = randn(dims);
        t.set_requires_grad(true);
        t
    }

    /// Finite-difference check of both operands of `a @ b` under a randomly
    /// weighted sum. A plain sum would be blind to some transposed-dim bugs.
    fn assert_matmul_gradients(a_dims: &[usize], b_dims: &[usize]) {
        crate::tensor::manual_seed(0x5EED);
        let a = operand(a_dims);
        let b = operand(b_dims);
        let spec = MatmulSpec::new(
            &crate::shape::Shape::derived(a_dims.to_vec()),
            &crate::shape::Shape::derived(b_dims.to_vec()),
        )
        .expect("test shapes must be valid");
        let w = randn(&spec.out_dims());

        let (max_a, mean_a, ok_a) =
            RawTensor::check_gradients(&a, |t| t.matmul(&b).elem_mul(&w).sum(), 1e-2, 2e-2);
        assert!(
            ok_a,
            "grad wrt A failed for {a_dims:?} @ {b_dims:?}: max={max_a:.3e} mean={mean_a:.3e}"
        );
        let (max_b, mean_b, ok_b) =
            RawTensor::check_gradients(&b, |t| a.matmul(t).elem_mul(&w).sum(), 1e-2, 2e-2);
        assert!(
            ok_b,
            "grad wrt B failed for {a_dims:?} @ {b_dims:?}: max={max_b:.3e} mean={mean_b:.3e}"
        );
    }

    #[test]
    fn test_matmul_2d_gradients() {
        assert_matmul_gradients(&[3, 4], &[4, 5]);
    }

    #[test]
    fn test_matmul_batched_equal_batch_gradients() {
        assert_matmul_gradients(&[2, 3, 4], &[2, 4, 5]);
    }

    #[test]
    fn test_matmul_batch_dims_broadcast_on_both_sides_gradients() {
        assert_matmul_gradients(&[2, 1, 3, 4], &[1, 3, 4, 5]);
    }

    #[test]
    fn test_matmul_batch_dim_broadcast_on_one_side_gradients() {
        assert_matmul_gradients(&[1, 3, 4], &[2, 4, 5]);
        assert_matmul_gradients(&[2, 3, 4], &[1, 4, 5]);
    }

    /// Reference GEMM, independent of `matmul_raw`: `(m,k) @ (k,n)` where element `(i, j)` of
    /// an operand is at `i * row_stride + j * col_stride`. A transpose is the same buffer
    /// with the strides swapped. Accumulates in f64, so it is a stricter oracle than the
    /// f32 kernels it checks.
    fn ref_gemm(
        a: &[f32],
        (a_rs, a_cs): (usize, usize),
        b: &[f32],
        (b_rs, b_cs): (usize, usize),
        (m, k, n): (usize, usize, usize),
    ) -> Vec<f32> {
        let mut c = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let acc: f64 = (0..k)
                    .map(|p| {
                        f64::from(fetch(a, i * a_rs + p * a_cs))
                            * f64::from(fetch(b, p * b_rs + j * b_cs))
                    })
                    .sum();
                *slot(&mut c, i * n + j) = acc as f32;
            }
        }
        c
    }

    /// Naive row-major (m,k) @ (k,n) reference.
    fn reference_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        ref_gemm(a, (k, 1), b, (n, 1), (m, k, n))
    }

    /// The oracle itself, against hand-computed values: `A = [[1,2,3],[4,5,6]]`,
    /// `B = [[7,8],[9,10],[11,12]]`, `A @ B = [[58,64],[139,154]]`, and the same product
    /// read through swapped strides from buffers holding `A^T` and `B^T`.
    #[test]
    fn ref_gemm_matches_hand_values_and_transposes_by_strides() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let expected = [58.0, 64.0, 139.0, 154.0];
        assert_eq!(ref_gemm(&a, (3, 1), &b, (2, 1), (2, 3, 2)), expected);
        let a_t = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let b_t = [7.0, 9.0, 11.0, 8.0, 10.0, 12.0];
        assert_eq!(ref_gemm(&a_t, (1, 2), &b_t, (1, 3), (2, 3, 2)), expected);
    }

    /// Flat index into `in_batch` of output batch `o`, broadcasting size-1 dims.
    fn batch_source(o: usize, out_batch: &[usize], in_batch: &[usize]) -> usize {
        let (mut rem, mut idx, mut stride) = (o, 0, 1);
        for (&out_dim, &in_dim) in out_batch.iter().zip(in_batch).rev() {
            let coord = rem % out_dim;
            rem /= out_dim;
            if in_dim != 1 {
                idx += coord * stride;
            }
            stride *= in_dim;
        }
        idx
    }

    /// Forward output and both gradients of `sum(w * (a @ b))`, from `ref_gemm` alone:
    /// `dA = W @ B^T` and `dB = A^T @ W` per output batch (transposes as swapped strides),
    /// summed into the input batch each output batch reads from.
    fn reference_forward_backward(
        (a, a_batch): (&[f32], &[usize]),
        (b, b_batch): (&[f32], &[usize]),
        w: &[f32],
        (m, k, n): (usize, usize, usize),
    ) -> [Vec<f32>; 3] {
        let out_batch: Vec<usize> = a_batch
            .iter()
            .zip(b_batch)
            .map(|(&x, &y)| x.max(y))
            .collect();
        let (size_a, size_b, size_c) = (m * k, k * n, m * n);
        let mut c = Vec::new();
        let (mut da, mut db) = (vec![0.0; a.len()], vec![0.0; b.len()]);
        for o in 0..out_batch.iter().product() {
            let a_off = batch_source(o, &out_batch, a_batch) * size_a;
            let b_off = batch_source(o, &out_batch, b_batch) * size_b;
            let a_o = span(a, a_off..a_off + size_a);
            let b_o = span(b, b_off..b_off + size_b);
            let w_o = span(w, o * size_c..(o + 1) * size_c);
            c.extend(ref_gemm(a_o, (k, 1), b_o, (n, 1), (m, k, n)));
            let da_o = ref_gemm(w_o, (n, 1), b_o, (1, n), (m, n, k));
            for (i, g) in da_o.into_iter().enumerate() {
                *slot(&mut da, a_off + i) += g;
            }
            let db_o = ref_gemm(a_o, (1, k), w_o, (n, 1), (k, m, n));
            for (i, g) in db_o.into_iter().enumerate() {
                *slot(&mut db, b_off + i) += g;
            }
        }
        [c, da, db]
    }

    /// `a @ b`'s output and both gradients under a randomly weighted loss match
    /// `reference_forward_backward`. Rank-1 operands are the `[1, k]` / `[k, 1]` matrices
    /// the vector arms compute, with the same flat layout.
    fn assert_matches_reference(a_dims: &[usize], b_dims: &[usize]) {
        crate::tensor::manual_seed(0x0E7A);
        let a = operand(a_dims);
        let b = operand(b_dims);
        let (a_batch, a_mat) = a_dims.split_at(a_dims.len().saturating_sub(2));
        let (b_batch, b_mat) = b_dims.split_at(b_dims.len().saturating_sub(2));
        let (m, k) = match *a_mat {
            [m, k] => (m, k),
            [k] => (1, k),
            _ => unreachable!("rank >= 1"),
        };
        let n = match *b_mat {
            [_, n] => n,
            [_] => 1,
            _ => unreachable!("rank >= 1"),
        };
        let out = a.matmul(&b);
        let w = randn(out.shape().truncate_rank());
        out.elem_mul(&w)
            .sum()
            .into_loss()
            .expect("scalar loss")
            .backward();

        let [c, da, db] = reference_forward_backward(
            (&a.data(), a_batch),
            (&b.data(), b_batch),
            &w.data(),
            (m, k, n),
        );
        let what = format!("{a_dims:?} @ {b_dims:?}");
        assert_close(&out.data(), &c, &format!("{what} forward"));
        assert_close(&a.grad().expect("a grad"), &da, &format!("{what} dA"));
        assert_close(&b.grad().expect("b grad"), &db, &format!("{what} dB"));
    }

    /// Past one 16x16 tile and not a multiple of 16 on any side (GEMM report §4).
    #[test]
    fn matmul_matches_reference_2d_past_one_tile() {
        assert_matches_reference(&[37, 19], &[19, 23]);
    }

    #[test]
    fn matmul_matches_reference_equal_batch() {
        assert_matches_reference(&[3, 5, 7], &[3, 7, 4]);
    }

    #[test]
    fn matmul_matches_reference_broadcast_batch() {
        assert_matches_reference(&[2, 1, 5, 6], &[1, 3, 6, 4]);
    }

    #[test]
    fn matmul_matches_reference_vector_arms() {
        assert_matches_reference(&[5, 7], &[7]);
        assert_matches_reference(&[7], &[7, 4]);
        assert_matches_reference(&[7], &[7]);
    }

    fn assert_close(actual: &[f32], expected: &[f32], what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what}: length");
        for (i, (x, y)) in actual.iter().zip(expected).enumerate() {
            assert!((x - y).abs() <= 1e-4, "{what}: index {i}: {x} vs {y}");
        }
    }

    // The three tests below are the fence for the contiguous-only contract: they pass today
    // because transpose/permute copy, and start guarding it once those return views.

    #[test]
    fn matmul_of_transposed_operand_matches_reference() {
        crate::tensor::manual_seed(0xA11C);
        // Non-square on every side so a cancelling transpose bug cannot hide.
        let a = randn(&[3, 4]);
        let b = randn(&[5, 4]);
        let out = a.matmul(&b.transpose());
        assert_eq!(out.shape(), vec![3, 5]);

        let b_t = RawTensor::transpose_2d(&b.data(), &[5, 4]);
        let expected = reference_matmul(&a.data(), &b_t, 3, 4, 5);
        assert_close(&out.data(), &expected, "a @ b^T");

        let double = a.transpose().transpose().matmul(&b.transpose());
        assert_close(&double.data(), &expected, "(a^T)^T @ b^T");
    }

    #[test]
    fn matmul_of_permuted_batched_operand_matches_reference() {
        use crate::shape::Axis;

        crate::tensor::manual_seed(0xBA7C);
        let (batch, m, k, n) = (2, 3, 4, 5);
        let a = randn(&[batch, m, k]);
        let b = randn(&[batch, n, k]);
        let out = a.matmul(&b.permute(&[Axis(0), Axis(2), Axis(1)]));
        assert_eq!(out.shape(), vec![batch, m, n]);

        let (a_data, b_data) = (a.data(), b.data());
        let mut expected = Vec::new();
        for (a_slice, b_slice) in a_data.chunks_exact(m * k).zip(b_data.chunks_exact(n * k)) {
            let b_t = RawTensor::transpose_2d(b_slice, &[n, k]);
            expected.extend(reference_matmul(a_slice, &b_t, m, k, n));
        }
        assert_close(&out.data(), &expected, "batched a @ permute(b)");
    }

    #[test]
    fn test_matmul_through_transpose_gradients() {
        crate::tensor::manual_seed(0x7A25);
        let a = operand(&[3, 4]);
        let b = operand(&[5, 4]);
        let w = randn(&[3, 5]);

        let (max_a, mean_a, ok_a) = RawTensor::check_gradients(
            &a,
            |t| t.matmul(&b.transpose()).elem_mul(&w).sum(),
            1e-2,
            2e-2,
        );
        assert!(ok_a, "grad wrt A failed: max={max_a:.3e} mean={mean_a:.3e}");
        let (max_b, mean_b, ok_b) = RawTensor::check_gradients(
            &b,
            |t| a.matmul(&t.transpose()).elem_mul(&w).sum(),
            1e-2,
            2e-2,
        );
        assert!(ok_b, "grad wrt B failed: max={max_b:.3e} mean={mean_b:.3e}");
    }

    #[test]
    #[should_panic(expected = "A has the wrong length")]
    fn matmul_raw_rejects_short_a() {
        let _ = RawTensor::matmul_raw(&[1.0; 5], &[1.0; 6], 2, 3, 2);
    }

    #[test]
    #[should_panic(expected = "B has the wrong length")]
    fn matmul_raw_rejects_long_b() {
        let _ = RawTensor::matmul_raw(&[1.0; 6], &[1.0; 7], 2, 3, 2);
    }

    #[test]
    fn test_matmul_rejects_unequal_rank_b4_repro() {
        // B4: [2,3,4] @ [2,1,4,6] was accepted and gave a 36-100% wrong gradient.
        let a = operand(&[2, 3, 4]);
        let b = operand(&[2, 1, 4, 6]);
        let msg = panic_message(|| a.matmul(&b));
        assert!(msg.contains("equal rank"), "{msg}");
        let msg = panic_message(|| b.matmul(&a));
        assert!(msg.contains("equal rank"), "{msg}");
    }

    #[test]
    fn test_matmul_rejects_3d_against_2d() {
        let a = operand(&[2, 3, 4]);
        let b = operand(&[4, 5]);
        let msg = panic_message(|| a.matmul(&b));
        assert!(msg.contains("equal rank"), "{msg}");
    }

    #[test]
    fn test_matmul_rejects_inner_dim_mismatch() {
        let a = operand(&[3, 4]);
        let b = operand(&[5, 6]);
        let msg = panic_message(|| a.matmul(&b));
        assert!(msg.contains("inner dims"), "{msg}");
    }

    #[test]
    fn test_matmul_rejects_batch_dims_that_do_not_broadcast() {
        let a = operand(&[2, 3, 4]);
        let b = operand(&[3, 4, 5]);
        let msg = panic_message(|| a.matmul(&b));
        assert!(msg.contains("do not broadcast"), "{msg}");
    }
}

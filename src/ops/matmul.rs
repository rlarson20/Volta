use crate::autograd::{GradFn, edge};
use crate::device::Device;
use crate::shape::Shape;
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
        let transposed = RawTensor::transpose_2d(&out_grad.data, &out_grad.shape);
        let dim0 = out_grad.shape.first().copied().unwrap_or(1);
        let dim1 = out_grad.shape.get(1).copied().unwrap_or(1);
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
    #[allow(clippy::many_single_char_names, reason = "Numerical code")]
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
    /// Dimension mismatch, or matrix operands of unequal rank (see [`MatmulSpec`])
    #[allow(clippy::too_many_lines)]
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
        match (shape_a.len(), shape_b.len()) {
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

                edge(
                    [self_t.clone(), other.clone()],
                    Storage::cpu(result_data),
                    Shape::derived(vec![m]),
                    Device::CPU,
                    GradFn::MatMul(MatMulGradFn::vector()),
                )
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

                edge(
                    [self_t.clone(), other.clone()],
                    Storage::cpu(result_data),
                    Shape::derived(vec![p]),
                    Device::CPU,
                    GradFn::MatMul(MatMulGradFn::vector()),
                )
            }
            (1, 1) => {
                // Dot product: (n,) @ (n,) -> scalar
                let n = shape_a.first().copied().unwrap_or(1);
                let n2 = shape_b.first().copied().unwrap_or(1);
                assert_eq!(n, n2, "Dot product dimension mismatch: ({n}) @ ({n2})");

                let sum: f32 = data_a.iter().zip(&data_b).map(|(a, b)| a * b).sum();
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
                let spec = MatmulSpec::new(
                    &Shape::derived(shape_a.clone()),
                    &Shape::derived(shape_b.clone()),
                )
                .unwrap_or_else(|e| panic!("{e}"));

                if !spec.is_batched() {
                    // Standard 2D matmul: (m,k) @ (k,n) -> (m,n)
                    //
                    // If both inputs live on the same GPU device, try the GPU path first.
                    // Fallback to the existing CPU implementation if anything fails.
                    #[cfg(feature = "gpu")]
                    {
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
                    }

                    // CPU fallback: BLAS/Accelerate or matrixmultiply on host data.
                    let result_data = Self::matmul_raw(&data_a, &data_b, spec.m, spec.k, spec.n);
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
                let data_a_expanded =
                    Self::broadcast_to(&data_a, &spec.a_dims(), &spec.a_expanded_dims());
                let data_b_expanded =
                    Self::broadcast_to(&data_b, &spec.b_dims(), &spec.b_expanded_dims());

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

        assert_eq!(shape.len(), 2, "Transpose expects 2D tensor");

        let transposed_data = Self::transpose_2d(&data, &shape);
        let dim0 = shape.first().copied().unwrap_or(1);
        let dim1 = shape.get(1).copied().unwrap_or(1);
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
    #[allow(clippy::too_many_lines, clippy::many_single_char_names)]
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
        #[cfg(feature = "gpu")]
        {
            // For GPU backward, we need:
            // - out_grad on GPU
            // - The corresponding input (x for grad_x, y for grad_y) on GPU
            // - Same device for all involved tensors
            let gpu_available = out_grad.device.is_gpu()
                && ((x.device.is_gpu() && y.device.is_gpu())
                    || (!x.requires_grad || !y.requires_grad));

            // Only the plain 2D×2D case runs on GPU; everything else uses the CPU path below.
            if gpu_available && let Some(spec) = self.spec.as_ref().filter(|s| !s.is_batched()) {
                let grad_x = if x.requires_grad {
                    RawTensor::gpu_matmul_backward_a(&out_grad.data, &y.data, spec).map(|storage| {
                        RawTensor::new_with_storage(storage, &x.shape, x.device.clone(), false)
                    })
                } else {
                    None
                };

                let grad_y = if y.requires_grad {
                    RawTensor::gpu_matmul_backward_b(&x.data, &out_grad.data, spec).map(|storage| {
                        RawTensor::new_with_storage(storage, &y.shape, y.device.clone(), false)
                    })
                } else {
                    None
                };

                // Return early only if *every* required gradient was computed. A `None` for a
                // tensor that needs one would silently freeze that parameter.
                if (!x.requires_grad || grad_x.is_some()) && (!y.requires_grad || grad_y.is_some())
                {
                    return vec![grad_x, grad_y];
                }
                log::warn!(
                    "GPU matmul backward failed for a required gradient; falling back to CPU"
                );
            }
        }

        // For z = x @ y where x: (m,n), y: (n,p), z: (m,p)
        // ∂L/∂x = ∂L/∂z @ y^T  -> (m,p) @ (p,n) = (m,n)
        // ∂L/∂y = x^T @ ∂L/∂z  -> (n,m) @ (m,p) = (n,p)

        let grad_x = if x.requires_grad {
            match (&self.spec, x.shape.len(), y.shape.len()) {
                (Some(spec), _, _) => Some(RawTensor::new(
                    Self::matrix_grad_a(spec, &out_grad.data, &y.data),
                    &x.shape,
                    false,
                )),
                (None, 2, 1) => {
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
                (None, 1, 2) => {
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
                (None, 1, 1) => {
                    // Dot: (n,) @ (n,) -> scalar
                    // ∂L/∂x = out_grad * y
                    let og = out_grad.data.first().copied().unwrap_or(0.0);
                    let grad_data: Vec<f32> = y.data.iter().map(|&v| og * v).collect();
                    Some(RawTensor::new(grad_data, &x.shape, false))
                }
                (None, rx, ry) => {
                    unreachable!("vector matmul backward with ranks {rx} and {ry}")
                }
            }
        } else {
            None
        };

        let grad_y = if y.requires_grad {
            match (&self.spec, x.shape.len(), y.shape.len()) {
                (Some(spec), _, _) => Some(RawTensor::new(
                    Self::matrix_grad_b(spec, &x.data, &out_grad.data),
                    &y.shape,
                    false,
                )),
                (None, 2, 1) => {
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
                (None, 1, 2) => {
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
                (None, 1, 1) => {
                    // Dot: (n,) @ (n,) -> scalar
                    let og = out_grad.data.first().copied().unwrap_or(0.0);
                    let grad_data: Vec<f32> = x.data.iter().map(|&u| og * u).collect();
                    Some(RawTensor::new(grad_data, &y.shape, false))
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
    use crate::spec::MatmulSpec;
    use crate::tensor::{RawTensor, Tensor, TensorOps, randn};

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

    /// Naive row-major (m,k) @ (k,n) reference, independent of `matmul_raw`.
    fn reference_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        a.chunks_exact(k)
            .take(m)
            .flat_map(|row| {
                (0..n).map(move |j| {
                    row.iter()
                        .zip(b.iter().skip(j).step_by(n))
                        .map(|(x, y)| x * y)
                        .sum()
                })
            })
            .collect()
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

use crate::ops::gpu_ops::MuonStep;
use crate::real::{Real, dispatch_real};
use crate::storage::Storage;
use crate::tensor::{RawTensor, Tensor};

/// Coefficients of the quintic `p(x) = a·x + b·x³ + c·x⁵` that Muon iterates on the
/// singular values (Jordan et al.'s reference implementation). They are tuned for the
/// slope at zero, not for convergence to 1: after five steps the singular values land
/// roughly in `[0.7, 1.2]`, which is close enough to orthogonal for an optimizer.
const NS_A: f32 = 3.4445;
/// Second coefficient of the Newton-Schulz quintic (see `NS_A`).
const NS_B: f32 = -4.7750;
/// Third coefficient of the Newton-Schulz quintic (see `NS_A`).
const NS_C: f32 = 2.0315;
/// Added to the Frobenius norm before dividing by it, so a zero gradient stays zero.
const NS_EPS: f32 = 1e-7;

/// Muon: momentum, then an approximately orthogonalized update.
///
/// Each matrix-shaped parameter's momentum is pushed through a few Newton-Schulz
/// iterations that drive its singular values toward 1, and the result (scaled by
/// `sqrt(max(1, rows / cols))`) is the step. This follows the reference implementation's
/// quintic iteration and shape scaling.
///
/// A parameter is treated as the matrix `[dim0, product of the other dims]`, so a conv
/// weight `[out, in, kh, kw]` is `out × (in·kh·kw)`, as in the reference. Volta's
/// `Linear` stores `[in, out]`, the transpose of PyTorch, so for it the shape scale is
/// `sqrt(max(1, in / out))` (the orthogonalization itself does not depend on the
/// orientation). A parameter that is not a matrix (rank below 2, or one row or column)
/// is updated with plain momentum and no orthogonalization or scaling; the reference
/// optimizes such parameters with Adam instead.
///
/// Parameters may live on the CPU or the GPU. A GPU parameter's momentum, Newton-Schulz,
/// and update all run on the device, with no host round-trip.
pub struct Muon {
    /// Parameters being optimized.
    pub(super) params: Vec<Tensor>,
    /// Learning rate.
    lr: f32,
    /// Momentum coefficient.
    momentum: f32,
    /// Whether to use Nesterov momentum.
    nesterov: bool,
    /// Newton-Schulz iterations per step.
    ns_steps: usize,
    /// Momentum buffer per parameter, allocated on the parameter's first step (`None` until
    /// then), so a model converted with `to_device` or `to_dtype` before that step is
    /// stepped in its new device and dtype. A conversion after the first step panics instead.
    velocity: Vec<Option<Storage>>,
}

impl Muon {
    /// Create a Muon optimizer.
    ///
    /// # Arguments
    /// * `params` - Parameters to optimize (any float dtype, CPU or GPU)
    /// * `lr` - Learning rate. The step is orthogonalized, so its size does not follow the
    ///   gradient's; the reference uses about 0.02
    /// * `momentum` - Momentum coefficient (typical: 0.95)
    /// * `nesterov` - Use the Nesterov update `momentum·v + g` instead of `v`
    /// * `ns_steps` - Newton-Schulz iterations (typical: 5)
    ///
    /// # Panics
    /// If a hyperparameter is invalid; see [`Self::try_new`].
    #[must_use]
    #[track_caller]
    pub fn new(
        params: Vec<Tensor>,
        lr: f32,
        momentum: f32,
        nesterov: bool,
        ns_steps: usize,
    ) -> Self {
        Self::try_new(params, lr, momentum, nesterov, ns_steps)
            .unwrap_or_else(|e| panic!("Muon: {e}"))
    }

    /// [`Self::new`], checking the hyperparameters first.
    ///
    /// # Errors
    /// `VoltaError::InvalidParameter` if `lr` or `momentum` is negative or NaN, or if the
    /// same tensor is listed twice in `params` (#101).
    pub fn try_new(
        params: Vec<Tensor>,
        lr: f32,
        momentum: f32,
        nesterov: bool,
        ns_steps: usize,
    ) -> crate::Result<Self> {
        super::check_unique(&params)?;
        super::check_non_negative("learning rate", lr)?;
        super::check_non_negative("momentum", momentum)?;
        let velocity = params.iter().map(|_| None).collect();
        Ok(Self {
            params,
            lr,
            momentum,
            nesterov,
            ns_steps,
            velocity,
        })
    }

    /// Clear every parameter's gradient.
    pub fn zero_grad(&self) {
        for param in &self.params {
            param.zero_grad();
        }
    }

    /// Approximately orthogonalize the `rows × cols` row-major matrix `g` in place with
    /// `steps` quintic Newton-Schulz iterations.
    ///
    /// `g` is normalized by its Frobenius norm first (`‖g‖_F ≥ ‖g‖₂`, so every singular
    /// value starts in `[0, 1]`, inside the basin of the iteration). A tall matrix is
    /// transposed so the Gram matrix `X Xᵀ` is on the smaller side, and transposed back.
    ///
    /// `coeffs` is the quintic's `[a, b, c]` and `eps` the normalization epsilon; `step`
    /// passes the same values to the GPU kernels.
    fn newton_schulz_with<T: Real>(
        g: &mut [T],
        rows: usize,
        cols: usize,
        steps: usize,
        coeffs: [T; 3],
        eps: T,
    ) {
        let [a_coeff, b_coeff, c_coeff] = coeffs;
        let transposed = rows > cols;
        let (m, n) = if transposed {
            (cols, rows)
        } else {
            (rows, cols)
        };

        let mut x = if transposed {
            RawTensor::transpose_2d(g, &[rows, cols])
        } else {
            g.to_vec()
        };

        let norm = x.iter().map(|&v| v * v).sum::<T>().sqrt();
        let inv = T::from_f64(1.0) / (norm + eps);
        for v in &mut x {
            *v *= inv;
        }

        for _ in 0..steps {
            // A = X Xᵀ (m×m), B = b·A + c·A², X ← a·X + B·X
            let xt = RawTensor::transpose_2d(&x, &[m, n]);
            let a = T::gemm(&x, &xt, m, n, m);
            let a2 = T::gemm(&a, &a, m, m, m);
            let b: Vec<T> = a
                .iter()
                .zip(&a2)
                .map(|(&a, &a2)| b_coeff * a + c_coeff * a2)
                .collect();
            let bx = T::gemm(&b, &x, m, m, n);
            x = x
                .iter()
                .zip(&bx)
                .map(|(&x, &bx)| a_coeff * x + bx)
                .collect();
        }

        let result = if transposed {
            RawTensor::transpose_2d(&x, &[m, n])
        } else {
            x
        };
        g.copy_from_slice(&result);
    }

    /// [`Self::newton_schulz_with`] with the reference coefficients and epsilon.
    #[cfg(test)]
    pub(crate) fn newton_schulz(g: &mut [f32], rows: usize, cols: usize, steps: usize) {
        Self::newton_schulz_with(g, rows, cols, steps, [NS_A, NS_B, NS_C], NS_EPS);
    }

    /// Perform one optimization step on every parameter that has a gradient.
    ///
    /// # Panics
    /// If a GPU kernel cannot run: the parameter would otherwise be silently left
    /// untrained, as in `SGD` and `Adam`.
    pub fn step(&mut self) {
        // Each parameter steps in its own dtype (a mixed list is fine, as in PyTorch);
        // each must have a kernel. Checked for all before any is updated.
        for param in &self.params {
            crate::real::compute_dtype(|| "Muon step".to_string(), &[param.dtype()]);
        }
        for (param, velocity) in self.params.iter().zip(self.velocity.iter_mut()) {
            let mut p = param.borrow_mut();
            let Some(grad) = p.grad.clone() else {
                continue;
            };
            // The step rewrites the parameter in place (S9).
            p.bump_version();
            // Allocated on the parameter's first step, on its current device and dtype.
            let velocity = velocity.get_or_insert_with(|| super::zeros_like(&p));

            // The matrix view: [dim0, everything else]. A vector is (n, 1) and a scalar
            // is (1, 1).
            let (rows, cols) = match p.shape.rank() {
                0 => (1, 1),
                1 => (p.shape.dim_at(0), 1),
                _ => (p.shape.dim_at(0), p.shape.numel_from(1)),
            };

            // Only a real matrix is orthogonalized and shape-scaled.
            let orthogonalize = rows > 1 && cols > 1;
            let scale = if orthogonalize {
                (rows as f32 / cols as f32).max(1.0).sqrt()
            } else {
                1.0
            };

            // One description of the step, read by the CPU and the GPU path alike.
            let step = MuonStep {
                lr: self.lr,
                momentum: self.momentum,
                nesterov: self.nesterov,
                ns_steps: self.ns_steps,
                rows,
                cols,
                orthogonalize,
                scale,
                coeffs: [NS_A, NS_B, NS_C],
                eps: NS_EPS,
            };
            step.validate(p.data.len());

            if p.device.is_gpu() {
                RawTensor::gpu_muon_step(&mut p.data, &grad, velocity, &step)
                    .expect("GPU Muon step failed: the parameter was not updated");
                continue;
            }

            dispatch_real!(p.data.dtype(), T => {
            let widen = |v: f32| T::from_f64(f64::from(v));
            let (lr, momentum, scale, eps) =
                (widen(step.lr), widen(step.momentum), widen(step.scale), widen(step.eps));
            let coeffs = step.coeffs.map(widen);
            let grad = grad.host::<T>();
            assert_eq!(
                velocity.dtype(),
                p.data.dtype(),
                "Muon velocity dtype; convert the model before the optimizer's first step"
            );
            let mut velocity = velocity.host_mut::<T>();

            // v ← momentum·v + g
            for (v, &g) in velocity.iter_mut().zip(grad.iter()) {
                *v = momentum.mul_add(*v, g);
            }

            // Nesterov looks one momentum step ahead: update = momentum·v + g
            let mut update: Vec<T> = if step.nesterov {
                velocity
                    .iter()
                    .zip(grad.iter())
                    .map(|(&v, &g)| momentum * v + g)
                    .collect()
            } else {
                velocity.to_vec()
            };

            if step.orthogonalize {
                Self::newton_schulz_with(
                    &mut update,
                    step.rows,
                    step.cols,
                    step.ns_steps,
                    coeffs,
                    eps,
                );
            }

            // w -= lr · scale · update
            for (w, u) in p
                .data
                .host_mut::<T>()
                .iter_mut()
                .zip(&update)
            {
                *w -= lr * scale * *u;
            }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Storage;
    use crate::index::fetch;

    /// A seeded random `rows × cols` matrix as a flat row-major vector.
    fn random_matrix(rows: usize, cols: usize) -> Vec<f32> {
        crate::tensor::randn(&[rows, cols]).borrow().data.to_vec()
    }

    /// Largest `|(X Xᵀ − I)[i][j]|` over the Gram matrix of the smaller side of `x`
    /// (`rows × cols`). It bounds the spectral norm of `X Xᵀ − I` from below by the
    /// entries, and is 0 for exactly orthonormal rows (or columns).
    fn gram_deviation(x: &[f32], rows: usize, cols: usize) -> f32 {
        let (m, n, x) = if rows > cols {
            (cols, rows, RawTensor::transpose_2d(x, &[rows, cols]))
        } else {
            (rows, cols, x.to_vec())
        };
        let xt = RawTensor::transpose_2d(&x, &[m, n]);
        let gram = RawTensor::matmul_raw(&x, &xt, m, n, m);
        let mut worst = 0.0f32;
        for i in 0..m {
            for j in 0..m {
                let want = if i == j { 1.0 } else { 0.0 };
                worst = worst.max((fetch(&gram, i * m + j) - want).abs());
            }
        }
        worst
    }

    #[test]
    fn test_muon_step() {
        // Simple test: Ensure parameters change after a step
        let x = RawTensor::new(vec![1.0, -1.0, 0.5, 0.5], &[2, 2], true);
        // Artificial gradient
        x.set_grad(Storage::cpu(vec![0.1, 0.1, -0.1, -0.1]));

        let mut opt = Muon::new(vec![x.clone()], 0.1, 0.9, true, 5);

        let data_before = x.borrow().data.clone();
        opt.step();
        let data_after = x.borrow().data.clone();

        // Verify data changed
        assert_ne!(data_before, data_after);

        // Verify Newton-Schulz didn't explode values (regularization property)
        for v in data_after.to_host_f32() {
            assert!(v.abs() < 2.0);
        }
    }

    #[test]
    fn newton_schulz_makes_rectangular_matrices_nearly_orthonormal() {
        // A random gradient is far from orthogonal: normalized by its Frobenius norm, the
        // Gram matrix's diagonal is about 1/m. Five quintic steps bring every singular
        // value near 1, so X Xᵀ is near the identity in both orientations. (The quintic
        // settles in about [0.7, 1.2] rather than at 1, so the bound is loose; the test
        // below is the one that tells it from the classic cubic iteration.)
        crate::tensor::manual_seed(0x5EED);
        for (rows, cols) in [(8, 12), (12, 8), (6, 24), (24, 6), (16, 16)] {
            let mut g = random_matrix(rows, cols);
            let before = {
                let norm = g.iter().map(|v| v * v).sum::<f32>().sqrt();
                let unit: Vec<f32> = g.iter().map(|v| v / norm).collect();
                gram_deviation(&unit, rows, cols)
            };
            Muon::newton_schulz(&mut g, rows, cols, 5);
            let after = gram_deviation(&g, rows, cols);
            assert!(
                before > 0.5,
                "{rows}x{cols}: the unorthogonalized matrix should be far from orthonormal, got {before}"
            );
            assert!(
                after < 0.6,
                "{rows}x{cols}: X Xᵀ should be near I after Newton-Schulz, worst entry off by {after}"
            );
        }
    }

    #[test]
    fn newton_schulz_recovers_a_nearly_vanished_direction() {
        // One row shrunk 20x: after the Frobenius normalization its singular value is
        // about 5e-3. The quintic's slope at zero (3.44) lifts it into the band within
        // five steps; the classic iteration 1.5x - 0.5 x xᵀx grows it by only 1.5 per step
        // (about 0.04 after five) and leaves X Xᵀ[0][0] near 0.
        crate::tensor::manual_seed(0xD1CE);
        for (rows, cols) in [(8, 12), (12, 8)] {
            let mut g = random_matrix(rows, cols);
            // Shrink the first row of a wide matrix, or the first column of a tall one: the
            // short side's index 0.
            if rows <= cols {
                for j in 0..cols {
                    *crate::index::slot(&mut g, j) *= 0.05;
                }
            } else {
                for i in 0..rows {
                    *crate::index::slot(&mut g, i * cols) *= 0.05;
                }
            }
            Muon::newton_schulz(&mut g, rows, cols, 5);
            let deviation = gram_deviation(&g, rows, cols);
            assert!(
                deviation < 0.6,
                "{rows}x{cols}: the shrunken direction was not recovered, worst entry off by {deviation}"
            );
        }
    }

    #[test]
    fn newton_schulz_is_transpose_equivariant() {
        // NS(Gᵀ) = NS(G)ᵀ: the tall case runs the wide computation on the transpose.
        crate::tensor::manual_seed(0x7A11);
        let (rows, cols) = (5, 9);
        let wide = random_matrix(rows, cols);
        let mut tall = RawTensor::transpose_2d(&wide, &[rows, cols]);
        let mut wide_out = wide;
        Muon::newton_schulz(&mut wide_out, rows, cols, 5);
        Muon::newton_schulz(&mut tall, cols, rows, 5);
        let wide_t = RawTensor::transpose_2d(&wide_out, &[rows, cols]);
        assert_eq!(wide_t.len(), tall.len());
        for (a, b) in wide_t.iter().zip(&tall) {
            assert!((a - b).abs() < 1e-6, "tall {b} vs wide transposed {a}");
        }
    }

    #[test]
    fn newton_schulz_leaves_a_zero_gradient_at_zero() {
        // 0 / (0 + 1e-7) = 0, and the quintic maps 0 to 0: no NaN from the normalization.
        let mut g = vec![0.0; 12];
        Muon::newton_schulz(&mut g, 3, 4, 5);
        assert!(g.iter().all(|&v| v == 0.0));
    }

    /// One step with `lr = 1`, no momentum, and no Nesterov, returning how far each entry
    /// moved (`before - after`, which is the update).
    fn one_step_delta(shape: &[usize], grad: &[f32]) -> Vec<f32> {
        let numel: usize = shape.iter().product();
        let p = RawTensor::new(vec![0.0; numel], shape, true);
        p.set_grad(Storage::cpu(grad.to_vec()));
        let mut opt = Muon::new(vec![p.clone()], 1.0, 0.0, false, 5);
        opt.step();
        let after = p.borrow().data.to_vec();
        after.iter().map(|w| -w).collect()
    }

    #[test]
    fn step_applies_an_orthogonalized_update() {
        // lr = 1 and no momentum, so the movement is the update itself: for a wide matrix
        // (scale 1) its rows are near orthonormal, which the raw gradient's are not.
        crate::tensor::manual_seed(0x0A7B);
        let grad = random_matrix(4, 8);
        let delta = one_step_delta(&[4, 8], &grad);
        assert!(
            gram_deviation(&delta, 4, 8) < 0.6,
            "the step should move along near-orthonormal rows"
        );
        // The gradient direction itself is nowhere near orthonormal, so this isn't a
        // property of the input.
        let norm = grad.iter().map(|v| v * v).sum::<f32>().sqrt();
        let unit: Vec<f32> = grad.iter().map(|v| v / norm).collect();
        assert!(gram_deviation(&unit, 4, 8) > 0.5);
    }

    #[test]
    fn tall_matrix_update_is_scaled_by_sqrt_of_the_aspect_ratio() {
        // The same gradient as a wide [4, 8] and as a tall [8, 4] (its transpose): the
        // orthogonalized directions are transposes of each other, and the tall one is
        // scaled by sqrt(8 / 4). The wide one has scale 1.
        crate::tensor::manual_seed(0xA5C1);
        let wide_grad = random_matrix(4, 8);
        let tall_grad = RawTensor::transpose_2d(&wide_grad, &[4, 8]);
        let wide = one_step_delta(&[4, 8], &wide_grad);
        let tall = one_step_delta(&[8, 4], &tall_grad);
        let wide_t = RawTensor::transpose_2d(&wide, &[4, 8]);
        assert_eq!(wide_t.len(), tall.len());
        for (w, t) in wide_t.iter().zip(&tall) {
            assert!(
                (w * 2.0_f32.sqrt() - t).abs() < 1e-5,
                "tall update {t} should be sqrt(2) x wide update {w}"
            );
        }
    }

    #[test]
    fn vector_parameters_get_plain_momentum() {
        // A vector is not orthogonalized and not scaled: w -= lr * g.
        let delta = one_step_delta(&[3], &[0.5, -2.0, 4.0]);
        assert_eq!(delta, vec![0.5, -2.0, 4.0]);
        // A single-column matrix is a vector too.
        let delta = one_step_delta(&[3, 1], &[0.5, -2.0, 4.0]);
        assert_eq!(delta, vec![0.5, -2.0, 4.0]);
    }

    #[test]
    fn parameters_without_a_gradient_are_untouched() {
        let p = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        let mut opt = Muon::new(vec![p.clone()], 0.1, 0.9, false, 5);
        opt.step();
        assert_eq!(p.borrow().data.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }
}

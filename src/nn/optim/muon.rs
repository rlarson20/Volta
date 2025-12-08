use crate::tensor::{RawTensor, Tensor};

/// Muon: Momentum Orthogonal Optimizer
///
/// Uses Newton-Schulz iteration to orthogonalize the update steps,
/// useful for large-scale training where standard adaptive methods usually struggle
/// or for specific architectures.
pub struct Muon {
    params: Vec<Tensor>,
    lr: f32,
    momentum: f32,
    nesterov: bool,
    ns_steps: usize,
    velocity: Vec<Vec<f32>>,
}

impl Muon {
    pub fn new(
        params: Vec<Tensor>,
        lr: f32,
        momentum: f32,
        nesterov: bool,
        ns_steps: usize,
    ) -> Self {
        let velocity = params
            .iter()
            .map(|p| vec![0.0; p.borrow().data.len()])
            .collect();
        Muon {
            params,
            lr,
            momentum,
            nesterov,
            ns_steps,
            velocity,
        }
    }

    pub fn zero_grad(&self) {
        for param in &self.params {
            param.borrow_mut().grad = None;
        }
    }

    /// Newton-Schulz iteration: X_{k+1} = 0.5 * `X_k` * (3I - `X_k^T` * `X_k`)
    ///
    /// Operated on flattened (M, N) matrices.
    fn newton_schulz(&self, g: &mut [f32], rows: usize, cols: usize) {
        let steps = self.ns_steps;

        // Helper for inplace: X = A * B
        // We re-use matmul logic from RawTensor

        // Normalize first: x /= x.norm() + epsilon (approx spectral norm)
        // Simple Frobenius norm as proxy or just standard deviation scaling generally used in simple implementations
        // For Muon, usually we assume the inputs are somewhat regulated, but let's compute Frobenius for stability
        let sum_sq: f32 = g.iter().map(|x| x * x).sum();
        let norm = (sum_sq + 1e-8).sqrt();
        if norm > 1e-8 {
            for x in g.iter_mut() {
                *x /= norm;
            }
        }

        // Iteration
        for _ in 0..steps {
            // A = X^T
            let xt = RawTensor::transpose_2d(g, &[rows, cols]);

            // B = X^T * X
            // (Cols, Rows) * (Rows, Cols) -> (Cols, Cols)
            let xt_x = RawTensor::matmul_raw(&xt, g, cols, rows, cols);

            // C = 3I - B
            let mut c = vec![0.0; cols * cols];
            for i in 0..cols {
                for j in 0..cols {
                    let val = xt_x[i * cols + j];
                    if i == j {
                        c[i * cols + j] = 3.0 - val;
                    } else {
                        c[i * cols + j] = -val;
                    }
                }
            }

            // New X = 0.5 * X * C
            // (Rows, Cols) * (Cols, Cols) -> (Rows, Cols)
            let next_x = RawTensor::matmul_raw(g, &c, rows, cols, cols);

            for (i, val) in next_x.iter().enumerate() {
                g[i] = val * 0.5;
            }
        }

        // Rescale? Usually Muon keeps it normalized, acting as "normalized update".
        // We apply LR outside.
    }

    pub fn step(&mut self) {
        for (i, param) in self.params.iter().enumerate() {
            let mut p = param.borrow_mut();
            if let Some(grad) = &p.grad {
                let shape = &p.shape;

                // Muon logic:
                // 1. Flatten params to 2D (keeping output dim as row usually)
                // If vector (1D), treat as (Rows, 1)
                let (rows, cols) = if shape.len() < 2 {
                    (shape[0], 1)
                } else {
                    // For Linear: [In, Out] -> Muon treats as [In, Out] usually or [Out, In]
                    // For Conv2d: [Out, In, K, K] -> flatten to [Out, In*K*K]
                    let rows = shape[0];
                    let cols = shape[1..].iter().product();
                    (rows, cols)
                };

                // Update buffer
                let velocity = &mut self.velocity[i];
                for (v, &g) in velocity.iter_mut().zip(grad.iter()) {
                    *v = self.momentum * *v + g;
                }

                // Prepare update: Nesterov or standard momentum
                let mut update = if self.nesterov {
                    // v = mu * v + g
                    // update = mu * v + g
                    // Here we approximate the update target
                    velocity
                        .iter()
                        .zip(grad.iter())
                        .map(|(v, g)| self.momentum * v + g)
                        .collect::<Vec<f32>>()
                } else {
                    velocity.clone()
                };

                // Orthogonalize update
                if rows > 1 && cols > 1 {
                    self.newton_schulz(&mut update, rows, cols);
                }

                // Apply to weights: w -= lr * update
                for (w, u) in p.data.iter_mut().zip(update.iter()) {
                    *w -= self.lr * u * 0.1; // Scaling factor often needed for Muon stability vs Adam
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RawTensor;
    use crate::Storage;

    #[test]
    fn test_muon_step() {
        // Simple test: Ensure parameters change after a step
        let x = RawTensor::new(vec![1.0, -1.0, 0.5, 0.5], &[2, 2], true);
        // Artificial gradient
        x.borrow_mut().grad = Some(Storage::cpu(vec![0.1, 0.1, -0.1, -0.1]));

        let mut opt = Muon::new(vec![x.clone()], 0.1, 0.9, true, 5);

        let data_before = x.borrow().data.clone();
        opt.step();
        let data_after = x.borrow().data.clone();

        // Verify data changed
        assert_ne!(data_before, data_after);

        // Verify Newton-Schulz didn't explode values (regularization property)
        for v in data_after.iter() {
            assert!(v.abs() < 2.0);
        }
    }
}

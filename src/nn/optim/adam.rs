use crate::tensor::Tensor;

pub struct Adam {
    params: Vec<Tensor>,
    lr: f32,
    betas: (f32, f32),
    eps: f32,
    m: Vec<Vec<f32>>, // 1st moment
    v: Vec<Vec<f32>>, // 2nd moment
    t: usize,         // timestep
}

impl Adam {
    pub fn new(params: Vec<Tensor>, lr: f32, betas: (f32, f32), eps: f32) -> Self {
        let m = params
            .iter()
            .map(|p| vec![0.0; p.borrow().data.len()])
            .collect();
        let v = params
            .iter()
            .map(|p| vec![0.0; p.borrow().data.len()])
            .collect();
        Adam {
            params,
            lr,
            betas,
            eps,
            m,
            v,
            t: 0,
        }
    }

    pub fn zero_grad(&self) {
        for param in &self.params {
            param.borrow_mut().grad = None;
        }
    }

    pub fn step(&mut self) {
        self.t += 1;
        for (i, param) in self.params.iter().enumerate() {
            let mut p = param.borrow_mut();
            if let Some(grad) = &p.grad {
                // Update biased moments
                for j in 0..grad.len() {
                    self.m[i][j] = self.betas.0 * self.m[i][j] + (1.0 - self.betas.0) * grad[j];
                    self.v[i][j] =
                        self.betas.1 * self.v[i][j] + (1.0 - self.betas.1) * grad[j].powi(2);
                }

                // Bias correction
                let m_hat_scale = 1.0 / (1.0 - self.betas.0.powi(self.t as i32));
                let v_hat_scale = 1.0 / (1.0 - self.betas.1.powi(self.t as i32));

                // Update parameters
                for j in 0..p.data.len() {
                    let m_hat = self.m[i][j] * m_hat_scale;
                    let v_hat = self.v[i][j] * v_hat_scale;
                    p.data[j] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
                }
            }
        }
    }
}

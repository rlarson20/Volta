use crate::io::StateDict;
use crate::nn::Module;
use crate::tensor::{RawTensor, Tensor, TensorOps};
use rand::Rng;

pub struct Dropout {
    p: f32,
    training: bool,
}

impl Dropout {
    /// Create a new Dropout layer
    ///
    /// # Arguments
    /// * `p` - Probability of an element being zeroed out (default: 0.5)
    /// # Panics
    /// dropout prob must be in \[0,1\]
    #[must_use]
    pub fn new(p: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&p),
            "Dropout probability must be in [0, 1]"
        );
        Self { p, training: true }
    }
}

impl Module for Dropout {
    fn forward(&self, x: &Tensor) -> Tensor {
        if !self.training || self.p == 0.0 {
            return x.clone();
        }

        let keep_prob = 1.0 - self.p;
        let scale = 1.0 / keep_prob;

        let (shape, device) = {
            let x_borrow = x.borrow();
            (x_borrow.shape.clone(), x_borrow.device.clone())
        };
        let size: usize = shape.iter().product();

        // Generate mask: 1 with prob (1-p), 0 with prob p
        let mask_data: Vec<f32> = crate::tensor::with_rng(|rng| {
            (0..size)
                .map(|_| {
                    if rng.random::<f32>() < keep_prob {
                        scale
                    } else {
                        0.0
                    }
                })
                .collect()
        });

        let mask = RawTensor::new(mask_data, &shape, false).to_device(device);

        // Apply mask: x * mask
        x.elem_mul(&mask)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }

    fn state_dict(&self) -> StateDict {
        StateDict::new()
    }

    fn load_state_dict(&mut self, _state: &StateDict) {
        // Stateless
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }
}

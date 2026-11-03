use crate::io::StateDict;
use crate::nn::{Layer, Module};
use crate::tensor::Tensor;
use rand::Rng;

/// Zeroes each element with probability `p` in training mode, scaling the rest by `1 / (1 - p)`; the identity in evaluation mode.
pub struct Dropout {
    /// Probability that an element is zeroed.
    p: f32,
    /// Whether dropout is active.
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
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }

    fn state_dict(&self) -> StateDict {
        StateDict::new()
    }

    fn load_state_dict(&mut self, _state: &StateDict) -> crate::Result<()> {
        // Stateless
        Ok(())
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }
}

impl Layer for Dropout {
    fn forward(&self, x: &Tensor) -> Tensor {
        if !self.training || self.p == 0.0 {
            return x.clone();
        }

        let keep_prob = 1.0 - self.p;
        let scale = 1.0 / keep_prob;

        let shape = x.borrow().shape.clone();
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

        // On x's device and in its dtype; the f32 scale widens exactly.
        let mask = x.constant_like(mask_data, shape.truncate_rank());

        // Apply mask: x * mask
        x.elem_mul(&mask)
    }
}

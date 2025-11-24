use crate::io::StateDict;
use crate::nn::Module;
use crate::tensor::{Tensor, TensorOps};

/// Flattens the input tensor into a 2D tensor (batch_size, remaining_features).
///
/// Assumes the first dimension is the batch dimension and flattens all subsequent dimensions.
/// Input shape: (B, D1, D2, ...)
/// Output shape: (B, D1 * D2 * ...)
pub struct Flatten;

impl Flatten {
    pub fn new() -> Self {
        Flatten
    }
}

impl Default for Flatten {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Flatten {
    fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.borrow().shape.clone();
        if shape.len() < 2 {
            // Already flat or scalar
            return x.clone();
        }

        let batch_size = shape[0];
        let flattened_size: usize = shape[1..].iter().product();

        x.reshape(&[batch_size, flattened_size])
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RawTensor;

    #[test]
    fn test_flatten() {
        let flatten = Flatten::new();
        // B=2, C=2, H=2, W=2 -> 2 x 8
        let x = RawTensor::zeros(&[2, 2, 2, 2]);
        let y = flatten.forward(&x);

        assert_eq!(y.borrow().shape, vec![2, 8]);
    }
}

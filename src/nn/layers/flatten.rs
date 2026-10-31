use crate::io::StateDict;
use crate::nn::{Layer, Module};
use crate::tensor::Tensor;

/// Flattens the input tensor into a 2D tensor (`batch_size`, `remaining_features`).
///
/// Assumes the first dimension is the batch dimension and flattens all subsequent dimensions.
/// Input shape: (B, D1, D2, ...)
/// Output shape: (B, D1 * D2 * ...)
pub struct Flatten;

impl Flatten {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Default for Flatten {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Flatten {
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
}

impl Layer for Flatten {
    fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.borrow().shape.clone();
        if shape.rank() < 2 {
            // Already flat or scalar
            return x.clone();
        }

        let batch_size = shape.dim_at(0);
        let flattened_size = shape.numel_from(1);

        x.reshape(&[batch_size, flattened_size])
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

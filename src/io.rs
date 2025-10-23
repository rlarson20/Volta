use crate::{Linear, Tensor};
use bincode::{Decode, Encode, config};
use std::fs::File;
use std::io::{Error, Read, Result, Write};

// Serializable representation of tensor data
#[derive(Encode, Decode, Clone)]
struct TensorData {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl TensorData {
    fn from_tensor(t: &Tensor) -> Self {
        let borrowed = t.borrow();
        TensorData {
            data: borrowed.data.clone(),
            shape: borrowed.shape.clone(),
        }
    }

    fn to_tensor(&self, requires_grad: bool) -> Tensor {
        crate::RawTensor::new(self.data.clone(), &self.shape, requires_grad)
    }
}

// Serializable Linear layer
#[derive(Encode, Decode)]
pub struct LinearState {
    weight: TensorData,
    bias: Option<TensorData>,
}

impl Linear {
    pub fn save(&self, path: &str) -> Result<()> {
        let state = LinearState {
            weight: TensorData::from_tensor(&self.weight),
            bias: self.bias.as_ref().map(TensorData::from_tensor),
        };
        let mut file = File::create(path)?;
        let encoded = bincode::encode_to_vec(&state, config::standard()).map_err(Error::other)?;
        file.write_all(&encoded)?;
        Ok(())
    }

    pub fn load(&mut self, path: &str) -> Result<()> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let (state, _): (LinearState, _) =
            bincode::decode_from_slice(&buffer, config::standard()).map_err(Error::other)?;

        // Update weights
        self.weight = state.weight.to_tensor(true);
        self.bias = state.bias.map(|b| b.to_tensor(true));
        Ok(())
    }
}

#[cfg(test)]
mod io_tests {
    use crate::Linear as VoltaLinear;
    use crate::RawTensor;
    #[test]
    fn test_save_load_linear() {
        let layer = VoltaLinear::new(3, 2, true);

        // Forward pass before save
        let x = RawTensor::new(vec![1.0, 2.0, 3.0], &[1, 3], false);
        let y1 = layer.forward(&x);

        // Save
        layer.save("/tmp/test_linear.bin").unwrap();

        // Create new layer and load
        let mut layer2 = VoltaLinear::new(3, 2, true);
        layer2.load("/tmp/test_linear.bin").unwrap();

        // Forward pass after load should match
        let y2 = layer2.forward(&x);

        for (a, b) in y1.borrow().data.iter().zip(&y2.borrow().data) {
            assert!((a - b).abs() < 1e-6, "Loaded weights don't match");
        }
    }
}

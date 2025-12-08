use crate::tensor::Tensor;
use bincode::{Decode, Encode, config};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Error, Read, Result, Write};

pub type StateDict = BTreeMap<String, TensorData>;

// Serializable representation of tensor data
#[derive(Encode, Decode, Clone)]
pub struct TensorData {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl TensorData {
    pub fn from_tensor(t: &Tensor) -> Self {
        let borrowed = t.borrow();
        TensorData {
            data: borrowed.data.to_vec(),
            shape: borrowed.shape.clone(),
        }
    }

    pub fn to_tensor(&self, requires_grad: bool) -> Tensor {
        crate::RawTensor::new(self.data.clone(), &self.shape, requires_grad)
    }
}

pub fn save_state_dict(state: &StateDict, path: &str) -> Result<()> {
    let mut file = File::create(path)?;
    let encoded = bincode::encode_to_vec(state, config::standard()).map_err(Error::other)?;
    file.write_all(&encoded)?;
    Ok(())
}
pub fn load_state_dict(path: &str) -> Result<StateDict> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    let (state, _): (StateDict, _) =
        bincode::decode_from_slice(&buffer, config::standard()).map_err(Error::other)?;
    Ok(state)
}

#[cfg(test)]
mod io_tests {
    use super::*;
    use crate::nn::Module;
    use crate::{Linear, ReLU, Sequential};

    #[test]
    fn test_save_load_sequential() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(2, 3, true)),
            Box::new(ReLU),
            Box::new(Linear::new(3, 1, true)),
        ]);

        let path = std::env::temp_dir().join("test_seq.bin");
        let path_str = path.to_str().unwrap();

        // Save
        let state = model.state_dict();
        save_state_dict(&state, path_str).unwrap();

        // Load new model
        let mut model2 = Sequential::new(vec![
            Box::new(Linear::new(2, 3, true)),
            Box::new(ReLU),
            Box::new(Linear::new(3, 1, true)),
        ]);

        // Verify weights are different initially
        let p1 = model.parameters();
        let p2 = model2.parameters();
        assert_ne!(p1[0].borrow().data, p2[0].borrow().data);

        // Load
        let loaded_state = load_state_dict(path_str).unwrap();
        model2.load_state_dict(&loaded_state);

        // Verify weights match
        let p1 = model.parameters();
        let p2 = model2.parameters();
        for (t1, t2) in p1.iter().zip(p2.iter()) {
            assert_eq!(t1.borrow().data, t2.borrow().data);
        }
    }
}

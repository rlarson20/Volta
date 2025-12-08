use crate::nn::Module;
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

/// Summary of differences between two state dicts.
///
/// Intended for debugging and tooling: `expected` is usually taken from
/// `model.state_dict()`, and `loaded` is what was deserialized or passed in.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct StateDictDiff {
    /// Keys that exist in `expected` but are missing from `loaded`.
    pub missing_keys: Vec<String>,
    /// Keys that exist in `loaded` but not in `expected`.
    pub unexpected_keys: Vec<String>,
    /// Keys present in both, but with differing shapes:
    /// `(key, expected_shape, loaded_shape)`.
    pub shape_mismatches: Vec<(String, Vec<usize>, Vec<usize>)>,
}

impl StateDictDiff {
    /// Returns true if there are no missing, unexpected, or shape‑mismatched keys.
    pub fn is_empty(&self) -> bool {
        self.missing_keys.is_empty()
            && self.unexpected_keys.is_empty()
            && self.shape_mismatches.is_empty()
    }
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

/// Compute a diff between an "expected" and a "loaded" state dict.
///
/// Typical usage:
/// - `expected` = `model.state_dict()` from the current architecture
/// - `loaded`   = state loaded from disk or another model
///
/// This function **does not mutate any tensors** and is purely informational.
pub fn diff_state_dict(expected: &StateDict, loaded: &StateDict) -> StateDictDiff {
    let mut diff = StateDictDiff::default();

    // 1. Missing keys and shape mismatches
    for (key, expected_td) in expected.iter() {
        match loaded.get(key) {
            None => diff.missing_keys.push(key.clone()),
            Some(actual_td) => {
                if expected_td.shape != actual_td.shape {
                    diff.shape_mismatches.push((
                        key.clone(),
                        expected_td.shape.clone(),
                        actual_td.shape.clone(),
                    ));
                }
            }
        }
    }

    // 2. Unexpected keys present only in `loaded`
    for key in loaded.keys() {
        if !expected.contains_key(key) {
            diff.unexpected_keys.push(key.clone());
        }
    }

    diff
}

/// Load a state dict and report which keys were missing/unexpected or mismatched.
///
/// This helper computes the diff between what the module currently expects
/// (`module.state_dict()`) and what was provided before delegating to
/// `module.load_state_dict`. It is safe to call even when some keys are missing,
/// since we inspect the diff up front.
pub fn load_state_dict_checked<M: Module + ?Sized>(
    module: &mut M,
    state: &StateDict,
) -> StateDictDiff {
    let expected = module.state_dict();
    let diff = diff_state_dict(&expected, state);
    module.load_state_dict(state);
    diff
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

    #[test]
    fn test_state_dict_diff_reports_mismatches() {
        // Simple Linear layer: expected state dict has "weight" and "bias".
        let layer = Linear::new(2, 3, true);
        let expected = layer.state_dict();

        // Start from a clone and deliberately introduce:
        // - one missing key ("bias"),
        // - one unexpected key ("extra"),
        // - one shape mismatch on "weight".
        let mut loaded = expected.clone();

        // Remove "bias" to make it a missing key.
        loaded.remove("bias");

        // Add an unexpected key.
        loaded.insert(
            "extra".to_string(),
            TensorData {
                data: vec![0.0],
                shape: vec![1],
            },
        );

        // Corrupt the shape of "weight".
        if let Some(td) = loaded.get_mut("weight") {
            td.shape = vec![999];
        }

        let diff = diff_state_dict(&expected, &loaded);
        assert!(!diff.is_empty());

        assert!(diff.missing_keys.contains(&"bias".to_string()));
        assert!(diff.unexpected_keys.contains(&"extra".to_string()));

        assert!(
            diff.shape_mismatches
                .iter()
                .any(|(k, _exp, _act)| k == "weight")
        );
    }
}

use crate::dtype::DType;
use crate::nn::Module;
use crate::storage::Storage;
use crate::tensor::Tensor;
use bincode::{Decode, Encode, config};
use safetensors::SafeTensors;
use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::io::{Error, Read, Result, Write};
use std::path::Path;

pub mod mapping;

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

impl fmt::Display for StateDictDiff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return write!(f, "StateDictDiff(empty)");
        }

        let mut sections = Vec::new();

        if !self.missing_keys.is_empty() {
            sections.push(format!("missing: {}", self.missing_keys.join(", ")));
        }

        if !self.unexpected_keys.is_empty() {
            sections.push(format!("unexpected: {}", self.unexpected_keys.join(", ")));
        }

        if !self.shape_mismatches.is_empty() {
            let entries = self
                .shape_mismatches
                .iter()
                .map(|(key, expected, loaded)| {
                    format!("{} (expected {:?}, found {:?})", key, expected, loaded)
                })
                .collect::<Vec<_>>()
                .join("; ");
            sections.push(format!("shape mismatches: {}", entries));
        }

        write!(f, "StateDictDiff {{ {} }}", sections.join("; "))
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

// ========== SafeTensors Support ==========

/// Convert SafeTensors dtype to Volta DType
fn safetensors_dtype_to_volta(dtype: safetensors::Dtype) -> DType {
    match dtype {
        safetensors::Dtype::F16 => DType::F16,
        safetensors::Dtype::BF16 => DType::BF16,
        safetensors::Dtype::F32 => DType::F32,
        safetensors::Dtype::F64 => DType::F64,
        safetensors::Dtype::I32 => DType::I32,
        safetensors::Dtype::I64 => DType::I64,
        safetensors::Dtype::U8 => DType::U8,
        safetensors::Dtype::BOOL => DType::Bool,
        _ => DType::F32, // Fallback for unsupported types
    }
}

/// Convert Volta DType to SafeTensors dtype
fn volta_dtype_to_safetensors(dtype: DType) -> safetensors::Dtype {
    match dtype {
        DType::F16 => safetensors::Dtype::F16,
        DType::BF16 => safetensors::Dtype::BF16,
        DType::F32 => safetensors::Dtype::F32,
        DType::F64 => safetensors::Dtype::F64,
        DType::I32 => safetensors::Dtype::I32,
        DType::I64 => safetensors::Dtype::I64,
        DType::U8 => safetensors::Dtype::U8,
        DType::Bool => safetensors::Dtype::BOOL,
    }
}

/// Load a SafeTensors file into a StateDict (converts all tensors to f32)
///
/// This is the simplest API for loading pretrained models. All tensors are
/// converted to f32 to match the existing StateDict format.
///
/// For native dtype loading, use `load_safetensors_raw()`.
pub fn load_safetensors<P: AsRef<Path>>(path: P) -> Result<StateDict> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let tensors = SafeTensors::deserialize(&buffer)
        .map_err(|e| Error::other(format!("SafeTensors parse error: {}", e)))?;

    let mut state_dict = StateDict::new();

    for (name, tensor) in tensors.tensors() {
        let shape: Vec<usize> = tensor.shape().to_vec();
        let dtype = safetensors_dtype_to_volta(tensor.dtype());

        // Create Storage from raw bytes and convert to f32
        let storage = Storage::from_bytes(tensor.data().to_vec(), dtype);
        let f32_data = storage.to_f32_vec();

        state_dict.insert(
            name.to_string(),
            TensorData {
                data: f32_data,
                shape,
            },
        );
    }

    Ok(state_dict)
}

/// Tensor data with native dtype support for SafeTensors
#[derive(Clone)]
pub struct TypedTensorData {
    /// Raw bytes of tensor data
    pub data: Vec<u8>,
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: DType,
}

impl TypedTensorData {
    /// Convert to f32 TensorData
    pub fn to_tensor_data(&self) -> TensorData {
        let storage = Storage::from_bytes(self.data.clone(), self.dtype);
        TensorData {
            data: storage.to_f32_vec(),
            shape: self.shape.clone(),
        }
    }

    /// Create a Storage with native dtype
    pub fn to_storage(&self) -> Storage {
        Storage::from_bytes(self.data.clone(), self.dtype)
    }

    /// Get number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Load a SafeTensors file with native dtypes preserved
///
/// Returns a map of tensor names to TypedTensorData, preserving the original
/// dtype (F16, BF16, etc.) without conversion.
pub fn load_safetensors_raw<P: AsRef<Path>>(path: P) -> Result<BTreeMap<String, TypedTensorData>> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let tensors = SafeTensors::deserialize(&buffer)
        .map_err(|e| Error::other(format!("SafeTensors parse error: {}", e)))?;

    let mut result = BTreeMap::new();

    for (name, tensor) in tensors.tensors() {
        result.insert(
            name.to_string(),
            TypedTensorData {
                data: tensor.data().to_vec(),
                shape: tensor.shape().to_vec(),
                dtype: safetensors_dtype_to_volta(tensor.dtype()),
            },
        );
    }

    Ok(result)
}

/// Save a StateDict to SafeTensors format
///
/// All tensors are saved as F32 since StateDict uses f32.
pub fn save_safetensors<P: AsRef<Path>>(state: &StateDict, path: P) -> Result<()> {
    use safetensors::tensor::{Dtype, TensorView};

    let tensors: Vec<(String, TensorView<'_>)> = state
        .iter()
        .map(|(name, td)| {
            let view =
                TensorView::new(Dtype::F32, td.shape.clone(), bytemuck::cast_slice(&td.data))
                    .expect("Failed to create TensorView");
            (name.clone(), view)
        })
        .collect();

    let bytes = safetensors::tensor::serialize(tensors, None)
        .map_err(|e| Error::other(format!("SafeTensors serialize error: {}", e)))?;

    let mut file = File::create(path)?;
    file.write_all(&bytes)?;
    Ok(())
}

/// Save typed tensor data to SafeTensors format with native dtypes
pub fn save_safetensors_typed<P: AsRef<Path>>(
    tensors: &BTreeMap<String, TypedTensorData>,
    path: P,
) -> Result<()> {
    use safetensors::tensor::TensorView;

    let tensor_views: Vec<(String, TensorView<'_>)> = tensors
        .iter()
        .map(|(name, td)| {
            let view = TensorView::new(
                volta_dtype_to_safetensors(td.dtype),
                td.shape.clone(),
                &td.data,
            )
            .expect("Failed to create TensorView");
            (name.clone(), view)
        })
        .collect();

    let bytes = safetensors::tensor::serialize(tensor_views, None)
        .map_err(|e| Error::other(format!("SafeTensors serialize error: {}", e)))?;

    let mut file = File::create(path)?;
    file.write_all(&bytes)?;
    Ok(())
}

/// Load a state dict from a file and apply transformations
///
/// This is a convenience wrapper around `load_state_dict` that applies
/// a StateDictMapper transformation before returning the result.
///
/// # Example
/// ```no_run
/// use volta::io::{load_state_dict_with_mapping, mapping::StateDictMapper};
///
/// let mapper = StateDictMapper::new()
///     .strip_prefix("model.")
///     .transpose_pattern("weight");
///
/// let state = load_state_dict_with_mapping("model.bin", &mapper)?;
/// # Ok::<(), std::io::Error>(())
/// ```
pub fn load_state_dict_with_mapping<P: AsRef<Path>>(
    path: P,
    mapper: &mapping::StateDictMapper,
) -> Result<StateDict> {
    let state = load_state_dict(path.as_ref().to_str().unwrap())?;
    Ok(mapper.map(state))
}

/// Load a safetensors file and apply transformations
///
/// This is a convenience wrapper around `load_safetensors` that applies
/// a StateDictMapper transformation before returning the result.
///
/// # Example
/// ```no_run
/// use volta::io::{load_safetensors_with_mapping, mapping::StateDictMapper};
///
/// let mapper = StateDictMapper::new()
///     .rename("fc1.weight", "encoder.weight")
///     .transpose("encoder.weight");
///
/// let state = load_safetensors_with_mapping("pytorch_model.safetensors", &mapper)?;
/// # Ok::<(), std::io::Error>(())
/// ```
pub fn load_safetensors_with_mapping<P: AsRef<Path>>(
    path: P,
    mapper: &mapping::StateDictMapper,
) -> Result<StateDict> {
    let state = load_safetensors(path)?;
    Ok(mapper.map(state))
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
    #[test]
    fn test_state_dict_diff_display() {
        let mut diff = StateDictDiff::default();
        diff.missing_keys.push("0.bias".into());
        diff.unexpected_keys.push("extra".into());
        diff.shape_mismatches
            .push(("0.weight".into(), vec![2, 3], vec![3, 2]));

        let message = diff.to_string();
        assert!(message.contains("missing: 0.bias"));
        assert!(message.contains("unexpected: extra"));
        assert!(message.contains("shape mismatches:"));
        assert!(message.contains("0.weight"));
        assert!(message.contains("[2, 3]"));
        assert!(message.contains("[3, 2]"));
    }

    #[test]
    fn test_safetensors_roundtrip() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(2, 3, true)),
            Box::new(ReLU),
            Box::new(Linear::new(3, 1, true)),
        ]);

        let path = std::env::temp_dir().join("test_safetensors.safetensors");

        // Save to SafeTensors
        let state = model.state_dict();
        save_safetensors(&state, &path).unwrap();

        // Load from SafeTensors
        let loaded_state = load_safetensors(&path).unwrap();

        // Verify keys match
        assert_eq!(state.len(), loaded_state.len());
        for (key, td) in state.iter() {
            let loaded_td = loaded_state.get(key).expect("Key should exist");
            assert_eq!(td.shape, loaded_td.shape);
            // Check data (with small tolerance for f32 roundtrip)
            for (a, b) in td.data.iter().zip(loaded_td.data.iter()) {
                assert!((a - b).abs() < 1e-6, "Data mismatch for key {}", key);
            }
        }
    }

    #[test]
    fn test_safetensors_raw_dtypes() {
        use crate::DType;

        // Create typed tensor data
        let mut tensors = BTreeMap::new();

        // F32 tensor
        let f32_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        tensors.insert(
            "f32_tensor".to_string(),
            TypedTensorData {
                data: bytemuck::cast_slice(&f32_data).to_vec(),
                shape: vec![2, 2],
                dtype: DType::F32,
            },
        );

        // F16 tensor
        let f16_data: Vec<half::f16> = [1.0, 2.0, 3.0, 4.0]
            .iter()
            .map(|&x| half::f16::from_f32(x))
            .collect();
        tensors.insert(
            "f16_tensor".to_string(),
            TypedTensorData {
                data: bytemuck::cast_slice(&f16_data).to_vec(),
                shape: vec![2, 2],
                dtype: DType::F16,
            },
        );

        let path = std::env::temp_dir().join("test_typed.safetensors");

        // Save with native dtypes
        save_safetensors_typed(&tensors, &path).unwrap();

        // Load with native dtypes
        let loaded = load_safetensors_raw(&path).unwrap();

        // Verify dtypes are preserved
        assert_eq!(loaded.get("f32_tensor").unwrap().dtype, DType::F32);
        assert_eq!(loaded.get("f16_tensor").unwrap().dtype, DType::F16);

        // Verify shapes
        assert_eq!(loaded.get("f32_tensor").unwrap().shape, vec![2, 2]);
        assert_eq!(loaded.get("f16_tensor").unwrap().shape, vec![2, 2]);

        // Verify data conversion
        let f32_loaded = loaded.get("f32_tensor").unwrap().to_storage().to_f32_vec();
        assert_eq!(f32_loaded, f32_data);

        let f16_loaded = loaded.get("f16_tensor").unwrap().to_storage().to_f32_vec();
        for (a, b) in f16_loaded.iter().zip([1.0f32, 2.0, 3.0, 4.0].iter()) {
            assert!((a - b).abs() < 0.01, "F16 conversion mismatch");
        }
    }
}

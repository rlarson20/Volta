use crate::dtype::DType;
use crate::error::OrPanic;
use crate::nn::Module;
use crate::storage::{Element, Storage};
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::io::{Error, Read, Result, Write};
use std::path::Path;

/// Key remapping between Volta and external (e.g. PyTorch) state dicts.
pub mod mapping;

/// Parameter name to saved values.
pub type StateDict = BTreeMap<String, TensorData>;

/// Serializable copy of one tensor: its values in their own dtype, and its shape.
///
/// `data` holds the elements' bytes exactly as [`Storage::as_bytes`] gives them
/// (native-endian, which is little-endian on every platform Volta builds for, the same
/// layout safetensors uses), so a save and load is bit-exact for every dtype. The fields
/// are public, so nothing stops a caller from building an entry whose byte count does not
/// match `shape` and `dtype`; [`TensorData::to_storage`] checks that before decoding, and
/// the loaders check every entry they read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorData {
    /// Row-major element bytes: `shape.iter().product::<usize>() * dtype.size_of()` of them.
    pub data: Vec<u8>,
    /// Dimension sizes, outermost first.
    pub shape: Vec<usize>,
    /// Element type of `data`.
    pub dtype: DType,
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
    #[must_use]
    pub const fn is_empty(&self) -> bool {
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
                    format!("{key} (expected {expected:?}, found {loaded:?})")
                })
                .collect::<Vec<_>>()
                .join("; ");
            sections.push(format!("shape mismatches: {entries}"));
        }

        write!(f, "StateDictDiff {{ {} }}", sections.join("; "))
    }
}

impl TensorData {
    /// Copies a tensor's values (in its own dtype) and shape out of the graph. A GPU
    /// tensor is read through its host copy.
    #[must_use]
    pub fn from_tensor(t: &Tensor) -> Self {
        let borrowed = t.borrow();
        Self {
            data: borrowed.data.as_bytes().to_vec(),
            shape: borrowed.shape.truncate_rank().to_vec(),
            dtype: borrowed.data.dtype(),
        }
    }

    /// An f32 entry from row-major values. The common case, and unambiguous with float
    /// literals (unlike [`Self::from_typed`], where `vec![1.0]` would infer f64).
    #[must_use]
    pub fn from_f32(values: Vec<f32>, shape: Vec<usize>) -> Self {
        Self::from_typed(values, shape)
    }

    /// An entry from row-major values of any element type; the dtype follows `T`.
    #[must_use]
    pub fn from_typed<T: Element>(values: Vec<T>, shape: Vec<usize>) -> Self {
        Self {
            data: Storage::from_typed(values).as_bytes().to_vec(),
            shape,
            dtype: T::DTYPE,
        }
    }

    /// Number of elements `shape` describes.
    ///
    /// # Panics
    /// If the product of `shape` overflows `usize`. [`Self::to_storage`] reports that as
    /// an error instead.
    #[must_use]
    #[track_caller]
    pub fn numel(&self) -> usize {
        checked_numel(&self.shape).expect("TensorData::numel: shape product overflows usize")
    }

    /// Checks that `data` holds exactly the bytes `shape` and `dtype` need, and that every
    /// `Bool` byte is 0 or 1. Allocates nothing. `key` names the entry in the error.
    ///
    /// # Errors
    /// `VoltaError::StateDataLength` (counted in bytes; `expected` is `usize::MAX` when the
    /// shape's size overflows) if the byte count is wrong, and
    /// `VoltaError::InvalidParameter` if a `Bool` byte is neither 0 nor 1.
    pub fn validate(&self, key: &str) -> crate::Result<()> {
        let expected = checked_numel(&self.shape)
            .and_then(|n| n.checked_mul(self.dtype.size_of()))
            .unwrap_or(usize::MAX);
        if self.data.len() != expected {
            return Err(crate::VoltaError::StateDataLength {
                key: key.to_string(),
                expected,
                len: self.data.len(),
            });
        }
        if self.dtype == DType::Bool
            && let Some((i, b)) = self.data.iter().enumerate().find(|&(_, &b)| b > 1)
        {
            return Err(crate::VoltaError::InvalidParameter(format!(
                "state dict entry '{key}': bool element {i} is the byte {b}, not 0 or 1"
            )));
        }
        Ok(())
    }

    /// Decodes `data` as `dtype`, after [`Self::validate`]. `key` names the entry in the
    /// error.
    ///
    /// # Errors
    /// Whatever [`Self::validate`] returns; nothing is allocated before it passes.
    pub fn to_storage(&self, key: &str) -> crate::Result<Storage> {
        self.validate(key)?;
        Storage::try_from_bytes(&self.data, self.dtype).map_err(|e| {
            crate::VoltaError::InvalidParameter(format!("state dict entry '{key}': {e}"))
        })
    }

    /// The values in their own dtype, without conversion. `T` names the dtype, as in
    /// [`Tensor::to_vec_as`].
    ///
    /// # Errors
    /// Whatever [`Self::to_storage`] returns, and `VoltaError::DTypeMismatch` if `T` is
    /// not `dtype`.
    pub fn to_vec_as<T: Element>(&self) -> crate::Result<Vec<T>> {
        self.to_storage("")?.as_host::<T>().map(<[T]>::to_vec)
    }

    /// Rebuilds a CPU tensor from the saved values, in the saved dtype, with the given
    /// `requires_grad`.
    ///
    /// # Errors
    /// Whatever [`Self::validate`] returns, and whatever [`crate::shape::Shape::from_dims`]
    /// returns for `shape` (a zero dim, or too many elements).
    pub fn try_to_tensor(&self, requires_grad: bool) -> crate::Result<Tensor> {
        let storage = self.to_storage("")?;
        let shape = crate::shape::Shape::from_dims(self.shape.clone())?;
        crate::RawTensor::try_new_with_storage(storage, shape, crate::Device::CPU, requires_grad)
    }

    /// Panicking form of [`Self::try_to_tensor`].
    ///
    /// # Panics
    /// Whenever `try_to_tensor` returns an error.
    #[must_use]
    #[track_caller]
    pub fn to_tensor(&self, requires_grad: bool) -> Tensor {
        self.try_to_tensor(requires_grad).or_panic("to_tensor")
    }
}

/// Product of `shape`, or `None` if it overflows `usize`.
fn checked_numel(shape: &[usize]) -> Option<usize> {
    shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d))
}

/// Checks every entry of a state dict read from outside the program.
///
/// # Errors
/// `InvalidData`, wrapping the [`crate::VoltaError`] from [`TensorData::to_storage`], for
/// the first malformed entry.
fn validate_state(state: &StateDict) -> Result<()> {
    for (key, td) in state {
        td.validate(key)
            .map_err(|e| Error::new(std::io::ErrorKind::InvalidData, e))?;
    }
    Ok(())
}

/// Compute a diff between an "expected" and a "loaded" state dict.
///
/// Typical usage:
/// - `expected` = `model.state_dict()` from the current architecture
/// - `loaded`   = state loaded from disk or another model
///
/// This function **does not mutate any tensors** and is purely informational.
#[must_use]
pub fn diff_state_dict(expected: &StateDict, loaded: &StateDict) -> StateDictDiff {
    let mut diff = StateDictDiff::default();

    // 1. Missing keys and shape mismatches
    for (key, expected_td) in expected {
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
/// since we inspect the diff up front. Entries whose shape mismatches are reported
/// in the diff and *not* loaded, so a wrong shape is not an error here.
///
/// # Errors
///
/// `VoltaError::StateDataLength` if an entry whose shape matches holds the wrong
/// number of values for it. Shape mismatches never reach the load; they are in the
/// returned diff. As with [`Module::load_state_dict`], the module may be partly loaded
/// on `Err`.
pub fn load_state_dict_checked<M: Module + ?Sized>(
    module: &mut M,
    state: &StateDict,
) -> crate::Result<StateDictDiff> {
    let expected = module.state_dict();
    let diff = diff_state_dict(&expected, state);
    let loadable: StateDict = state
        .iter()
        .filter(|(key, _)| !diff.shape_mismatches.iter().any(|(k, _, _)| k == *key))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    module.load_state_dict(&loadable)?;
    Ok(diff)
}

// ========== SafeTensors Support ==========

/// Convert a `SafeTensors` dtype to Volta's, or `None` if Volta has no such dtype.
const fn safetensors_dtype_to_volta(dtype: safetensors::Dtype) -> Option<DType> {
    match dtype {
        safetensors::Dtype::F16 => Some(DType::F16),
        safetensors::Dtype::BF16 => Some(DType::BF16),
        safetensors::Dtype::F32 => Some(DType::F32),
        safetensors::Dtype::F64 => Some(DType::F64),
        safetensors::Dtype::I32 => Some(DType::I32),
        safetensors::Dtype::I64 => Some(DType::I64),
        safetensors::Dtype::U8 => Some(DType::U8),
        safetensors::Dtype::BOOL => Some(DType::Bool),
        // Listed so a new safetensors dtype is a deliberate choice, not a silent f32 (S18).
        safetensors::Dtype::F4
        | safetensors::Dtype::F6_E2M3
        | safetensors::Dtype::F6_E3M2
        | safetensors::Dtype::I8
        | safetensors::Dtype::F8_E5M2
        | safetensors::Dtype::F8_E4M3
        | safetensors::Dtype::F8_E8M0
        | safetensors::Dtype::I16
        | safetensors::Dtype::U16
        | safetensors::Dtype::U32
        | safetensors::Dtype::C64
        | safetensors::Dtype::U64
        | _ => None,
    }
}

/// Volta's dtype for one safetensors entry.
///
/// # Errors
/// `InvalidData` if the dtype has no Volta equivalent. Reading it as f32 bytes instead
/// gave plausible wrong values (S18).
fn entry_dtype(name: &str, tensor: &safetensors::tensor::TensorView<'_>) -> Result<DType> {
    safetensors_dtype_to_volta(tensor.dtype()).ok_or_else(|| {
        Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "SafeTensors entry '{name}' has dtype {:?}, which Volta does not support \
                 (supported: F16, BF16, F32, F64, I32, I64, U8, BOOL)",
                tensor.dtype()
            ),
        )
    })
}

/// Convert Volta `DType` to `SafeTensors` dtype
const fn volta_dtype_to_safetensors(dtype: DType) -> safetensors::Dtype {
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

/// Load a `SafeTensors` file into a `StateDict`, each entry in its stored dtype.
///
/// The values are not converted: [`Module::load_state_dict`] casts each entry to its
/// parameter's dtype.
/// # Errors
/// The file can't be read or parsed, or an entry has a dtype Volta does not support
/// (`InvalidData`; I8, U16, U32, U64, the f8 formats, ...).
pub fn load_safetensors<P: AsRef<Path>>(path: P) -> Result<StateDict> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let tensors = SafeTensors::deserialize(&buffer)
        .map_err(|e| Error::other(format!("SafeTensors parse error: {e}")))?;

    let mut state_dict = StateDict::new();

    for (name, tensor) in tensors.tensors() {
        let entry = TensorData {
            data: tensor.data().to_vec(),
            shape: tensor.shape().to_vec(),
            dtype: entry_dtype(&name, &tensor)?,
        };
        state_dict.insert(name, entry);
    }
    // safetensors has checked each byte length against shape and dtype; this repeats it
    // (and checks bool bytes) so a mismatch is an error here too, not a later panic.
    validate_state(&state_dict)?;

    Ok(state_dict)
}

/// Tensor data with native dtype support for `SafeTensors`
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
    /// The same entry as a [`TensorData`], in the same dtype (the bytes are copied, not
    /// converted).
    ///
    /// # Errors
    /// Whatever [`TensorData::validate`] returns for it.
    pub fn to_tensor_data(&self) -> crate::Result<TensorData> {
        let td = TensorData {
            data: self.data.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype,
        };
        td.validate("")?;
        Ok(td)
    }

    /// Storage holding the data in its own dtype.
    ///
    /// # Errors
    /// `InvalidParameter` if the bytes don't decode as `dtype` (see
    /// [`Storage::try_from_bytes`]).
    pub fn to_storage(&self) -> crate::Result<Storage> {
        Storage::try_from_bytes(&self.data, self.dtype)
    }

    /// Get number of elements
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Load a `SafeTensors` file with native dtypes preserved
///
/// Returns a map of tensor names to `TypedTensorData`, preserving the original
/// dtype (F16, BF16, etc.) without conversion.
/// # Errors
/// The file can't be read or parsed, or an entry has a dtype Volta does not support
/// (`InvalidData`; I8, U16, U32, U64, the f8 formats, ...).
pub fn load_safetensors_raw<P: AsRef<Path>>(path: P) -> Result<BTreeMap<String, TypedTensorData>> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let tensors = SafeTensors::deserialize(&buffer)
        .map_err(|e| Error::other(format!("SafeTensors parse error: {e}")))?;

    let mut result = BTreeMap::new();

    for (name, tensor) in tensors.tensors() {
        let dtype = entry_dtype(&name, &tensor)?;
        result.insert(
            name.clone(),
            TypedTensorData {
                data: tensor.data().to_vec(),
                shape: tensor.shape().to_vec(),
                dtype,
            },
        );
    }

    Ok(result)
}

/// Save a `StateDict` to `SafeTensors` format, each entry in its own dtype.
/// # Errors
/// The file can't be written, or (`InvalidData`, wrapping the [`crate::VoltaError`] from
/// [`TensorData::validate`]) an entry's byte count does not match its shape and dtype.
pub fn save_safetensors<P: AsRef<Path>>(state: &StateDict, path: P) -> Result<()> {
    use safetensors::tensor::TensorView;

    validate_state(state)?;
    let tensors = state
        .iter()
        .map(|(name, td)| {
            TensorView::new(
                volta_dtype_to_safetensors(td.dtype),
                td.shape.clone(),
                &td.data,
            )
            .map(|view| (name.clone(), view))
            .map_err(|e| {
                Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("SafeTensors entry '{name}': {e}"),
                )
            })
        })
        .collect::<Result<Vec<(String, TensorView<'_>)>>>()?;

    let bytes = safetensors::tensor::serialize(tensors, None)
        .map_err(|e| Error::other(format!("SafeTensors serialize error: {e}")))?;

    let mut file = File::create(path)?;
    file.write_all(&bytes)?;
    Ok(())
}

/// Save typed tensor data to `SafeTensors` format with native dtypes
///
/// # Panics
/// If an entry's byte count does not match its shape and dtype (`TensorView::new` fails).
///
/// # Errors
/// If the file can't be created or written, or the tensors can't be serialized.
#[track_caller]
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
        .map_err(|e| Error::other(format!("SafeTensors serialize error: {e}")))?;

    let mut file = File::create(path)?;
    file.write_all(&bytes)?;
    Ok(())
}

/// Load a safetensors file and apply transformations
///
/// This is a convenience wrapper around `load_safetensors` that applies
/// a `StateDictMapper` transformation before returning the result.
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
/// # Errors
/// Same as [`load_safetensors`], and `InvalidData` wrapping the
/// [`VoltaError`](crate::VoltaError) if the mapper fails ([`mapping::StateDictMapper::try_map`]).
pub fn load_safetensors_with_mapping<P: AsRef<Path>>(
    path: P,
    mapper: &mapping::StateDictMapper,
) -> Result<StateDict> {
    let state = load_safetensors(path)?;
    mapper
        .try_map(state)
        .map_err(|e| Error::new(std::io::ErrorKind::InvalidData, e))
}

#[cfg(test)]
mod io_tests {
    use super::*;
    use crate::nn::Module;
    use crate::{Linear, RawTensor, ReLU, Sequential};

    /// A temp-file path unique to this process, removed on drop (also when the test
    /// fails). A fixed name would race with another test process running at once.
    struct TempFile(std::path::PathBuf);

    impl TempFile {
        fn new(name: &str) -> Self {
            Self(std::env::temp_dir().join(format!("volta_{}_{name}", std::process::id())))
        }

        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            // Best effort: the file does not exist if the test failed before saving.
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn test_save_load_sequential() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(2, 3, true)),
            Box::new(ReLU),
            Box::new(Linear::new(3, 1, true)),
        ]);

        let tmp = TempFile::new("test_seq.safetensors");

        // Save
        let state = model.state_dict();
        save_safetensors(&state, tmp.path()).unwrap();

        // Load new model
        let mut model2 = Sequential::new(vec![
            Box::new(Linear::new(2, 3, true)),
            Box::new(ReLU),
            Box::new(Linear::new(3, 1, true)),
        ]);

        // Verify weights are different initially
        let p1 = model.parameters();
        let p2 = model2.parameters();
        assert_ne!(
            p1.first().unwrap().borrow().data,
            p2.first().unwrap().borrow().data
        );

        // Load
        let loaded_state = load_safetensors(tmp.path()).unwrap();
        model2
            .load_state_dict(&loaded_state)
            .expect("saved state loads");

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
            TensorData::from_f32(vec![0.0], vec![1]),
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

        let tmp = TempFile::new("test_safetensors.safetensors");
        let path = tmp.path();

        // Save to SafeTensors
        let state = model.state_dict();
        save_safetensors(&state, path).unwrap();

        // Load from SafeTensors
        let loaded_state = load_safetensors(path).unwrap();

        // Every entry comes back with the same dtype, shape and bytes.
        assert_eq!(state, loaded_state);
    }

    /// A two-layer model converted to `dtype`, so `state_dict()` has entries in it.
    fn model_in(dtype: DType) -> Sequential {
        let mut model = Sequential::new(vec![
            Box::new(Linear::new(3, 4, true)),
            Box::new(ReLU),
            Box::new(Linear::new(4, 2, true)),
        ]);
        model.to_dtype(dtype);
        model
    }

    /// Saves `dtype` model's state dict with `save`, loads it with `load` into a fresh
    /// model of the same dtype, and checks the dtype is kept and the values are bit-exact,
    /// in the loaded dict and in the loaded model's parameters (A8, #53).
    fn assert_round_trip(
        dtype: DType,
        name: &str,
        save: fn(&StateDict, &std::path::Path),
        load: fn(&std::path::Path) -> StateDict,
    ) {
        let model = model_in(dtype);
        let state = model.state_dict();
        assert_eq!(state.len(), 4);
        for (key, td) in &state {
            assert_eq!(td.dtype, dtype, "state_dict() entry {key}");
        }

        let tmp = TempFile::new(name);
        save(&state, tmp.path());
        let loaded = load(tmp.path());
        for (key, td) in &state {
            let back = loaded.get(key).expect("key survives the round trip");
            assert_eq!(back.dtype, dtype, "{key}: dtype");
            assert_eq!(back.shape, td.shape, "{key}: shape");
            assert_eq!(back.data, td.data, "{key}: bytes");
        }
        assert_eq!(loaded.len(), state.len());

        let mut fresh = model_in(dtype);
        fresh
            .load_state_dict(&loaded)
            .expect("same architecture loads");
        let (want, got) = (model.parameters(), fresh.parameters());
        assert_eq!(want.len(), got.len());
        for (w, g) in want.iter().zip(&got) {
            assert_eq!(g.dtype(), dtype);
            assert_eq!(g.borrow().data.as_bytes(), w.borrow().data.as_bytes());
        }
    }

    /// The safetensors saver as `assert_round_trip` takes it.
    fn safetensors_save(state: &StateDict, path: &std::path::Path) {
        save_safetensors(state, path).expect("save");
    }

    /// The safetensors loader as `assert_round_trip` takes it.
    fn safetensors_load(path: &std::path::Path) -> StateDict {
        load_safetensors(path).expect("load")
    }

    #[test]
    fn f64_state_dict_round_trips_through_safetensors() {
        assert_round_trip(
            DType::F64,
            "f64.safetensors",
            safetensors_save,
            safetensors_load,
        );
    }

    #[test]
    fn f16_state_dict_round_trips_through_safetensors() {
        assert_round_trip(
            DType::F16,
            "f16.safetensors",
            safetensors_save,
            safetensors_load,
        );
    }

    #[test]
    fn bf16_state_dict_round_trips_through_safetensors() {
        assert_round_trip(
            DType::BF16,
            "bf16.safetensors",
            safetensors_save,
            safetensors_load,
        );
    }

    /// f64 values that f32 cannot hold survive a save and load unchanged: the entry is not
    /// narrowed on the way (A8).
    #[test]
    fn f64_values_beyond_f32_precision_survive() {
        let vals = vec![0.1f64, 1.0 + 1e-12, -3.0e-300, 1.0e300];
        let t = RawTensor::from_typed(vals.clone(), &[2, 2]);
        let td = TensorData::from_tensor(&t);
        let mut state = StateDict::new();
        state.insert("w".to_string(), td);
        let tmp = TempFile::new("f64_precision.safetensors");
        safetensors_save(&state, tmp.path());
        let back = safetensors_load(tmp.path());
        assert_eq!(back["w"].to_vec_as::<f64>().expect("f64 entry"), vals);
    }

    /// Loading an entry into a parameter of another dtype casts it, explicitly, as
    /// before: an f64 checkpoint loads into an f32 model, rounded to nearest.
    #[test]
    fn loading_into_another_dtype_casts() {
        let src = model_in(DType::F64);
        let mut dst = model_in(DType::F32);
        dst.load_state_dict(&src.state_dict())
            .expect("loads with a cast");
        for (s, d) in src.parameters().iter().zip(&dst.parameters()) {
            assert_eq!(d.dtype(), DType::F32);
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the expected value is the f64 rounded to f32, as the cast does"
            )]
            let want: Vec<f32> = s
                .to_vec_as::<f64>()
                .expect("f64 source")
                .iter()
                .map(|&x| x as f32)
                .collect();
            assert_eq!(d.to_vec_as::<f32>().expect("f32 destination"), want);
        }
    }

    /// A safetensors file whose entry holds fewer bytes than its shape and dtype need is
    /// rejected on load, not returned to panic later. `save_safetensors` refuses to write
    /// such an entry, so the file is built by hand: an 8-byte little-endian header length,
    /// the JSON header, then 31 data bytes for an f64 `[2, 2]` entry that needs 32.
    #[test]
    fn safetensors_load_rejects_a_corrupted_length() {
        let header = br#"{"w":{"dtype":"F64","shape":[2,2],"data_offsets":[0,31]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[0u8; 31]);
        let tmp = TempFile::new("corrupt.safetensors");
        std::fs::write(tmp.path(), &bytes).expect("write");

        let err = load_safetensors(tmp.path()).expect_err("a short entry is an error");
        assert!(err.to_string().contains("SafeTensors parse error"), "{err}");
    }

    /// A program-built entry with the wrong byte count is an `Err` from
    /// `load_state_dict` and from `save_safetensors`, and leaves the parameter alone.
    #[test]
    fn malformed_entry_is_an_error_everywhere() {
        let mut layer = Linear::new(2, 2, false);
        let before = layer.weight.data();
        let mut td = TensorData::from_f32(vec![1.0; 4], vec![2, 2]);
        td.data.pop();
        let mut state = StateDict::new();
        state.insert("weight".to_string(), td);

        let err = layer.load_state_dict(&state).expect_err("short entry");
        assert!(
            matches!(
                err,
                crate::VoltaError::StateDataLength {
                    expected: 16,
                    len: 15,
                    ..
                }
            ),
            "{err:?}"
        );
        assert_eq!(layer.weight.data(), before);

        let tmp = TempFile::new("malformed.safetensors");
        let err = save_safetensors(&state, tmp.path()).expect_err("short entry");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    /// `to_tensor` rebuilds the tensor in the saved dtype, and `try_to_tensor` refuses a
    /// malformed entry instead of panicking.
    #[test]
    fn to_tensor_keeps_the_dtype_and_try_form_rejects_bad_entries() {
        let vals: Vec<half::f16> = [0.5f32, -1.25, 3.0]
            .iter()
            .map(|&x| half::f16::from_f32(x))
            .collect();
        let t = RawTensor::from_typed(vals.clone(), &[3]);
        let back = TensorData::from_tensor(&t).to_tensor(false);
        assert_eq!(back.dtype(), DType::F16);
        assert_eq!(back.to_vec_as::<half::f16>().expect("f16"), vals);

        let mut bad = TensorData::from_f32(vec![1.0; 4], vec![2, 2]);
        bad.data.truncate(12);
        assert!(matches!(
            bad.try_to_tensor(false),
            Err(crate::VoltaError::StateDataLength { .. })
        ));
    }

    /// A bool entry whose byte is not 0 or 1 fails validation.
    #[test]
    fn bool_entry_with_a_bad_byte_is_rejected() {
        let td = TensorData {
            data: vec![0, 1, 2],
            shape: vec![3],
            dtype: DType::Bool,
        };
        assert!(matches!(
            td.validate("mask"),
            Err(crate::VoltaError::InvalidParameter(_))
        ));
    }

    /// `StateDictMapper`'s transpose moves whole elements, so a half entry keeps its dtype
    /// and its values land transposed (square and asymmetric, so a missing transpose
    /// shows).
    #[test]
    fn mapper_transposes_a_bf16_entry() {
        let vals: Vec<half::bf16> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .map(|&x| half::bf16::from_f32(x))
            .collect();
        let mut state = StateDict::new();
        state.insert(
            "fc.weight".to_string(),
            TensorData::from_typed(vals, vec![2, 2]),
        );
        let mapped = mapping::StateDictMapper::new()
            .transpose_linear("fc")
            .map(state);
        let w = &mapped["fc.weight"];
        assert_eq!(w.dtype, DType::BF16);
        let got: Vec<f32> = w
            .to_vec_as::<half::bf16>()
            .expect("bf16 entry")
            .iter()
            .map(|x| x.to_f32())
            .collect();
        assert_eq!(got, vec![1.0, 3.0, 2.0, 4.0]);
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

        let tmp = TempFile::new("test_typed.safetensors");
        let path = tmp.path();

        // Save with native dtypes
        save_safetensors_typed(&tensors, path).unwrap();

        // Load with native dtypes
        let loaded = load_safetensors_raw(path).unwrap();

        // Verify dtypes are preserved
        assert_eq!(loaded.get("f32_tensor").unwrap().dtype, DType::F32);
        assert_eq!(loaded.get("f16_tensor").unwrap().dtype, DType::F16);

        // Verify shapes
        assert_eq!(loaded.get("f32_tensor").unwrap().shape, vec![2, 2]);
        assert_eq!(loaded.get("f16_tensor").unwrap().shape, vec![2, 2]);

        // Verify data conversion
        let f32_loaded = loaded
            .get("f32_tensor")
            .unwrap()
            .to_storage()
            .expect("valid f32 bytes")
            .to_f32_vec();
        assert_eq!(f32_loaded, f32_data);

        let f16_loaded = loaded
            .get("f16_tensor")
            .unwrap()
            .to_storage()
            .expect("valid f16 bytes")
            .to_f32_vec();
        for (a, b) in f16_loaded.iter().zip([1.0f32, 2.0, 3.0, 4.0].iter()) {
            assert!((a - b).abs() < 0.01, "F16 conversion mismatch");
        }
    }
}

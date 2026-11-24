use crate::dtype::DType;
use crate::error::OrPanic;
use crate::nn::Module;
use crate::storage::{Element, Storage};
use crate::tensor::Tensor;
use crate::{Result, VoltaError};
use safetensors::SafeTensors;
use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

/// Key remapping between Volta and external (e.g. PyTorch) state dicts.
pub mod mapping;

/// Parameter name to saved values.
pub type StateDict = BTreeMap<String, TensorData>;

/// Serializable copy of one tensor: its values in their own dtype, and its shape.
///
/// `data` holds the elements' bytes exactly as [`Storage::as_bytes`] gives them
/// (native-endian, which is little-endian on every platform Volta builds for, the same
/// layout safetensors uses), so a save and load is bit-exact for every dtype. Build one with
/// [`TensorData::from_f32`], [`TensorData::from_typed`] or [`TensorData::from_tensor`]; the
/// struct is `#[non_exhaustive]`, so it can gain fields without a breaking change. The
/// fields are public to read and write, so nothing stops a caller from editing an entry
/// until its byte count no longer matches `shape` and `dtype`; [`TensorData::to_storage`]
/// checks that before decoding, and the loaders check every entry they read.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
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
#[non_exhaustive]
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

    /// Checks that `shape` is one a tensor can have (no zero dim, at most
    /// [`crate::shape::MAX_NUMEL`] elements), that `data` holds exactly the bytes `shape`
    /// and `dtype` need, and that every `Bool` byte is 0 or 1. Allocates nothing. `key`
    /// names the entry in the error.
    ///
    /// # Errors
    /// `VoltaError::InvalidParameter` if `shape` has a zero dim or more than `MAX_NUMEL`
    /// elements, `VoltaError::StateDataLength` (counted in bytes; `expected` is
    /// `usize::MAX` when the shape's size overflows) if the byte count is wrong, and
    /// `VoltaError::InvalidParameter` if a `Bool` byte is neither 0 nor 1.
    pub fn validate(&self, key: &str) -> crate::Result<()> {
        validate_entry(key, &self.data, &self.shape, self.dtype)
    }

    /// Decodes `data` as `dtype`, after [`Self::validate`]. `key` names the entry in the
    /// error.
    ///
    /// # Errors
    /// Whatever [`Self::validate`] returns; nothing is allocated before it passes.
    pub fn to_storage(&self, key: &str) -> crate::Result<Storage> {
        self.validate(key)?;
        // `try_from_bytes` returns only `InvalidParameter`; name the entry in its message.
        #[expect(
            clippy::wildcard_enum_match_arm,
            reason = "only InvalidParameter is rewrapped; any other error passes through unchanged"
        )]
        Storage::try_from_bytes(&self.data, self.dtype).map_err(|e| match e {
            crate::VoltaError::InvalidParameter(d) => crate::VoltaError::invalid(
                "TensorData::to_storage",
                Some("data"),
                format!("state dict entry '{key}': {}", d.message),
            ),
            other => other,
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
    /// Whatever [`Self::validate`] returns, whatever [`crate::shape::Shape::from_dims`]
    /// returns for `shape` (a zero dim, or too many elements), and
    /// `VoltaError::NotDifferentiable` if `requires_grad` is true for an integer or bool
    /// entry, as [`Tensor::set_requires_grad`] refuses it (#458).
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

/// The check behind [`TensorData::validate`], on borrowed parts.
fn validate_entry(key: &str, data: &[u8], shape: &[usize], dtype: DType) -> crate::Result<()> {
    // The shape first: a zero dim makes numel 0, so `[0, usize::MAX]` with no bytes would
    // pass the byte check below and then hang code that loops over a dim (#459). Volta's
    // tensors have no zero dims (`Shape::from_dims`), so neither do its state dicts.
    if shape.contains(&0) {
        return Err(crate::VoltaError::invalid(
            "TensorData::validate",
            Some("shape"),
            format!("state dict entry '{key}': shape {shape:?} has a zero dimension"),
        ));
    }
    if checked_numel(shape).is_some_and(|n| n > crate::shape::MAX_NUMEL) {
        return Err(crate::VoltaError::invalid(
            "TensorData::validate",
            Some("shape"),
            format!(
                "state dict entry '{key}': shape {shape:?} has more than {} elements",
                crate::shape::MAX_NUMEL
            ),
        ));
    }
    let expected = checked_numel(shape)
        .and_then(|n| n.checked_mul(dtype.size_of()))
        .unwrap_or(usize::MAX);
    if data.len() != expected {
        return Err(crate::VoltaError::StateDataLength {
            key: key.to_string(),
            expected,
            len: data.len(),
        });
    }
    if dtype == DType::Bool
        && let Some((i, b)) = data.iter().enumerate().find(|&(_, &b)| b > 1)
    {
        return Err(crate::VoltaError::invalid(
            "TensorData::validate",
            Some("data"),
            format!("state dict entry '{key}': bool element {i} is the byte {b}, not 0 or 1"),
        ));
    }
    Ok(())
}

/// Checks every entry of a state dict read from outside the program.
///
/// # Errors
/// The [`VoltaError`] from [`TensorData::validate`] for the first malformed entry; it
/// names the entry's key.
fn validate_state(state: &StateDict) -> Result<()> {
    for (key, td) in state {
        td.validate(key)?;
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
/// a lenient load (as `module.load_state_dict_partial`). It is safe to call even when some keys are missing,
/// since we inspect the diff up front. Entries whose shape mismatches are reported
/// in the diff and *not* loaded, so a wrong shape is not an error here.
///
/// # Errors
///
/// `VoltaError::StateDataLength` if an entry whose shape matches holds the wrong
/// number of values for it. Shape mismatches never reach the load; they are in the
/// returned diff. As with [`Module::load_state_dict`], every entry that would load is
/// checked before any is written, so on `Err` the module is unchanged.
pub fn load_state_dict_checked<M: Module + ?Sized>(
    module: &mut M,
    state: &StateDict,
) -> crate::Result<StateDictDiff> {
    let expected = module.state_dict();
    let diff = diff_state_dict(&expected, state);
    let mut plan = crate::nn::LoadPlan::of(&*module, state);
    plan.skip(|key| diff.shape_mismatches.iter().any(|(k, _, _)| k == key));
    plan.apply()?;
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
/// [`VoltaError::Decode`] naming `path` and the entry if the dtype has no Volta
/// equivalent. Reading it as f32 bytes instead gave plausible wrong values (S18).
fn entry_dtype(
    path: &Path,
    name: &str,
    tensor: &safetensors::tensor::TensorView<'_>,
) -> Result<DType> {
    safetensors_dtype_to_volta(tensor.dtype()).ok_or_else(|| {
        VoltaError::decode(
            "load_safetensors",
            path,
            Some(name),
            format!(
                "dtype {:?} has no Volta equivalent (supported: F16, BF16, F32, F64, I32, \
                 I64, U8, BOOL)",
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
/// - [`VoltaError::Io`] if the file can't be opened or read.
/// - [`VoltaError::Decode`], carrying the path, if it is not a valid safetensors file, or (with
///   the entry's key) an entry has a dtype Volta does not support (I8, U16, U32, U64, the
///   f8 formats, ...).
/// - The [`TensorData::validate`] error, naming the key, for an entry whose bytes don't
///   match its shape and dtype.
pub fn load_safetensors<P: AsRef<Path>>(path: P) -> Result<StateDict> {
    let path = path.as_ref();
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let tensors = SafeTensors::deserialize(&buffer).map_err(|e| {
        VoltaError::decode(
            "load_safetensors",
            path,
            None,
            format!("not a valid safetensors file: {e}"),
        )
    })?;

    let mut state_dict = StateDict::new();

    for (name, tensor) in tensors.tensors() {
        let entry = TensorData {
            data: tensor.data().to_vec(),
            shape: tensor.shape().to_vec(),
            dtype: entry_dtype(path, &name, &tensor)?,
        };
        state_dict.insert(name, entry);
    }
    // safetensors has checked each byte length against shape and dtype; this repeats it
    // (and checks bool bytes) so a mismatch is an error here too, not a later panic.
    validate_state(&state_dict)?;

    Ok(state_dict)
}

/// Save a `StateDict` to `SafeTensors` format, each entry in its own dtype.
///
/// The file is written to a temporary file next to `path`, synced, and renamed over
/// `path`, so a failed or interrupted save leaves an existing file at `path` intact.
/// # Errors
/// - The [`TensorData::validate`] error, naming the key, if an entry's byte count does not
///   match its shape and dtype; nothing is written.
/// - [`VoltaError::InvalidParameter`] if `path` does not name a file.
/// - [`VoltaError::Io`] if the file can't be written.
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
            // `validate_state` checked the byte count, so this is a mismatch between the
            // two libraries' rules, not a bad entry the caller can fix.
            .map_err(|e| {
                VoltaError::OpError(crate::error::ErrorDetail::new(
                    "save_safetensors",
                    "state",
                    format!("entry '{name}': {e}"),
                ))
            })
        })
        .collect::<Result<Vec<(String, TensorView<'_>)>>>()?;

    let bytes = safetensors::tensor::serialize(tensors, None).map_err(|e| {
        VoltaError::OpError(crate::error::ErrorDetail::new(
            "save_safetensors",
            None,
            format!("serialize: {e}"),
        ))
    })?;

    let path = path.as_ref();
    if path.file_name().is_none() {
        return Err(VoltaError::invalid(
            "save_safetensors",
            "path",
            format!("{} does not name a file", path.display()),
        ));
    }
    Ok(write_atomically(path, |file| file.write_all(&bytes))?)
}

/// Distinguishes temporary files made by concurrent saves in one process.
static TEMP_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Write `path` through a temporary file in the same directory: `write` fills it, it is
/// flushed to disk with `sync_all`, then renamed over `path`. A failed write, a full disk or
/// a crash before the rename leaves any existing file at `path` as it was (#395). On an
/// error the temporary file is removed.
fn write_atomically(
    path: &Path,
    write: impl FnOnce(&mut File) -> std::io::Result<()>,
) -> std::io::Result<()> {
    let file_name = path.file_name().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("{} does not name a file", path.display()),
        )
    })?;
    let dir = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    let mut tmp_name = std::ffi::OsString::from(".");
    tmp_name.push(file_name);
    tmp_name.push(format!(
        ".{}.{}.tmp",
        std::process::id(),
        TEMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let tmp = dir.join(tmp_name);

    // `create_new` refuses to reuse a path some other writer already holds. The file is
    // closed at the end of the block, before the rename.
    let written = {
        let mut file = File::options().write(true).create_new(true).open(&tmp)?;
        write(&mut file).and_then(|()| file.sync_all())
    };
    let result = written.and_then(|()| std::fs::rename(&tmp, path));
    if let Err(e) = result {
        if let Err(cleanup) = std::fs::remove_file(&tmp) {
            log::warn!(
                "could not remove temporary file {}: {cleanup}",
                tmp.display()
            );
        }
        return Err(e);
    }
    // Persist the rename itself. Directories can't be opened as files on Windows, and Miri
    // supports `fsync` only on regular files (#425), so the sanitizer job skips this step.
    #[cfg(all(unix, not(miri)))]
    File::open(dir)?.sync_all()?;
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
/// # Ok::<(), volta::VoltaError>(())
/// ```
/// # Errors
/// Same as [`load_safetensors`], and the mapper's error
/// ([`mapping::StateDictMapper::try_map`]).
pub fn load_safetensors_with_mapping<P: AsRef<Path>>(
    path: P,
    mapper: &mapping::StateDictMapper,
) -> Result<StateDict> {
    let state = load_safetensors(path)?;
    mapper.try_map(state)
}

#[cfg(test)]
mod io_tests {
    use super::*;
    use crate::RawTensor;
    use crate::nn::Module;
    use crate::nn::{Linear, ReLU, Sequential};

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
        assert!(
            matches!(
                &err,
                crate::VoltaError::Decode { path, key: None, detail }
                    if path == tmp.path() && detail.op == "load_safetensors"
            ),
            "{err:?}"
        );
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
        assert!(
            matches!(&err, crate::VoltaError::StateDataLength { key, .. } if key == "weight"),
            "{err:?}"
        );
    }

    /// #110: `save_safetensors` returns `StateDataLength` for an entry whose byte count
    /// does not match its shape (it used to panic in `TensorView::new`), and creates no file.
    #[test]
    fn save_safetensors_rejects_a_length_mismatch_and_writes_nothing() {
        let mut entry = TensorData::from_f32(vec![0.0; 4], vec![2, 2]);
        entry.data.truncate(12);
        let mut state = StateDict::new();
        state.insert("w".to_string(), entry);
        let tmp = TempFile::new("mismatch.safetensors");
        let err = save_safetensors(&state, tmp.path()).expect_err("12 bytes for [2, 2]");
        assert!(
            matches!(&err, crate::VoltaError::StateDataLength { key, expected: 16, len: 12 } if key == "w"),
            "{err:?}"
        );
        assert!(!tmp.path().exists(), "no file is written for a bad entry");
    }

    /// #395: a save that fails mid-write leaves the previous file byte-for-byte intact
    /// and removes its temporary file.
    #[test]
    fn failed_write_leaves_the_existing_file_intact() {
        let dir = std::env::temp_dir().join(format!("volta_{}_atomic_fail", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let target = dir.join("model.safetensors");
        let mut state = StateDict::new();
        state.insert(
            "w".to_string(),
            TensorData::from_f32(vec![1.0, 2.0], vec![2]),
        );
        save_safetensors(&state, &target).expect("first save");
        let before = std::fs::read(&target).expect("read");

        let err = write_atomically(&target, |file| {
            file.write_all(b"partial")?;
            Err(std::io::Error::other("injected write failure"))
        })
        .expect_err("injected failure");
        assert_eq!(err.to_string(), "injected write failure");

        assert_eq!(std::fs::read(&target).expect("read"), before);
        let names: Vec<_> = std::fs::read_dir(&dir)
            .expect("list")
            .map(|e| e.expect("entry").file_name())
            .collect();
        assert_eq!(names, vec![std::ffi::OsString::from("model.safetensors")]);
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    /// #395: a successful save replaces an existing file, leaves no temporary file, and
    /// loads back as the new state.
    #[test]
    fn save_replaces_an_existing_file() {
        let dir = std::env::temp_dir().join(format!("volta_{}_atomic_ok", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let target = dir.join("model.safetensors");
        let mut old = StateDict::new();
        old.insert(
            "w".to_string(),
            TensorData::from_f32(vec![1.0, 2.0], vec![2]),
        );
        save_safetensors(&old, &target).expect("first save");
        let mut new = StateDict::new();
        new.insert(
            "v".to_string(),
            TensorData::from_f32(vec![3.0, 4.0, 5.0], vec![3]),
        );
        save_safetensors(&new, &target).expect("second save");

        let loaded = load_safetensors(&target).expect("load");
        assert_eq!(loaded.keys().collect::<Vec<_>>(), vec!["v"]);
        assert_eq!(
            loaded["v"].to_vec_as::<f32>().expect("f32"),
            vec![3.0, 4.0, 5.0]
        );
        assert_eq!(std::fs::read_dir(&dir).expect("list").count(), 1);
        std::fs::remove_dir_all(&dir).expect("cleanup");
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

    /// #459: a zero dim or more than `MAX_NUMEL` elements fails validation before the
    /// byte count is looked at, so an empty `[0, usize::MAX]` entry is refused; saving
    /// one is refused too.
    #[test]
    fn validate_rejects_zero_dims_and_oversized_shapes() {
        for shape in [vec![0], vec![2, 0], vec![0, usize::MAX]] {
            let td = TensorData::from_typed(Vec::<f32>::new(), shape.clone());
            let err = td.validate("w").expect_err("zero dim");
            assert!(
                err.to_string().contains("zero dimension"),
                "{shape:?}: {err}"
            );
            assert!(err.to_string().contains("'w'"), "{err}");
        }
        let big = TensorData {
            data: vec![],
            shape: vec![crate::shape::MAX_NUMEL + 1],
            dtype: DType::U8,
        };
        let err = big.validate("w").expect_err("oversized");
        assert!(err.to_string().contains("more than"), "{err}");
        let state = StateDict::from([(
            "w".to_string(),
            TensorData::from_typed(Vec::<f32>::new(), vec![0, 4]),
        )]);
        let dir = std::env::temp_dir().join(format!("volta-459-{}", std::process::id()));
        assert!(save_safetensors(&state, &dir).is_err());
        assert!(!dir.exists(), "nothing written");
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

    /// Each entry keeps its own dtype through a save and load, bit for bit.
    #[test]
    fn safetensors_round_trip_keeps_f32_and_f16() {
        let f32_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let f16_data: Vec<half::f16> = [1.0f32, -2.5, 3.0, 65504.0]
            .iter()
            .map(|&x| half::f16::from_f32(x))
            .collect();
        let mut state = StateDict::new();
        state.insert(
            "f32_tensor".to_string(),
            TensorData::from_f32(f32_data.clone(), vec![2, 2]),
        );
        state.insert(
            "f16_tensor".to_string(),
            TensorData::from_typed(f16_data.clone(), vec![2, 2]),
        );

        let tmp = TempFile::new("round_trip_dtypes.safetensors");
        save_safetensors(&state, tmp.path()).unwrap();
        let loaded = load_safetensors(tmp.path()).unwrap();

        assert_eq!(loaded, state);
        let f32_entry = loaded.get("f32_tensor").unwrap();
        assert_eq!(
            (f32_entry.dtype, f32_entry.shape.clone()),
            (DType::F32, vec![2, 2])
        );
        assert_eq!(f32_entry.to_vec_as::<f32>().unwrap(), f32_data);
        let f16_entry = loaded.get("f16_tensor").unwrap();
        assert_eq!(
            (f16_entry.dtype, f16_entry.shape.clone()),
            (DType::F16, vec![2, 2])
        );
        assert_eq!(f16_entry.to_vec_as::<half::f16>().unwrap(), f16_data);
    }
}

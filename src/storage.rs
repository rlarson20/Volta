//! Tensor storage: a typed host buffer, or an f32 GPU buffer.
//!
//! CPU data is a [`HostBuf`]: one `Arc<[T]>` per dtype, so alignment and element type are
//! guaranteed by the type rather than by a tag next to a byte buffer. Kernels read their
//! inputs through `host::<T>()` for `T` = f32 or f64 (`crate::real`), which panics on any
//! other dtype rather than converting it. GPU storage is f32-only until ADDENDUM §11 P8.

use crate::VoltaError;
use crate::device::Device;
use crate::dtype::DType;
#[cfg(feature = "gpu")]
use crate::gpu::{GpuBuffer, is_gpu_available};
use bytemuck::cast_slice;
use half::{bf16, f16};
use std::borrow::Cow;
use std::sync::Arc;

#[cfg(feature = "gpu")]
use std::sync::OnceLock;

/// CPU element data, one variant per [`DType`].
///
/// The buffer is shared: `clone` bumps a refcount instead of copying. The only write path
/// is [`Storage::as_host_mut`], which copies first if the buffer is shared
/// (copy-on-write), so a write to one storage is never visible through another.
#[non_exhaustive]
#[derive(Clone)]
pub enum HostBuf {
    /// IEEE half precision.
    F16(Arc<[f16]>),
    /// bfloat16.
    BF16(Arc<[bf16]>),
    /// IEEE single precision.
    F32(Arc<[f32]>),
    /// IEEE double precision.
    F64(Arc<[f64]>),
    /// 32-bit signed integers.
    I32(Arc<[i32]>),
    /// 64-bit signed integers.
    I64(Arc<[i64]>),
    /// 8-bit unsigned integers.
    U8(Arc<[u8]>),
    /// Booleans.
    Bool(Arc<[bool]>),
}

impl HostBuf {
    /// The dtype of the elements.
    #[must_use]
    pub const fn dtype(&self) -> DType {
        match self {
            Self::F16(_) => DType::F16,
            Self::BF16(_) => DType::BF16,
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
            Self::U8(_) => DType::U8,
            Self::Bool(_) => DType::Bool,
        }
    }

    /// Number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::F16(d) => d.len(),
            Self::BF16(d) => d.len(),
            Self::F32(d) => d.len(),
            Self::F64(d) => d.len(),
            Self::I32(d) => d.len(),
            Self::I64(d) => d.len(),
            Self::U8(d) => d.len(),
            Self::Bool(d) => d.len(),
        }
    }

    /// True if there are no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The elements' native-endian bytes, borrowed.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::F16(d) => cast_slice(d),
            Self::BF16(d) => cast_slice(d),
            Self::F32(d) => cast_slice(d),
            Self::F64(d) => cast_slice(d),
            Self::I32(d) => cast_slice(d),
            Self::I64(d) => cast_slice(d),
            Self::U8(d) => d,
            Self::Bool(d) => cast_slice(d),
        }
    }
}

/// Sealing for [`Element`]: only the eight dtypes' Rust types implement it.
mod sealed {
    /// Not nameable outside this module.
    pub trait Sealed {}
}

/// A Rust type that is the element type of one [`DType`].
///
/// Sealed: implemented for `f16`, `bf16`, `f32`, `f64`, `i32`, `i64`, `u8`, and `bool`.
pub trait Element: sealed::Sealed + Copy + Send + Sync + 'static {
    /// The dtype this type stores.
    const DTYPE: DType;
    /// Borrow `buf` as `[Self]` if it holds this type.
    fn view(buf: &HostBuf) -> Option<&[Self]>;
    /// Borrow `buf` mutably as `[Self]`, copying it first if it is shared.
    fn view_mut(buf: &mut HostBuf) -> Option<&mut [Self]>;
    /// Wrap a buffer of this type.
    fn wrap(data: Arc<[Self]>) -> HostBuf;
}

/// Implements [`Element`] for one type and its `HostBuf` variant.
macro_rules! element {
    ($t:ty, $variant:ident) => {
        impl sealed::Sealed for $t {}
        impl Element for $t {
            const DTYPE: DType = DType::$variant;
            fn view(buf: &HostBuf) -> Option<&[Self]> {
                match buf {
                    HostBuf::$variant(d) => Some(d),
                    _ => None,
                }
            }
            fn view_mut(buf: &mut HostBuf) -> Option<&mut [Self]> {
                match buf {
                    HostBuf::$variant(d) => Some(Arc::make_mut(d)),
                    _ => None,
                }
            }
            fn wrap(data: Arc<[Self]>) -> HostBuf {
                HostBuf::$variant(data)
            }
        }
    };
}

element!(f16, F16);
element!(bf16, BF16);
element!(f32, F32);
element!(f64, F64);
element!(i32, I32);
element!(i64, I64);
element!(u8, U8);
element!(bool, Bool);

/// Storage backend for tensor data
///
/// This enum allows tensors to store their data either on CPU (a typed [`HostBuf`]) or on
/// GPU (an f32 `GpuBuffer`).
pub enum Storage {
    /// CPU storage: a typed, shared, copy-on-write buffer.
    Cpu(HostBuf),

    /// GPU storage - data lives in GPU memory, always f32.
    ///
    /// Shared like the CPU variant: `clone` bumps a refcount, and the only write path is
    /// `Storage::gpu_buffer_mut`, which copies the buffer first if it is shared. So a
    /// shared buffer never changes, and the host copy cached beside it never goes stale.
    #[cfg(feature = "gpu")]
    Gpu(Arc<GpuStorage>),
}

/// The shared half of GPU [`Storage`]: the device buffer and its lazily read host copy.
///
/// The host copy is a write-once cell. It is filled by the first host read and dropped only
/// by a writer that holds the sole reference, so a reference into it cannot outlive a change.
#[cfg(feature = "gpu")]
pub struct GpuStorage {
    /// The device buffer.
    buffer: GpuBuffer,
    /// Host copy, filled by the first host read.
    host: OnceLock<Vec<f32>>,
}

#[cfg(feature = "gpu")]
impl GpuStorage {
    /// Wraps `buffer` with an empty host copy.
    fn new(buffer: GpuBuffer) -> Self {
        Self {
            buffer,
            host: OnceLock::new(),
        }
    }
}

impl Clone for Storage {
    fn clone(&self) -> Self {
        match self {
            Self::Cpu(buf) => Self::Cpu(buf.clone()),
            #[cfg(feature = "gpu")]
            Self::Gpu(shared) => Self::Gpu(Arc::clone(shared)),
        }
    }
}

/// Decode native-endian `bytes` as `T`s; the copy handles any alignment.
fn pod_from_bytes<T: bytemuck::Pod>(bytes: &[u8]) -> Arc<[T]> {
    Arc::from(bytemuck::pod_collect_to_vec::<u8, T>(bytes))
}

/// Cast exact float values (every source float widens to `f64` without loss) to `target`.
///
/// # Panics
/// On a NaN, infinite, or out-of-range value bound for an integer dtype.
#[track_caller]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "narrowing is the cast: f64 to f32 rounds to nearest, and float to integer \
              truncates after the range check"
)]
fn cast_float(values: impl Iterator<Item = f64>, target: DType) -> HostBuf {
    // Torch rounds f64 to half precision through f32 (two roundings); so does this.
    match target {
        DType::F16 => HostBuf::F16(values.map(|x| f16::from_f32(x as f32)).collect()),
        DType::BF16 => HostBuf::BF16(values.map(|x| bf16::from_f32(x as f32)).collect()),
        DType::F32 => HostBuf::F32(values.map(|x| x as f32).collect()),
        DType::F64 => HostBuf::F64(values.collect()),
        DType::I32 => HostBuf::I32(
            truncate_checked(values, -2_147_483_648.0, 2_147_483_648.0, target)
                .map(|t| t as i32)
                .collect(),
        ),
        DType::I64 => HostBuf::I64(
            truncate_checked(
                values,
                -9_223_372_036_854_775_808.0,
                9.223_372_036_854_776e18,
                target,
            )
            .map(|t| t as i64)
            .collect(),
        ),
        DType::U8 => HostBuf::U8(
            truncate_checked(values, 0.0, 256.0, target)
                .map(|t| t as u8)
                .collect(),
        ),
        DType::Bool => HostBuf::Bool(values.map(|x| x != 0.0).collect()),
    }
}

/// Truncate each value toward zero, panicking unless it lands in `[lo, hi)`. NaN fails
/// both comparisons, so it panics too.
#[track_caller]
fn truncate_checked(
    values: impl Iterator<Item = f64>,
    lo: f64,
    hi: f64,
    target: DType,
) -> impl Iterator<Item = f64> {
    let caller = std::panic::Location::caller();
    values.enumerate().map(move |(i, x)| {
        let t = x.trunc();
        assert!(
            t >= lo && t < hi,
            "cast to {target} at {caller}: element {i} is {x}, outside {target}'s range \
             (PyTorch leaves this undefined)"
        );
        t
    })
}

/// Cast exact integer values (every source integer and bool widens to `i128`) to `target`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "integer narrowing wraps (two's complement), as torch's static_cast does; \
              integer to float rounds to nearest"
)]
fn cast_int(values: impl Iterator<Item = i128>, target: DType) -> HostBuf {
    // Torch converts integers to half precision through f32; so does this.
    match target {
        DType::F16 => HostBuf::F16(values.map(|x| f16::from_f32(x as f32)).collect()),
        DType::BF16 => HostBuf::BF16(values.map(|x| bf16::from_f32(x as f32)).collect()),
        DType::F32 => HostBuf::F32(values.map(|x| x as f32).collect()),
        DType::F64 => HostBuf::F64(values.map(|x| x as f64).collect()),
        DType::I32 => HostBuf::I32(values.map(|x| x as i32).collect()),
        DType::I64 => HostBuf::I64(values.map(|x| x as i64).collect()),
        DType::U8 => HostBuf::U8(values.map(|x| x as u8).collect()),
        DType::Bool => HostBuf::Bool(values.map(|x| x != 0).collect()),
    }
}

impl Storage {
    // ========== Constructors ==========

    /// CPU storage from f32 data.
    #[must_use]
    pub fn cpu(data: Vec<f32>) -> Self {
        Self::from_typed(data)
    }

    /// CPU storage from data of any [`Element`] type; the dtype follows `T`.
    #[must_use]
    pub fn from_typed<T: Element>(data: Vec<T>) -> Self {
        Self::Cpu(T::wrap(Arc::from(data)))
    }

    /// CPU storage decoded from native-endian bytes (panicking form of
    /// [`Self::try_from_bytes`]).
    ///
    /// # Panics
    /// Whenever `try_from_bytes` returns an error.
    #[must_use]
    #[track_caller]
    pub fn from_bytes(bytes: &[u8], dtype: DType) -> Self {
        Self::try_from_bytes(bytes, dtype).unwrap_or_else(|e| panic!("from_bytes: {e}"))
    }

    /// CPU storage decoded from native-endian bytes, as safetensors stores them on every
    /// platform Volta builds for (all little-endian). The bytes are copied, so they need no
    /// alignment.
    ///
    /// # Errors
    ///
    /// `InvalidParameter` if the byte length is not a multiple of the dtype's size, or a
    /// `Bool` byte is neither 0 nor 1.
    pub fn try_from_bytes(bytes: &[u8], dtype: DType) -> Result<Self, VoltaError> {
        if !bytes.len().is_multiple_of(dtype.size_of()) {
            return Err(VoltaError::InvalidParameter(format!(
                "byte length {} is not a multiple of the {dtype} size {}",
                bytes.len(),
                dtype.size_of()
            )));
        }
        let buf = match dtype {
            DType::F16 => HostBuf::F16(pod_from_bytes(bytes)),
            DType::BF16 => HostBuf::BF16(pod_from_bytes(bytes)),
            DType::F32 => HostBuf::F32(pod_from_bytes(bytes)),
            DType::F64 => HostBuf::F64(pod_from_bytes(bytes)),
            DType::I32 => HostBuf::I32(pod_from_bytes(bytes)),
            DType::I64 => HostBuf::I64(pod_from_bytes(bytes)),
            DType::U8 => HostBuf::U8(Arc::from(bytes)),
            DType::Bool => {
                if let Some((i, b)) = bytes.iter().enumerate().find(|&(_, &b)| b > 1) {
                    return Err(VoltaError::InvalidParameter(format!(
                        "bool element {i} is the byte {b}, not 0 or 1"
                    )));
                }
                HostBuf::Bool(bytes.iter().map(|&b| b == 1).collect())
            }
        };
        Ok(Self::Cpu(buf))
    }

    /// Create new GPU storage from f32 data.
    ///
    /// The host copy is read lazily - data is only copied back from GPU
    /// when read via `to_host_f32()` or similar methods. This avoids
    /// duplicating data in CPU memory unnecessarily.
    ///
    /// # Panics
    /// If no GPU is available, or the buffer can't be allocated. It never returns CPU
    /// storage in place of GPU storage.
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn gpu(data: Vec<f32>) -> Self {
        assert!(is_gpu_available(), "Storage::gpu: no GPU is available");
        let buffer = GpuBuffer::from_slice(&data)
            .unwrap_or_else(|| panic!("Storage::gpu: GPU buffer allocation failed"));
        Self::from_gpu_buffer(buffer)
    }

    /// Not-gpu stub.
    ///
    /// # Panics
    /// Always: there is no GPU without the `gpu` feature.
    #[cfg(not(feature = "gpu"))]
    #[must_use]
    pub fn gpu(_data: Vec<f32>) -> Self {
        panic!("Storage::gpu: no GPU is available (built without the `gpu` feature)")
    }

    /// Wrap a freshly computed device buffer as unshared GPU storage with an empty host copy.
    #[cfg(feature = "gpu")]
    #[must_use]
    pub(crate) fn from_gpu_buffer(buffer: GpuBuffer) -> Self {
        Self::Gpu(Arc::new(GpuStorage::new(buffer)))
    }

    // ========== Dtype Access ==========

    /// Get the data type of this storage
    #[must_use]
    pub const fn dtype(&self) -> DType {
        match self {
            Self::Cpu(buf) => buf.dtype(),
            #[cfg(feature = "gpu")]
            Self::Gpu(_) => DType::F32,
        }
    }

    /// Create new zero-filled f32 storage on the specified device
    ///
    /// # Panics
    /// If `device` is a GPU and no GPU is available, or the buffer can't be allocated.
    #[must_use]
    pub fn new_zeros(len: usize, device: &Device) -> Self {
        match device {
            Device::CPU => Self::cpu(vec![0.0; len]),
            Device::GPU(_) => {
                #[cfg(feature = "gpu")]
                {
                    assert!(
                        is_gpu_available(),
                        "new_zeros({device}): no GPU is available"
                    );
                    let buffer = GpuBuffer::zeros(len).unwrap_or_else(|| {
                        panic!("new_zeros({device}): GPU buffer allocation failed")
                    });
                    Self::from_gpu_buffer(buffer)
                }
                #[cfg(not(feature = "gpu"))]
                {
                    panic!(
                        "new_zeros({device}): no GPU is available (built without the `gpu` feature)"
                    )
                }
            }
        }
    }

    // ========== Host Access ==========
    //
    // There is no `Deref`/`Index` on `Storage` (DECISIONS.md, "No `Deref` on `Storage`"):
    // every host read or write is spelled at the call site, so a GPU sync or a
    // GPU-storage panic is visible where it happens.

    /// Borrow CPU data as `[T]`. No transfer: GPU storage is an error, not a sync.
    ///
    /// # Errors
    ///
    /// `DeviceError` if the storage is on the GPU (use [`Self::to_host_f32`]), and
    /// `DTypeMismatch` if the dtype is not `T`'s.
    pub fn as_host<T: Element>(&self) -> Result<&[T], VoltaError> {
        match self {
            Self::Cpu(buf) => T::view(buf).ok_or(VoltaError::DTypeMismatch {
                expected: T::DTYPE,
                actual: buf.dtype(),
            }),
            #[cfg(feature = "gpu")]
            Self::Gpu(_) => Err(VoltaError::DeviceError(
                "as_host on GPU storage; use to_host_f32 for an explicit transfer".to_string(),
            )),
        }
    }

    /// Mutable CPU data as `[T]`, copy-on-write.
    ///
    /// This is the one host write path. If another storage shares the buffer, it is
    /// copied first, so the write never shows through a clone; a unique owner writes in
    /// place.
    ///
    /// # Errors
    ///
    /// `DeviceError` if the storage is on the GPU, and `DTypeMismatch` if the dtype is
    /// not `T`'s.
    pub fn as_host_mut<T: Element>(&mut self) -> Result<&mut [T], VoltaError> {
        match self {
            Self::Cpu(buf) => {
                let actual = buf.dtype();
                T::view_mut(buf).ok_or(VoltaError::DTypeMismatch {
                    expected: T::DTYPE,
                    actual,
                })
            }
            #[cfg(feature = "gpu")]
            Self::Gpu(_) => Err(VoltaError::DeviceError(
                "as_host_mut on GPU storage; GPU storage has no host write path".to_string(),
            )),
        }
    }

    /// Borrow CPU F32 data: [`Self::as_host`] for `f32`.
    ///
    /// # Errors
    ///
    /// As [`Self::as_host`].
    pub fn as_host_f32(&self) -> Result<&[f32], VoltaError> {
        self.as_host::<f32>()
    }

    /// Mutable CPU F32 data: [`Self::as_host_mut`] for `f32`.
    ///
    /// # Errors
    ///
    /// As [`Self::as_host_mut`].
    pub fn as_host_f32_mut(&mut self) -> Result<&mut [f32], VoltaError> {
        self.as_host_mut::<f32>()
    }

    /// Host view of F32 data: the explicit, greppable host transfer every kernel reads
    /// through.
    ///
    /// CPU storage is borrowed with no copy. GPU storage syncs to the host on first read and
    /// borrows the cached copy.
    ///
    /// # Panics
    ///
    /// If the dtype is not F32: another dtype reaching an f32 read is a loud failure, not a
    /// silent conversion. Use [`Self::to_f32_vec`] to decode other dtypes on purpose.
    #[must_use]
    #[track_caller]
    pub fn to_host_f32(&self) -> &[f32] {
        match self {
            Self::Cpu(buf) => f32::view(buf).unwrap_or_else(|| {
                panic!(
                    "{}",
                    VoltaError::DTypeMismatch {
                        expected: DType::F32,
                        actual: buf.dtype(),
                    }
                )
            }),
            #[cfg(feature = "gpu")]
            Self::Gpu(shared) => shared.host.get_or_init(|| shared.buffer.to_vec()),
        }
    }

    /// Host view of a kernel's input as `T` (`f32` or `f64`): [`Self::to_host_f32`] for
    /// `f32` (which syncs GPU storage), the CPU buffer for `f64`. F16 and BF16 storage is
    /// decoded (exactly) into an owned `f32` copy: half types compute in the `f32` kernel
    /// (ADDENDUM §11), and [`Self::from_compute`] rounds the result back.
    ///
    /// # Panics
    /// If the dtype is neither `T`'s nor (for `T = f32`) a half type. Kernels check dtypes
    /// at the op's entry (`real::compute_dtype`), so this is the backstop.
    #[must_use]
    #[track_caller]
    pub(crate) fn host<T: crate::real::Real>(&self) -> Cow<'_, [T]> {
        match (T::DTYPE, self) {
            // Half to f64 to `T = f32` is exact both ways.
            (DType::F32, Self::Cpu(HostBuf::F16(d))) => {
                Cow::Owned(d.iter().map(|x| T::from_f64(x.to_f64())).collect())
            }
            (DType::F32, Self::Cpu(HostBuf::BF16(d))) => {
                Cow::Owned(d.iter().map(|x| T::from_f64(x.to_f64())).collect())
            }
            (DType::F32, _) => Cow::Borrowed(cast_slice(self.to_host_f32())),
            _ => Cow::Borrowed(self.as_host::<T>().unwrap_or_else(|e| panic!("{e}"))),
        }
    }

    /// Mutable host view of this CPU storage as `T`: in place for `T`'s own dtype. F16 and
    /// BF16 storage (for `T = f32`) is decoded into an f32 copy that is rounded back into
    /// this storage when the view drops, so an in-place update (an optimizer step, running
    /// statistics) computes in f32 and rounds once, like every half op.
    ///
    /// # Panics
    /// If the storage is not CPU storage of `T`'s dtype or (for `T = f32`) a half type.
    #[track_caller]
    pub(crate) fn host_mut<T: crate::real::Real>(&mut self) -> HostMut<'_, T> {
        if self.dtype() == T::DTYPE {
            HostMut::Native(self.as_host_mut::<T>().unwrap_or_else(|e| panic!("{e}")))
        } else {
            let wide = self.host::<T>().into_owned();
            HostMut::Wide { wide, dst: self }
        }
    }

    /// A kernel's output computed in `T`, stored as `dtype`: `T`'s own dtype as is, and
    /// F16/BF16 from `f32` rounded to nearest-even (PyTorch's CPU half ops compute in f32
    /// and round the output).
    ///
    /// # Panics
    /// If `dtype` is neither `T`'s nor (for `T = f32`) a half type.
    #[must_use]
    #[track_caller]
    pub(crate) fn from_compute<T: crate::real::Real>(dtype: DType, data: Vec<T>) -> Self {
        let s = Self::from_typed(data);
        match (T::DTYPE, dtype) {
            (a, b) if a == b => s,
            (DType::F32, DType::F16 | DType::BF16) => s.cast(dtype),
            (a, b) => panic!("from_compute: a {a} kernel cannot produce {b}"),
        }
    }

    /// Owned copy of [`Self::to_host_f32`]: the f32 data, for a kernel that needs a `Vec`.
    ///
    /// # Panics
    ///
    /// If the dtype is not F32, like `to_host_f32`. [`Self::to_f32_vec`] decodes any dtype.
    #[must_use]
    #[track_caller]
    pub fn to_vec(&self) -> Vec<f32> {
        self.to_host_f32().to_vec()
    }

    /// The elements' native-endian bytes. GPU storage syncs through the host copy.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Cpu(buf) => buf.as_bytes(),
            #[cfg(feature = "gpu")]
            Self::Gpu(_) => cast_slice(self.to_host_f32()),
        }
    }

    // ========== Conversion ==========

    /// Every element decoded to `f32`, for display, serialization, and comparison. Lossy
    /// for `F64` and for `I32`/`I64` beyond 2^24; `Bool` is 0.0 or 1.0. Not a kernel input:
    /// kernels read [`Self::to_host_f32`], which refuses other dtypes.
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        reason = "decoding to f32 for display narrows by definition"
    )]
    pub fn to_f32_vec(&self) -> Vec<f32> {
        let buf = match self {
            Self::Cpu(buf) => buf,
            #[cfg(feature = "gpu")]
            Self::Gpu(_) => return self.to_host_f32().to_vec(),
        };
        match buf {
            HostBuf::F16(d) => d.iter().map(|x| x.to_f32()).collect(),
            HostBuf::BF16(d) => d.iter().map(|x| x.to_f32()).collect(),
            HostBuf::F32(d) => d.to_vec(),
            HostBuf::F64(d) => d.iter().map(|&x| x as f32).collect(),
            HostBuf::I32(d) => d.iter().map(|&x| x as f32).collect(),
            HostBuf::I64(d) => d.iter().map(|&x| x as f32).collect(),
            HostBuf::U8(d) => d.iter().map(|&x| f32::from(x)).collect(),
            HostBuf::Bool(d) => d.iter().map(|&x| f32::from(u8::from(x))).collect(),
        }
    }

    // ========== Casting ==========

    /// This storage converted to `target`, following PyTorch's CPU `x.to(dtype)`
    /// (`tests/torch_ref_casts.rs` checks every pair bit for bit):
    ///
    /// - float to float rounds to nearest, ties to even. `F64` to `F16`/`BF16` rounds
    ///   through `f32` first, as torch does, so a value just above a half-way point can
    ///   round down.
    /// - integer to `F16`/`BF16` also goes through `f32`; integer to `F32`/`F64` rounds once.
    /// - float to integer truncates toward zero. A NaN, an infinity, or a value outside the
    ///   integer's range panics: torch leaves it undefined, and saturating would be a
    ///   plausible wrong number.
    /// - integer to integer wraps (two's complement), as torch's `static_cast` does.
    /// - anything to `Bool` is `x != 0` (NaN is true); `Bool` to anything is 0 or 1.
    ///
    /// Same dtype returns a clone that shares the buffer.
    ///
    /// # Panics
    ///
    /// On GPU storage (f32-only until ADDENDUM §11 P8), and on a float-to-integer value
    /// out of range, as above.
    #[must_use]
    #[track_caller]
    pub fn cast(&self, target: DType) -> Self {
        if self.dtype() == target {
            return self.clone();
        }
        let buf = match self {
            Self::Cpu(buf) => buf,
            #[cfg(feature = "gpu")]
            Self::Gpu(_) => panic!("cast to {target}: GPU storage is f32-only (ADDENDUM §11 P8)"),
        };
        Self::Cpu(match buf {
            HostBuf::F16(d) => cast_float(d.iter().map(|x| f64::from(x.to_f32())), target),
            HostBuf::BF16(d) => cast_float(d.iter().map(|x| f64::from(x.to_f32())), target),
            HostBuf::F32(d) => cast_float(d.iter().map(|&x| f64::from(x)), target),
            HostBuf::F64(d) => cast_float(d.iter().copied(), target),
            HostBuf::I32(d) => cast_int(d.iter().map(|&x| i128::from(x)), target),
            HostBuf::I64(d) => cast_int(d.iter().map(|&x| i128::from(x)), target),
            HostBuf::U8(d) => cast_int(d.iter().map(|&x| i128::from(x)), target),
            HostBuf::Bool(d) => cast_int(d.iter().map(|&x| i128::from(x)), target),
        })
    }

    // ========== Length and Properties ==========

    /// Get the number of elements
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Cpu(buf) => buf.len(),
            #[cfg(feature = "gpu")]
            Self::Gpu(shared) => shared.buffer.len(),
        }
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if this is GPU storage
    #[must_use]
    pub const fn is_gpu(&self) -> bool {
        match self {
            Self::Cpu(_) => false,
            #[cfg(feature = "gpu")]
            Self::Gpu(_) => true,
        }
    }

    /// Move to a specific device
    ///
    /// # Panics
    /// If `device` is a GPU and no GPU is available (it never returns CPU storage for a
    /// GPU request), or the storage is not F32: GPU storage is f32-only, and converting
    /// on the way would change the dtype silently.
    #[must_use]
    pub fn to_device(&self, device: &Device) -> Self {
        match device {
            Device::CPU => match self {
                // Already host memory: share the buffer.
                Self::Cpu(_) => self.clone(),
                #[cfg(feature = "gpu")]
                Self::Gpu(_) => Self::cpu(self.to_host_f32().to_vec()),
            },
            Device::GPU(_) => {
                #[cfg(feature = "gpu")]
                {
                    assert!(
                        is_gpu_available(),
                        "to_device({device}): no GPU is available"
                    );
                    match self {
                        // Already on the (only) GPU: share the buffer.
                        Self::Gpu(_) => self.clone(),
                        Self::Cpu(buf) => {
                            assert!(
                                buf.dtype() == DType::F32,
                                "to_device({device}): GPU storage is f32-only, got {}",
                                buf.dtype()
                            );
                            Self::gpu(self.to_host_f32().to_vec())
                        }
                    }
                }
                #[cfg(not(feature = "gpu"))]
                {
                    panic!(
                        "to_device({device}): no GPU is available (built without the `gpu` feature)"
                    )
                }
            }
        }
    }

    /// Get the GPU buffer if this is GPU storage
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn gpu_buffer(&self) -> Option<&GpuBuffer> {
        match self {
            Self::Gpu(shared) => Some(&shared.buffer),
            Self::Cpu(_) => None,
        }
    }

    /// The GPU buffer, ready to be written: the one GPU write path.
    ///
    /// If another storage shares the buffer, it is copied on the device first (copy-on-write,
    /// like [`Self::as_host_f32_mut`]), so a write never shows through a clone. The host copy
    /// is dropped, because the buffer is about to change. `None` for CPU storage, or if the
    /// device copy fails.
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_buffer_mut(&mut self) -> Option<&GpuBuffer> {
        let Self::Gpu(shared) = self else {
            return None;
        };
        if Arc::get_mut(shared).is_none() {
            let copy = shared.buffer.copy_region(0, shared.buffer.len())?;
            *shared = Arc::new(GpuStorage::new(copy));
        }
        let unique = Arc::get_mut(shared).expect("storage is unshared after the copy");
        unique.host.take();
        Some(&unique.buffer)
    }

    /// Release the host copy of GPU data, to reduce memory usage.
    ///
    /// The host copy is read again on the next host access. If the buffer is shared, the
    /// copy is shared too and stays: other storages may be reading it, and it is still
    /// correct because a shared buffer never changes. For CPU storage this is a no-op.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut storage = Storage::gpu(vec![1.0, 2.0, 3.0]);
    /// let _ = storage.to_host_f32(); // Reads the host copy
    /// storage.invalidate_cpu_cache(); // Releases it
    /// ```
    #[cfg(feature = "gpu")]
    pub fn invalidate_cpu_cache(&mut self) {
        if let Self::Gpu(shared) = self
            && let Some(unique) = Arc::get_mut(shared)
        {
            unique.host.take();
        }
    }
}

// ========== Comparison ==========

impl PartialEq for Storage {
    fn eq(&self, other: &Self) -> bool {
        if self.dtype() != other.dtype() {
            return false;
        }
        self.as_bytes() == other.as_bytes()
    }
}

impl PartialEq<Vec<f32>> for Storage {
    fn eq(&self, other: &Vec<f32>) -> bool {
        if self.dtype() != DType::F32 {
            return false;
        }
        self.to_host_f32() == other.as_slice()
    }
}

impl PartialEq<Storage> for Vec<f32> {
    fn eq(&self, other: &Storage) -> bool {
        other == self
    }
}

impl std::fmt::Debug for Storage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu(buf) => write!(f, "Storage::Cpu({} elements, {})", buf.len(), buf.dtype()),
            #[cfg(feature = "gpu")]
            Self::Gpu(shared) => {
                write!(f, "Storage::Gpu({} elements, f32)", shared.buffer.len())
            }
        }
    }
}

/// A mutable host view from [`Storage::host_mut`].
pub(crate) enum HostMut<'a, T: crate::real::Real> {
    /// The storage's own elements.
    Native(&'a mut [T]),
    /// A decoded f32 copy of half storage, rounded back into `dst` on drop.
    Wide {
        /// The decoded elements.
        wide: Vec<T>,
        /// The half storage they came from.
        dst: &'a mut Storage,
    },
}

impl<T: crate::real::Real> std::ops::Deref for HostMut<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        match self {
            Self::Native(s) => s,
            Self::Wide { wide, .. } => wide,
        }
    }
}

impl<T: crate::real::Real> std::ops::DerefMut for HostMut<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        match self {
            Self::Native(s) => s,
            Self::Wide { wide, .. } => wide,
        }
    }
}

impl<T: crate::real::Real> Drop for HostMut<'_, T> {
    fn drop(&mut self) {
        if let Self::Wide { wide, dst } = self {
            **dst = Storage::from_compute(dst.dtype(), std::mem::take(wide));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::fetch;

    #[test]
    fn test_storage_f32() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let storage = Storage::cpu(data.clone());

        assert_eq!(storage.dtype(), DType::F32);
        assert_eq!(storage.len(), 4);
        assert_eq!(storage.as_host_f32().expect("cpu f32"), &data[..]);
        let host = storage.as_host_f32().expect("cpu f32");
        assert_eq!(fetch(host, 0), 1.0);
        assert_eq!(fetch(host, 3), 4.0);
    }

    /// Every element type round-trips through `from_typed`, `as_host`, and the bytes.
    #[test]
    fn every_dtype_round_trips() {
        fn check<T: Element + PartialEq + std::fmt::Debug>(data: Vec<T>) {
            let s = Storage::from_typed(data.clone());
            assert_eq!(s.dtype(), T::DTYPE);
            assert_eq!(s.len(), data.len());
            assert_eq!(s.as_host::<T>().expect("own dtype"), &data[..]);
            let back = Storage::try_from_bytes(s.as_bytes(), T::DTYPE).expect("own bytes");
            assert_eq!(back.as_host::<T>().expect("own dtype"), &data[..]);
        }
        check(vec![f16::from_f32(1.5), f16::from_f32(-2.0)]);
        check(vec![bf16::from_f32(1.5), bf16::from_f32(-2.0)]);
        check(vec![1.5f32, -2.0]);
        check(vec![1.5f64, -2.0, 1e300]);
        check(vec![i32::MIN, 0, i32::MAX]);
        check(vec![i64::MIN, 0, i64::MAX]);
        check(vec![0u8, 255]);
        check(vec![true, false, true]);
    }

    /// Bytes at an odd offset decode correctly: the copy does not need alignment.
    #[test]
    fn from_bytes_accepts_unaligned_input() {
        let mut bytes = vec![0u8];
        bytes.extend(1.25f64.to_ne_bytes());
        bytes.extend((-3.0f64).to_ne_bytes());
        let tail = bytes.get(1..).expect("17 bytes");
        let s = Storage::try_from_bytes(tail, DType::F64).expect("16 bytes of f64");
        assert_eq!(s.as_host::<f64>().expect("f64"), &[1.25, -3.0]);
    }

    #[test]
    fn from_bytes_rejects_ragged_length() {
        assert!(matches!(
            Storage::try_from_bytes(&[0, 0, 0], DType::F32),
            Err(VoltaError::InvalidParameter(_))
        ));
    }

    /// A bool byte other than 0 or 1 is an error, not `true`.
    #[test]
    fn from_bytes_rejects_bool_byte_above_one() {
        let err = Storage::try_from_bytes(&[1, 0, 2], DType::Bool).expect_err("byte 2");
        assert!(err.to_string().contains("element 2"), "{err}");
    }

    #[test]
    fn to_f32_vec_decodes_every_dtype() {
        assert_eq!(
            Storage::from_typed(vec![f16::from_f32(1.5)]).to_f32_vec(),
            [1.5]
        );
        assert_eq!(Storage::from_typed(vec![-2.5f64]).to_f32_vec(), [-2.5]);
        assert_eq!(Storage::from_typed(vec![-7i64]).to_f32_vec(), [-7.0]);
        assert_eq!(Storage::from_typed(vec![200u8]).to_f32_vec(), [200.0]);
        assert_eq!(
            Storage::from_typed(vec![true, false]).to_f32_vec(),
            [1.0, 0.0]
        );
    }

    #[test]
    fn cpu_clone_shares_buffer() {
        let a = Storage::cpu(vec![1.0, 2.0, 3.0]);
        let b = a.clone();
        assert_eq!(a.as_bytes().as_ptr(), b.as_bytes().as_ptr());
    }

    #[test]
    fn cpu_write_after_clone_copies() {
        let a = Storage::cpu(vec![1.0, 2.0, 3.0]);
        let mut b = a.clone();
        *crate::index::slot(b.as_host_f32_mut().expect("f32 cpu storage"), 1) = 9.0;
        assert_eq!(a.as_host_f32().expect("cpu f32"), &[1.0, 2.0, 3.0]);
        assert_eq!(b.as_host_f32().expect("cpu f32"), &[1.0, 9.0, 3.0]);
        assert_ne!(a.as_bytes().as_ptr(), b.as_bytes().as_ptr());
    }

    /// Copy-on-write holds for every dtype, not just f32.
    #[test]
    fn non_f32_write_after_clone_copies() {
        let a = Storage::from_typed(vec![1i64, 2]);
        let mut b = a.clone();
        *crate::index::slot(b.as_host_mut::<i64>().expect("i64"), 0) = 9;
        assert_eq!(a.as_host::<i64>().expect("i64"), &[1, 2]);
        assert_eq!(b.as_host::<i64>().expect("i64"), &[9, 2]);
    }

    #[test]
    fn cpu_unique_write_is_in_place() {
        let mut a = Storage::cpu(vec![1.0, 2.0, 3.0]);
        let before = a.as_bytes().as_ptr();
        *crate::index::slot(a.as_host_f32_mut().expect("f32 cpu storage"), 0) = 5.0;
        assert_eq!(a.as_bytes().as_ptr(), before);
        assert_eq!(fetch(a.as_host_f32().expect("cpu f32"), 0), 5.0);
    }

    #[test]
    fn as_host_rejects_other_dtype() {
        let s = Storage::from_typed(vec![f16::from_f32(1.0)]);
        assert!(matches!(
            s.as_host_f32(),
            Err(VoltaError::DTypeMismatch {
                expected: DType::F32,
                actual: DType::F16,
            })
        ));
        assert!(matches!(
            s.as_host::<bf16>(),
            Err(VoltaError::DTypeMismatch {
                expected: DType::BF16,
                actual: DType::F16,
            })
        ));
    }

    #[test]
    fn to_host_f32_borrows_cpu_f32() {
        let s = Storage::cpu(vec![1.0, 2.0]);
        let host = s.to_host_f32();
        assert_eq!(host.as_ptr().cast::<u8>(), s.as_bytes().as_ptr());
    }

    /// The kernel read refuses another dtype instead of converting it (the old behavior).
    #[test]
    #[should_panic(expected = "DType mismatch: expected F32, got F64")]
    fn to_host_f32_refuses_f64() {
        let _ = Storage::from_typed(vec![1.0f64]).to_host_f32();
    }

    #[test]
    fn as_host_f32_mut_rejects_non_f32() {
        let mut s = Storage::from_typed(vec![f16::from_f32(1.0)]);
        assert!(matches!(
            s.as_host_f32_mut(),
            Err(VoltaError::DTypeMismatch { .. })
        ));
    }

    /// GPU storage has no borrowed host view without an explicit transfer.
    /// Skips without an adapter, so it is not verified in CPU-only CI.
    #[cfg(feature = "gpu")]
    #[test]
    fn as_host_f32_rejects_gpu() {
        if !crate::gpu::is_gpu_available() {
            return;
        }
        let mut s = Storage::gpu(vec![1.0, 2.0]);
        assert!(s.is_gpu());
        assert!(matches!(s.as_host_f32(), Err(VoltaError::DeviceError(_))));
        assert!(matches!(
            s.as_host_f32_mut(),
            Err(VoltaError::DeviceError(_))
        ));
        assert_eq!(s.to_host_f32(), &[1.0, 2.0]);
    }

    /// A non-f32 storage is refused at the GPU boundary instead of being upcast.
    /// Skips without an adapter (the no-GPU panic comes first), so it is not verified in
    /// CPU-only CI.
    #[cfg(feature = "gpu")]
    #[test]
    fn to_device_gpu_refuses_non_f32() {
        if !crate::gpu::is_gpu_available() {
            return;
        }
        let gpu = Device::gpu().expect("adapter present");
        let result = std::panic::catch_unwind(|| Storage::from_typed(vec![1.0f64]).to_device(&gpu));
        assert!(result.is_err(), "f64 storage moved to the GPU");
    }
}

#[cfg(all(test, not(feature = "gpu")))]
mod no_gpu_tests {
    use super::*;

    /// Without a GPU, asking for one panics instead of returning CPU storage.
    #[test]
    #[should_panic(expected = "no GPU is available")]
    fn to_device_gpu_without_gpu_panics() {
        let _ = Storage::cpu(vec![1.0]).to_device(&Device::GPU("x".to_string()));
    }
}

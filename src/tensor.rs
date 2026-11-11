use crate::Storage;
use crate::autograd::{CastGradFn, GradFn, ToDeviceGradFn, edge};
use crate::device::Device;
use crate::device_policy::{Pass, on_gap};
use crate::dtype::DType;
use crate::index::{fetch, slot, span};
use crate::ops::binary::BinaryOp;
use crate::ops::compare::CmpOp;
use crate::ops::reduce::ReduceOp;
use crate::ops::ternary::TernaryOp;
use crate::ops::unary::UnaryOp;
use crate::real::{Real, compute_dtype, dispatch_real};
use crate::shape::{Axis, Shape};
use crate::spec::PadValue;
use crate::storage::Element;
use crate::{Result, VoltaError};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use std::cell::RefCell;
use std::rc::Rc;

/// Wrapper struct for a reference-counted, interior-mutable tensor.
///
/// We use `Rc<RefCell<RawTensor>>` to allow multiple references to the same tensor
/// (needed for computation graphs) while still allowing mutation (for gradient accumulation).
///
/// **Note for production**: This is single-threaded only. For multi-threading,
/// replace with `Arc<Mutex<RawTensor>>`.
#[derive(Clone, Debug)]
pub struct Tensor {
    /// The shared node; private so downstream code cannot reach `RawTensor`'s fields.
    inner: Rc<RefCell<RawTensor>>,
}

impl Tensor {
    /// Whether gradients are tracked for this tensor.
    #[must_use]
    pub fn requires_grad(&self) -> bool {
        self.inner.borrow().requires_grad
    }

    /// The element type of this tensor's data.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.inner.borrow().data.dtype()
    }

    /// Crate-internal forwarding methods to the underlying `RefCell<RawTensor>`.
    ///
    /// Before this type existed, `Tensor` *was* `Rc<RefCell<RawTensor>>`, so every
    /// `t.borrow()` / `t.borrow_mut()` / `t.as_ptr()` call in the codebase resolved
    /// through `Rc`'s `Deref` impl straight to `RefCell`'s inherent methods of the
    /// same name. Wrapping `Rc<RefCell<RawTensor>>` in a struct removes that `Deref`
    /// chain, so we re-supply methods with the *same names* here rather than
    /// renaming them — Rust resolves a method call by (name, receiver type), so this
    /// keeps every existing internal call site compiling unchanged instead of
    /// forcing a rename across every op/layer file.
    ///
    /// The opacity §1.1 actually cares about comes from `pub(crate)`, not from a
    /// different method name: internal code (autograd, ops, nn) can still reach
    /// `RawTensor`'s fields this way, but an external crate depending on `volta`
    /// cannot — `inner` and these methods are invisible outside this crate, so
    /// `x.borrow_mut().requires_grad = true` from downstream code is simply gone.
    pub(crate) fn borrow(&self) -> std::cell::Ref<'_, RawTensor> {
        self.inner.borrow()
    }

    /// Mutably borrows the underlying `RawTensor`; see [`Tensor::borrow`].
    pub(crate) fn borrow_mut(&self) -> std::cell::RefMut<'_, RawTensor> {
        self.inner.borrow_mut()
    }

    /// Pointer identity for the graph-traversal visited-set in `RawTensor::run_backward`.
    pub(crate) fn as_ptr(&self) -> *const RawTensor {
        self.inner.as_ptr()
    }

    /// Wrap an already-built `RawTensor` value into a `Tensor` handle.
    ///
    /// For the rare case where code outside this module needs a `Tensor` around a
    /// `RawTensor` it already has in hand (e.g. wrapping a `&RawTensor` gradient
    /// argument so a `Tensor` method can be called on it) — `inner` being private
    /// means a `Tensor { inner: .. }` struct literal only compiles inside this
    /// module, so this is the crate-wide way to do it instead.
    #[must_use]
    pub(crate) fn from_raw(raw: RawTensor) -> Self {
        Self {
            inner: Rc::new(RefCell::new(raw)),
        }
    }

    /// This tensor's shape.
    ///
    /// Read-only — unlike `requires_grad`/`grad_fn`/`parents`, nothing about graph
    /// correctness depends on who can *see* a shape, only on who can overwrite the
    /// fields that encode graph structure. Safe to expose without qualification.
    #[must_use]
    pub fn shape(&self) -> Shape {
        self.inner.borrow().shape.clone()
    }

    /// Overwrite this tensor's values from a state-dict entry (H7).
    ///
    /// Keeps the tensor's shape, device, `requires_grad` and graph metadata: only the
    /// values change. A GPU-resident tensor stays on the GPU.
    ///
    /// # Errors
    ///
    /// `VoltaError::StateShapeMismatch` if `src.shape` differs from this tensor's shape,
    /// `VoltaError::StateDataLength` if `src.data.len()` does not match it, and
    /// `VoltaError::DTypeMismatch` if this tensor is not a float dtype. The f32 values are
    /// cast to this tensor's dtype (exact for f64), as PyTorch's `load_state_dict` does.
    /// `name` is the state-dict key, reported in the error. Nothing is written on error.
    pub(crate) fn load_weight(&self, name: &str, src: &crate::io::TensorData) -> Result<()> {
        let mut t = self.inner.borrow_mut();
        // `TensorData` is f32 and is cast to the destination's dtype below, as PyTorch's
        // `load_state_dict` (`copy_`) does. Only a float destination can be a parameter.
        if !t.data.dtype().is_float() {
            return Err(VoltaError::DTypeMismatch {
                expected: DType::F32,
                actual: t.data.dtype(),
            });
        }
        // `TensorData.shape` is the serialized `Vec`; compare it at the io boundary.
        if src.shape.as_slice() != t.shape.truncate_rank() {
            return Err(VoltaError::StateShapeMismatch {
                key: name.to_string(),
                expected: t.shape.truncate_rank().to_vec(),
                actual: src.shape.clone(),
            });
        }
        let numel = t.shape.numel();
        if src.data.len() != numel {
            return Err(VoltaError::StateDataLength {
                key: name.to_string(),
                expected: numel,
                len: src.data.len(),
            });
        }
        let cpu = Storage::cpu(src.data.clone()).cast(t.data.dtype());
        t.data = cpu.to_device(&t.device);
        Ok(())
    }

    /// This tensor's data, decoded to `f32` regardless of underlying dtype/device.
    #[must_use]
    pub fn data(&self) -> Vec<f32> {
        self.inner.borrow().data.to_f32_vec()
    }

    /// Converts this tensor's data, and its gradient if any, to `dtype` in place, keeping
    /// it in the graph: the parameter cast behind [`crate::nn::Module::to_dtype`]. Values
    /// follow [`Storage::cast`].
    ///
    /// # Panics
    /// If the tensor is on the GPU and `dtype` cannot live there (F64 or an integer type),
    /// or `dtype` is not a float dtype while the tensor requires grad.
    #[track_caller]
    pub(crate) fn cast_in_place(&self, dtype: DType) {
        let mut t = self.inner.borrow_mut();
        assert!(
            !t.requires_grad || dtype.is_float(),
            "to_dtype: {}",
            VoltaError::NotDifferentiable { dtype }
        );
        t.data = t.data.cast(dtype);
        let grad = t.grad.clone();
        drop(t);
        // Through `set_grad`, one of the sanctioned writers of `grad` (invariant 3).
        if let Some(g) = grad {
            self.set_grad(g.cast(dtype));
        }
    }

    /// This tensor's data in its own dtype, without conversion: the lossless counterpart of
    /// [`Self::data`] (which decodes to `f32`). `T` names the dtype, e.g.
    /// `t.to_vec_as::<f64>()`. A GPU tensor is copied to the host.
    ///
    /// # Errors
    ///
    /// `VoltaError::DTypeMismatch` if `T` is not this tensor's dtype. Use
    /// [`Self::to_dtype`] first to convert on purpose.
    pub fn to_vec_as<T: Element>(&self) -> Result<Vec<T>> {
        let t = self.inner.borrow();
        let host = if t.data.is_gpu() {
            t.data.to_device(&Device::CPU)
        } else {
            t.data.clone()
        };
        host.as_host::<T>().map(<[T]>::to_vec)
    }

    /// The value of a single-element tensor, decoded to `f32` regardless of dtype/device.
    ///
    /// Use this to read a loss or any other scalar. It is stricter than
    /// `data().first()`, which would quietly return the first of many elements.
    ///
    /// # Panics
    ///
    /// If the tensor does not hold exactly one element (a `[1, 1]` tensor is fine).
    #[must_use]
    #[track_caller]
    pub fn item(&self) -> f32 {
        match self.data().as_slice() {
            [value] => *value,
            other => panic!(
                "item() needs a single-element tensor, got {} elements (shape {:?})",
                other.len(),
                self.inner.borrow().shape
            ),
        }
    }

    /// The device this tensor's data lives on.
    #[must_use]
    pub fn device(&self) -> Device {
        self.inner.borrow().device.clone()
    }

    /// This tensor's gradient as raw `Storage` (e.g. to check `.is_gpu()`), if any.
    ///
    /// Prefer `Tensor::grad()` (returns `Option<Vec<f32>>`) unless you
    /// specifically need the `Storage` handle rather than decoded values.
    #[must_use]
    pub fn grad_storage(&self) -> Option<Storage> {
        self.inner.borrow().grad.clone()
    }

    /// A constant (no grad) on this tensor's device and in its dtype, for layers that
    /// combine a fixed value with their input. Building it on the CPU, or in f32 next to an
    /// f64 input, would trip the mixed-device or mixed-dtype check. The values are given as
    /// f32 and widened exactly.
    ///
    /// # Panics
    /// If `data.len()` != the product of `shape`.
    pub(crate) fn constant_like(&self, data: Vec<f32>, shape: &[usize]) -> Self {
        let c = RawTensor::new(data, shape, false);
        let dtype = self.dtype();
        let c = if dtype.is_float() && dtype != DType::F32 {
            c.to_dtype(dtype)
        } else {
            c
        };
        c.to_device(self.device())
    }

    /// Set `requires_grad` on a leaf tensor.
    ///
    /// # Panics
    /// Panics if called on a tensor that already has a `grad_fn` attached —
    /// flipping the flag on a non-leaf tensor would contradict the invariant
    /// `edge()` maintains (`requires_grad` is derived from inputs, not hand-set,
    /// on anything that isn't a leaf). This is the same rule PyTorch enforces on
    /// `.requires_grad = ...` for non-leaf tensors, and it's an unconditional
    /// `assert!` rather than `debug_assert!` deliberately: if this method is public,
    /// the check needs to actually hold in release builds too, or it's the same
    /// "invariant as convention, not enforcement" problem this whole wrapper exists
    /// to close.
    ///
    /// Also panics if `on` is true for an integer or bool tensor, which has no gradient
    /// (PyTorch raises here too).
    #[track_caller]
    pub fn set_requires_grad(&self, on: bool) {
        let mut t = self.inner.borrow_mut();
        assert!(
            !(on && t.grad_fn.is_some()),
            "set_requires_grad called on a non-leaf tensor"
        );
        assert!(
            !on || t.data.dtype().is_float(),
            "set_requires_grad: {}",
            VoltaError::NotDifferentiable {
                dtype: t.data.dtype()
            }
        );
        t.requires_grad = on;
    }

    /// Return a new leaf tensor sharing this tensor's data but detached from the graph.
    #[must_use]
    pub fn detach(&self) -> Self {
        let t = self.inner.borrow();
        Self {
            inner: Rc::new(RefCell::new(RawTensor {
                data: t.data.clone(),
                shape: t.shape.clone(),
                grad: None,
                requires_grad: false,
                grad_fn: None,
                parents: vec![],
                device: t.device.clone(),
            })),
        }
    }

    /// Accumulate a gradient contribution from `op`'s backward into this tensor.
    ///
    /// One of the two sanctioned writers of `grad` outside of initialization
    /// — this one adds a contribution in during `backward()`; `set_grad`
    /// overwrites directly for test seeding.
    ///
    /// It is also the backward device check (`crate::device_policy`): a gradient that
    /// `op` computed on the CPU for a GPU tensor is a gap, which panics under
    /// `DevicePolicy::Strict` and moves to the GPU under `PreferGpu`.
    ///
    /// # Panics
    /// On a backward gap under `Strict`, or if `op` produced a GPU gradient for a CPU
    /// tensor (an op bug).
    pub(crate) fn accumulate_grad(&self, g: Storage, op: &GradFn) {
        self.accumulate_named(g, || op.op_name());
    }

    /// [`Tensor::accumulate_grad`] for a contribution that no `GradFn` produced (the
    /// seed of a root that is itself a leaf); `source` names it in errors.
    pub(crate) fn accumulate_named(&self, g: Storage, source: impl Fn() -> String) {
        let g = self.route_grad(g, &source);
        let mut t = self.inner.borrow_mut();
        match t.grad {
            None => t.grad = Some(g),
            Some(ref mut existing) => RawTensor::accumulate_grad(existing, &g),
        }
    }

    /// Checks a gradient contribution for this tensor and puts it on the tensor's device,
    /// without storing it. Backward uses this for the per-pass gradient of an op output,
    /// which is never written to `grad` (only a leaf keeps one).
    ///
    /// # Panics
    /// As [`Tensor::accumulate_grad`]: a dtype mismatch, a backward gap under `Strict`, or a
    /// GPU gradient for a CPU tensor.
    pub(crate) fn route_grad(&self, g: Storage, source: &impl Fn() -> String) -> Storage {
        let t = self.inner.borrow();
        assert!(
            g.dtype() == t.data.dtype(),
            "{}: produced a {} gradient for a {} tensor (a gradient has its tensor's dtype)",
            source(),
            g.dtype(),
            t.data.dtype()
        );
        match (t.device.is_gpu(), g.is_gpu()) {
            (true, false) => {
                on_gap(&source(), Pass::Backward);
                g.to_device(&t.device)
            }
            (false, true) => panic!("{}: produced a GPU gradient for a CPU tensor", source()),
            (true, true) | (false, false) => g,
        }
    }

    /// Directly set this tensor's gradient, overwriting any existing value.
    ///
    /// This is the second sanctioned writer of `grad` outside initialization —
    /// `accumulate_grad` (used by `backward()`) adds a contribution in; this one
    /// replaces it outright. Intended for tests that need to seed a specific
    /// gradient without running a full forward+backward pass (e.g. verifying
    /// optimizer arithmetic in isolation). Moves `grad` onto this tensor's device
    /// first, so a gradient never sits on another device than its tensor. This is an
    /// explicit transfer, so the device policy doesn't apply (unlike `accumulate_grad`,
    /// where a CPU gradient for a GPU tensor is a gap).
    ///
    /// # Panics
    /// If `grad`'s dtype differs from the tensor's: a gradient has its tensor's dtype.
    #[track_caller]
    #[expect(
        clippy::needless_pass_by_value,
        reason = "takes ownership by design: the caller hands over the buffer it just built, and borrowing would make every caller clone"
    )]
    pub fn set_grad(&self, grad: Storage) {
        let mut t = self.inner.borrow_mut();
        assert!(
            grad.dtype() == t.data.dtype(),
            "set_grad: {}",
            VoltaError::DTypeMismatch {
                expected: t.data.dtype(),
                actual: grad.dtype(),
            }
        );
        let device = t.device.clone();
        t.grad = Some(grad.to_device(&device));
    }

    // pub(crate) fn snapshot(&self) -> TensorSnapshot<'static> { /* owned clone of fields */ }
    // Deferred: no consumer yet and `TensorSnapshot` doesn't exist. Revisit alongside
    // the `load_state_dict` rewrite described in ADDENDUM.md §1.1.
}

// ===== RAW TENSOR STRUCTURE =====

/// The core tensor structure containing data and gradient tracking
///
/// This is wrapped in `Rc<RefCell<>>` to create the public `Tensor` type.
/// Fields:
/// - `data`: CPU: flat `Vec<f32>` of actual values (row-major order), GPU: `Arc<GpuBuffer>`
/// - `shape`: dimensions, e.g. [batch, channels, height, width]
/// - `grad`: accumulated gradient (Some if `requires_grad`, None otherwise)
/// - `requires_grad`: whether to track gradients for this tensor
/// - `grad_fn`: function to compute parent gradients during backward
/// - `parents`: input tensors that this tensor depends on
/// - `device`: where computation happens (CPU/GPU)
pub struct RawTensor {
    /// Element storage (CPU bytes or a GPU buffer).
    pub data: Storage,
    /// Dimensions, e.g. `[B, C, H, W]`.
    pub shape: Shape,
    /// Gradient with respect to `data`; `None` until one is accumulated.
    pub grad: Option<Storage>,
    /// Whether gradients are tracked for this tensor.
    pub requires_grad: bool,
    /// Backward function, present when this tensor is the result of an op.
    pub(crate) grad_fn: Option<GradFn>,
    /// Inputs this tensor was computed from (the graph's edges).
    pub parents: Vec<Tensor>,
    /// Where `data` lives.
    pub device: Device,
}

// ===== RNG HELPERS =====

thread_local! {
    static GLOBAL_RNG: RefCell<Option<StdRng>> = const { RefCell::new(None) };
}

/// Set the seed for the thread-local random number generator
pub fn manual_seed(seed: u64) {
    GLOBAL_RNG.with(|rng| {
        *rng.borrow_mut() = Some(StdRng::seed_from_u64(seed));
    });
}

/// Run a closure with the current RNG: the seeded `StdRng` after [`manual_seed`], else
/// the thread's `ThreadRng` (not reproducible).
///
/// # Panics
///
/// If `f` itself calls `with_rng`. The inner call could not reach the seeded RNG, and
/// falling back to the thread RNG would make a seeded run silently irreproducible.
pub(crate) fn with_rng<F, T>(f: F) -> T
where
    F: FnOnce(&mut dyn rand::RngCore) -> T,
{
    GLOBAL_RNG.with(|rng_cell| {
        let mut borrow = rng_cell
            .try_borrow_mut()
            .expect("with_rng called reentrantly: the closure must not draw from the global RNG through another with_rng call");
        match borrow.as_mut() {
            Some(rng) => f(rng),
            None => f(&mut rand::rng()),
        }
    })
}

impl Clone for RawTensor {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            shape: self.shape.clone(),
            grad: self.grad.clone(),
            requires_grad: self.requires_grad,
            grad_fn: self.grad_fn.clone(),
            parents: self.parents.clone(),
            device: self.device.clone(),
        }
    }
}

impl std::fmt::Debug for RawTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("requires_grad", &self.requires_grad)
            .field("has_grad", &self.grad.is_some())
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

// ===== TENSOR CONSTRUCTORS =====
impl RawTensor {
    /// Create a new tensor from data and shape (fallible)
    ///
    /// # Errors
    ///
    /// Returns `VoltaError::InvalidParameter` if `shape` has a zero dimension or
    /// more than `MAX_NUMEL` elements (see [`Shape::from_dims`])
    /// Returns `VoltaError::ShapeDataMismatch` if data length doesn't match shape
    pub fn try_new(data: Vec<f32>, shape: &[usize], requires_grad: bool) -> Result<Tensor> {
        Self::try_new_typed(data, shape, requires_grad)
    }

    /// A CPU leaf tensor of any [`Element`] type; its dtype follows `T`.
    ///
    /// The one leaf constructor: `try_new` and `try_from_typed` wrap it.
    ///
    /// # Errors
    ///
    /// As [`Self::try_new`], plus `VoltaError::NotDifferentiable` if `requires_grad` is
    /// true for an integer or bool `T`.
    pub(crate) fn try_new_typed<T: Element>(
        data: Vec<T>,
        shape: &[usize],
        requires_grad: bool,
    ) -> Result<Tensor> {
        if requires_grad && !T::DTYPE.is_float() {
            return Err(VoltaError::NotDifferentiable { dtype: T::DTYPE });
        }
        let shape = Shape::from_dims(shape)?;
        if data.len() != shape.numel() {
            return Err(VoltaError::ShapeDataMismatch {
                shape: shape.truncate_rank().to_vec(),
                elements: shape.numel(),
                len: data.len(),
            });
        }

        let raw = Self {
            data: Storage::from_typed(data),
            shape,
            grad: None,
            requires_grad,
            grad_fn: None,
            parents: vec![],
            device: Device::CPU,
        };
        Ok(Tensor {
            inner: Rc::new(RefCell::new(raw)),
        })
    }

    /// Create a new tensor from data and shape
    ///
    /// # Arguments
    /// * `data` - Flat vector of values (length must equal product of shape dimensions)
    /// * `shape` - Dimensions of the tensor
    /// * `requires_grad` - Whether to track gradients for backpropagation
    ///
    /// # Panics
    /// If `shape` is invalid or `data.len()` != `shape.product()`. Use [`Self::try_new`]
    /// to get an error instead.
    #[allow(
        clippy::new_ret_no_self,
        reason = "Public wrapper around private struct"
    )]
    #[must_use]
    #[track_caller]
    pub fn new(data: Vec<f32>, shape: &[usize], requires_grad: bool) -> Tensor {
        Self::try_new(data, shape, requires_grad).unwrap_or_else(|e| panic!("new: {e}"))
    }

    /// Create a new tensor from Storage and shape (internal use for GPU ops, fallible)
    #[allow(dead_code)]
    pub(crate) fn try_new_with_storage(
        data: crate::storage::Storage,
        shape: Shape,
        device: Device,
        requires_grad: bool,
    ) -> Result<Tensor> {
        if data.len() != shape.numel() {
            return Err(VoltaError::ShapeDataMismatch {
                shape: shape.truncate_rank().to_vec(),
                elements: shape.numel(),
                len: data.len(),
            });
        }

        let raw = Self {
            data,
            shape,
            grad: None,
            requires_grad,
            grad_fn: None,
            parents: vec![],
            device,
        };
        Ok(Tensor {
            inner: Rc::new(RefCell::new(raw)),
        })
    }

    /// Create a new tensor from Storage and shape (internal use for GPU ops)
    ///
    /// # Panics
    /// If `data.len()` != `shape.numel()`, or if the storage isn't where `device` says
    /// it is (a device label is a promise; see `crate::device_policy`).
    #[allow(dead_code)]
    pub(crate) fn new_with_storage(
        data: crate::storage::Storage,
        shape: Shape,
        device: Device,
        requires_grad: bool,
    ) -> Tensor {
        assert_eq!(
            data.is_gpu(),
            device.is_gpu(),
            "tensor labeled {device} but its storage is on the {}",
            if data.is_gpu() { "GPU" } else { "CPU" }
        );
        Self::try_new_with_storage(data, shape, device, requires_grad)
            .expect("Data length must match shape")
    }

    /// CPU tensor from data and an already-validated `Shape` (op and backward code).
    ///
    /// # Panics
    ///
    /// If `data.len()` != `shape.numel()`.
    #[cfg(test)]
    pub(crate) fn from_shape(data: Vec<f32>, shape: Shape, requires_grad: bool) -> Tensor {
        Self::new_with_storage(Storage::cpu(data), shape, Device::CPU, requires_grad)
    }

    /// CPU gradient (no grad of its own) of `dtype` from a kernel's `T` result and an
    /// already-validated `Shape`: the typed twin of [`Self::from_shape`] for backward code,
    /// rounded to a half dtype like any kernel output (`Storage::from_compute`).
    ///
    /// # Panics
    ///
    /// If `data.len()` != `shape.numel()`, or `dtype` cannot hold a `T` result.
    pub(crate) fn from_shape_compute<T: crate::real::Real>(
        dtype: DType,
        data: Vec<T>,
        shape: Shape,
    ) -> Tensor {
        Self::new_with_storage(
            Storage::from_compute(dtype, data),
            shape,
            Device::CPU,
            false,
        )
    }

    /// Element count of a shape that came from the caller, checked before anything is
    /// allocated (zero dimension, overflow, or more than `MAX_NUMEL` elements).
    pub(crate) fn checked_numel(shape: &[usize]) -> Result<usize> {
        Ok(Shape::from_dims(shape)?.numel())
    }

    /// Create a tensor filled with zeros (fallible)
    ///
    /// # Errors
    ///
    /// `VoltaError::InvalidParameter` if `shape` has a zero dimension or more than
    /// `MAX_NUMEL` elements (see [`Shape::from_dims`])
    pub fn try_zeros(shape: &[usize]) -> Result<Tensor> {
        Self::try_new(vec![0.0; Self::checked_numel(shape)?], shape, false)
    }

    /// Create a tensor filled with zeros
    ///
    /// # Panics
    ///
    /// If `shape` is invalid. Use [`Self::try_zeros`] to get an error instead.
    #[must_use]
    #[track_caller]
    pub fn zeros(shape: &[usize]) -> Tensor {
        Self::try_zeros(shape).unwrap_or_else(|e| panic!("zeros: {e}"))
    }

    /// Create a tensor filled with ones (fallible)
    ///
    /// # Errors
    ///
    /// Same as [`Self::try_zeros`]
    pub fn try_ones(shape: &[usize]) -> Result<Tensor> {
        Self::try_new(vec![1.0; Self::checked_numel(shape)?], shape, false)
    }

    /// Create a tensor filled with ones
    ///
    /// # Panics
    ///
    /// If `shape` is invalid. Use [`Self::try_ones`] to get an error instead.
    #[must_use]
    #[track_caller]
    pub fn ones(shape: &[usize]) -> Tensor {
        Self::try_ones(shape).unwrap_or_else(|e| panic!("ones: {e}"))
    }

    /// Create a tensor with random values uniformly distributed in [0, 1) (fallible)
    ///
    /// # Errors
    ///
    /// Same as [`Self::try_zeros`]
    pub fn try_rand(shape: &[usize]) -> Result<Tensor> {
        let size = Self::checked_numel(shape)?;
        let data: Vec<f32> = with_rng(|rng| (0..size).map(|_| rng.random::<f32>()).collect());
        Self::try_new(data, shape, false)
    }

    /// Create a tensor with random values uniformly distributed in [0, 1)
    ///
    /// # Panics
    ///
    /// If `shape` is invalid. Use [`Self::try_rand`] to get an error instead.
    #[must_use]
    #[track_caller]
    pub fn rand(shape: &[usize]) -> Tensor {
        Self::try_rand(shape).unwrap_or_else(|e| panic!("rand: {e}"))
    }

    /// Create a tensor with values from standard normal distribution N(0, 1) (fallible)
    ///
    /// # Errors
    ///
    /// Same as [`Self::try_zeros`]
    pub fn try_randn(shape: &[usize]) -> Result<Tensor> {
        let size = Self::checked_numel(shape)?;
        let normal = Normal::new(0.0, 1.0).map_err(|e| {
            VoltaError::InvalidParameter(format!("Invalid normal distribution: {e}"))
        })?;
        let data: Vec<f32> = with_rng(|rng| (0..size).map(|_| normal.sample(rng)).collect());

        Self::try_new(data, shape, false)
    }

    /// Create a tensor with values from standard normal distribution N(0, 1)
    ///
    /// # Panics
    ///
    /// If `shape` is invalid. Use [`Self::try_randn`] to get an error instead.
    #[must_use]
    #[track_caller]
    pub fn randn(shape: &[usize]) -> Tensor {
        Self::try_randn(shape).unwrap_or_else(|e| panic!("randn: {e}"))
    }

    /// Create a tensor filled with random values from N(0, 1) with the same shape as the input
    #[must_use]
    pub fn randn_like(tensor: &Tensor) -> Tensor {
        let shape = tensor.borrow().shape.clone();
        Self::randn(shape.truncate_rank())
    }

    /// Xavier uniform initialization (fallible)
    ///
    /// Samples weights uniformly from [-limit, limit] where
    /// limit = sqrt(6 / (`fan_in` + `fan_out`))
    ///
    /// This helps maintain gradient variance across layers. A rank-1 shape `[n]` is
    /// treated as `[n, 1]` (a bias vector).
    ///
    /// # Errors
    ///
    /// `VoltaError::InvalidParameter` if `shape` is not rank 1 or rank 2, or is invalid
    /// as in [`Self::try_zeros`]
    pub fn try_xavier_uniform(shape: &[usize]) -> Result<Tensor> {
        // A rank-1 shape is a bias vector: one fan-out per element. Any other rank has no
        // [in, out] reading, so it fails here instead of later as a data/shape mismatch.
        let (fan_in, fan_out) = match *shape {
            [n] => (n, 1),
            [fan_in, fan_out] => (fan_in, fan_out),
            _ => {
                return Err(VoltaError::InvalidParameter(format!(
                    "xavier_uniform expects a rank-1 or rank-2 shape, got {shape:?}"
                )));
            }
        };
        let size = Self::checked_numel(shape)?;
        let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
        let data: Vec<f32> =
            with_rng(|rng| (0..size).map(|_| rng.random_range(-limit..limit)).collect());

        Self::try_new(data, shape, false)
    }

    /// Xavier uniform initialization
    ///
    /// Samples weights uniformly from [-limit, limit] where
    /// limit = sqrt(6 / (`fan_in` + `fan_out`)). A rank-1 shape `[n]` is treated as
    /// `[n, 1]` (a bias vector).
    ///
    /// # Panics
    ///
    /// If `shape` is not rank 1 or rank 2, or is invalid. Use [`Self::try_xavier_uniform`]
    /// to get an error instead.
    #[must_use]
    #[track_caller]
    pub fn xavier_uniform(shape: &[usize]) -> Tensor {
        Self::try_xavier_uniform(shape).unwrap_or_else(|e| panic!("xavier_uniform: {e}"))
    }

    /// He (Kaiming) normal initialization suited for `ReLU` networks. (fallible)
    ///
    /// Draws samples from `N(0, sqrt(2 / fan_in))` where `fan_in`
    /// is the number of input connections.
    ///
    /// # Errors
    ///
    /// `VoltaError::InvalidParameter` if `shape` is empty, or is invalid as in
    /// [`Self::try_zeros`]
    pub fn try_he_initialization(shape: &[usize]) -> Result<Tensor> {
        if shape.is_empty() {
            return Err(VoltaError::InvalidParameter(
                "He initialization requires at least one dimension".to_string(),
            ));
        }
        // After this, every dimension is non-zero, so `fan_in` below is positive.
        let size = Self::checked_numel(shape)?;

        let fan_in = match shape.len() {
            1 | 2 => fetch(shape, 0), // Linear weights: [in, out]
            _ => span(shape, 1..shape.len()).iter().product(), // Conv weights: [out, in, kH, kW, ...]
        };

        let std_val: f32 = 2.0 / fan_in as f32;
        let std = std_val.sqrt();
        let normal = Normal::new(0.0, std)
            .map_err(|e| VoltaError::InvalidParameter(format!("Invalid He std: {e}")))?;
        let data: Vec<f32> = with_rng(|rng| (0..size).map(|_| normal.sample(rng)).collect());

        Self::try_new(data, shape, false)
    }

    /// He (Kaiming) normal initialization suited for `ReLU` networks.
    ///
    /// Draws samples from `N(0, sqrt(2 / fan_in))` where `fan_in`
    /// is the number of input connections.
    ///
    /// # Panics
    ///
    /// If `shape` is empty or invalid. Use [`Self::try_he_initialization`] to get an
    /// error instead.
    #[must_use]
    #[track_caller]
    pub fn he_initialization(shape: &[usize]) -> Tensor {
        Self::try_he_initialization(shape).unwrap_or_else(|e| panic!("he_initialization: {e}"))
    }
}

impl RawTensor {
    /// Return the GPU device shared by `tensors` if every tensor lives on the same GPU.
    ///
    /// This avoids accidentally invoking GPU kernels when inputs are on mixed devices.
    pub(crate) fn common_gpu_device(tensors: &[&Tensor]) -> Option<Device> {
        let first = tensors.first()?;
        let first_device = first.borrow().device.clone();
        if !first_device.is_gpu() {
            return None;
        }
        for tensor in tensors.iter().skip(1) {
            let device = tensor.borrow().device.clone();
            if device != first_device {
                return None;
            }
        }
        Some(first_device)
    }
}

#[cfg(feature = "gpu")]
impl RawTensor {
    /// Clear any CPU-side cached copy of GPU data
    ///
    /// This releases CPU memory for GPU tensors that have been read back to CPU.
    /// The cache will be repopulated on the next CPU access if needed.
    ///
    /// Useful for reducing memory pressure after GPU operations are complete
    /// and the CPU copy is no longer needed. Call this between benchmark groups
    /// or after training steps to reduce memory pressure.
    ///
    /// For CPU tensors, this is a no-op.
    ///
    /// # Example
    ///
    /// From outside the crate this is reached through
    /// `volta::gpu::invalidate_all_tensor_caches`:
    ///
    /// ```no_run
    /// use volta::{Device, RawTensor, gpu::invalidate_all_tensor_caches};
    ///
    /// let t = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], false)
    ///     .to_device(Device::gpu().expect("a GPU"));
    /// let _ = t.data(); // Populates the CPU cache
    /// invalidate_all_tensor_caches(&[&t]); // Releases the CPU copy
    /// ```
    pub fn invalidate_gpu_cache(&mut self) {
        self.data.invalidate_cpu_cache();
        if let Some(ref mut grad) = self.grad {
            grad.invalidate_cpu_cache();
        }
    }
}

// ===== LOSS FUNCTIONS =====
impl RawTensor {
    /// Mean squared error: `mean((pred - target)^2)`.
    #[must_use]
    pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Tensor {
        // An f32 op under `amp::autocast` (PyTorch's losses): half inputs are cast up.
        let (pred, target) = (&crate::amp::to_f32(pred), &crate::amp::to_f32(target));
        let diff = pred.sub(target);
        let squared = diff.elem_mul(&diff);
        squared.mean()
    }

    /// Cross entropy from raw logits `[N, C]`, averaged over the batch (applies
    /// `log_softmax` along axis 1). Like PyTorch's `cross_entropy`, `targets` is either
    /// class indices, an I64 or I32 tensor `[N]` with values in `0..C`, or class
    /// probabilities (one-hot or soft) `[N, C]` in the logits' dtype.
    ///
    /// # Panics
    /// If integer targets are not `[N]` or hold an index outside `0..C` (there is no
    /// `ignore_index`), or float targets don't match the logits (`elem_mul`'s checks).
    #[must_use]
    #[track_caller]
    pub fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Tensor {
        // An f32 op under `amp::autocast` (PyTorch's losses): half inputs are cast up.
        let (logits, targets) = (&crate::amp::to_f32(logits), &crate::amp::to_f32(targets));
        // Use log_softmax for numerical stability
        let log_probs = Self::log_softmax(logits, Axis(1));
        Self::nll_loss_named(&log_probs, targets, "cross_entropy_loss")
    }

    /// Negative log likelihood from log-probabilities `[N, C]`, averaged over the batch:
    /// [`Self::cross_entropy_loss`] without its `log_softmax`. `targets` is class indices
    /// (I64 or I32, `[N]`) or class probabilities (`[N, C]`), as there.
    ///
    /// # Panics
    /// As [`Self::cross_entropy_loss`].
    #[must_use]
    #[track_caller]
    pub fn nll_loss(log_probs: &Tensor, targets: &Tensor) -> Tensor {
        // An f32 op under `amp::autocast` (PyTorch's losses): half inputs are cast up.
        let (log_probs, targets) = (&crate::amp::to_f32(log_probs), &crate::amp::to_f32(targets));
        Self::nll_loss_named(log_probs, targets, "nll_loss")
    }

    /// `-sum(targets * log_probs, dim=1).mean()` for probability targets, as PyTorch
    /// computes it (so `0 * -inf` is NaN there too). Class-index targets select
    /// `log_probs[i, t_i]` through a one-hot `where` mask instead of multiplying by it, as
    /// PyTorch's gather does: a `-inf` log-probability on another class (a `-inf` logit)
    /// is never touched, so it can't turn the loss into NaN (S8), and the gradient flows
    /// to the picked entries only. `op` names the caller in panics.
    #[track_caller]
    fn nll_loss_named(log_probs: &Tensor, targets: &Tensor, op: &str) -> Tensor {
        let picked = if targets.dtype().is_int() {
            let lp = log_probs.shape();
            let [n, c] = lp.fixed::<2>();
            let ts = targets.shape();
            assert!(
                ts.rank() == 1 && ts.numel() == n,
                "{op}: class-index targets must be [N] = [{n}] for logits [{n}, {c}], got {:?}",
                ts.truncate_rank()
            );
            let classes: Vec<i64> = match targets.dtype() {
                DType::I64 => targets.to_vec_as::<i64>().expect("I64 checked"),
                DType::I32 => targets
                    .to_vec_as::<i32>()
                    .expect("I32 checked")
                    .into_iter()
                    .map(i64::from)
                    .collect(),
                other @ (DType::U8
                | DType::F16
                | DType::BF16
                | DType::F32
                | DType::F64
                | DType::Bool) => {
                    panic!("{op}: class-index targets must be I64 or I32, got {other}")
                }
            };
            let mut one_hot = vec![0.0f32; n * c];
            for (i, &t) in classes.iter().enumerate() {
                let class = usize::try_from(t)
                    .ok()
                    .filter(|&k| k < c)
                    .unwrap_or_else(|| panic!("{op}: target {t} out of range for {c} classes"));
                *slot(&mut one_hot, i * c + class) = 1.0;
            }
            // A no-grad Bool constant on the log-probs' device; the zeros come from the
            // unselected branch, never from a product with a log-probability.
            let mask = Self::new(one_hot, &[n, c], false)
                .to_dtype(DType::Bool)
                .to_device(log_probs.device());
            let zeros = log_probs.constant_like(vec![0.0; n * c], &[n, c]);
            mask.where_op(log_probs, &zeros)
        } else {
            targets.elem_mul(log_probs)
        };
        let sum = Self::sum_dim(&picked, Axis(1), false);
        sum.neg().mean()
    }

    /// KL divergence between N(mu, sigma) and N(0, 1)
    ///
    /// Used in VAE loss: KL(q(z|x) || p(z))
    /// Formula: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    ///
    /// The sum runs over every element, the batch dimension included, so the result
    /// grows with the batch size. Divide by the batch size for a per-sample term, and
    /// match it to the reconstruction term's reduction (`mse_loss` is a mean).
    ///
    /// # Arguments
    /// * `mu` - Mean of learned distribution
    /// * `logvar` - Log variance of learned distribution
    #[must_use]
    pub fn kl_divergence_gaussian(mu: &Tensor, logvar: &Tensor) -> Tensor {
        // An f32 op under `amp::autocast` (PyTorch's losses): half inputs are cast up.
        let (mu, logvar) = (&crate::amp::to_f32(mu), &crate::amp::to_f32(logvar));
        // 1 + logvar
        let mu_shape = mu.borrow().shape.clone();
        let one = mu.constant_like(vec![1.0; mu_shape.numel()], mu_shape.truncate_rank());
        let term1 = one.add(logvar);

        // mu^2
        let mu_sq = mu.elem_mul(mu);

        // exp(logvar)
        let var = logvar.exp();

        // 1 + logvar - mu^2 - exp(logvar)
        let sum_terms = term1.sub(&mu_sq).sub(&var);

        // -0.5 * sum(...)
        let half = mu.constant_like(vec![0.5], &[1]);
        sum_terms.sum().elem_mul(&half).neg()
    }

    /// Binary cross entropy on probabilities, averaged over every element: PyTorch's
    /// `binary_cross_entropy`, `-mean(y * log(p) + (1 - y) * log(1 - p))` with each `log`
    /// floored at -100, so `p = 0` or `p = 1` gives a finite loss. The gradient is
    /// PyTorch's `(p - y) / max(p * (1 - p), 1e-12)` over the element count.
    ///
    /// `pred` holds probabilities (typically a sigmoid's output) and `target` labels or
    /// soft targets in `[0, 1]`. From logits, [`Self::bce_with_logits_loss`] is more
    /// accurate.
    ///
    /// # Panics
    /// If the shapes differ, `target` requires grad while grad is enabled, or a
    /// prediction is outside `[0, 1]` or NaN (all as in PyTorch). The range check reads
    /// `pred` back to the host.
    #[must_use]
    #[track_caller]
    pub fn bce_loss(pred: &Tensor, target: &Tensor) -> Tensor {
        // An f32 op under `amp::autocast` (PyTorch's losses): half inputs are cast up.
        let (pred, target) = (&crate::amp::to_f32(pred), &crate::amp::to_f32(target));
        Self::bce_elementwise(pred, target).mean()
    }

    /// Binary Cross Entropy with logits (numerically stable)
    ///
    /// Combines sigmoid and BCE using the stable form
    /// `max(x, 0) - x*y + ln(1 + exp(-|x|))`, which never exponentiates a positive
    /// value, so large `|x|` neither overflows nor produces NaN gradients.
    /// The gradient is `sigmoid(x) - y` (divided by the element count for the mean).
    ///
    /// # Arguments
    /// * `logits` - Raw network outputs (before sigmoid)
    /// * `target` - Target labels (0 or 1)
    #[must_use]
    pub fn bce_with_logits_loss(logits: &Tensor, target: &Tensor) -> Tensor {
        // An f32 op under `amp::autocast` (PyTorch's losses): half inputs are cast up.
        let (logits, target) = (&crate::amp::to_f32(logits), &crate::amp::to_f32(target));
        let shape = logits.borrow().shape.clone();
        let zeros = logits.constant_like(vec![0.0; shape.numel()], shape.truncate_rank());
        let one = logits.constant_like(vec![1.0; shape.numel()], shape.truncate_rank());

        // max(x, 0); ties at x == 0 route gradient to `logits`, giving relu'(0) = 1.
        let pos_part = logits.max_elem(&zeros);
        // |x| = max(x, -x)
        let abs_logits = logits.max_elem(&logits.neg());
        // ln(1 + exp(-|x|)); exp argument is <= 0, so it cannot overflow.
        let softplus_term = one.add(&abs_logits.neg().exp()).log();
        let xy_term = logits.elem_mul(target);

        pos_part.sub(&xy_term).add(&softplus_term).mean()
    }
}

// ===== SOFTMAX & AXIS REDUCTIONS =====

/// Gradient for `sum_dim`: broadcast ones back to input shape
#[derive(Clone)]
pub(crate) struct SumDimGradFn {
    /// Shape of the tensor that was summed.
    input_shape: Shape,
    /// Axis that was summed.
    dim: usize,
    /// Whether the summed axis was kept with size 1.
    keepdim: bool,
}

impl SumDimGradFn {
    /// Broadcasts `out_grad` back along the summed axis.
    pub(crate) fn backward(
        &self,
        out_grad: &RawTensor,
        _parents: &[Tensor],
    ) -> Vec<Option<Tensor>> {
        dispatch_real!(out_grad.data.dtype(), T => self.backward_typed::<T>(out_grad))
    }

    /// [`Self::backward`] in the gradient's dtype `T`.
    fn backward_typed<T: Real>(&self, out_grad: &RawTensor) -> Vec<Option<Tensor>> {
        let grad_data = out_grad.data.host::<T>();
        let grad_shape = &out_grad.shape;

        // If keepdim=false, we need to unsqueeze the dimension back
        let expanded_shape = if self.keepdim {
            grad_shape.clone()
        } else {
            grad_shape.unsqueeze(self.dim)
        };

        // Broadcast gradient back to input shape
        // Each output element contributed to by input_shape[dim] elements
        let size = self.input_shape.numel();
        let mut result = vec![T::ZERO; size];

        let grad_strides = expanded_shape.strides();
        let grad_strides = grad_strides.truncate_rank();

        for i in 0..size {
            // Get input coordinates
            let mut coords = vec![0; self.input_shape.rank()];
            let mut rem = i;
            for (d, dim_sz) in self.input_shape.iter().enumerate().rev() {
                *slot(&mut coords, d) = rem % dim_sz;
                rem /= dim_sz;
            }

            // Map to gradient coordinates (zero out the summed dimension)
            let mut grad_coords = coords.clone();
            *slot(&mut grad_coords, self.dim) = 0;

            let grad_idx: usize = grad_coords
                .iter()
                .zip(grad_strides)
                .map(|(c, s)| c * s)
                .sum();
            *slot(&mut result, i) = fetch(&grad_data, grad_idx);
        }

        vec![Some(RawTensor::from_shape_compute(
            out_grad.data.dtype(),
            result,
            self.input_shape.clone(),
        ))]
    }
}

/// Gradient for `max_dim`: sparse gradient to max elements only
#[derive(Clone)]
pub(crate) struct MaxDimGradFn {
    /// Shape of the tensor the max was taken over.
    input_shape: Shape,
    /// Linear index, in the input, of each maximum element.
    max_indices: Vec<usize>,
    /// Axis the max was taken along.
    dim: usize,
    /// Whether the reduced axis was kept with size 1.
    keepdim: bool,
}

impl MaxDimGradFn {
    /// Routes `out_grad` to the maximum elements; every other element gets zero.
    pub(crate) fn backward(
        &self,
        out_grad: &RawTensor,
        _parents: &[Tensor],
    ) -> Vec<Option<Tensor>> {
        dispatch_real!(out_grad.data.dtype(), T => self.backward_typed::<T>(out_grad))
    }

    /// [`Self::backward`] in the gradient's dtype `T`.
    fn backward_typed<T: Real>(&self, out_grad: &RawTensor) -> Vec<Option<Tensor>> {
        let grad_data = out_grad.data.host::<T>();
        let grad_shape = &out_grad.shape;

        let expanded_shape = if self.keepdim {
            grad_shape.clone()
        } else {
            grad_shape.unsqueeze(self.dim)
        };

        let size = self.input_shape.numel();
        let mut result = vec![T::ZERO; size];

        // Only max elements receive gradient
        let grad_strides = expanded_shape.strides();
        let grad_strides = grad_strides.truncate_rank();

        for (out_idx, &max_lin_idx) in self.max_indices.iter().enumerate() {
            // Convert output index to coordinates in expanded shape
            let mut grad_coords = vec![0; expanded_shape.rank()];
            let mut rem = out_idx;
            for (d, dim_sz) in expanded_shape.iter().enumerate().rev() {
                *slot(&mut grad_coords, d) = rem % dim_sz;
                rem /= dim_sz;
            }

            let grad_idx: usize = grad_coords
                .iter()
                .zip(grad_strides)
                .map(|(c, s)| c * s)
                .sum();
            *slot(&mut result, max_lin_idx) = fetch(&grad_data, grad_idx);
        }

        vec![Some(RawTensor::from_shape_compute(
            out_grad.data.dtype(),
            result,
            self.input_shape.clone(),
        ))]
    }
}

impl RawTensor {
    /// Sum along a specific axis
    ///
    /// # Arguments
    /// * `dim` - Axis to reduce (0-indexed)
    /// * `keepdim` - If true, keep reduced dimension as size 1
    ///
    /// # Examples
    /// ```
    /// use volta::{Axis, RawTensor};
    /// let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
    /// assert_eq!(x.sum_dim(Axis(1), false).shape(), vec![2]); // [6, 15]
    /// assert_eq!(x.sum_dim(Axis(1), true).shape(), vec![2, 1]); // [[6], [15]]
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `VoltaError::DimensionOutOfBounds` if dim is >= tensor rank
    pub fn try_sum_dim(self_t: &Tensor, dim: Axis, keepdim: bool) -> Result<Tensor> {
        let dtype = compute_dtype(|| "SumDim".to_string(), &[self_t.dtype()]);
        let (data, shape) = {
            let s = self_t.borrow();
            (s.data.clone(), s.shape.clone())
        };

        let dim_size = shape.dim(dim)?;
        let dim = dim.0;
        let out_shape = shape.with_dim(dim, 1); // intermediate shape before squeeze
        let out_size = out_shape.numel();

        // Optimized stride-based reduction: O(1) per element instead of O(rank) coordinate conversion
        // View data as: [outer_size, dim_size, inner_size] where we sum over dim_size
        let outer_size = shape.numel_before(dim);
        let inner_size = shape.numel_from(dim + 1);

        // Sum over the target dimension using stride arithmetic, in the input's dtype.
        let storage = dispatch_real!(dtype, T => {
            let mut result = vec![T::ZERO; out_size];
            let data = data.host::<T>();
            for outer in 0..outer_size {
                for inner in 0..inner_size {
                    let out_idx = outer * inner_size + inner;
                    let mut sum = T::ZERO;
                    for k in 0..dim_size {
                        let in_idx = outer * dim_size * inner_size + k * inner_size + inner;
                        sum += fetch(&data, in_idx);
                    }
                    *slot(&mut result, out_idx) = sum;
                }
            }
            Storage::from_compute(dtype, result)
        });

        // Squeeze dimension if keepdim=false
        let final_shape = if keepdim {
            out_shape
        } else {
            out_shape.squeeze(dim)
        };

        // Computed on the CPU (there is no GPU kernel): `edge()` applies the device
        // policy when the input is on the GPU.
        Ok(edge(
            [self_t.clone()],
            storage,
            final_shape,
            Device::CPU,
            GradFn::SumDim(SumDimGradFn {
                input_shape: shape,
                dim,
                keepdim,
            }),
        ))
    }

    /// # Arguments
    /// * `dim` - Axis to reduce (0-indexed)
    /// * `keepdim` - If true, keep reduced dimension as size 1
    ///
    /// # Examples
    /// ```
    /// use volta::{Axis, RawTensor};
    /// let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
    /// assert_eq!(x.sum_dim(Axis(1), false).shape(), vec![2]); // [6, 15]
    /// assert_eq!(x.sum_dim(Axis(1), true).shape(), vec![2, 1]); // [[6], [15]]
    /// ```
    /// # Panics
    /// dim out of bounds
    #[must_use]
    pub fn sum_dim(self_t: &Tensor, dim: Axis, keepdim: bool) -> Tensor {
        Self::try_sum_dim(self_t, dim, keepdim).expect("Dimension out of bounds")
    }

    /// Copy the tensor to `device`, as a differentiable op (PyTorch's `.to(device)`).
    ///
    /// The result is a new node whose parent is `self_t`, so backward moves its gradient
    /// to `self_t`'s device and into `self_t`'s gradient. It carries no gradient of its
    /// own, and on a leaf it is not a leaf. To make a leaf on another device, move a
    /// tensor that doesn't require grad and then set it:
    /// `RawTensor::new(data, shape, false).to_device(d)` followed by `set_requires_grad(true)`.
    /// To move a layer's parameters in place, use `Module::to_device`.
    ///
    /// Returns `self_t` itself when it is already on `device`.
    ///
    /// # Panics
    /// If `device` is a GPU and no GPU is available, or it names a GPU other than this
    /// machine's adapter (`Device::gpu()`).
    #[must_use]
    pub fn to_device(self_t: &Tensor, device: Device) -> Tensor {
        // Fast path: already on requested device
        {
            let t = self_t.borrow();
            if t.device == device {
                return self_t.clone();
            }
        }

        // A GPU that isn't this machine's adapter is an error, like `cuda:7` on a one-GPU
        // machine, so device equality can compare names (multi-GPU-ready).
        if device.is_gpu()
            && let Some(real) = Device::gpu()
            && real != device
        {
            panic!("to_device({device}): no such GPU; the available GPU is {real}");
        }

        let (data, shape, source) = {
            let t = self_t.borrow();
            (t.data.to_device(&device), t.shape.clone(), t.device.clone())
        };
        edge(
            [self_t.clone()],
            data,
            shape,
            device,
            GradFn::ToDevice(ToDeviceGradFn { source }),
        )
    }

    /// Converts to `dtype` as a differentiable op, like PyTorch's `x.to(dtype)`. The values
    /// follow [`Storage::cast`]; backward casts the gradient back to `self_t`'s dtype.
    /// A cast to an integer or bool dtype carries no gradient.
    ///
    /// Returns `self_t` itself when it already has `dtype`, as PyTorch does.
    ///
    /// # Panics
    /// If `self_t` is on the GPU and `dtype` cannot live there (F64 or an integer type;
    /// casts between F32, F16, BF16 and Bool run on the device), and on a float-to-integer
    /// value that is NaN, infinite, or out of range (see [`Storage::cast`]).
    #[must_use]
    #[track_caller]
    pub fn to_dtype(self_t: &Tensor, dtype: DType) -> Tensor {
        let (data, shape, device, source) = {
            let t = self_t.borrow();
            if t.data.dtype() == dtype {
                return self_t.clone();
            }
            (
                t.data.cast(dtype),
                t.shape.clone(),
                t.device.clone(),
                t.data.dtype(),
            )
        };
        edge(
            [self_t.clone()],
            data,
            shape,
            device,
            GradFn::Cast(CastGradFn { source }),
        )
    }

    /// Max along a specific axis
    ///
    /// Returns maximum value along dimension and stores indices for backward pass.
    /// # Panics
    /// dim out of bounds
    #[must_use]
    pub fn max_dim(self_t: &Tensor, dim: Axis, keepdim: bool) -> Tensor {
        let dtype = compute_dtype(|| "MaxDim".to_string(), &[self_t.dtype()]);
        let (data, shape) = {
            let s = self_t.borrow();
            (s.data.clone(), s.shape.clone())
        };

        let dim_size = shape.dim(dim).unwrap_or_else(|e| panic!("max_dim: {e}"));
        let dim = dim.0;
        let out_shape = shape.with_dim(dim, 1);
        let out_size = out_shape.numel();
        let mut max_indices = vec![0; out_size]; // track which index won

        // Optimized stride-based reduction: O(1) per element instead of O(rank) coordinate conversion
        // View data as: [outer_size, dim_size, inner_size] where we find max over dim_size
        let outer_size = shape.numel_before(dim);
        let inner_size = shape.numel_from(dim + 1);

        // Find max over the target dimension using stride arithmetic, in the input's dtype.
        let storage = dispatch_real!(dtype, T => {
            let mut result = vec![T::NEG_INFINITY; out_size];
            let data = data.host::<T>();
            for outer in 0..outer_size {
                for inner in 0..inner_size {
                    let out_idx = outer * inner_size + inner;
                    for k in 0..dim_size {
                        let in_idx = outer * dim_size * inner_size + k * inner_size + inner;
                        let in_val = fetch(&data, in_idx);
                        if in_val > fetch(&result, out_idx) {
                            *slot(&mut result, out_idx) = in_val;
                            *slot(&mut max_indices, out_idx) = in_idx;
                        }
                    }
                }
            }
            Storage::from_compute(dtype, result)
        });

        let final_shape = if keepdim {
            out_shape
        } else {
            out_shape.squeeze(dim)
        };

        // Computed on the CPU (there is no GPU kernel): `edge()` applies the device
        // policy when the input is on the GPU.
        edge(
            [self_t.clone()],
            storage,
            final_shape,
            Device::CPU,
            GradFn::MaxDim(MaxDimGradFn {
                input_shape: shape,
                max_indices,
                dim,
                keepdim,
            }),
        )
    }

    /// Numerically stable softmax along `dim` (subtracts the max first).
    #[must_use]
    pub fn softmax(self_t: &Tensor, dim: Axis) -> Tensor {
        let max = Self::max_dim(self_t, dim, true);
        let shifted = self_t.sub(&max);
        let exp_x = shifted.exp();
        let sum_exp = Self::sum_dim(&exp_x, dim, true);
        exp_x.div(&sum_exp)
    }
    /// `LogSoftmax` along a specific axis (numerically stable)
    /// `log(exp(x_i)` / `sum(exp(x_j))`) = `x_i` - `log(sum(exp(x_j)))`
    /// Uses `LogSumExp` trick: log(sum(exp(x))) = m + log(sum(exp(x-m)))
    #[must_use]
    pub fn log_softmax(self_t: &Tensor, dim: Axis) -> Tensor {
        let max = Self::max_dim(self_t, dim, true);
        let shifted = self_t.sub(&max);
        let exp_x = shifted.exp();
        let sum_exp = Self::sum_dim(&exp_x, dim, true);
        let log_sum = sum_exp.log();
        let log_sum_plus_max = log_sum.add(&max);
        self_t.sub(&log_sum_plus_max)
    }

    /// Mean along a specific axis
    ///
    /// Implemented as `sum_dim(dim)` / size(dim)
    /// # Panics
    /// dim out of bounds
    #[must_use]
    pub fn mean_dim(self_t: &Tensor, dim: Axis, keepdim: bool) -> Tensor {
        let shape = self_t.borrow().shape.clone();

        let n = shape.dim(dim).unwrap_or_else(|e| panic!("mean_dim: {e}")) as f32;
        let sum = Self::sum_dim(self_t, dim, keepdim);
        sum.div(&self_t.constant_like(vec![n], &[1]))
    }
}

// ===== NUMERICAL GRADIENT CHECKING =====

impl RawTensor {
    /// Check gradients numerically using finite differences
    ///
    /// For each parameter, we compute:
    ///
    /// Analytical gradient: What our `backward()` computes
    /// Numerical gradient: (f(x+ε) - f(x-ε)) / (2ε)
    ///
    /// The central difference formula is more accurate than forward difference.
    ///
    /// # Arguments
    /// * `tensor` - The input tensor whose gradients to check
    /// * `loss_fn` - Function that computes a scalar loss from the tensor
    /// * `epsilon` - Step size for finite differences (typically 1e-5 to 1e-2)
    /// * `tolerance` - Maximum acceptable relative error (typically 1e-3 to 1e-2)
    ///
    /// # Returns
    /// (`max_error`, `mean_error`, passed)
    ///
    /// # Side effects
    /// `tensor`'s gradient is cleared first, so a gradient left over from an earlier
    /// `backward()` can't skew the result, and afterwards holds the analytic gradient.
    /// Every other tensor the loss reaches (layer parameters) gets its gradient back
    /// as it was before the call, including having none. The one thing not restored
    /// is layer state that `loss_fn` updates: the finite-difference evaluations run
    /// `loss_fn` 2N more times, so `BatchNorm` in training mode keeps updating its
    /// running statistics. The result is unaffected, because training mode normalizes
    /// with batch statistics; put the layer in eval mode, or rebuild it, if the
    /// statistics matter afterwards.
    ///
    /// # Panics
    /// If `loss_fn` doesn't return a scalar that requires grad, or `tensor` ends up
    /// without a gradient
    pub fn check_gradients<F>(
        tensor: &Tensor,
        loss_fn: F,
        epsilon: f32,
        tolerance: f32,
    ) -> (f32, f32, bool)
    where
        F: Fn(&Tensor) -> Tensor,
    {
        // Compute analytical gradient. The backward pass adds into every tensor the loss
        // reaches (a layer's parameters among them), so save theirs first and put them
        // back; only `tensor` keeps the result.
        Self::zero_grad(tensor);
        let loss = loss_fn(tensor);
        let saved = crate::autograd::GradSnapshot::take(&loss, tensor);
        loss.into_loss()
            .expect("loss_fn must return a scalar that requires grad")
            .backward();
        saved.restore();

        // Everything below is in f64: the gradient and the losses are widened exactly, and
        // the input is perturbed in its own dtype (f32 or f64), so an f64 tensor gets an
        // f64-accurate check.
        let analytical_grad: Vec<f64> = tensor
            .grad_storage()
            .expect("Tensor must have gradient")
            .cast(DType::F64)
            .as_host::<f64>()
            .expect("cast to f64")
            .to_vec();
        let mut numerical_grad = vec![0.0f64; analytical_grad.len()];

        let original_data = tensor.borrow().data.clone();
        let original_shape = tensor.borrow().shape.clone();
        let requires_grad = tensor.borrow().requires_grad;
        let dtype = compute_dtype(|| "check_gradients".to_string(), &[original_data.dtype()]);
        let loss_value = |loss: &Tensor| {
            let l = loss.borrow();
            assert!(
                !l.data.is_empty(),
                "check_gradients: loss must be scalar, got shape {:?}",
                l.shape
            );
            fetch(
                l.data
                    .cast(DType::F64)
                    .as_host::<f64>()
                    .expect("cast to f64"),
                0,
            )
        };

        // Compute numerical gradient for each element. These perturbed forward passes
        // are for finite-difference comparison only — scoped in `no_grad()` so they
        // can't attach graph nodes or otherwise contaminate the analytical pass above
        // (ADDENDUM.md §1.5).
        {
            let _no_grad = crate::autograd::no_grad();
            dispatch_real!(dtype, T => {
                let step = T::from_f64(f64::from(epsilon));
                let perturbed = |i: usize, up: bool| {
                    let mut data = original_data.clone();
                    {
                        let mut host = data.host_mut::<T>();
                        let x = slot(&mut host, i);
                        if up {
                            *x += step;
                        } else {
                            *x -= step;
                        }
                    }
                    Self::new_with_storage(data, original_shape.clone(), Device::CPU, requires_grad)
                };
                for i in 0..original_data.len() {
                    let val_plus = loss_value(&loss_fn(&perturbed(i, true)));
                    let val_minus = loss_value(&loss_fn(&perturbed(i, false)));
                    //central diff
                    *slot(&mut numerical_grad, i) =
                        (val_plus - val_minus) / (2.0 * f64::from(epsilon));
                }
            });
        }

        // Compute errors (in f64; reported as f32, the signature's type)
        let mut max_error: f64 = 0.0;
        let mut total_error: f64 = 0.0;

        for (i, (&analytical, &numerical)) in
            analytical_grad.iter().zip(&numerical_grad).enumerate()
        {
            let error = (analytical - numerical).abs();
            // When both gradients are extremely small, pure relative error becomes noisy
            // and can falsely flag otherwise correct gradients (e.g., BatchNorm sum outputs
            // that should be exactly zero). In that regime, we compare absolute error instead.
            let scale = analytical.abs().max(numerical.abs());
            let relative_error = if scale > 1e-3 { error / scale } else { error };

            max_error = max_error.max(relative_error);
            total_error += relative_error;

            if relative_error > f64::from(tolerance) {
                log::debug!(
                    "Gradient mismatch at index {i}: analytical={analytical:.6e}, numerical={numerical:.6e}, error={relative_error:.6e}",
                );
            }
        }

        let mean_error = total_error / analytical_grad.len() as f64;
        let passed = max_error < f64::from(tolerance);

        #[expect(
            clippy::cast_possible_truncation,
            reason = "the errors are reported in the signature's f32"
        )]
        (max_error as f32, mean_error as f32, passed)
    }

    /// Simplified gradient checker with default parameters
    ///
    /// Uses epsilon=1e-2 and tolerance=1e-3, which work well for most cases.
    pub fn check_gradients_simple<F>(tensor: &Tensor, loss_fn: F) -> bool
    where
        F: Fn(&Tensor) -> Tensor,
    {
        let (max_err, mean_err, passed) = Self::check_gradients(
            tensor, loss_fn, 1e-2, // epsilon
            1e-3, // tolerance
        );

        if !passed {
            log::warn!("Gradient check FAILED: max_error={max_err:.6e}, mean_error={mean_err:.6e}");
        }

        passed
    }
}

impl RawTensor {
    /// Clear any cached gradient on this tensor.
    ///
    /// This mirrors `Module::zero_grad` but works on individual tensors so
    /// training loops that own raw tensors (not wrapped in modules) can
    /// reset gradients before a new backward pass.
    pub fn zero_grad(self_t: &Tensor) {
        // The one writer that clears `grad` (invariant 3).
        self_t.borrow_mut().grad = None;
    }
}

// ===== TENSOR OPS =====

impl Tensor {
    /// Elementwise sum, with broadcasting.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        RawTensor::binary_op(self, other, BinaryOp::Add)
    }

    /// Elementwise difference, with broadcasting.
    #[must_use]
    pub fn sub(&self, other: &Self) -> Self {
        RawTensor::binary_op(self, other, BinaryOp::Sub)
    }

    /// Elementwise product, with broadcasting.
    #[must_use]
    pub fn elem_mul(&self, other: &Self) -> Self {
        RawTensor::binary_op(self, other, BinaryOp::Mul)
    }

    /// Elementwise quotient, with broadcasting.
    #[must_use]
    pub fn div(&self, other: &Self) -> Self {
        RawTensor::binary_op(self, other, BinaryOp::Div)
    }

    /// Elementwise maximum, with broadcasting.
    #[must_use]
    pub fn max_elem(&self, other: &Self) -> Self {
        RawTensor::binary_op(self, other, BinaryOp::Max)
    }

    /// Elementwise remainder, with broadcasting.
    #[must_use]
    pub fn modulo(&self, other: &Self) -> Self {
        RawTensor::binary_op(self, other, BinaryOp::Mod)
    }

    /// Elementwise `self < other` as a Bool tensor, with broadcasting (`RawTensor::compare`).
    #[must_use]
    pub fn lt(&self, other: &Self) -> Self {
        RawTensor::compare(self, other, CmpOp::Lt)
    }

    /// Elementwise `self <= other` as a Bool tensor, with broadcasting.
    #[must_use]
    pub fn le(&self, other: &Self) -> Self {
        RawTensor::compare(self, other, CmpOp::Le)
    }

    /// Elementwise `self > other` as a Bool tensor, with broadcasting.
    #[must_use]
    pub fn gt(&self, other: &Self) -> Self {
        RawTensor::compare(self, other, CmpOp::Gt)
    }

    /// Elementwise `self >= other` as a Bool tensor, with broadcasting.
    #[must_use]
    pub fn ge(&self, other: &Self) -> Self {
        RawTensor::compare(self, other, CmpOp::Ge)
    }

    /// Elementwise `self == other` as a Bool tensor, with broadcasting (PyTorch's `eq`;
    /// named like `elem_mul` so it can't be mistaken for `PartialEq::eq`).
    #[must_use]
    pub fn elem_eq(&self, other: &Self) -> Self {
        RawTensor::compare(self, other, CmpOp::Eq)
    }

    /// Elementwise `self != other` as a Bool tensor, with broadcasting.
    #[must_use]
    pub fn elem_ne(&self, other: &Self) -> Self {
        RawTensor::compare(self, other, CmpOp::Ne)
    }

    /// Negation, elementwise.
    #[must_use]
    pub fn neg(&self) -> Self {
        RawTensor::unary_op(self, UnaryOp::Neg)
    }

    /// Reciprocal, elementwise.
    #[must_use]
    pub fn recip(&self) -> Self {
        RawTensor::unary_op(self, UnaryOp::Recip)
    }

    /// Square root, elementwise.
    #[must_use]
    pub fn sqrt(&self) -> Self {
        RawTensor::unary_op(self, UnaryOp::Sqrt)
    }

    /// Base-2 exponential, elementwise.
    #[must_use]
    pub fn exp2(&self) -> Self {
        RawTensor::unary_op(self, UnaryOp::Exp2)
    }

    /// Base-2 logarithm, elementwise.
    #[must_use]
    pub fn log2(&self) -> Self {
        RawTensor::unary_op(self, UnaryOp::Log2)
    }

    /// Natural exponential, elementwise.
    #[must_use]
    pub fn exp(&self) -> Self {
        RawTensor::unary_op(self, UnaryOp::Exp)
    }

    /// Natural logarithm, elementwise.
    #[must_use]
    pub fn log(&self) -> Self {
        RawTensor::unary_op(self, UnaryOp::Log)
    }

    /// Sine, elementwise.
    #[must_use]
    pub fn sin(&self) -> Self {
        RawTensor::unary_op(self, UnaryOp::Sin)
    }

    /// Cosine, elementwise.
    #[must_use]
    pub fn cos(&self) -> Self {
        RawTensor::unary_op(self, UnaryOp::Cos)
    }

    /// Hyperbolic tangent, elementwise.
    #[must_use]
    pub fn tanh(&self) -> Self {
        RawTensor::unary_op(self, UnaryOp::Tanh)
    }

    /// Logistic sigmoid, elementwise.
    #[must_use]
    pub fn sigmoid(&self) -> Self {
        RawTensor::unary_op(self, UnaryOp::Sigmoid)
    }

    /// Rectified linear unit, elementwise.
    #[must_use]
    pub fn relu(&self) -> Self {
        RawTensor::unary_op(self, UnaryOp::ReLU)
    }

    /// Error function, elementwise.
    #[must_use]
    pub fn erf(&self) -> Self {
        RawTensor::unary_op(self, UnaryOp::Erf)
    }

    /// Sum of all elements, as a scalar.
    #[must_use]
    pub fn sum(&self) -> Self {
        RawTensor::reduce_op(self, ReduceOp::Sum)
    }

    /// Maximum of all elements, as a scalar.
    #[must_use]
    pub fn max_reduce(&self) -> Self {
        RawTensor::reduce_op(self, ReduceOp::Max)
    }

    /// Mean of all elements, as a scalar.
    #[must_use]
    pub fn mean(&self) -> Self {
        RawTensor::reduce_op(self, ReduceOp::Mean)
    }

    /// Fused `self * y + z`.
    #[must_use]
    pub fn mulacc(&self, y: &Self, z: &Self) -> Self {
        RawTensor::ternary_op(self, y, z, TernaryOp::MulAcc)
    }

    /// Selects `x` where `self` (a Bool condition) is true, else `y`, with broadcasting
    /// (PyTorch's `where`). `x` and `y` may have any dtype; branches of different dtypes
    /// promote (`DTypePolicy`). Gradients flow to the selected float branch. On the GPU,
    /// equal shapes run a kernel and broadcasting is a device-policy gap.
    ///
    /// # Panics
    /// If the condition is not Bool, the inputs are on different devices, or the
    /// branches' dtypes differ under `DTypePolicy::Strict`.
    #[must_use]
    pub fn where_op(&self, x: &Self, y: &Self) -> Self {
        RawTensor::ternary_op(self, x, y, TernaryOp::Where)
    }

    /// Reshapes to `new_shape`; the element count must match.
    ///
    /// # Panics
    ///
    /// If `new_shape` is invalid or has a different element count. Use
    /// [`Self::try_reshape`] to get an error instead.
    #[must_use]
    #[track_caller]
    pub fn reshape(&self, new_shape: &[usize]) -> Self {
        RawTensor::reshape(self, new_shape)
    }

    /// Reshapes to `new_shape` (fallible).
    ///
    /// # Errors
    ///
    /// `VoltaError::InvalidParameter` if `new_shape` is invalid (see [`Shape::from_dims`])
    /// or does not have the same element count as `self`.
    pub fn try_reshape(&self, new_shape: &[usize]) -> Result<Self> {
        RawTensor::try_reshape(self, new_shape)
    }

    /// Permutes the dimensions by `axes`.
    #[must_use]
    pub fn permute(&self, axes: &[Axis]) -> Self {
        RawTensor::permute(self, axes)
    }

    /// Broadcasts to `new_shape`.
    ///
    /// # Panics
    ///
    /// If `new_shape` is invalid or not reachable by expanding size-1 dimensions. Use
    /// [`Self::try_expand`] to get an error instead.
    #[must_use]
    #[track_caller]
    pub fn expand(&self, new_shape: &[usize]) -> Self {
        RawTensor::expand(self, new_shape)
    }

    /// Broadcasts to `new_shape` (fallible).
    ///
    /// # Errors
    ///
    /// `VoltaError::InvalidParameter` if `new_shape` is invalid or has a different rank,
    /// `VoltaError::BroadcastError` if a dimension is neither `1` nor equal to the target.
    pub fn try_expand(&self, new_shape: &[usize]) -> Result<Self> {
        RawTensor::try_expand(self, new_shape)
    }

    /// Pads each dimension by `(before, after)`, filling the border with `value`.
    #[must_use]
    pub fn pad_with(&self, padding: &[(usize, usize)], value: PadValue) -> Self {
        RawTensor::pad_with(self, padding, value)
    }

    /// Slices each dimension to the half-open `(start, end)` range.
    #[must_use]
    pub fn shrink(&self, ranges: &[(usize, usize)]) -> Self {
        RawTensor::shrink(self, ranges)
    }

    /// Takes every `strides[i]`-th element along each dimension.
    #[must_use]
    pub fn stride_op(&self, strides: &[usize]) -> Self {
        RawTensor::stride_op(self, strides)
    }

    /// Converts to `dtype` as a differentiable op: backward casts the gradient back to this
    /// tensor's dtype. See [`RawTensor::to_dtype`].
    ///
    /// # Panics
    /// As [`RawTensor::to_dtype`].
    #[must_use]
    #[track_caller]
    pub fn to_dtype(&self, dtype: DType) -> Self {
        RawTensor::to_dtype(self, dtype)
    }

    /// Copies the tensor to `device` as a differentiable op: backward moves the gradient
    /// back to this tensor. See [`RawTensor::to_device`].
    ///
    /// # Panics
    /// If `device` is a GPU and no GPU is available, or it names a GPU other than this
    /// machine's adapter (`Device::gpu()`).
    #[must_use]
    pub fn to_device(&self, device: Device) -> Self {
        RawTensor::to_device(self, device)
    }

    /// Matrix product.
    #[must_use]
    pub fn matmul(&self, other: &Self) -> Self {
        RawTensor::matmul(self, other)
    }

    /// Transposes a 2-D tensor.
    #[must_use]
    pub fn transpose(&self) -> Self {
        RawTensor::transpose(self)
    }

    /// The accumulated gradient decoded to `f32` (like [`Self::data`]), or `None` if
    /// there is none. Lossy for an f64 gradient; [`Self::grad_storage`] keeps the dtype.
    #[must_use]
    pub fn grad(&self) -> Option<Vec<f32>> {
        self.borrow()
            .grad
            .as_ref()
            .map(super::storage::Storage::to_f32_vec)
    }

    /// Clears the accumulated gradient.
    pub fn zero_grad(&self) {
        RawTensor::zero_grad(self);
    }

    /// Sum along `dim`.
    #[must_use]
    pub fn sum_dim(&self, dim: Axis, keepdim: bool) -> Self {
        RawTensor::sum_dim(self, dim, keepdim)
    }

    /// Maximum along `dim`.
    #[must_use]
    pub fn max_dim(&self, dim: Axis, keepdim: bool) -> Self {
        RawTensor::max_dim(self, dim, keepdim)
    }

    /// Mean along `dim`.
    #[must_use]
    pub fn mean_dim(&self, dim: Axis, keepdim: bool) -> Self {
        RawTensor::mean_dim(self, dim, keepdim)
    }

    /// Softmax along `dim`.
    #[must_use]
    pub fn softmax(&self, dim: Axis) -> Self {
        RawTensor::softmax(self, dim)
    }

    /// Log-softmax along `dim`.
    #[must_use]
    pub fn log_softmax(&self, dim: Axis) -> Self {
        RawTensor::log_softmax(self, dim)
    }
}

// ===== DataLoaders =====

/// Iterator over mini-batches (optionally shuffled) of `(data, targets)` tensors.
pub struct DataLoader {
    /// All samples, flattened row-major.
    data: Vec<f32>,
    /// All targets, flattened row-major.
    targets: Vec<f32>,
    /// Shape of one sample.
    data_shape: Vec<usize>,
    /// Shape of one target.
    target_shape: Vec<usize>,
    /// Samples per batch; the last batch may be smaller.
    batch_size: usize,
    /// Whether `reset` reshuffles the sample order.
    shuffle: bool,
    /// Order in which samples are visited.
    indices: Vec<usize>,
    /// Position in `indices` of the next batch.
    current: usize,
    /// If set, each batch is moved to this device (GPU prefetch).
    device: Option<crate::device::Device>,
}

impl DataLoader {
    /// Builds a loader over `data` and `targets`, whose sample count is `data.len()` divided by the elements of `data_shape`. Shuffles once if `shuffle`.
    #[must_use]
    pub fn new(
        data: Vec<f32>,
        targets: Vec<f32>,
        data_shape: &[usize],   // e.g., [28, 28] for MNIST
        target_shape: &[usize], // e.g., [10] for one-hot
        batch_size: usize,
        shuffle: bool,
    ) -> Self {
        let num_samples = data.len() / data_shape.iter().product::<usize>();
        let mut indices: Vec<usize> = (0..num_samples).collect();

        if shuffle {
            use rand::seq::SliceRandom;
            with_rng(|rng| indices.shuffle(rng));
        }

        Self {
            data,
            targets,
            data_shape: data_shape.to_vec(),
            target_shape: target_shape.to_vec(),
            batch_size,
            shuffle,
            indices,
            current: 0,
            device: None, // Default: no GPU prefetch
        }
    }

    /// Enable automatic GPU prefetch for batches
    ///
    /// When set, batches will automatically be transferred to the specified device.
    /// This avoids manual `.to_device()` calls in the training loop.
    ///
    /// # Arguments
    /// * `device` - Device to prefetch batches to (typically GPU)
    ///
    /// # Example
    /// ```no_run
    /// # use volta::{DataLoader, Device};
    /// # #[cfg(feature = "gpu")]
    /// # {
    /// # let data = vec![0.0; 28 * 28 * 100];
    /// # let targets = vec![0.0; 10 * 100];
    /// let device = Device::gpu().expect("GPU required");
    /// let dataloader = DataLoader::new(data, targets, &[28, 28], &[10], 64, true)
    ///     .with_device(device);
    /// // Batches will now be automatically on GPU
    /// # }
    /// ```
    #[must_use]
    pub fn with_device(mut self, device: crate::device::Device) -> Self {
        self.device = Some(device);
        self
    }

    /// Rewinds to the first batch, reshuffling if the loader shuffles.
    pub fn reset(&mut self) {
        self.current = 0;
        if self.shuffle {
            use rand::seq::SliceRandom;
            with_rng(|rng| self.indices.shuffle(rng));
        }
    }
}

impl Iterator for DataLoader {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.indices.len() {
            return None;
        }

        let end = (self.current + self.batch_size).min(self.indices.len());
        let batch_indices = self.indices.get(self.current..end)?;
        let actual_batch = batch_indices.len();

        let sample_size: usize = self.data_shape.iter().product();
        let target_size: usize = self.target_shape.iter().product();

        // Gather batch
        let mut batch_data = Vec::with_capacity(actual_batch * sample_size);
        let mut batch_targets = Vec::with_capacity(actual_batch * target_size);

        for &idx in batch_indices {
            let data_start = idx * sample_size;
            let target_start = idx * target_size;

            batch_data.extend_from_slice(span(&self.data, data_start..data_start + sample_size));
            batch_targets.extend_from_slice(span(
                &self.targets,
                target_start..target_start + target_size,
            ));
        }

        self.current = end;

        let mut batch_shape = vec![actual_batch];
        batch_shape.extend_from_slice(&self.data_shape);

        let mut target_batch_shape = vec![actual_batch];
        target_batch_shape.extend_from_slice(&self.target_shape);

        let data_tensor = RawTensor::new(batch_data, &batch_shape, false);
        let target_tensor = RawTensor::new(batch_targets, &target_batch_shape, false);

        // Transfer to device if GPU prefetch is enabled
        if let Some(ref device) = self.device {
            Some((
                data_tensor.to_device(device.clone()),
                target_tensor.to_device(device.clone()),
            ))
        } else {
            Some((data_tensor, target_tensor))
        }
    }
}

// ===== PUBLIC API EXPORTS =====
pub use RawTensor as new_tensor;

// Tensor constructors
/// Free-function form of [`RawTensor::zeros`].
#[must_use]
pub fn zeros(shape: &[usize]) -> Tensor {
    RawTensor::zeros(shape)
}

/// Free-function form of [`RawTensor::ones`].
#[must_use]
pub fn ones(shape: &[usize]) -> Tensor {
    RawTensor::ones(shape)
}

/// Free-function form of [`RawTensor::rand`].
#[must_use]
pub fn rand(shape: &[usize]) -> Tensor {
    RawTensor::rand(shape)
}

/// Free-function form of [`RawTensor::randn`].
#[must_use]
pub fn randn(shape: &[usize]) -> Tensor {
    RawTensor::randn(shape)
}

/// Free-function form of [`RawTensor::randn_like`].
#[must_use]
pub fn randn_like(tensor: &Tensor) -> Tensor {
    RawTensor::randn_like(tensor)
}

// Loss functions
/// Free-function form of [`RawTensor::mse_loss`].
#[must_use]
pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    RawTensor::mse_loss(pred, target)
}

/// Free-function form of [`RawTensor::cross_entropy_loss`].
#[must_use]
#[track_caller]
pub fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Tensor {
    RawTensor::cross_entropy_loss(logits, targets)
}

/// Free-function form of [`RawTensor::nll_loss`].
#[must_use]
#[track_caller]
pub fn nll_loss(log_probs: &Tensor, targets: &Tensor) -> Tensor {
    RawTensor::nll_loss(log_probs, targets)
}

/// Free-function form of [`RawTensor::kl_divergence_gaussian`].
#[must_use]
pub fn kl_divergence_gaussian(mu: &Tensor, logvar: &Tensor) -> Tensor {
    RawTensor::kl_divergence_gaussian(mu, logvar)
}

/// Free-function form of [`RawTensor::bce_loss`].
#[must_use]
#[track_caller]
pub fn bce_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    RawTensor::bce_loss(pred, target)
}

/// Free-function form of [`RawTensor::bce_with_logits_loss`].
#[must_use]
pub fn bce_with_logits_loss(logits: &Tensor, target: &Tensor) -> Tensor {
    RawTensor::bce_with_logits_loss(logits, target)
}

// Axis reductions
/// Free-function form of [`RawTensor::sum_dim`].
#[must_use]
pub fn sum_dim(t: &Tensor, dim: Axis, keepdim: bool) -> Tensor {
    RawTensor::sum_dim(t, dim, keepdim)
}

/// Free-function form of [`RawTensor::max_dim`].
#[must_use]
pub fn max_dim(t: &Tensor, dim: Axis, keepdim: bool) -> Tensor {
    RawTensor::max_dim(t, dim, keepdim)
}

/// Free-function form of [`RawTensor::softmax`].
#[must_use]
pub fn softmax(t: &Tensor, dim: Axis) -> Tensor {
    RawTensor::softmax(t, dim)
}

// Gradient checking
/// Free-function form of [`RawTensor::check_gradients`].
pub fn check_gradients<F>(
    tensor: &Tensor,
    loss_fn: F,
    epsilon: f32,
    tolerance: f32,
) -> (f32, f32, bool)
where
    F: Fn(&Tensor) -> Tensor,
{
    RawTensor::check_gradients(tensor, loss_fn, epsilon, tolerance)
}

/// Free-function form of [`RawTensor::check_gradients_simple`].
pub fn check_gradients_simple<F>(tensor: &Tensor, loss_fn: F) -> bool
where
    F: Fn(&Tensor) -> Tensor,
{
    RawTensor::check_gradients_simple(tensor, loss_fn)
}

#[cfg(test)]
mod tensor_tests {
    use super::*;

    /// A nested `with_rng` used to fall back to the thread RNG without a word, so a
    /// seeded run stopped being reproducible. It is a bug in the caller and panics.
    #[test]
    #[should_panic(expected = "reentrant")]
    fn with_rng_reentrant_call_panics() {
        manual_seed(1);
        with_rng(|_| with_rng(|_| ()));
    }

    /// Shapes that reach the library from outside (JSON, a UI, a config file) must come
    /// back as an `Err` before anything is allocated: `[usize::MAX, 2]` used to overflow
    /// `shape.iter().product()` and `[MAX_NUMEL + 1]` allocated first and checked later.
    #[test]
    fn try_constructors_reject_bad_shapes_before_allocating() {
        use crate::shape::MAX_NUMEL;
        let bad: [&[usize]; 3] = [&[usize::MAX, 2], &[0, 3], &[MAX_NUMEL + 1]];
        for shape in bad {
            let results = [
                ("try_zeros", RawTensor::try_zeros(shape)),
                ("try_ones", RawTensor::try_ones(shape)),
                ("try_empty", RawTensor::try_empty(shape)),
                ("try_constant", RawTensor::try_constant(2.0, shape)),
                ("try_rand", RawTensor::try_rand(shape)),
                ("try_randn", RawTensor::try_randn(shape)),
                (
                    "try_he_initialization",
                    RawTensor::try_he_initialization(shape),
                ),
            ];
            for (name, result) in results {
                assert!(
                    matches!(result, Err(VoltaError::InvalidParameter(_))),
                    "{name}({shape:?}) should be InvalidParameter"
                );
            }
        }
        assert!(matches!(
            RawTensor::try_xavier_uniform(&[usize::MAX, 2]),
            Err(VoltaError::InvalidParameter(_))
        ));
    }

    #[test]
    fn try_xavier_uniform_rejects_unsupported_rank() {
        assert!(matches!(
            RawTensor::try_xavier_uniform(&[2, 3, 4]),
            Err(VoltaError::InvalidParameter(_))
        ));
        assert!(matches!(
            RawTensor::try_xavier_uniform(&[]),
            Err(VoltaError::InvalidParameter(_))
        ));
    }

    #[test]
    fn try_from_vec_reports_a_length_mismatch() {
        assert!(matches!(
            RawTensor::try_from_vec(vec![1.0, 2.0, 3.0], &[2, 2]),
            Err(VoltaError::ShapeDataMismatch { .. })
        ));
        let t = RawTensor::try_from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert_eq!(t.shape().truncate_rank(), &[2, 2]);
        assert_eq!(t.data(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    /// Each panicking constructor is its `try_` twin plus a panic, so a valid shape gives
    /// the same tensor (random ones: under the same seed) from both.
    #[test]
    fn try_constructors_match_the_panicking_forms() {
        type Pair = (fn(&[usize]) -> Result<Tensor>, fn(&[usize]) -> Tensor);
        assert_eq!(RawTensor::try_zeros(&[2, 3]).unwrap().data(), vec![0.0; 6]);
        assert_eq!(RawTensor::try_ones(&[2, 3]).unwrap().data(), vec![1.0; 6]);
        assert_eq!(RawTensor::try_empty(&[4]).unwrap().data(), vec![0.0; 4]);
        assert_eq!(
            RawTensor::try_constant(2.5, &[3]).unwrap().data(),
            vec![2.5; 3]
        );

        let pairs: [Pair; 4] = [
            (RawTensor::try_rand, RawTensor::rand),
            (RawTensor::try_randn, RawTensor::randn),
            (RawTensor::try_xavier_uniform, RawTensor::xavier_uniform),
            (
                RawTensor::try_he_initialization,
                RawTensor::he_initialization,
            ),
        ];
        for (try_form, panicking) in pairs {
            manual_seed(3);
            let a = try_form(&[3, 4]).unwrap().data();
            manual_seed(3);
            let b = panicking(&[3, 4]).data();
            assert_eq!(a, b);
        }
    }

    #[test]
    #[should_panic(expected = "zero dimension")]
    fn panicking_constructor_says_why() {
        let _ = RawTensor::zeros(&[0, 3]);
    }

    #[test]
    fn try_reshape_validates_the_target_shape() {
        let t = RawTensor::try_from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        // `[usize::MAX, 2]` wraps to `usize::MAX * 2 mod 2^64 = usize::MAX - 1`, never 6.
        for bad in [&[4, 2][..], &[0, 6], &[usize::MAX, 2], &[]] {
            assert!(
                matches!(t.try_reshape(bad), Err(VoltaError::InvalidParameter(_))),
                "try_reshape({bad:?}) should be InvalidParameter"
            );
        }
        let r = t.try_reshape(&[3, 2]).unwrap();
        assert_eq!(r.shape().truncate_rank(), &[3, 2]);
        assert_eq!(r.data(), t.data());
    }

    #[test]
    fn try_expand_validates_the_target_shape() {
        let t = RawTensor::try_from_vec(vec![1.0, 2.0], &[1, 2]).unwrap();
        assert!(matches!(
            t.try_expand(&[3]),
            Err(VoltaError::InvalidParameter(_))
        ));
        assert!(matches!(
            t.try_expand(&[3, 5]),
            Err(VoltaError::BroadcastError(..))
        ));
        assert!(matches!(
            t.try_expand(&[0, 2]),
            Err(VoltaError::InvalidParameter(_))
        ));
        let e = t.try_expand(&[3, 2]).unwrap();
        assert_eq!(e.data(), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }

    /// The first two epochs' sample order, from a loader built right after seeding.
    fn shuffled_orders(seed: u64) -> (Vec<f32>, Vec<f32>) {
        let n = 20;
        manual_seed(seed);
        let data = (0..n).map(|i| i as f32).collect();
        let mut loader = DataLoader::new(data, vec![0.0; n], &[1], &[1], n, true);
        let first = loader.next().unwrap().0.data();
        loader.reset();
        let second = loader.next().unwrap().0.data();
        (first, second)
    }

    /// `DataLoader` shuffles through the global RNG, so `manual_seed` fixes its order
    /// (it used `rand::rng()` and ignored the seed).
    #[test]
    fn dataloader_shuffle_follows_manual_seed() {
        let (first, second) = shuffled_orders(7);
        assert_eq!((first.clone(), second.clone()), shuffled_orders(7));

        let identity: Vec<f32> = (0..20).map(|i| i as f32).collect();
        assert_ne!(first, identity, "the first epoch should be shuffled");
        assert_ne!(first, second, "reset should reshuffle");
        assert_ne!(first, shuffled_orders(8).0, "another seed, another order");
    }

    /// `check_gradients` clears the input's existing gradient before its analytic pass.
    /// Without that, the gradient left by the earlier `backward()` (and by the first
    /// check) is added in and a correct gradient fails, the audit's "probe E".
    #[test]
    fn check_gradients_ignores_preexisting_input_grad() {
        let x = RawTensor::new(vec![0.3, -1.2, 2.0, 0.7], &[4], true);
        let w = RawTensor::new(vec![1.5, -0.5, 0.25, 2.0], &[4], true);
        x.elem_mul(&w).sum().into_loss().unwrap().backward();
        assert!(x.grad().is_some(), "setup: x needs a stale gradient");

        let loss_fn = |t: &Tensor| t.elem_mul(&w).elem_mul(t).sum();
        let (first_err, _, first_passed) = RawTensor::check_gradients(&x, loss_fn, 1e-2, 1e-3);
        let (second_err, _, second_passed) = RawTensor::check_gradients(&x, loss_fn, 1e-2, 1e-3);
        assert!(first_passed, "first check failed: max_err={first_err}");
        assert!(second_passed, "repeat check failed: max_err={second_err}");
    }

    /// The analytic pass of `check_gradients` used to add its gradient into every other
    /// leaf the loss reads, so checking an input through a layer changed the layer's
    /// parameter gradients. They are put back: a preset gradient is unchanged, and a
    /// parameter with none still has none.
    #[test]
    fn check_gradients_leaves_other_leaf_grads_alone() {
        manual_seed(0x6A4D);
        let x = RawTensor::randn(&[4, 3]);
        x.set_requires_grad(true);
        let weight = RawTensor::randn(&[3, 2]);
        weight.set_requires_grad(true);
        let bias = RawTensor::randn(&[2]);
        bias.set_requires_grad(true);
        let preset = vec![0.5, -1.0, 2.0, 0.25, -0.75, 1.5];
        weight.set_grad(Storage::cpu(preset.clone()));
        let loss_weights = RawTensor::randn(&[4, 2]);

        let loss_fn = |t: &Tensor| t.matmul(&weight).add(&bias).elem_mul(&loss_weights).sum();
        let (max_err, _, passed) = RawTensor::check_gradients(&x, loss_fn, 0.5, 2e-2);
        assert!(passed, "the check itself should pass: max_err={max_err}");

        assert_eq!(weight.grad(), Some(preset), "preset gradient was changed");
        assert!(
            bias.grad().is_none(),
            "a parameter with no gradient got one"
        );
        assert!(x.grad().is_some(), "the input keeps the analytic gradient");
    }

    /// `kl_divergence_gaussian` is `-0.5 * sum(1 + logvar - mu^2 - exp(logvar))` over every
    /// element. Closed-form value on two latents, then finite-difference gradients for
    /// `mu` and `logvar` from a random state.
    #[test]
    fn test_kl_divergence_gaussian_value_and_gradcheck() {
        // mu = [0, 1], logvar = [0, 0]: terms are 0 and -1, so KL = -0.5 * -1 = 0.5.
        let mu = RawTensor::new(vec![0.0, 1.0], &[1, 2], false);
        let logvar = RawTensor::new(vec![0.0, 0.0], &[1, 2], false);
        assert!((kl_divergence_gaussian(&mu, &logvar).item() - 0.5).abs() < 1e-6);

        manual_seed(0x4B1D);
        let mu = RawTensor::randn(&[4, 3]);
        mu.set_requires_grad(true);
        let logvar = RawTensor::randn(&[4, 3]);
        logvar.set_requires_grad(true);

        let (mu_err, _, mu_ok) =
            RawTensor::check_gradients(&mu, |m| kl_divergence_gaussian(m, &logvar), 1e-2, 2e-2);
        assert!(mu_ok, "d KL / d mu: max_err={mu_err}");
        let (lv_err, _, lv_ok) =
            RawTensor::check_gradients(&logvar, |l| kl_divergence_gaussian(&mu, l), 1e-2, 2e-2);
        assert!(lv_ok, "d KL / d logvar: max_err={lv_err}");
    }

    /// A rank-1 shape is a bias vector: `n` values bounded by `sqrt(6 / (n + 1))`.
    #[test]
    fn xavier_uniform_rank_one_is_a_bias_vector() {
        let t = RawTensor::xavier_uniform(&[8]);
        let limit = (6.0f32 / 9.0).sqrt();
        let data = t.data();
        assert_eq!(data.len(), 8);
        assert!(data.iter().all(|v| v.abs() <= limit));
    }

    /// Rank 3 has no `[in, out]` reading. It used to default `fan_out` to the second dim
    /// and fail later as a data/shape mismatch; it now fails up front.
    #[test]
    #[should_panic(expected = "xavier_uniform expects a rank-1 or rank-2 shape, got [2, 3, 4]")]
    fn xavier_uniform_rejects_rank_three() {
        let _ = RawTensor::xavier_uniform(&[2, 3, 4]);
    }

    #[test]
    fn test_he_initialization_linear_shape() {
        let t = RawTensor::he_initialization(&[64, 32]);
        let b = t.borrow();
        assert_eq!(b.shape, vec![64, 32]);
        assert_eq!(b.data.len(), 64 * 32);
        assert!(b.data.to_host_f32().iter().any(|v| v.abs() > 0.0));
    }

    #[test]
    fn test_he_initialization_conv_shape() {
        let t = RawTensor::he_initialization(&[16, 3, 3, 3]);
        let b = t.borrow();
        assert_eq!(b.shape, vec![16, 3, 3, 3]);
        assert_eq!(b.data.len(), 16 * 3 * 3 * 3);
        assert!(b.data.to_host_f32().iter().any(|v| v.abs() > 0.0));
    }

    /// `item` reads a rank-0-style `[1]` tensor and a `[1, 1]` tensor.
    #[test]
    fn test_item_reads_single_element() {
        assert_eq!(RawTensor::new(vec![2.5], &[1], false).item(), 2.5);
        assert_eq!(RawTensor::new(vec![-4.0], &[1, 1], false).item(), -4.0);
    }

    /// `item` of a reduction is the reduced value.
    #[test]
    fn test_item_of_sum() {
        let t = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], false);
        assert_eq!(t.sum().item(), 6.0);
    }

    /// `item` on a multi-element tensor panics instead of returning the first element.
    #[test]
    #[should_panic(expected = "single-element tensor")]
    fn test_item_panics_on_multiple_elements() {
        let _ = RawTensor::new(vec![1.0, 2.0], &[2], false).item();
    }
}

#[cfg(all(test, feature = "gpu"))]
mod load_weight_tests {
    use super::*;
    use crate::io::TensorData;

    /// H7: loading into a GPU-resident tensor must keep it on the GPU with the new values.
    #[test]
    fn test_load_weight_preserves_gpu_storage() {
        if !crate::is_gpu_available() {
            eprintln!("SKIPPED: no GPU available");
            return;
        }
        let dev = Device::gpu().expect("GPU availability checked above");
        let t = RawTensor::new(vec![0.0; 6], &[2, 3], true).to_device(dev.clone());
        assert!(t.borrow().data.is_gpu(), "precondition: tensor on GPU");

        let vals = vec![1.5, -2.0, 3.25, 0.5, -0.75, 4.0];
        t.load_weight(
            "w",
            &TensorData {
                data: vals.clone(),
                shape: vec![2, 3],
            },
        )
        .expect("matching entry loads");

        assert!(t.borrow().data.is_gpu(), "load_weight demoted GPU to CPU");
        assert_eq!(t.borrow().device, dev);
        assert_eq!(t.data(), vals);
        assert!(t.requires_grad());
    }
}

#[cfg(test)]
mod load_weight_dtype_tests {
    use super::*;
    use crate::io::TensorData;

    /// `TensorData` is f32: loading it into an f64 tensor casts the values (exactly), as
    /// PyTorch's `load_state_dict` does, and keeps the tensor f64.
    #[test]
    fn load_weight_casts_to_the_destination_dtype() {
        let t = RawTensor::from_typed(vec![1.0f64, 2.0], &[2]);
        let src = TensorData {
            data: vec![0.1, 6.0],
            shape: vec![2],
        };
        t.load_weight("w", &src).expect("f64 destination");
        assert_eq!(t.dtype(), DType::F64);
        assert_eq!(
            t.to_vec_as::<f64>().expect("f64"),
            vec![f64::from(0.1f32), 6.0]
        );
    }

    /// An integer tensor is not a parameter: loading into one is an error.
    #[test]
    fn load_weight_refuses_an_integer_tensor() {
        let t = RawTensor::from_typed(vec![1i64, 2], &[2]);
        let src = TensorData {
            data: vec![5.0, 6.0],
            shape: vec![2],
        };
        let err = t.load_weight("w", &src).expect_err("i64 destination");
        assert!(matches!(err, VoltaError::DTypeMismatch { .. }), "{err}");
        assert_eq!(t.to_vec_as::<i64>().expect("i64"), vec![1, 2]);
    }
}

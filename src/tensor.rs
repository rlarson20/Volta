use crate::autograd::{CastGradFn, GradFn, ToDeviceGradFn, edge};
use crate::device::Device;
use crate::device_policy::{Pass, on_gap};
use crate::dtype::DType;
use crate::error::OrPanic;
use crate::index::{fetch, slot, span};
use crate::ops::binary::BinaryOp;
use crate::ops::compare::CmpOp;
use crate::ops::max_rule::dim_scan_replaces;
use crate::ops::reduce::{ReduceOp, pairwise_sum_by};
use crate::ops::ternary::TernaryOp;
use crate::ops::unary::UnaryOp;
use crate::real::{Real, compute_dtype, dispatch_real};
use crate::shape::{Axis, Shape};
/// The fill value of [`Tensor::pad_with`].
pub use crate::spec::PadValue;
use crate::storage::Element;
use crate::storage::Storage;
use crate::{Result, VoltaError};
use rand::RngExt;
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
/// Single-threaded by design (DECISIONS.md, "Stay synchronous").
#[derive(Clone)]
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

    /// This tensor's storage, for a `GradFn::backward` that builds its gradient with tensor
    /// ops (#268): a refcount bump, and the handle is dropped.
    pub(crate) fn into_grad_storage(self) -> crate::storage::Storage {
        self.inner.borrow().data.clone()
    }

    /// Whether `self` and `other` are handles to the same tensor (a `clone()` of a
    /// `Tensor` shares it), not merely equal values. A weight tied between two layers
    /// is the same tensor in both.
    #[must_use]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        std::rc::Rc::ptr_eq(&self.inner, &other.inner)
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

    /// Checks that the state-dict entry `src` can overwrite this tensor's values, without
    /// decoding it: the check [`crate::nn::LoadPlan`] makes on every entry before it
    /// writes any (#65).
    ///
    /// # Errors
    ///
    /// `VoltaError::DTypeMismatch` if this tensor is not a float dtype (only a float
    /// tensor can be a parameter), `VoltaError::StateShapeMismatch` if `src.shape`
    /// differs from this tensor's shape, and `VoltaError::StateDataLength` if `src.data`
    /// does not hold the bytes `src.shape` and `src.dtype` need (`InvalidParameter` for a
    /// bool byte that is not 0 or 1). `name` is the state-dict key, reported in the error.
    pub(crate) fn check_load(&self, name: &str, src: &crate::io::TensorData) -> Result<()> {
        let t = self.inner.borrow();
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
        // The byte count against `src`'s own shape and dtype, and bool bytes.
        src.validate(name)
    }

    /// Overwrite this tensor's values from a state-dict entry (H7).
    ///
    /// Keeps the tensor's shape, device, `requires_grad` and graph metadata: only the
    /// values change. A GPU-resident tensor stays on the GPU. The entry's values are cast
    /// from `src.dtype` to this tensor's dtype (following [`Storage::cast`]), as
    /// PyTorch's `load_state_dict` (`copy_`) does.
    ///
    /// # Errors
    ///
    /// Those of [`Self::check_load`], which runs first; nothing is written on error.
    pub(crate) fn load_weight(&self, name: &str, src: &crate::io::TensorData) -> Result<()> {
        self.check_load(name, src)?;
        let saved = src.to_storage(name)?;
        let mut t = self.inner.borrow_mut();
        // The explicit cast: the entry's dtype need not be the parameter's.
        let cpu = saved.cast(t.data.dtype());
        t.data = cpu.to_device(&t.device);
        t.bump_version();
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
        t.bump_version();
        let grad = t.grad.clone();
        drop(t);
        // Through `set_grad`, one of the sanctioned writers of `grad` (invariant 3).
        if let Some(g) = grad {
            self.set_grad(g.cast(dtype));
        }
    }

    /// Rewrites this tensor's values in place, outside the graph: an optimizer step, a
    /// running-statistics update. `f` gets the storage and may change its values but not
    /// its length, dtype or device; changing one of them panics after `f` returns. The
    /// version is bumped first (S9), so a backward pass that saved these values detects
    /// the write.
    ///
    /// `f` must not touch this tensor through another handle (a borrow of it would panic
    /// as a `RefCell` double borrow): read what it needs, such as the gradient, before.
    ///
    /// # Panics
    /// If `f` changes the storage's length, dtype or device kind.
    #[track_caller]
    pub(crate) fn update_data<R>(&self, f: impl FnOnce(&mut Storage) -> R) -> R {
        let mut t = self.inner.borrow_mut();
        t.bump_version();
        let (len, dtype, gpu) = (t.data.len(), t.data.dtype(), t.data.is_gpu());
        let out = f(&mut t.data);
        assert!(
            t.data.len() == len && t.data.dtype() == dtype && t.data.is_gpu() == gpu,
            "update_data: the update changed the storage from {len} {dtype:?} elements \
             (gpu: {gpu}) to {} {:?} (gpu: {}); it may change only values",
            t.data.len(),
            t.data.dtype(),
            t.data.is_gpu(),
        );
        out
    }

    /// Replaces this tensor's storage with `data`, keeping shape, `requires_grad` and
    /// graph metadata: a test setting known weights. Bumps the version (S9).
    ///
    /// # Panics
    /// If `data`'s length or dtype differs from the current storage's, or it is on the
    /// other side of the CPU/GPU split from this tensor's device.
    #[cfg(test)]
    #[track_caller]
    pub(crate) fn replace_data(&self, data: Storage) {
        self.update_data(|d| *d = data);
    }

    /// Moves this tensor's data (and its gradient) to `device` in place, keeping graph
    /// metadata: [`crate::nn::Module::to_device`]'s per-parameter step. Unlike
    /// [`Self::to_device`] it builds no new tensor. Bumps the version (S9).
    ///
    /// # Panics
    /// If `device` is a GPU and none is available ([`Storage::to_device`]).
    #[track_caller]
    pub(crate) fn move_to_device(&self, device: &Device) {
        {
            let mut t = self.inner.borrow_mut();
            t.data = t.data.to_device(device);
            t.bump_version();
            t.device = device.clone();
        }
        // `set_grad` moves an existing gradient onto the new device (invariant 3).
        if let Some(g) = self.grad_storage() {
            self.set_grad(g);
        }
    }

    /// Drops the host copies a GPU tensor's data and gradient cache; a no-op on the CPU.
    /// The public entry point is [`crate::gpu::invalidate_all_tensor_caches`].
    #[cfg(feature = "gpu")]
    pub(crate) fn invalidate_gpu_cache(&self) {
        self.inner.borrow_mut().invalidate_gpu_cache();
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
    /// Panics if called on a tensor that already has a `grad_fn` attached, with either
    /// value of `on` (#134). Flipping the flag on a non-leaf tensor would contradict the invariant
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
        // Both directions: turning it off on a non-leaf left a node with a `grad_fn` but
        // `requires_grad == false`, which cut the graph if set before use and was ignored by
        // `backward` if set after (#134). PyTorch raises in both directions too.
        assert!(
            t.grad_fn.is_none(),
            "set_requires_grad({on}) called on a non-leaf tensor: you can only change \
             requires_grad on a leaf; use detach() to get a tensor outside the graph"
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
                version: 0,
                saved_versions: vec![],
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
    /// If `grad`'s dtype or length differs from the tensor's (see [`Tensor::try_set_grad`]).
    #[track_caller]
    pub fn set_grad(&self, grad: Storage) {
        self.try_set_grad(grad).or_panic("set_grad");
    }

    /// Fallible [`Tensor::set_grad`]: checks `grad` against the tensor before moving it
    /// onto the tensor's device, and leaves the existing gradient untouched on error.
    ///
    /// # Errors
    /// - `VoltaError::DTypeMismatch` if `grad`'s dtype differs from the tensor's: a gradient
    ///   has its tensor's dtype.
    /// - `VoltaError::ShapeDataMismatch` if `grad` does not hold exactly one element per
    ///   tensor element. An optimizer reads the parameter and its gradient at one index, so a
    ///   short or long gradient would update only part of the parameter (#324).
    #[expect(
        clippy::needless_pass_by_value,
        reason = "takes ownership by design: the caller hands over the buffer it just built, and borrowing would make every caller clone"
    )]
    pub fn try_set_grad(&self, grad: Storage) -> Result<()> {
        let mut t = self.inner.borrow_mut();
        if grad.dtype() != t.data.dtype() {
            return Err(VoltaError::DTypeMismatch {
                expected: t.data.dtype(),
                actual: grad.dtype(),
            });
        }
        let elements = t.shape.numel();
        if grad.len() != elements {
            return Err(VoltaError::ShapeDataMismatch {
                shape: t.shape.truncate_rank().to_vec(),
                elements,
                len: grad.len(),
            });
        }
        let device = t.device.clone();
        t.grad = Some(grad.to_device(&device));
        Ok(())
    }

    // pub(crate) fn snapshot(&self) -> TensorSnapshot<'static> { /* owned clone of fields */ }
    // Deferred: no consumer yet and `TensorSnapshot` doesn't exist. Revisit alongside
    // the `load_state_dict` rewrite described in ADDENDUM.md §1.1.
}

// ===== RAW TENSOR STRUCTURE =====

/// The core tensor structure containing data and gradient tracking
///
/// This is wrapped in `Rc<RefCell<>>` to create the public `Tensor` type. The type is
/// public because constructors and ops are its associated functions
/// (`RawTensor::new(..) -> Tensor`); its fields are crate-private, and `Tensor`'s methods
/// (`shape()`, `data()`, `grad()`, `device()`, ...) read them from outside. Fields:
/// - `data`: a [`Storage`]: a typed host buffer of any [`DType`] (row-major), or a GPU
///   buffer (f32, f16, bf16 or bool) with an optional cached host copy
/// - `shape`: dimensions, e.g. [batch, channels, height, width]
/// - `grad`: accumulated gradient (Some if `requires_grad`, None otherwise)
/// - `requires_grad`: whether to track gradients for this tensor
/// - `grad_fn`: function to compute parent gradients during backward
/// - `parents`: input tensors that this tensor depends on
/// - `device`: where computation happens (CPU/GPU)
pub struct RawTensor {
    /// Element storage (CPU bytes or a GPU buffer).
    pub(crate) data: Storage,
    /// Dimensions, e.g. `[B, C, H, W]`.
    pub(crate) shape: Shape,
    /// Gradient with respect to `data`; `None` until one is accumulated.
    pub(crate) grad: Option<Storage>,
    /// Whether gradients are tracked for this tensor.
    pub(crate) requires_grad: bool,
    /// Backward function, present when this tensor is the result of an op.
    pub(crate) grad_fn: Option<GradFn>,
    /// Inputs this tensor was computed from (the graph's edges).
    pub(crate) parents: Vec<Tensor>,
    /// Where `data` lives.
    pub(crate) device: Device,
    /// Bumped by every in-place write to `data` (an optimizer step, `load_weight`, a dtype
    /// or device move), as PyTorch's version counter is. `edge()` records it for each input
    /// a backward pass reads, and backward panics if it has moved (S9).
    pub(crate) version: u64,
    /// For an op output, one entry per parent: the parent's `version` at forward time if
    /// this op's backward reads that parent's values, else `None`.
    pub(crate) saved_versions: Vec<Option<u64>>,
}

impl Drop for RawTensor {
    /// Drops the graph above this node iteratively (#91).
    ///
    /// The default drop recurses through `parents`, one stack frame per node, so the
    /// last handle to a long chain (a running loss total, a long RNN unroll) overflowed
    /// the stack and aborted the process. Here each parent that this node holds the
    /// last reference to hands its own parents to a work list before it drops, so every
    /// node drops with no parents left and the depth stays constant.
    fn drop(&mut self) {
        let mut pending = std::mem::take(&mut self.parents);
        while let Some(parent) = pending.pop() {
            // `Err` means another handle still holds the parent: dropping ours only
            // decrements the count, which does not recurse.
            if let Ok(cell) = Rc::try_unwrap(parent.inner) {
                pending.append(&mut cell.into_inner().parents);
            }
        }
    }
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
    F: FnOnce(&mut dyn rand::Rng) -> T,
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

impl RawTensor {
    /// Records an in-place write to `data`, so a backward pass that saved this tensor's
    /// values can tell they changed (S9). Every in-place writer calls it.
    pub(crate) const fn bump_version(&mut self) {
        self.version += 1;
    }
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
            version: self.version,
            saved_versions: self.saved_versions.clone(),
        }
    }
}

impl std::fmt::Debug for RawTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape.truncate_rank())
            .field("dtype", &self.data.dtype())
            .field("requires_grad", &self.requires_grad)
            .field("has_grad", &self.grad.is_some())
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

/// The tensor's shape, dtype, device and gradient state, not its values (#116): the
/// `RawTensor` behind the handle, without the `Rc<RefCell<..>>` wrapping.
impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&*self.inner.borrow(), f)
    }
}

/// Largest analytic-gradient magnitude that [`RawTensor::check_gradients`] still treats
/// as zero when it rejects an all-zero gradient (#461). A loss that is constant in its
/// input gets a backward of rounding noise (about 1e-8 in f32), while a real gradient
/// through a weighted loss is many orders larger.
const DEGENERATE_GRAD: f64 = 1e-6;

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
            version: 0,
            saved_versions: vec![],
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
        Self::try_new(data, shape, requires_grad).or_panic("new")
    }

    /// Create a new tensor from Storage and shape (internal use for GPU ops, fallible).
    ///
    /// Every leaf reaches here or [`Self::try_new_typed`], so both refuse `requires_grad`
    /// on a dtype that has no gradient (#458).
    ///
    /// # Errors
    /// `VoltaError::NotDifferentiable` if `requires_grad` is true for an integer or bool
    /// `data`, and `VoltaError::ShapeDataMismatch` if `data.len()` != `shape.numel()`.
    pub(crate) fn try_new_with_storage(
        data: crate::storage::Storage,
        shape: Shape,
        device: Device,
        requires_grad: bool,
    ) -> Result<Tensor> {
        if requires_grad && !data.dtype().is_float() {
            return Err(VoltaError::NotDifferentiable {
                dtype: data.dtype(),
            });
        }
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
            version: 0,
            saved_versions: vec![],
        };
        Ok(Tensor {
            inner: Rc::new(RefCell::new(raw)),
        })
    }

    /// Create a new tensor from Storage and shape (internal use for GPU ops)
    ///
    /// # Panics
    /// If `data.len()` != `shape.numel()`, if `requires_grad` is true for an integer or
    /// bool `data`, or if the storage isn't where `device` says it is (a device label is
    /// a promise; see `crate::device_policy`).
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
        Self::try_new_with_storage(data, shape, device, requires_grad).or_panic("new_with_storage")
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
        Self::try_zeros(shape).or_panic("zeros")
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
        Self::try_ones(shape).or_panic("ones")
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
        Self::try_rand(shape).or_panic("rand")
    }

    /// Create a tensor with values from standard normal distribution N(0, 1) (fallible)
    ///
    /// # Errors
    ///
    /// Same as [`Self::try_zeros`]
    pub fn try_randn(shape: &[usize]) -> Result<Tensor> {
        let size = Self::checked_numel(shape)?;
        let normal = Normal::new(0.0, 1.0).map_err(|e| {
            VoltaError::invalid(
                "randn",
                Some("shape"),
                format!("invalid normal distribution: {e}"),
            )
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
        Self::try_randn(shape).or_panic("randn")
    }

    /// Create a tensor filled with random values from N(0, 1) with the same shape, dtype
    /// and device as the input, as PyTorch's `randn_like` (#99). The values are drawn in
    /// f32 and cast, as `constant_like` does. The result does not require grad.
    ///
    /// # Panics
    /// If `tensor` has an integer or bool dtype (there is no normal sample of one; PyTorch
    /// raises too).
    #[must_use]
    #[track_caller]
    pub fn randn_like(tensor: &Tensor) -> Tensor {
        let dtype = tensor.dtype();
        assert!(
            dtype.is_float(),
            "randn_like: {dtype:?} is not a float dtype (normal samples need one)"
        );
        let shape = tensor.borrow().shape.clone();
        let r = Self::randn(shape.truncate_rank());
        let r = if dtype == DType::F32 {
            r
        } else {
            r.to_dtype(dtype)
        };
        r.to_device(tensor.device())
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
                return Err(VoltaError::invalid(
                    "xavier_uniform",
                    Some("shape"),
                    format!("expects a rank-1 or rank-2 shape, got {shape:?}"),
                ));
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
        Self::try_xavier_uniform(shape).or_panic("xavier_uniform")
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
            return Err(VoltaError::invalid(
                "he_initialization",
                Some("shape"),
                "requires at least one dimension".to_string(),
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
        let normal = Normal::new(0.0, std).map_err(|e| {
            VoltaError::invalid(
                "he_initialization",
                Some("shape"),
                format!("invalid std: {e}"),
            )
        })?;
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
        Self::try_he_initialization(shape).or_panic("he_initialization")
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
    /// For CPU tensors, this is a no-op. The public entry point is
    /// [`crate::gpu::invalidate_all_tensor_caches`].
    pub(crate) fn invalidate_gpu_cache(&mut self) {
        self.data.invalidate_cpu_cache();
        if let Some(ref mut grad) = self.grad {
            grad.invalidate_cpu_cache();
        }
    }
}

// ===== LOSS FUNCTIONS =====
impl RawTensor {
    /// Panics unless `a` and `b` have the same shape. The elementwise losses never
    /// broadcast: `[N, 1]` against `[N]` would give an `[N, N]` matrix whose mean is
    /// returned as the loss (#95).
    #[track_caller]
    fn assert_same_shape(op: &str, a: (&str, &Tensor), b: (&str, &Tensor)) {
        let (sa, sb) = (a.1.shape(), b.1.shape());
        assert!(
            sa == sb,
            "{op}: {} and {} must have the same shape (no broadcasting), got {:?} and {:?}",
            a.0,
            b.0,
            sa.truncate_rank(),
            sb.truncate_rank()
        );
    }

    /// Mean squared error: `mean((pred - target)^2)`.
    ///
    /// # Panics
    /// If `pred` and `target` differ in shape. They are never broadcast; PyTorch only
    /// warns, and a silent `[N, N]` mean is the wrong number (#95).
    #[must_use]
    #[track_caller]
    pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Tensor {
        // An f32 op under `amp::autocast` (PyTorch's losses): half inputs are cast up.
        let (pred, target) = (&crate::amp::to_f32(pred), &crate::amp::to_f32(target));
        Self::assert_same_shape("mse_loss", ("pred", pred), ("target", target));
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
    /// `ignore_index`), or float targets are not exactly `[N, C]`, the logits' shape. Float
    /// targets are never broadcast, so float class indices `[N]` are rejected rather than
    /// read as probabilities; class indices must be an I64 or I32 tensor.
    #[must_use]
    #[track_caller]
    pub fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Tensor {
        // An f32 op under `amp::autocast` (PyTorch's losses): half inputs are cast up.
        let (logits, targets) = (&crate::amp::to_f32(logits), &crate::amp::to_f32(targets));
        let rank = logits.borrow().shape.rank();
        assert!(
            rank == 2,
            "cross_entropy_loss: logits must be [N, C], got rank {rank} ({:?})",
            logits.borrow().shape.truncate_rank()
        );
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
    /// computes it (so `0 * -inf` is NaN there too). Class-index targets gather
    /// `log_probs[i, t_i]` instead of multiplying by a one-hot, as PyTorch does: a `-inf`
    /// log-probability on another class (a `-inf` logit) is never touched, so it can't
    /// turn the loss into NaN (S8), and the gradient flows to the picked entries only.
    /// `op` names the caller in panics.
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
            // Checked here, before `gather`, so the error names the loss and the class count.
            for &t in &classes {
                assert!(
                    usize::try_from(t).is_ok_and(|k| k < c),
                    "{op}: target {t} out of range for {c} classes"
                );
            }
            // `log_probs[i, t_i]` for each row: `[N, 1]`, O(N) where a one-hot mask was
            // O(N·C) (#78). The index stays on the host, as `gather` reads it.
            let index = Self::from_typed(classes, &[n, 1]);
            let picked = log_probs.gather(Axis(1), &index);
            return picked.neg().mean();
        } else {
            // Exact match, no broadcast: float class indices `[N]` broadcast against
            // `[N, C]` whenever N == C or N == 1 and give a number that is no cross entropy.
            let (ts, lp) = (targets.shape(), log_probs.shape());
            assert!(
                ts == lp,
                "{op}: float targets must match the logits' shape {:?}, got {:?} \
                 (class indices must be an I64/I32 tensor)",
                lp.truncate_rank(),
                ts.truncate_rank()
            );
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
    ///
    /// # Panics
    /// If `mu` and `logvar` differ in shape (they are never broadcast).
    #[must_use]
    #[track_caller]
    pub fn kl_divergence_gaussian(mu: &Tensor, logvar: &Tensor) -> Tensor {
        // An f32 op under `amp::autocast` (PyTorch's losses): half inputs are cast up.
        let (mu, logvar) = (&crate::amp::to_f32(mu), &crate::amp::to_f32(logvar));
        Self::assert_same_shape("kl_divergence_gaussian", ("mu", mu), ("logvar", logvar));
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
    /// The tail term is `ln(1 + e)`, not `log1p(e)` (there is no `log1p` op yet; #348), so
    /// it is rounded to the f32 spacing of 1 (`1.19e-7`): from about `|x| = 15.5` it reads
    /// `1.19e-7`, and from about `|x| = 16.6` (`exp(-|x|) < 2^-24`) exactly 0. The absolute
    /// error per element stays below about `6e-8`, and the gradient is not affected
    /// (`-3.059022e-7` at `x = 15`, `y = 1`, the exact value to f32). What changes is the
    /// loss value of a confident prediction: `3.576e-7` at `x = 15` (PyTorch `3.059e-7`)
    /// and exactly 0 from `x = 17` (PyTorch `4.14e-8`).
    ///
    /// # Arguments
    /// * `logits` - Raw network outputs (before sigmoid)
    /// * `target` - Target labels (0 or 1)
    ///
    /// # Panics
    /// If `logits` and `target` differ in shape, as in PyTorch (no broadcasting).
    #[must_use]
    #[track_caller]
    pub fn bce_with_logits_loss(logits: &Tensor, target: &Tensor) -> Tensor {
        // An f32 op under `amp::autocast` (PyTorch's losses): half inputs are cast up.
        let (logits, target) = (&crate::amp::to_f32(logits), &crate::amp::to_f32(target));
        Self::assert_same_shape(
            "bce_with_logits_loss",
            ("logits", logits),
            ("target", target),
        );
        let shape = logits.borrow().shape.clone();
        let zeros = logits.constant_like(vec![0.0; shape.numel()], shape.truncate_rank());
        let one = logits.constant_like(vec![1.0; shape.numel()], shape.truncate_rank());

        // max(x, 0); `max_elem` splits a tie in half, so at x == 0 this passes 1/2.
        let pos_part = logits.max_elem(&zeros);
        // |x| = max(x, -x); at x == 0 the halves cancel, so |x| passes 0 there and the
        // gradient at 0 is 1/2 - y = sigmoid(0) - y.
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
    /// Whether the sum was divided by the axis length (`mean_dim`).
    mean: bool,
}

impl SumDimGradFn {
    /// The op's name: `MeanDim` for `mean_dim`, else `SumDim`.
    pub(crate) fn name(&self) -> &'static str {
        if self.mean { "MeanDim" } else { "SumDim" }
    }

    /// Broadcasts `out_grad` back along the summed axis.
    pub(crate) fn backward(
        &self,
        out_grad: &RawTensor,
        _parents: &[Tensor],
    ) -> Vec<Option<Storage>> {
        dispatch_real!(out_grad.data.dtype(), T => self.backward_typed::<T>(out_grad))
    }

    /// [`Self::backward`] in the gradient's dtype `T`.
    fn backward_typed<T: Real>(&self, out_grad: &RawTensor) -> Vec<Option<Storage>> {
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
        let divisor = T::from_usize(if self.mean {
            self.input_shape.dim_at(self.dim)
        } else {
            1
        });

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
            *slot(&mut result, i) = fetch(&grad_data, grad_idx) / divisor;
        }

        vec![Some(Storage::from_compute(out_grad.data.dtype(), result))]
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
    ) -> Vec<Option<Storage>> {
        dispatch_real!(out_grad.data.dtype(), T => self.backward_typed::<T>(out_grad))
    }

    /// [`Self::backward`] in the gradient's dtype `T`.
    fn backward_typed<T: Real>(&self, out_grad: &RawTensor) -> Vec<Option<Storage>> {
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

        vec![Some(Storage::from_compute(out_grad.data.dtype(), result))]
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
        Self::reduce_dim(self_t, dim, keepdim, false)
    }

    /// `sum_dim`, or with `mean` the sum divided by the axis length, both in one pass:
    /// each output is computed in the compute type (f32 for a half tensor) and rounded to
    /// the input's dtype once, so a half mean does not overflow where its sum would (#316).
    fn reduce_dim(self_t: &Tensor, dim: Axis, keepdim: bool, mean: bool) -> Result<Tensor> {
        let op = if mean { "MeanDim" } else { "SumDim" };
        let dtype = compute_dtype(|| op.to_string(), &[self_t.dtype()]);
        let (data, shape) = {
            let s = self_t.borrow();
            (s.data.clone(), s.shape.clone())
        };

        let dim_size = shape.dim(dim)?;
        let divisor = if mean { dim_size } else { 1 };
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
            let divisor = T::from_usize(divisor);
            for outer in 0..outer_size {
                for inner in 0..inner_size {
                    let out_idx = outer * inner_size + inner;
                    // Pairwise, not a running sum: a long axis would stall (S6).
                    let sum = pairwise_sum_by(dim_size, |k| {
                        fetch(&data, outer * dim_size * inner_size + k * inner_size + inner)
                    });
                    *slot(&mut result, out_idx) = sum / divisor;
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
                mean,
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
    ///
    /// # Panics
    /// If `dim` is not less than the tensor's rank.
    #[must_use]
    #[track_caller]
    pub fn sum_dim(self_t: &Tensor, dim: Axis, keepdim: bool) -> Tensor {
        Self::try_sum_dim(self_t, dim, keepdim).or_panic("sum_dim")
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
    /// As [`Self::try_to_device`] errs: if `device` is a GPU and no GPU is available, it
    /// names a GPU other than this machine's adapter (`Device::gpu()`), the tensor's dtype
    /// can't live on the GPU, or the tensor is larger than the GPU can bind.
    #[must_use]
    #[track_caller]
    pub fn to_device(self_t: &Tensor, device: Device) -> Tensor {
        Self::try_to_device(self_t, device).or_panic("")
    }

    /// [`Self::to_device`], with an error in place of each panic.
    ///
    /// # Errors
    /// [`VoltaError::DeviceError`] if `device` is a GPU and no GPU is available, or it
    /// names a GPU other than this machine's adapter, or as [`Storage::try_to_device`]: the
    /// dtype is F64 or an integer type, or the tensor is over the GPU's storage binding
    /// limit ([`crate::gpu::max_tensor_bytes`], #306). Nothing is allocated first.
    pub fn try_to_device(self_t: &Tensor, device: Device) -> Result<Tensor> {
        // Fast path: already on requested device
        {
            let t = self_t.borrow();
            if t.device == device {
                return Ok(self_t.clone());
            }
        }

        // A GPU that isn't this machine's adapter is an error, like `cuda:7` on a one-GPU
        // machine, so device equality can compare names (multi-GPU-ready).
        if device.is_gpu()
            && let Some(real) = Device::gpu()
            && real != device
        {
            return Err(VoltaError::device(
                "to_device",
                Some("device"),
                format!("no such GPU {device}; the available GPU is {real}"),
            ));
        }

        let (data, shape, source) = {
            let t = self_t.borrow();
            (
                t.data.try_to_device(&device)?,
                t.shape.clone(),
                t.device.clone(),
            )
        };
        Ok(edge(
            [self_t.clone()],
            data,
            shape,
            device,
            GradFn::ToDevice(ToDeviceGradFn { source }),
        ))
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
    /// Returns maximum value along dimension and stores indices for backward pass. Like
    /// `torch.max(x, dim)`, a tie goes to the first index, a NaN in the row makes the result
    /// NaN (the first NaN wins), and all of the gradient goes to that one element.
    ///
    /// # Panics
    /// If `dim` is not less than the tensor's rank, or the dtype is not a float.
    #[must_use]
    #[track_caller]
    pub fn max_dim(self_t: &Tensor, dim: Axis, keepdim: bool) -> Tensor {
        let dtype = compute_dtype(|| "MaxDim".to_string(), &[self_t.dtype()]);
        let (data, shape) = {
            let s = self_t.borrow();
            (s.data.clone(), s.shape.clone())
        };

        let dim_size = shape.dim(dim).or_panic("max_dim");
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
            let mut result = vec![T::ZERO; out_size];
            let data = data.host::<T>();
            for outer in 0..outer_size {
                for inner in 0..inner_size {
                    let out_idx = outer * inner_size + inner;
                    // Each row starts from its own first element, so a row with no
                    // winner (all -inf) keeps its gradient in that row (S2).
                    let first = outer * dim_size * inner_size + inner;
                    let (mut best, mut best_idx) = (fetch(&data, first), first);
                    for k in 1..dim_size {
                        let in_idx = outer * dim_size * inner_size + k * inner_size + inner;
                        let in_val = fetch(&data, in_idx);
                        if dim_scan_replaces(in_val, best) {
                            (best, best_idx) = (in_val, in_idx);
                        }
                    }
                    *slot(&mut result, out_idx) = best;
                    *slot(&mut max_indices, out_idx) = best_idx;
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
    ///
    /// # Panics
    /// If `dim` is not less than the tensor's rank, or the dtype is not a float (as
    /// [`Self::max_dim`]).
    #[must_use]
    pub fn softmax(self_t: &Tensor, dim: Axis) -> Tensor {
        // Checked under this op's name; `max_dim` would report its own (#142).
        let _ = self_t.borrow().shape.dim(dim).or_panic("softmax");
        let max = Self::max_dim(self_t, dim, true);
        let shifted = self_t.sub(&max);
        let exp_x = shifted.exp();
        let sum_exp = Self::sum_dim(&exp_x, dim, true);
        exp_x.div(&sum_exp)
    }
    /// `LogSoftmax` along a specific axis (numerically stable)
    /// `log(exp(x_i)` / `sum(exp(x_j))`) = `x_i` - `log(sum(exp(x_j)))`
    /// Uses `LogSumExp` trick: log(sum(exp(x))) = m + log(sum(exp(x-m)))
    ///
    /// # Panics
    /// If `dim` is not less than the tensor's rank, or the dtype is not a float (as
    /// [`Self::max_dim`]).
    #[must_use]
    pub fn log_softmax(self_t: &Tensor, dim: Axis) -> Tensor {
        // Checked under this op's name; `max_dim` would report its own (#142).
        let _ = self_t.borrow().shape.dim(dim).or_panic("log_softmax");
        let max = Self::max_dim(self_t, dim, true);
        let shifted = self_t.sub(&max);
        let exp_x = shifted.exp();
        let sum_exp = Self::sum_dim(&exp_x, dim, true);
        let log_sum = sum_exp.log();
        // `shifted - log_sum`, not `x - (log_sum + max)`: adding `max` back rounds `log_sum`
        // to the spacing of `max`, an error up to half an ulp of the largest logit (#315).
        shifted.sub(&log_sum)
    }

    /// Mean along a specific axis.
    ///
    /// One op, as `mean` is: each mean is the pairwise sum divided by the axis length
    /// before it is rounded, so a half tensor's mean does not overflow where its sum
    /// would (#316). The gradient is `out_grad / n`, broadcast along the axis.
    ///
    /// # Panics
    /// If `dim` is not less than the tensor's rank, or the dtype is not a float.
    #[must_use]
    #[track_caller]
    pub fn mean_dim(self_t: &Tensor, dim: Axis, keepdim: bool) -> Tensor {
        Self::reduce_dim(self_t, dim, keepdim, true).or_panic("mean_dim")
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
    /// # Devices
    /// A GPU tensor is checked on the GPU: each perturbed input is built from a host copy
    /// and moved to `tensor`'s device, so `loss_fn` runs the GPU forward and backward
    /// kernels, and the gradient and losses are read back through the host.
    ///
    /// # Non-finite gradients
    /// A NaN or infinite relative error at any element (from a NaN or infinite
    /// analytic or numerical gradient, as when `loss_fn` evaluates an op outside its
    /// domain) fails the check: `passed` is `false` and `max_error` is that NaN or
    /// infinity, not the largest finite error.
    ///
    /// # Degenerate losses
    /// An analytic gradient whose every element is at most 1e-6 in magnitude fails the
    /// check (#461). A loss that doesn't depend on `tensor`, such as `softmax(x).sum()`
    /// (each row sums to 1) or a `BatchNorm` output's `.sum()`, has that gradient for
    /// any backward, right or wrong, so a pass would prove nothing. Weight the output
    /// with a fixed random tensor instead: `(y * w).sum()`.
    ///
    /// # Panics
    /// If `tensor` is f16 or bf16: a finite-difference step can't be resolved at half
    /// precision, so the check would fail correct gradients. Cast the inputs to f32 or
    /// f64 first (torch's `gradcheck` warns about non-double inputs; Volta refuses).
    /// Also if `loss_fn` doesn't return a scalar that requires grad, or `tensor` ends
    /// up without a gradient.
    pub fn check_gradients<F>(
        tensor: &Tensor,
        loss_fn: F,
        epsilon: f32,
        tolerance: f32,
    ) -> (f32, f32, bool)
    where
        F: Fn(&Tensor) -> Tensor,
    {
        let input_dtype = tensor.dtype();
        assert!(
            !matches!(input_dtype, DType::F16 | DType::BF16),
            "check_gradients: input is {input_dtype:?}, where a finite difference is \
             meaningless (the step can't be resolved at half precision); cast the inputs \
             to f32 or f64 first"
        );

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
        // f64-accurate check. A GPU tensor's values are read through a host copy (GPU
        // storage holds no f64), and each perturbed input goes back to its device, so
        // `loss_fn` runs the GPU kernels (#124).
        let analytical_grad: Vec<f64> = tensor
            .grad_storage()
            .expect("Tensor must have gradient")
            .to_device(&Device::CPU)
            .cast(DType::F64)
            .as_host::<f64>()
            .expect("cast to f64")
            .to_vec();
        let mut numerical_grad = vec![0.0f64; analytical_grad.len()];

        let device = tensor.device();
        let original_data = tensor.borrow().data.to_device(&Device::CPU);
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
                    .to_device(&Device::CPU)
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
                    data.with_host_mut::<T, _>(|host| {
                        let x = slot(host, i);
                        if up {
                            *x += step;
                        } else {
                            *x -= step;
                        }
                    });
                    Self::new_with_storage(
                        data.to_device(&device),
                        original_shape.clone(),
                        device.clone(),
                        requires_grad,
                    )
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

        // Compute errors (in f64; reported as f32, the signature's type). `f64::max`
        // drops a NaN operand, so a non-finite error is tracked on its own: it fails the
        // check and is what `max_error` reports.
        let mut max_error: f64 = 0.0;
        let mut total_error: f64 = 0.0;
        let mut non_finite: Option<f64> = None;

        for (i, (&analytical, &numerical)) in
            analytical_grad.iter().zip(&numerical_grad).enumerate()
        {
            let error = (analytical - numerical).abs();
            // When both gradients are extremely small, pure relative error becomes noisy
            // and can falsely flag otherwise correct gradients (e.g., BatchNorm sum outputs
            // that should be exactly zero). In that regime, we compare absolute error instead.
            let scale = analytical.abs().max(numerical.abs());
            let relative_error = if scale > 1e-3 { error / scale } else { error };

            if relative_error.is_finite() {
                max_error = max_error.max(relative_error);
            } else if !non_finite.is_some_and(f64::is_nan) {
                // Keep the first NaN; otherwise the infinity.
                non_finite = Some(relative_error);
            }
            total_error += relative_error;

            if !relative_error.is_finite() || relative_error > f64::from(tolerance) {
                log::debug!(
                    "Gradient mismatch at index {i}: analytical={analytical:.6e}, numerical={numerical:.6e}, error={relative_error:.6e}",
                );
            }
        }

        let mean_error = total_error / analytical_grad.len() as f64;
        let max_error = non_finite.unwrap_or(max_error);
        // An all-zero analytic gradient agrees with any backward on a loss that is
        // constant in `tensor` (`softmax(x).sum()`, a normalized output's `.sum()`), so
        // it proves nothing and fails the check (#461). "Zero" allows rounding: such a
        // loss's gradient comes out around 1e-8 in f32, not exactly 0.
        let all_zero = analytical_grad.iter().all(|&g| g.abs() <= DEGENERATE_GRAD);
        if all_zero {
            log::warn!(
                "check_gradients: the analytic gradient is all zeros, so the check proves \
                 nothing; weight the loss, e.g. (y * w).sum() with a fixed random w"
            );
        }
        let passed = non_finite.is_none() && !all_zero && max_error < f64::from(tolerance);

        #[expect(
            clippy::cast_possible_truncation,
            reason = "the errors are reported in the signature's f32"
        )]
        (max_error as f32, mean_error as f32, passed)
    }

    /// Simplified gradient checker with default parameters
    ///
    /// Uses epsilon=1e-2 and tolerance=1e-3, which work well for most cases. A NaN or
    /// infinite gradient fails it, as in [`RawTensor::check_gradients`].
    ///
    /// # Panics
    /// As [`RawTensor::check_gradients`], including on an f16 or bf16 `tensor`.
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
    ///
    /// # Panics
    /// As [`RawTensor::binary_op`]: if the shapes don't broadcast, the operands are on
    /// different devices, the promoted dtype is not a float, or the dtypes differ under
    /// [`DTypePolicy::Strict`](crate::DTypePolicy::Strict).
    #[must_use]
    #[track_caller]
    pub fn add(&self, other: &Self) -> Self {
        RawTensor::binary_op(self, other, BinaryOp::Add)
    }

    /// Elementwise difference, with broadcasting.
    ///
    /// # Panics
    /// As [`RawTensor::binary_op`]: if the shapes don't broadcast, the operands are on
    /// different devices, the promoted dtype is not a float, or the dtypes differ under
    /// [`DTypePolicy::Strict`](crate::DTypePolicy::Strict).
    #[must_use]
    #[track_caller]
    pub fn sub(&self, other: &Self) -> Self {
        RawTensor::binary_op(self, other, BinaryOp::Sub)
    }

    /// Elementwise product, with broadcasting.
    ///
    /// # Panics
    /// As [`RawTensor::binary_op`]: if the shapes don't broadcast, the operands are on
    /// different devices, the promoted dtype is not a float, or the dtypes differ under
    /// [`DTypePolicy::Strict`](crate::DTypePolicy::Strict).
    #[must_use]
    #[track_caller]
    pub fn elem_mul(&self, other: &Self) -> Self {
        RawTensor::binary_op(self, other, BinaryOp::Mul)
    }

    /// Elementwise quotient, with broadcasting.
    ///
    /// # Panics
    /// As [`RawTensor::binary_op`]: if the shapes don't broadcast, the operands are on
    /// different devices, the promoted dtype is not a float, or the dtypes differ under
    /// [`DTypePolicy::Strict`](crate::DTypePolicy::Strict).
    #[must_use]
    #[track_caller]
    pub fn div(&self, other: &Self) -> Self {
        RawTensor::binary_op(self, other, BinaryOp::Div)
    }

    /// Elementwise maximum, with broadcasting, as `torch.maximum`: NaN if either input is
    /// NaN. A tie splits the gradient in half, and a NaN input gives both sides all of it.
    ///
    /// # Panics
    /// As [`RawTensor::binary_op`]: if the shapes don't broadcast, the operands are on
    /// different devices, the promoted dtype is not a float, or the dtypes differ under
    /// [`DTypePolicy::Strict`](crate::DTypePolicy::Strict).
    #[must_use]
    #[track_caller]
    pub fn max_elem(&self, other: &Self) -> Self {
        RawTensor::binary_op(self, other, BinaryOp::Max)
    }

    /// Elementwise remainder, with broadcasting.
    ///
    /// # Panics
    /// As [`RawTensor::binary_op`]: if the shapes don't broadcast, the operands are on
    /// different devices, the promoted dtype is not a float, or the dtypes differ under
    /// [`DTypePolicy::Strict`](crate::DTypePolicy::Strict).
    #[must_use]
    #[track_caller]
    pub fn modulo(&self, other: &Self) -> Self {
        RawTensor::binary_op(self, other, BinaryOp::Mod)
    }

    /// Elementwise `self < other` as a Bool tensor, with broadcasting (`RawTensor::compare`).
    ///
    /// # Panics
    /// As [`RawTensor::compare`]: if the shapes don't broadcast, the operands are on
    /// different devices, or the dtypes differ under [`DTypePolicy::Strict`](crate::DTypePolicy::Strict).
    #[must_use]
    #[track_caller]
    pub fn lt(&self, other: &Self) -> Self {
        RawTensor::compare(self, other, CmpOp::Lt)
    }

    /// Elementwise `self <= other` as a Bool tensor, with broadcasting.
    ///
    /// # Panics
    /// As [`RawTensor::compare`]: if the shapes don't broadcast, the operands are on
    /// different devices, or the dtypes differ under [`DTypePolicy::Strict`](crate::DTypePolicy::Strict).
    #[must_use]
    #[track_caller]
    pub fn le(&self, other: &Self) -> Self {
        RawTensor::compare(self, other, CmpOp::Le)
    }

    /// Elementwise `self > other` as a Bool tensor, with broadcasting.
    ///
    /// # Panics
    /// As [`RawTensor::compare`]: if the shapes don't broadcast, the operands are on
    /// different devices, or the dtypes differ under [`DTypePolicy::Strict`](crate::DTypePolicy::Strict).
    #[must_use]
    #[track_caller]
    pub fn gt(&self, other: &Self) -> Self {
        RawTensor::compare(self, other, CmpOp::Gt)
    }

    /// Elementwise `self >= other` as a Bool tensor, with broadcasting.
    ///
    /// # Panics
    /// As [`RawTensor::compare`]: if the shapes don't broadcast, the operands are on
    /// different devices, or the dtypes differ under [`DTypePolicy::Strict`](crate::DTypePolicy::Strict).
    #[must_use]
    #[track_caller]
    pub fn ge(&self, other: &Self) -> Self {
        RawTensor::compare(self, other, CmpOp::Ge)
    }

    /// Elementwise `self == other` as a Bool tensor, with broadcasting (PyTorch's `eq`;
    /// named like `elem_mul` so it can't be mistaken for `PartialEq::eq`).
    ///
    /// # Panics
    /// As [`RawTensor::compare`]: if the shapes don't broadcast, the operands are on
    /// different devices, or the dtypes differ under [`DTypePolicy::Strict`](crate::DTypePolicy::Strict).
    #[must_use]
    #[track_caller]
    pub fn elem_eq(&self, other: &Self) -> Self {
        RawTensor::compare(self, other, CmpOp::Eq)
    }

    /// Elementwise `self != other` as a Bool tensor, with broadcasting.
    ///
    /// # Panics
    /// As [`RawTensor::compare`]: if the shapes don't broadcast, the operands are on
    /// different devices, or the dtypes differ under [`DTypePolicy::Strict`](crate::DTypePolicy::Strict).
    #[must_use]
    #[track_caller]
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

    /// Rectified linear unit, elementwise. `relu(NaN)` is NaN, as in torch, and has no
    /// gradient.
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
    #[track_caller]
    pub fn sum(&self) -> Self {
        RawTensor::reduce_op(self, ReduceOp::Sum)
    }

    /// Maximum of all elements, as a scalar, as `torch.max()`: NaN if any element is NaN.
    /// The gradient is split evenly over every element equal to the maximum (every NaN,
    /// when the maximum is NaN).
    #[must_use]
    #[track_caller]
    pub fn max_reduce(&self) -> Self {
        RawTensor::reduce_op(self, ReduceOp::Max)
    }

    /// Mean of all elements, as a scalar.
    #[must_use]
    #[track_caller]
    pub fn mean(&self) -> Self {
        RawTensor::reduce_op(self, ReduceOp::Mean)
    }

    /// Fused `self * y + z`, on equal shapes (no broadcasting).
    ///
    /// # Panics
    /// As [`RawTensor::ternary_op`]: if the three shapes differ, the operands are on
    /// different devices, or the promoted dtype is not a float.
    #[must_use]
    #[track_caller]
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
    #[track_caller]
    pub fn where_op(&self, x: &Self, y: &Self) -> Self {
        RawTensor::ternary_op(self, x, y, TernaryOp::Where)
    }

    /// Values along `axis` picked by `index`, as `torch.gather(self, axis, index)`.
    ///
    /// `index` is an I64 or I32 tensor of the same rank as `self`, read on the host (the
    /// GPU holds no integer dtype, so a GPU input takes a CPU index).
    /// The output has `index`'s shape and `self`'s dtype, and
    /// `out[.., j, ..] = self[.., index[.., j, ..], ..]` with `j` on `axis`: for a
    /// `[N, C]` input and `axis = 1`, `out[i][j] = self[i][index[i][j]]`. On every other
    /// axis `index` may be shorter than `self` (it reads a prefix), never longer. Negative
    /// indices are refused, as in PyTorch.
    ///
    /// The gradient flows to the elements that were read, summed where one is read more
    /// than once; the indices carry none. There is no GPU kernel: GPU inputs are a
    /// device-policy gap (`DevicePolicy`).
    ///
    /// # Errors
    /// - [`VoltaError::DimensionOutOfBounds`] if `axis` is not below `self`'s rank.
    /// - [`VoltaError::DTypeMismatch`] if `index` is not I64 or I32.
    /// - [`VoltaError::InvalidParameter`] if `index` has another rank, is longer than
    ///   `self` on an axis other than `axis`, or holds a value outside `0..self.shape[axis]`.
    pub fn try_gather(&self, axis: Axis, index: &Self) -> Result<Self> {
        RawTensor::try_gather_op(self, axis, index)
    }

    /// [`Self::try_gather`], panicking on an error.
    ///
    /// # Panics
    /// Wherever [`Self::try_gather`] returns an error.
    #[must_use]
    #[track_caller]
    pub fn gather(&self, axis: Axis, index: &Self) -> Self {
        self.try_gather(axis, index).or_panic("gather")
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
    ///
    /// # Panics
    /// As [`RawTensor::permute`]: if `axes` is not a permutation of `0..rank`.
    #[must_use]
    #[track_caller]
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
    ///
    /// # Panics
    /// As [`RawTensor::pad_with`]: if `padding` does not have one entry per dimension, or
    /// the padded shape is invalid. Use [`RawTensor::try_pad_with`] to get an error instead.
    #[must_use]
    #[track_caller]
    pub fn pad_with(&self, padding: &[(usize, usize)], value: PadValue) -> Self {
        RawTensor::pad_with(self, padding, value)
    }

    /// Slices each dimension to the half-open `(start, end)` range.
    ///
    /// # Panics
    /// As [`RawTensor::shrink`]: if `ranges` does not have one entry per dimension, or a
    /// range is empty, reversed, or past the dimension's end.
    #[must_use]
    #[track_caller]
    pub fn shrink(&self, ranges: &[(usize, usize)]) -> Self {
        RawTensor::shrink(self, ranges)
    }

    /// Takes every `strides[i]`-th element along each dimension.
    ///
    /// # Panics
    /// As [`RawTensor::stride_op`]: if `strides` does not have one entry per dimension, or
    /// a stride is 0. Use [`RawTensor::try_stride_op`] to get an error instead.
    #[must_use]
    #[track_caller]
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
    /// As [`Self::try_to_device`] errs: if `device` is a GPU and no GPU is available, it
    /// names a GPU other than this machine's adapter (`Device::gpu()`), the dtype can't
    /// live on the GPU, or the tensor is larger than the GPU can bind.
    #[must_use]
    #[track_caller]
    pub fn to_device(&self, device: Device) -> Self {
        RawTensor::to_device(self, device)
    }

    /// [`Self::to_device`], with an error in place of each panic.
    ///
    /// # Errors
    /// As [`RawTensor::try_to_device`]: no GPU, another GPU's name, a dtype the GPU can't
    /// hold, or a tensor over the GPU's storage binding limit (#306).
    pub fn try_to_device(&self, device: Device) -> Result<Self> {
        RawTensor::try_to_device(self, device)
    }

    /// Matrix product.
    ///
    /// # Panics
    /// As [`RawTensor::matmul`]: on an inner-dimension mismatch, matrix operands of unequal
    /// rank, an output over [`crate::shape::MAX_NUMEL`] elements, or dtypes that differ or
    /// are not a float.
    #[must_use]
    #[track_caller]
    pub fn matmul(&self, other: &Self) -> Self {
        RawTensor::matmul(self, other)
    }

    /// Transposes a 2-D tensor.
    ///
    /// # Panics
    /// As [`RawTensor::transpose`]: if the tensor is not 2-D.
    #[must_use]
    #[track_caller]
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
    ///
    /// # Panics
    /// As [`RawTensor::sum_dim`]: if `dim` is not less than the rank. Use
    /// [`RawTensor::try_sum_dim`] to get an error instead.
    #[must_use]
    #[track_caller]
    pub fn sum_dim(&self, dim: Axis, keepdim: bool) -> Self {
        RawTensor::sum_dim(self, dim, keepdim)
    }

    /// Maximum along `dim`, as `torch.max(x, dim)`: a NaN in a lane makes it NaN, and the
    /// whole gradient goes to one element per lane (the first on a tie, the first NaN).
    ///
    /// # Panics
    /// As [`RawTensor::max_dim`]: if `dim` is not less than the rank, or the dtype is not
    /// a float.
    #[must_use]
    #[track_caller]
    pub fn max_dim(&self, dim: Axis, keepdim: bool) -> Self {
        RawTensor::max_dim(self, dim, keepdim)
    }

    /// Mean along `dim`.
    ///
    /// # Panics
    /// As [`RawTensor::mean_dim`]: if `dim` is not less than the rank, or the dtype is not
    /// a float.
    #[must_use]
    #[track_caller]
    pub fn mean_dim(&self, dim: Axis, keepdim: bool) -> Self {
        RawTensor::mean_dim(self, dim, keepdim)
    }

    /// Softmax along `dim`.
    ///
    /// # Panics
    /// As [`RawTensor::softmax`].
    #[must_use]
    pub fn softmax(&self, dim: Axis) -> Self {
        RawTensor::softmax(self, dim)
    }

    /// Log-softmax along `dim`.
    ///
    /// # Panics
    /// As [`RawTensor::log_softmax`].
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
    /// If set, `next` moves each batch to this device before returning it.
    device: Option<crate::device::Device>,
}

/// The shapes, sizes and position, not the samples.
impl std::fmt::Debug for DataLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DataLoader")
            .field("samples", &self.indices.len())
            .field("data_shape", &self.data_shape)
            .field("target_shape", &self.target_shape)
            .field("batch_size", &self.batch_size)
            .field("shuffle", &self.shuffle)
            .field("current", &self.current)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

impl DataLoader {
    /// Builds a loader over `data` and `targets`, whose sample count is `data.len()` divided by the elements of `data_shape`. Shuffles once if `shuffle`.
    ///
    /// # Panics
    /// Wherever [`DataLoader::try_new`] returns an error.
    #[must_use]
    #[track_caller]
    pub fn new(
        data: Vec<f32>,
        targets: Vec<f32>,
        data_shape: &[usize],   // e.g., [28, 28] for MNIST
        target_shape: &[usize], // e.g., [10] for one-hot
        batch_size: usize,
        shuffle: bool,
    ) -> Self {
        Self::try_new(data, targets, data_shape, target_shape, batch_size, shuffle)
            .or_panic("DataLoader::new")
    }

    /// As [`DataLoader::new`], returning an error instead of panicking.
    ///
    /// # Errors
    /// [`VoltaError::InvalidParameter`], checked before anything is allocated or shuffled, if:
    /// - `batch_size` is 0;
    /// - `data_shape` or `target_shape` is not a valid [`Shape`] (a zero dimension, or
    ///   over [`crate::shape::MAX_NUMEL`] elements);
    /// - `data.len()` is not a whole number of samples of `data_shape` (a ragged tail used
    ///   to be dropped without a word, #93);
    /// - `targets` does not hold exactly one target of `target_shape` per sample.
    pub fn try_new(
        data: Vec<f32>,
        targets: Vec<f32>,
        data_shape: &[usize],
        target_shape: &[usize],
        batch_size: usize,
        shuffle: bool,
    ) -> Result<Self> {
        if batch_size == 0 {
            return Err(VoltaError::invalid(
                "DataLoader::new",
                Some("batch_size"),
                "batch_size must be at least 1",
            ));
        }
        #[expect(
            clippy::wildcard_enum_match_arm,
            reason = "Shape::from_dims returns only InvalidParameter; any other error passes through unchanged"
        )]
        let numel = |name: &'static str, dims: &[usize]| {
            Shape::from_dims(dims.to_vec())
                .map(|s| s.numel())
                .map_err(|e| match e {
                    VoltaError::InvalidParameter(d) => VoltaError::invalid(
                        "DataLoader::new",
                        name,
                        format!("{name}: {}", d.message),
                    ),
                    other => other,
                })
        };
        let sample_size = numel("data_shape", data_shape)?;
        let target_size = numel("target_shape", target_shape)?;
        if !data.len().is_multiple_of(sample_size) {
            return Err(VoltaError::invalid(
                "DataLoader::new",
                Some("data"),
                format!(
                    "{} data values are not a whole number of samples of shape \
                 {data_shape:?} ({sample_size} values each)",
                    data.len()
                ),
            ));
        }
        let num_samples = data.len() / sample_size;
        let expected_targets = num_samples.checked_mul(target_size);
        if expected_targets != Some(targets.len()) {
            return Err(VoltaError::invalid(
                "DataLoader::new",
                Some("targets"),
                format!(
                    "{num_samples} samples of shape {data_shape:?} need {} target values of shape {target_shape:?}, got {}",
                    expected_targets
                        .map_or_else(|| "more than usize::MAX".to_string(), |n| n.to_string()),
                    targets.len()
                ),
            ));
        }
        let mut indices: Vec<usize> = (0..num_samples).collect();

        if shuffle {
            use rand::seq::SliceRandom;
            with_rng(|rng| indices.shuffle(rng));
        }

        Ok(Self {
            data,
            targets,
            data_shape: data_shape.to_vec(),
            target_shape: target_shape.to_vec(),
            batch_size,
            shuffle,
            indices,
            current: 0,
            device: None, // Default: batches stay on the CPU
        })
    }

    /// Move each batch to `device` as it is produced
    ///
    /// `next` builds the batch on the CPU, then calls `to_device` on the data and
    /// targets before returning them. This saves the `.to_device()` calls in the
    /// training loop. It is not a prefetch: the copy is synchronous, inside `next`,
    /// so it doesn't overlap with the previous step's compute.
    ///
    /// # Arguments
    /// * `device` - Device the batches are moved to (typically GPU)
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

        // Move to the target device, if one was set (synchronous)
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
///
/// # Panics
/// As [`RawTensor::mse_loss`].
#[must_use]
#[track_caller]
pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    RawTensor::mse_loss(pred, target)
}

/// Free-function form of [`RawTensor::cross_entropy_loss`].
///
/// # Panics
/// As [`RawTensor::cross_entropy_loss`].
#[must_use]
#[track_caller]
pub fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Tensor {
    RawTensor::cross_entropy_loss(logits, targets)
}

/// Free-function form of [`RawTensor::nll_loss`].
///
/// # Panics
/// As [`RawTensor::nll_loss`].
#[must_use]
#[track_caller]
pub fn nll_loss(log_probs: &Tensor, targets: &Tensor) -> Tensor {
    RawTensor::nll_loss(log_probs, targets)
}

/// Free-function form of [`RawTensor::kl_divergence_gaussian`].
///
/// # Panics
/// As [`RawTensor::kl_divergence_gaussian`].
#[must_use]
#[track_caller]
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
///
/// # Panics
/// As [`RawTensor::bce_with_logits_loss`].
#[must_use]
#[track_caller]
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

    /// #461: a loss that is constant in its input has a zero gradient for any backward,
    /// so the check fails it rather than pass a backward it never tested: `softmax.sum()`
    /// (each row sums to 1) and a loss that ignores the input. A weighted loss through
    /// the same op passes.
    #[test]
    fn check_gradients_fails_on_an_all_zero_gradient() {
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, -1.0, 0.5, 4.0], &[2, 3], true);
        let (_, _, passed) =
            RawTensor::check_gradients(&x, |t| t.softmax(Axis(1)).sum(), 1e-3, 1e-2);
        assert!(
            !passed,
            "softmax(x).sum() is constant, so its check must fail"
        );
        let (_, _, passed) = RawTensor::check_gradients(
            &x,
            |t| t.elem_mul(&RawTensor::zeros(&[2, 3])).sum(),
            1e-3,
            1e-2,
        );
        assert!(!passed, "an exactly zero gradient must fail too");
        let w = RawTensor::new(vec![0.3, -1.2, 0.7, 1.5, 0.2, -0.8], &[2, 3], false);
        assert!(RawTensor::check_gradients_simple(&x, |t| {
            t.softmax(Axis(1)).elem_mul(&w).sum()
        }));
    }

    /// sqrt of negatives: both gradients are NaN. `f64::max` used to drop the NaN, so the
    /// check reported `(0.0, NaN, true)` (#90).
    #[test]
    fn check_gradients_fails_when_both_gradients_are_nan() {
        let x = RawTensor::new(vec![-1.0, -2.0], &[2], true);
        let (max_err, _, passed) = RawTensor::check_gradients(&x, |t| t.sqrt().sum(), 1e-3, 1e-3);
        assert!(!passed, "a NaN gradient passed");
        assert!(max_err.is_nan(), "max_error should be NaN, got {max_err}");
        assert!(!RawTensor::check_gradients_simple(&x, |t| t.sqrt().sum()));
    }

    /// log of negatives: the analytic gradient `1/(x-5)` is finite, the numerical one is
    /// NaN (#90).
    #[test]
    fn check_gradients_fails_when_numerical_gradient_is_nan() {
        let x = RawTensor::new(vec![1.0, 2.0], &[2], true);
        let c = RawTensor::new(vec![5.0, 5.0], &[2], false);
        let loss_fn = |t: &Tensor| t.sub(&c).log().sum();
        let (max_err, _, passed) = RawTensor::check_gradients(&x, loss_fn, 1e-3, 1e-3);
        assert!(!passed, "a NaN numerical gradient passed");
        assert!(max_err.is_nan(), "max_error should be NaN, got {max_err}");
        assert!(!RawTensor::check_gradients_simple(&x, loss_fn));
    }

    /// An infinite analytic gradient fails too: `sqrt` at 0 has gradient `+inf` (and the
    /// numerical one steps to a negative input), so the error is non-finite (#90).
    #[test]
    fn check_gradients_fails_on_infinite_gradient() {
        let x = RawTensor::new(vec![0.0, 1.0], &[2], true);
        let (max_err, _, passed) = RawTensor::check_gradients(&x, |t| t.sqrt().sum(), 1e-3, 1e-3);
        assert!(!passed, "an infinite gradient passed");
        assert!(
            !max_err.is_finite(),
            "max_error should be non-finite, got {max_err}"
        );
    }

    /// f16 input: refused, since a finite difference can't resolve at half precision (A6).
    #[test]
    #[should_panic(expected = "cast the inputs to f32 or f64 first")]
    fn check_gradients_refuses_f16_input() {
        let x = RawTensor::new(vec![0.5, 1.5], &[2], false).to_dtype(DType::F16);
        x.set_requires_grad(true);
        let _ = RawTensor::check_gradients(&x, |t| t.elem_mul(t).sum(), 1e-2, 1e-2);
    }

    /// bf16 input: refused, as for f16 (A6).
    #[test]
    #[should_panic(expected = "cast the inputs to f32 or f64 first")]
    fn check_gradients_refuses_bf16_input() {
        let x = RawTensor::new(vec![0.5, 1.5], &[2], false).to_dtype(DType::BF16);
        x.set_requires_grad(true);
        let _ = RawTensor::check_gradients(&x, |t| t.elem_mul(t).sum(), 1e-2, 1e-2);
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
    #[should_panic(expected = "xavier_uniform: expects a rank-1 or rank-2 shape, got [2, 3, 4]")]
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
        if !crate::gpu::is_gpu_available() {
            eprintln!("SKIPPED: no GPU available");
            return;
        }
        let dev = Device::gpu().expect("GPU availability checked above");
        let t = RawTensor::new(vec![0.0; 6], &[2, 3], true).to_device(dev.clone());
        assert!(t.borrow().data.is_gpu(), "precondition: tensor on GPU");

        let vals = vec![1.5, -2.0, 3.25, 0.5, -0.75, 4.0];
        t.load_weight("w", &TensorData::from_f32(vals.clone(), vec![2, 3]))
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

    /// An f32 entry loaded into an f64 tensor is cast (exactly), as
    /// PyTorch's `load_state_dict` does, and keeps the tensor f64.
    #[test]
    fn load_weight_casts_to_the_destination_dtype() {
        let t = RawTensor::from_typed(vec![1.0f64, 2.0], &[2]);
        let src = TensorData::from_f32(vec![0.1, 6.0], vec![2]);
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
        let src = TensorData::from_f32(vec![5.0, 6.0], vec![2]);
        let err = t.load_weight("w", &src).expect_err("i64 destination");
        assert!(matches!(err, VoltaError::DTypeMismatch { .. }), "{err}");
        assert_eq!(t.to_vec_as::<i64>().expect("i64"), vec![1, 2]);
    }
}

use crate::Storage;
use crate::autograd::{GradFn, edge};
use crate::device::Device;
use crate::device_policy::{Pass, on_gap};
use crate::index::{fetch, slot, span};
use crate::ops::binary::BinaryOp;
use crate::ops::reduce::ReduceOp;
use crate::ops::ternary::TernaryOp;
use crate::ops::unary::UnaryOp;
use crate::shape::{Axis, Shape};
use crate::spec::PadValue;
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
    inner: Rc<RefCell<RawTensor>>, //keeps fields private
}

impl Tensor {
    #[must_use]
    pub fn requires_grad(&self) -> bool {
        self.inner.borrow().requires_grad
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
    /// TODO: investigate why i actually need this
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
    /// # Panics
    /// If `src.shape` differs from this tensor's shape, or `src.data.len()` does not
    /// match it. `name` is the state-dict key, used in the message.
    pub(crate) fn load_weight(&self, name: &str, src: &crate::io::TensorData) {
        let mut t = self.inner.borrow_mut();
        // `TensorData.shape` is the serialized `Vec`; compare it at the io boundary.
        assert_eq!(
            src.shape.as_slice(),
            t.shape.truncate_rank(),
            "load_state_dict: shape mismatch for '{name}'"
        );
        let numel = t.shape.numel();
        assert_eq!(
            src.data.len(),
            numel,
            "load_state_dict: '{name}' has {} values but shape {:?} needs {numel}",
            src.data.len(),
            t.shape
        );
        let cpu = Storage::cpu(src.data.clone());
        t.data = cpu.to_device(&t.device);
    }

    /// This tensor's data, decoded to `f32` regardless of underlying dtype/device.
    #[must_use]
    pub fn data(&self) -> Vec<f32> {
        self.inner.borrow().data.to_vec()
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

    /// A constant (no grad) on this tensor's device, for layers that combine a fixed value
    /// with their input. Building it on the CPU instead would trip the mixed-device check.
    ///
    /// # Panics
    /// If `data.len()` != the product of `shape`.
    pub(crate) fn constant_like(&self, data: Vec<f32>, shape: &[usize]) -> Self {
        RawTensor::new(data, shape, false).to_device(self.device())
    }

    /// Set `requires_grad` on a leaf tensor.
    ///
    /// # Panics
    /// Panics if called on a tensor that already has a `grad_fn` attached —
    /// flipping the flag on a non-leaf tensor would contradict the invariant
    /// `edge()` maintains (`requires_grad` is derived from inputs, not hand-set,
    /// on anything that isn't a leaf). This is the same rule `PyTorch` enforces on
    /// `.requires_grad = ...` for non-leaf tensors, and it's an unconditional
    /// `assert!` rather than `debug_assert!` deliberately: if this method is public,
    /// the check needs to actually hold in release builds too, or it's the same
    /// "invariant as convention, not enforcement" problem this whole wrapper exists
    /// to close.
    pub fn set_requires_grad(&self, on: bool) {
        let mut t = self.inner.borrow_mut();
        assert!(
            !(on && t.grad_fn.is_some()),
            "set_requires_grad called on a non-leaf tensor"
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
        let mut t = self.inner.borrow_mut();
        let g = match (t.device.is_gpu(), g.is_gpu()) {
            (true, false) => {
                on_gap(&op.op_name(), Pass::Backward);
                g.to_device(&t.device)
            }
            (false, true) => panic!("{}: produced a GPU gradient for a CPU tensor", op.op_name()),
            (true, true) | (false, false) => g,
        };
        match t.grad {
            None => t.grad = Some(g),
            Some(ref mut existing) => RawTensor::accumulate_grad(existing, &g),
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
    pub fn set_grad(&self, grad: Storage) {
        let mut t = self.inner.borrow_mut();
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
    pub data: Storage,         // Storage enum
    pub shape: Shape,          //tensor dims, eg [B,C,H,W]
    pub grad: Option<Storage>, //grad w.r.t tensor data, None if req_grad == false
    pub requires_grad: bool,
    pub(crate) grad_fn: Option<GradFn>, //func to compute grad, if result of op
    pub parents: Vec<Tensor>,           //refs to parent tensor on graph
    pub device: Device,                 //cpu/gpu
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

/// Run a closure with the current RNG (`ThreadRng` or specific `StdRng`)
pub(crate) fn with_rng<F, T>(f: F) -> T
where
    F: FnOnce(&mut dyn rand::RngCore) -> T,
{
    GLOBAL_RNG.with(|rng_cell| {
        if let Ok(mut borrow) = rng_cell.try_borrow_mut()
            && let Some(rng) = borrow.as_mut()
        {
            return f(rng);
        }

        // Fallback to thread RNG (not reproducible but safe)
        f(&mut rand::rng())
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
        let shape = Shape::from_dims(shape)?;
        if data.len() != shape.numel() {
            return Err(VoltaError::ShapeDataMismatch {
                shape: shape.truncate_rank().to_vec(),
                elements: shape.numel(),
                len: data.len(),
            });
        }

        let raw = Self {
            data: Storage::cpu(data),
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
    /// Panics if `data.len()` != `shape.product()`
    #[allow(
        clippy::new_ret_no_self,
        reason = "Public wrapper around private struct"
    )]
    #[must_use]
    pub fn new(data: Vec<f32>, shape: &[usize], requires_grad: bool) -> Tensor {
        Self::try_new(data, shape, requires_grad).expect("Data length must match shape")
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
    pub(crate) fn from_shape(data: Vec<f32>, shape: Shape, requires_grad: bool) -> Tensor {
        Self::new_with_storage(Storage::cpu(data), shape, Device::CPU, requires_grad)
    }

    /// Create a tensor filled with zeros
    #[must_use]
    pub fn zeros(shape: &[usize]) -> Tensor {
        let size = shape.iter().product();
        Self::new(vec![0.0; size], shape, false)
    }
    /// Create a tensor filled with ones
    #[must_use]
    pub fn ones(shape: &[usize]) -> Tensor {
        let size = shape.iter().product();
        Self::new(vec![1.0; size], shape, false)
    }
    /// Create a tensor with random values uniformly distributed in [0, 1)
    #[must_use]
    pub fn rand(shape: &[usize]) -> Tensor {
        let size = shape.iter().product();
        let data: Vec<f32> = with_rng(|rng| (0..size).map(|_| rng.random::<f32>()).collect());
        Self::new(data, shape, false)
    }
    /// Create a tensor with values from standard normal distribution N(0, 1) (fallible)
    ///
    /// # Errors
    ///
    /// Returns `VoltaError::InvalidParameter` if normal distribution parameters are invalid
    /// Propagates errors from `try_new` for shape/data mismatches
    pub fn try_randn(shape: &[usize]) -> Result<Tensor> {
        let size = shape.iter().product();
        let normal = Normal::new(0.0, 1.0).map_err(|e| {
            VoltaError::InvalidParameter(format!("Invalid normal distribution: {e}"))
        })?;
        let data: Vec<f32> = with_rng(|rng| (0..size).map(|_| normal.sample(rng)).collect());

        Self::try_new(data, shape, false)
    }

    /// Create a tensor with values from standard normal distribution N(0, 1)
    /// # Panics
    /// unwrap new `Normal` dist
    #[must_use]
    pub fn randn(shape: &[usize]) -> Tensor {
        Self::try_randn(shape).expect("Failed to create random tensor")
    }

    /// Create a tensor filled with random values from N(0, 1) with the same shape as the input
    #[must_use]
    pub fn randn_like(tensor: &Tensor) -> Tensor {
        let shape = tensor.borrow().shape.clone();
        Self::randn(shape.truncate_rank())
    }

    /// Xavier uniform initialization
    ///
    /// Samples weights uniformly from [-limit, limit] where
    /// limit = sqrt(6 / (`fan_in` + `fan_out`))
    ///
    /// This helps maintain gradient variance across layers. A rank-1 shape `[n]` is
    /// treated as `[n, 1]` (a bias vector).
    ///
    /// # Panics
    ///
    /// If `shape` is not rank 1 or rank 2.
    #[must_use]
    pub fn xavier_uniform(shape: &[usize]) -> Tensor {
        // A rank-1 shape is a bias vector: one fan-out per element. Any other rank has no
        // [in, out] reading, so it fails here instead of later as a data/shape mismatch.
        let (fan_in, fan_out) = match *shape {
            [n] => (n, 1),
            [fan_in, fan_out] => (fan_in, fan_out),
            _ => panic!("xavier_uniform expects a rank-1 or rank-2 shape, got {shape:?}"),
        };
        let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
        let data: Vec<f32> = with_rng(|rng| {
            (0..fan_in * fan_out)
                .map(|_| rng.random_range(-limit..limit))
                .collect()
        });

        Self::new(data, shape, false)
    }

    /// He (Kaiming) normal initialization suited for `ReLU` networks. (fallible)
    ///
    /// Draws samples from `N(0, sqrt(2 / fan_in))` where `fan_in`
    /// is the number of input connections.
    ///
    /// # Errors
    ///
    /// Returns `VoltaError::InvalidParameter` if shape is empty or `fan_in` is zero
    /// Propagates errors from normal distribution creation and tensor construction
    pub fn try_he_initialization(shape: &[usize]) -> Result<Tensor> {
        if shape.is_empty() {
            return Err(VoltaError::InvalidParameter(
                "He initialization requires at least one dimension".to_string(),
            ));
        }

        let fan_in = match shape.len() {
            1 | 2 => fetch(shape, 0), // Linear weights: [in, out]
            _ => shape.get(1..).unwrap_or(&[]).iter().product(), // Conv weights: [out, in, kH, kW, ...]
        };
        if fan_in == 0 {
            return Err(VoltaError::InvalidParameter(
                "fan_in must be positive for He initialization".to_string(),
            ));
        }

        let std_val: f32 = 2.0 / fan_in as f32;
        let std = std_val.sqrt();
        let normal = Normal::new(0.0, std)
            .map_err(|e| VoltaError::InvalidParameter(format!("Invalid He std: {e}")))?;
        let size: usize = shape.iter().product();
        let data: Vec<f32> = with_rng(|rng| (0..size).map(|_| normal.sample(rng)).collect());

        Self::try_new(data, shape, false)
    }

    /// He (Kaiming) normal initialization suited for `ReLU` networks.
    ///
    /// Draws samples from `N(0, sqrt(2 / fan_in))` where `fan_in`
    /// is the number of input connections.
    /// # Panics
    /// empty shape
    #[must_use]
    pub fn he_initialization(shape: &[usize]) -> Tensor {
        Self::try_he_initialization(shape).expect("He initialization failed")
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
    /// ```ignore
    /// let t = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], false)
    ///     .to_device(Device::gpu().unwrap());
    /// let _ = t.borrow().data.to_host_f32(); // Populates CPU cache
    /// t.borrow_mut().invalidate_gpu_cache(); // Releases CPU copy
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
    #[must_use]
    pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Tensor {
        let diff = pred.sub(target);
        let squared = diff.elem_mul(&diff);
        squared.mean()
    }

    #[must_use]
    pub fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Tensor {
        // Use log_softmax for numerical stability
        let log_probs = Self::log_softmax(logits, Axis(1));

        // -sum(targets * log_probs, dim=1).mean()
        let prod = targets.elem_mul(&log_probs);
        let sum = Self::sum_dim(&prod, Axis(1), false);
        sum.neg().mean()
    }

    /// Negative log likelihood loss
    /// Takes log-probabilities and one-hot targets
    /// Equivalent to `cross_entropy` but expects pre-computed log probabilities
    #[must_use]
    pub fn nll_loss(log_probs: &Tensor, targets: &Tensor) -> Tensor {
        // -sum(targets * log_probs, dim=1).mean()
        let prod = targets.elem_mul(log_probs);
        let sum = Self::sum_dim(&prod, Axis(1), false);
        sum.neg().mean()
    }

    /// KL divergence between N(mu, sigma) and N(0, 1)
    ///
    /// Used in VAE loss: KL(q(z|x) || p(z))
    /// Formula: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    ///
    /// # Arguments
    /// * `mu` - Mean of learned distribution
    /// * `logvar` - Log variance of learned distribution
    #[must_use]
    pub fn kl_divergence_gaussian(mu: &Tensor, logvar: &Tensor) -> Tensor {
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

    /// Binary Cross Entropy loss
    ///
    /// `BCE(y, y_hat) = -mean(y * log(y_hat) + (1-y) * log(1-y_hat))`
    ///
    /// # Arguments
    /// * `pred` - Predicted probabilities (should be in [0, 1], typically from sigmoid)
    /// * `target` - Target labels (0 or 1)
    ///
    /// # Note
    /// Predictions should be passed through sigmoid before calling this.
    /// For numerical stability with logits, use `bce_with_logits_loss` instead.
    #[must_use]
    pub fn bce_loss(pred: &Tensor, target: &Tensor) -> Tensor {
        // y * log(y_hat)
        let log_pred = pred.log();
        let term1 = target.elem_mul(&log_pred);

        // (1-y) * log(1-y_hat)
        let ones_like = |t: &Tensor| {
            let shape = t.borrow().shape.clone();
            t.constant_like(vec![1.0; shape.numel()], shape.truncate_rank())
        };
        let one_minus_target = ones_like(target).sub(target);
        let one_minus_pred = ones_like(pred).sub(pred);
        let log_one_minus_pred = one_minus_pred.log();
        let term2 = one_minus_target.elem_mul(&log_one_minus_pred);

        // -mean(term1 + term2)
        term1.add(&term2).mean().neg()
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
    input_shape: Shape,
    dim: usize,
    keepdim: bool,
}

impl SumDimGradFn {
    pub(crate) fn backward(
        &self,
        out_grad: &RawTensor,
        _parents: &[Tensor],
    ) -> Vec<Option<Tensor>> {
        let grad_data = &*out_grad.data.to_host_f32();
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
        let mut result = vec![0.0; size];

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
            *slot(&mut result, i) = fetch(grad_data, grad_idx);
        }

        vec![Some(RawTensor::from_shape(
            result,
            self.input_shape.clone(),
            false,
        ))]
    }
}

/// Gradient for `max_dim`: sparse gradient to max elements only
#[derive(Clone)]
pub(crate) struct MaxDimGradFn {
    input_shape: Shape,
    max_indices: Vec<usize>, // linear indices of max elements
    dim: usize,
    keepdim: bool,
}

impl MaxDimGradFn {
    pub(crate) fn backward(
        &self,
        out_grad: &RawTensor,
        _parents: &[Tensor],
    ) -> Vec<Option<Tensor>> {
        let grad_data = &*out_grad.data.to_host_f32();
        let grad_shape = &out_grad.shape;

        let expanded_shape = if self.keepdim {
            grad_shape.clone()
        } else {
            grad_shape.unsqueeze(self.dim)
        };

        let size = self.input_shape.numel();
        let mut result = vec![0.0; size];

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
            *slot(&mut result, max_lin_idx) = fetch(grad_data, grad_idx);
        }

        vec![Some(RawTensor::from_shape(
            result,
            self.input_shape.clone(),
            false,
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
        let (data, shape) = {
            let s = self_t.borrow();
            (s.data.clone(), s.shape.clone())
        };

        let dim_size = shape.dim(dim)?;
        let dim = dim.0;
        let out_shape = shape.with_dim(dim, 1); // intermediate shape before squeeze
        let out_size = out_shape.numel();
        let mut result = vec![0.0; out_size];

        // Optimized stride-based reduction: O(1) per element instead of O(rank) coordinate conversion
        // View data as: [outer_size, dim_size, inner_size] where we sum over dim_size
        let outer_size = shape.numel_before(dim);
        let inner_size = shape.numel_from(dim + 1);

        // Sum over the target dimension using stride arithmetic
        let data = data.to_host_f32();
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let out_idx = outer * inner_size + inner;
                let mut sum = 0.0;
                for k in 0..dim_size {
                    let in_idx = outer * dim_size * inner_size + k * inner_size + inner;
                    sum += fetch(&data, in_idx);
                }
                *slot(&mut result, out_idx) = sum;
            }
        }

        // Squeeze dimension if keepdim=false
        let final_shape = if keepdim {
            out_shape
        } else {
            out_shape.squeeze(dim)
        };

        // Computed on the CPU (there is no GPU kernel): `edge()` applies the device
        // policy when the input is on the GPU.
        let storage = Storage::cpu(result);

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

    /// Move tensor data to a different device.
    ///
    /// - `data` is moved via `Storage::to_device`.
    /// - The gradient, if any, moves with the data.
    /// - Autograd metadata (`requires_grad`, `grad_fn`, `parents`) is preserved,
    ///   so the returned tensor participates in the same computation graph.
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

        let t = self_t.borrow();

        let new_data = t.data.to_device(&device);
        let new_grad = t.grad.as_ref().map(|g| g.to_device(&device));

        let new_tensor = Self {
            data: new_data,
            shape: t.shape.clone(),
            grad: new_grad,
            requires_grad: t.requires_grad,
            grad_fn: t.grad_fn.clone(),
            parents: t.parents.clone(),
            device,
        };
        Tensor {
            inner: Rc::new(RefCell::new(new_tensor)),
        }
    }

    /// Max along a specific axis
    ///
    /// Returns maximum value along dimension and stores indices for backward pass.
    /// # Panics
    /// dim out of bounds
    #[must_use]
    pub fn max_dim(self_t: &Tensor, dim: Axis, keepdim: bool) -> Tensor {
        let (data, shape) = {
            let s = self_t.borrow();
            (s.data.clone(), s.shape.clone())
        };

        let dim_size = shape.dim(dim).unwrap_or_else(|e| panic!("max_dim: {e}"));
        let dim = dim.0;
        let out_shape = shape.with_dim(dim, 1);
        let out_size = out_shape.numel();

        let mut result = vec![f32::NEG_INFINITY; out_size];
        let mut max_indices = vec![0; out_size]; // track which index won

        // Optimized stride-based reduction: O(1) per element instead of O(rank) coordinate conversion
        // View data as: [outer_size, dim_size, inner_size] where we find max over dim_size
        let outer_size = shape.numel_before(dim);
        let inner_size = shape.numel_from(dim + 1);

        // Find max over the target dimension using stride arithmetic
        let data = data.to_host_f32();
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

        let final_shape = if keepdim {
            out_shape
        } else {
            out_shape.squeeze(dim)
        };

        // Computed on the CPU (there is no GPU kernel): `edge()` applies the device
        // policy when the input is on the GPU.
        let storage = Storage::cpu(result);

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
    /// Other leaves that `loss_fn` reads (layer parameters) are not cleared: the
    /// analytic pass adds its gradient to theirs. The finite-difference evaluations run
    /// `loss_fn` 2N more times, so stateful layers in training mode (`BatchNorm`
    /// running statistics) keep updating; the result is unaffected because training
    /// mode normalizes with batch statistics.
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
        // Compute analytical gradient
        Self::zero_grad(tensor);
        let loss = loss_fn(tensor)
            .into_loss()
            .expect("loss_fn must return a scalar that requires grad");
        loss.backward();

        let analytical_grad = tensor.grad().expect("Tensor must have gradient");
        let mut numerical_grad = vec![0.0; analytical_grad.len()];

        let original_data = tensor.borrow().data.clone();
        let original_shape = tensor.borrow().shape.clone();
        let requires_grad = tensor.borrow().requires_grad;

        // Compute numerical gradient for each element. These perturbed forward passes
        // are for finite-difference comparison only — scoped in `no_grad()` so they
        // can't attach graph nodes or otherwise contaminate the analytical pass above
        // (ADDENDUM.md §1.5).
        {
            let _no_grad = crate::autograd::no_grad();
            for i in 0..original_data.len() {
                //f(x + epsilon)
                let mut data_plus = original_data.clone();
                *slot(
                    data_plus
                        .as_host_f32_mut()
                        .expect("check_gradients: CPU F32 input"),
                    i,
                ) += epsilon;
                let tensor_plus =
                    Self::from_shape(data_plus.to_vec(), original_shape.clone(), requires_grad);
                let loss_plus = loss_fn(&tensor_plus);
                assert!(
                    !loss_plus.borrow().data.is_empty(),
                    "check_gradients: loss must be scalar, got shape {:?}",
                    loss_plus.borrow().shape
                );
                let val_plus = fetch(&loss_plus.borrow().data.to_host_f32(), 0);

                let mut data_minus = original_data.clone();
                *slot(
                    data_minus
                        .as_host_f32_mut()
                        .expect("check_gradients: CPU F32 input"),
                    i,
                ) -= epsilon;
                let tensor_minus =
                    Self::from_shape(data_minus.to_vec(), original_shape.clone(), requires_grad);
                let loss_minus = loss_fn(&tensor_minus);
                assert!(
                    !loss_minus.borrow().data.is_empty(),
                    "check_gradients: loss must be scalar, got shape {:?}",
                    loss_minus.borrow().shape
                );
                let val_minus = fetch(&loss_minus.borrow().data.to_host_f32(), 0);
                //central diff
                *slot(&mut numerical_grad, i) = (val_plus - val_minus) / (2.0 * epsilon);
            }
        }

        // Compute errors
        let mut max_error: f32 = 0.0;
        let mut total_error: f32 = 0.0;

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

            if relative_error > tolerance {
                log::debug!(
                    "Gradient mismatch at index {i}: analytical={analytical:.6e}, numerical={numerical:.6e}, error={relative_error:.6e}",
                );
            }
        }

        let mean_error = total_error / analytical_grad.len() as f32;
        let passed = max_error < tolerance;

        (max_error, mean_error, passed)
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

    /// Elementwise `self < other` as 1.0 or 0.0, with broadcasting.
    #[must_use]
    pub fn cmplt(&self, other: &Self) -> Self {
        RawTensor::binary_op(self, other, BinaryOp::Cmplt)
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

    /// Selects `x` where `self` is nonzero, else `y`.
    #[must_use]
    pub fn where_op(&self, x: &Self, y: &Self) -> Self {
        RawTensor::ternary_op(self, x, y, TernaryOp::Where)
    }

    /// Reshapes to `new_shape`; the element count must match.
    #[must_use]
    pub fn reshape(&self, new_shape: &[usize]) -> Self {
        RawTensor::reshape(self, new_shape)
    }

    /// Permutes the dimensions by `axes`.
    #[must_use]
    pub fn permute(&self, axes: &[Axis]) -> Self {
        RawTensor::permute(self, axes)
    }

    /// Broadcasts to `new_shape`.
    #[must_use]
    pub fn expand(&self, new_shape: &[usize]) -> Self {
        RawTensor::expand(self, new_shape)
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

    /// Moves the tensor and its gradient to `device`, keeping its autograd metadata.
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

    /// The accumulated gradient as host data, or `None` if there is none.
    #[must_use]
    pub fn grad(&self) -> Option<Vec<f32>> {
        self.borrow()
            .grad
            .as_ref()
            .map(super::storage::Storage::to_vec)
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

pub struct DataLoader {
    data: Vec<f32>,
    targets: Vec<f32>,
    data_shape: Vec<usize>, // per-sample shape
    target_shape: Vec<usize>,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>,
    current: usize,
    device: Option<crate::device::Device>, // Optional device for GPU prefetch
}

impl DataLoader {
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
            indices.shuffle(&mut rand::rng());
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

    pub fn reset(&mut self) {
        self.current = 0;
        if self.shuffle {
            use rand::seq::SliceRandom;
            self.indices.shuffle(&mut rand::rng());
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
#[must_use]
pub fn zeros(shape: &[usize]) -> Tensor {
    RawTensor::zeros(shape)
}

#[must_use]
pub fn ones(shape: &[usize]) -> Tensor {
    RawTensor::ones(shape)
}

#[must_use]
pub fn rand(shape: &[usize]) -> Tensor {
    RawTensor::rand(shape)
}

#[must_use]
pub fn randn(shape: &[usize]) -> Tensor {
    RawTensor::randn(shape)
}

#[must_use]
pub fn randn_like(tensor: &Tensor) -> Tensor {
    RawTensor::randn_like(tensor)
}

// Loss functions
#[must_use]
pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    RawTensor::mse_loss(pred, target)
}

#[must_use]
pub fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Tensor {
    RawTensor::cross_entropy_loss(logits, targets)
}

#[must_use]
pub fn nll_loss(log_probs: &Tensor, targets: &Tensor) -> Tensor {
    RawTensor::nll_loss(log_probs, targets)
}

#[must_use]
pub fn kl_divergence_gaussian(mu: &Tensor, logvar: &Tensor) -> Tensor {
    RawTensor::kl_divergence_gaussian(mu, logvar)
}

#[must_use]
pub fn bce_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    RawTensor::bce_loss(pred, target)
}

#[must_use]
pub fn bce_with_logits_loss(logits: &Tensor, target: &Tensor) -> Tensor {
    RawTensor::bce_with_logits_loss(logits, target)
}

// Axis reductions
#[must_use]
pub fn sum_dim(t: &Tensor, dim: Axis, keepdim: bool) -> Tensor {
    RawTensor::sum_dim(t, dim, keepdim)
}

#[must_use]
pub fn max_dim(t: &Tensor, dim: Axis, keepdim: bool) -> Tensor {
    RawTensor::max_dim(t, dim, keepdim)
}

#[must_use]
pub fn softmax(t: &Tensor, dim: Axis) -> Tensor {
    RawTensor::softmax(t, dim)
}

// Gradient checking
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

pub fn check_gradients_simple<F>(tensor: &Tensor, loss_fn: F) -> bool
where
    F: Fn(&Tensor) -> Tensor,
{
    RawTensor::check_gradients_simple(tensor, loss_fn)
}

#[cfg(test)]
mod tensor_tests {
    use super::*;

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
        );

        assert!(t.borrow().data.is_gpu(), "load_weight demoted GPU to CPU");
        assert_eq!(t.borrow().device, dev);
        assert_eq!(t.data(), vals);
        assert!(t.requires_grad());
    }
}

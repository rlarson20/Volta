use crate::Storage;
use crate::autograd::GradFn;
use crate::device::Device;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use std::cell::RefCell;
use std::rc::Rc;

/// Type alias for a reference-counted, interior-mutable tensor.
///
/// We use `Rc<RefCell<RawTensor>>` to allow multiple references to the same tensor
/// (needed for computation graphs) while still allowing mutation (for gradient accumulation).
///
/// **Note for production**: This is single-threaded only. For multi-threading,
/// replace with `Arc<Mutex<RawTensor>>`.
pub type Tensor = Rc<RefCell<RawTensor>>;

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
    pub shape: Vec<usize>,     //tensor dims, eg [B,C,H,W]
    pub grad: Option<Storage>, //grad w.r.t tensor data, None if req_grad == false
    pub requires_grad: bool,
    pub grad_fn: Option<Box<dyn GradFn>>, //func to compute grad, if result of op
    pub parents: Vec<Tensor>,             //refs to parent tensor on graph
    pub device: Device,                   //cpu/gpu
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
        RawTensor {
            data: self.data.clone(),
            shape: self.shape.clone(),
            grad: self.grad.clone(),
            requires_grad: self.requires_grad,
            grad_fn: self.grad_fn.as_ref().map(|gf| gf.clone_box()),
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
            .finish()
    }
}

// ===== TENSOR CONSTRUCTORS =====
impl RawTensor {
    /// Create a new tensor from data and shape
    ///
    /// # Arguments
    /// * `data` - Flat vector of values (length must equal product of shape dimensions)
    /// * `shape` - Dimensions of the tensor
    /// * `requires_grad` - Whether to track gradients for backpropagation
    ///
    /// # Panics
    /// Panics if `data.len()` != `shape.product()`
    #[must_use]
    pub fn new(data: Vec<f32>, shape: &[usize], requires_grad: bool) -> Tensor {
        assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "Data length must match shape"
        );
        // Additional validation for reasonable tensor sizes
        let total_size = shape.iter().product::<usize>();
        assert!(total_size > 0, "Cannot create tensor with zero elements");
        assert!(
            total_size <= 100_000_000,
            "Tensor too large (>100M elements) - check for memory issues"
        );

        let raw = RawTensor {
            data: Storage::cpu(data),
            shape: shape.to_vec(),
            grad: None,
            requires_grad,
            grad_fn: None,
            parents: vec![],
            device: Device::CPU,
        };
        Rc::new(RefCell::new(raw))
    }

    /// Create a new tensor from Storage and shape (internal use for GPU ops)
    #[allow(dead_code)]
    pub(crate) fn new_with_storage(
        data: crate::storage::Storage,
        shape: &[usize],
        device: Device,
        requires_grad: bool,
    ) -> Tensor {
        assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "Data length must match shape"
        );

        let raw = RawTensor {
            data,
            shape: shape.to_vec(),
            grad: None,
            requires_grad,
            grad_fn: None,
            parents: vec![],
            device,
        };
        Rc::new(RefCell::new(raw))
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
    /// Create a tensor with values from standard normal distribution N(0, 1)
    #[must_use]
    pub fn randn(shape: &[usize]) -> Tensor {
        let size = shape.iter().product();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let data: Vec<f32> = with_rng(|rng| (0..size).map(|_| normal.sample(rng)).collect());

        Self::new(data, shape, false)
    }

    /// Create a tensor filled with random values from N(0, 1) with the same shape as the input
    pub fn randn_like(tensor: &Tensor) -> Tensor {
        let shape = tensor.borrow().shape.clone();
        Self::randn(&shape)
    }

    /// Xavier uniform initialization
    ///
    /// Samples weights uniformly from [-limit, limit] where
    /// limit = sqrt(6 / (`fan_in` + `fan_out`))
    ///
    /// This helps maintain gradient variance across layers.
    #[must_use]
    pub fn xavier_uniform(shape: &[usize]) -> Tensor {
        let fan_in = shape[0];
        let fan_out = shape[1];
        let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
        let data: Vec<f32> = with_rng(|rng| {
            (0..fan_in * fan_out)
                .map(|_| rng.random_range(-limit..limit))
                .collect()
        });

        Self::new(data, shape, false)
    }

    /// He (Kaiming) normal initialization suited for `ReLU` networks.
    ///
    /// Draws samples from `N(0, sqrt(2 / fan_in))` where `fan_in`
    /// is the number of input connections.
    #[must_use]
    pub fn he_initialization(shape: &[usize]) -> Tensor {
        assert!(
            !shape.is_empty(),
            "He initialization requires at least one dimension"
        );

        let fan_in = match shape.len() {
            1 => shape[0],
            2 => shape[0],                    // Linear weights: [in, out]
            _ => shape[1..].iter().product(), // Conv weights: [out, in, kH, kW, ...]
        };
        assert!(fan_in > 0, "fan_in must be positive for He initialization");

        let std = (2.0 / fan_in as f32).sqrt();
        let normal = Normal::new(0.0, std).expect("valid He std");
        let size: usize = shape.iter().product();
        let data: Vec<f32> = with_rng(|rng| (0..size).map(|_| normal.sample(rng)).collect());

        Self::new(data, shape, false)
    }
}

#[cfg(feature = "gpu")]
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
    /// let _ = t.borrow().data.as_f32_slice(); // Populates CPU cache
    /// t.borrow_mut().invalidate_gpu_cache(); // Releases CPU copy
    /// ```
    pub fn invalidate_gpu_cache(&mut self) {
        self.data.invalidate_cpu_cache();
        if let Some(ref grad) = self.grad {
            grad.invalidate_cpu_cache();
        }
    }
}

// ===== LOSS FUNCTIONS =====
impl RawTensor {
    pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Tensor {
        let diff = pred.sub(target);
        let squared = diff.elem_mul(&diff);
        squared.mean()
    }

    pub fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Tensor {
        // Use log_softmax for numerical stability
        let log_probs = Self::log_softmax(logits, 1);

        // -sum(targets * log_probs, dim=1).mean()
        let prod = targets.elem_mul(&log_probs);
        let sum = Self::sum_dim(&prod, 1, false);
        sum.neg().mean()
    }

    /// Negative log likelihood loss
    /// Takes log-probabilities and one-hot targets
    /// Equivalent to cross_entropy but expects pre-computed log probabilities
    pub fn nll_loss(log_probs: &Tensor, targets: &Tensor) -> Tensor {
        // -sum(targets * log_probs, dim=1).mean()
        let prod = targets.elem_mul(log_probs);
        let sum = Self::sum_dim(&prod, 1, false);
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
    pub fn kl_divergence_gaussian(mu: &Tensor, logvar: &Tensor) -> Tensor {
        // 1 + logvar
        let one = RawTensor::ones(&mu.borrow().shape);
        let term1 = one.add(logvar);

        // mu^2
        let mu_sq = mu.elem_mul(mu);

        // exp(logvar)
        let var = logvar.exp();

        // 1 + logvar - mu^2 - exp(logvar)
        let sum_terms = term1.sub(&mu_sq).sub(&var);

        // -0.5 * sum(...)
        let half = RawTensor::new(vec![0.5], &[1], false);
        sum_terms.sum().elem_mul(&half).neg()
    }

    /// Binary Cross Entropy loss
    ///
    /// BCE(y, y_hat) = -mean(y * log(y_hat) + (1-y) * log(1-y_hat))
    ///
    /// # Arguments
    /// * `pred` - Predicted probabilities (should be in [0, 1], typically from sigmoid)
    /// * `target` - Target labels (0 or 1)
    ///
    /// # Note
    /// Predictions should be passed through sigmoid before calling this.
    /// For numerical stability with logits, use `bce_with_logits_loss` instead.
    pub fn bce_loss(pred: &Tensor, target: &Tensor) -> Tensor {
        // y * log(y_hat)
        let log_pred = pred.log();
        let term1 = target.elem_mul(&log_pred);

        // (1-y) * log(1-y_hat)
        let one = RawTensor::ones(&target.borrow().shape);
        let one_minus_target = one.sub(target);
        let one_minus_pred = RawTensor::ones(&pred.borrow().shape).sub(pred);
        let log_one_minus_pred = one_minus_pred.log();
        let term2 = one_minus_target.elem_mul(&log_one_minus_pred);

        // -mean(term1 + term2)
        term1.add(&term2).mean().neg()
    }

    /// Binary Cross Entropy with logits (numerically stable)
    ///
    /// Combines sigmoid and BCE in a numerically stable way.
    /// Formula: log(1 + exp(x)) - x*y
    ///
    /// # Arguments
    /// * `logits` - Raw network outputs (before sigmoid)
    /// * `target` - Target labels (0 or 1)
    pub fn bce_with_logits_loss(logits: &Tensor, target: &Tensor) -> Tensor {
        // log(1 + exp(x))
        let exp_logits = logits.exp();
        let one = RawTensor::ones(&logits.borrow().shape);
        let one_plus_exp = one.add(&exp_logits);
        let log_term = one_plus_exp.log();

        // x * y
        let xy_term = logits.elem_mul(target);

        // log(1 + exp(x)) - x*y
        log_term.sub(&xy_term).mean()
    }
}

// ===== SOFTMAX & AXIS REDUCTIONS =====

/// Gradient for `sum_dim`: broadcast ones back to input shape
struct SumDimGradFn {
    input_shape: Vec<usize>,
    dim: usize,
    keepdim: bool,
}

impl GradFn for SumDimGradFn {
    fn backward(&self, out_grad: &RawTensor, _parents: &[Tensor]) -> Vec<Option<Tensor>> {
        let grad_data = &out_grad.data;
        let grad_shape = &out_grad.shape;

        // If keepdim=false, we need to unsqueeze the dimension back
        let mut expanded_shape = grad_shape.clone();
        if !self.keepdim {
            expanded_shape.insert(self.dim, 1);
        }

        // Broadcast gradient back to input shape
        // Each output element contributed to by input_shape[dim] elements
        let size: usize = self.input_shape.iter().product();
        let mut result = vec![0.0; size];

        let _strides = RawTensor::compute_strides(&self.input_shape);
        let grad_strides = RawTensor::compute_strides(&expanded_shape);

        #[allow(clippy::needless_range_loop)]
        for i in 0..size {
            // Get input coordinates
            let mut coords = vec![0; self.input_shape.len()];
            let mut rem = i;
            for (d, &dim_sz) in self.input_shape.iter().enumerate().rev() {
                coords[d] = rem % dim_sz;
                rem /= dim_sz;
            }

            // Map to gradient coordinates (zero out the summed dimension)
            let mut grad_coords = coords.clone();
            grad_coords[self.dim] = 0;

            let grad_idx: usize = grad_coords
                .iter()
                .zip(&grad_strides)
                .map(|(c, s)| c * s)
                .sum();
            result[i] = grad_data[grad_idx];
        }

        vec![Some(RawTensor::new(result, &self.input_shape, false))]
    }

    fn clone_box(&self) -> Box<dyn GradFn> {
        Box::new(SumDimGradFn {
            input_shape: self.input_shape.clone(),
            dim: self.dim,
            keepdim: self.keepdim,
        })
    }
}

/// Gradient for `max_dim`: sparse gradient to max elements only
struct MaxDimGradFn {
    input_shape: Vec<usize>,
    max_indices: Vec<usize>, // linear indices of max elements
    dim: usize,
    keepdim: bool,
}

impl GradFn for MaxDimGradFn {
    fn backward(&self, out_grad: &RawTensor, _parents: &[Tensor]) -> Vec<Option<Tensor>> {
        let grad_data = &out_grad.data;
        let grad_shape = &out_grad.shape;

        let mut expanded_shape = grad_shape.clone();
        if !self.keepdim {
            expanded_shape.insert(self.dim, 1);
        }

        let size: usize = self.input_shape.iter().product();
        let mut result = vec![0.0; size];

        // Only max elements receive gradient
        let grad_strides = RawTensor::compute_strides(&expanded_shape);

        for (out_idx, &max_lin_idx) in self.max_indices.iter().enumerate() {
            // Convert output index to coordinates in expanded shape
            let mut grad_coords = vec![0; expanded_shape.len()];
            let mut rem = out_idx;
            for (d, &dim_sz) in expanded_shape.iter().enumerate().rev() {
                grad_coords[d] = rem % dim_sz;
                rem /= dim_sz;
            }

            let grad_idx: usize = grad_coords
                .iter()
                .zip(&grad_strides)
                .map(|(c, s)| c * s)
                .sum();
            result[max_lin_idx] = grad_data[grad_idx];
        }

        vec![Some(RawTensor::new(result, &self.input_shape, false))]
    }

    fn clone_box(&self) -> Box<dyn GradFn> {
        Box::new(MaxDimGradFn {
            input_shape: self.input_shape.clone(),
            max_indices: self.max_indices.clone(),
            dim: self.dim,
            keepdim: self.keepdim,
        })
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
    /// let x = `Tensor::new(vec`![1,2,3,4,5,6], &\[2,3\], true);
    /// `x.sum_dim(1`, false) // -> [6, 15] shape \[2\]
    /// `x.sum_dim(1`, true)  // -> [\[6\], \[15\]] shape \[2,1\]
    pub fn sum_dim(self_t: &Tensor, dim: usize, keepdim: bool) -> Tensor {
        let (data, shape, req_grad, device) = {
            let s = self_t.borrow();
            assert!(
                dim < s.shape.len(),
                "dim {} out of bounds for shape {:?}",
                dim,
                s.shape
            );
            (
                s.data.clone(),
                s.shape.clone(),
                s.requires_grad,
                s.device.clone(),
            )
        };

        let dim_size = shape[dim];
        let mut out_shape = shape.clone();
        out_shape[dim] = 1; // intermediate shape before squeeze
        let out_size: usize = out_shape.iter().product();
        let mut result = vec![0.0; out_size];

        // Optimized stride-based reduction: O(1) per element instead of O(rank) coordinate conversion
        // View data as: [outer_size, dim_size, inner_size] where we sum over dim_size
        let outer_size: usize = shape[..dim].iter().product();
        let inner_size: usize = shape[dim + 1..].iter().product();

        // Sum over the target dimension using stride arithmetic
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let out_idx = outer * inner_size + inner;
                let mut sum = 0.0;
                for k in 0..dim_size {
                    let in_idx = outer * dim_size * inner_size + k * inner_size + inner;
                    sum += data[in_idx];
                }
                result[out_idx] = sum;
            }
        }

        // Squeeze dimension if keepdim=false
        let final_shape = if keepdim {
            out_shape
        } else {
            out_shape
                .iter()
                .enumerate()
                .filter(|(d, _)| *d != dim)
                .map(|(_, &sz)| sz)
                .collect()
        };

        let out = Self::new(result, &final_shape, req_grad);
        // Place the reduction result on the same logical device as the input.
        {
            let mut ob = out.borrow_mut();
            ob.data = ob.data.to_device(&device);
            ob.device = device;
        }

        if req_grad {
            out.borrow_mut().parents = vec![self_t.clone()];
            out.borrow_mut().grad_fn = Some(Box::new(SumDimGradFn {
                input_shape: shape,
                dim,
                keepdim,
            }));
        }
        out
    }

    /// Move tensor data to a different device.
    ///
    /// - `data` is moved via `Storage::to_device`.
    /// - Gradients remain on CPU storage for now, since the autograd engine and
    ///   all gradient accumulation logic are CPU-only.
    /// - Autograd metadata (`requires_grad`, `grad_fn`, `parents`) is preserved,
    ///   so the returned tensor participates in the same computation graph.
    pub fn to_device(self_t: &Tensor, device: Device) -> Tensor {
        // Fast path: already on requested device
        {
            let t = self_t.borrow();
            if t.device == device {
                return self_t.clone();
            }
        }

        let t = self_t.borrow();

        let new_data = t.data.to_device(&device);
        // Keep gradients on CPU for now – backward and accumulation are CPU-only.
        let new_grad = t.grad.clone();

        let new_tensor = RawTensor {
            data: new_data,
            shape: t.shape.clone(),
            grad: new_grad,
            requires_grad: t.requires_grad,
            grad_fn: t.grad_fn.as_ref().map(|gf| gf.clone_box()),
            parents: t.parents.clone(),
            device,
        };
        Rc::new(RefCell::new(new_tensor))
    }

    /// Max along a specific axis
    ///
    /// Returns maximum value along dimension and stores indices for backward pass.
    pub fn max_dim(self_t: &Tensor, dim: usize, keepdim: bool) -> Tensor {
        let (data, shape, req_grad, device) = {
            let s = self_t.borrow();
            assert!(
                dim < s.shape.len(),
                "dim {} out of bounds for shape {:?}",
                dim,
                s.shape
            );
            (
                s.data.clone(),
                s.shape.clone(),
                s.requires_grad,
                s.device.clone(),
            )
        };

        let dim_size = shape[dim];
        let mut out_shape = shape.clone();
        out_shape[dim] = 1;
        let out_size: usize = out_shape.iter().product();

        let mut result = vec![f32::NEG_INFINITY; out_size];
        let mut max_indices = vec![0; out_size]; // track which index won

        // Optimized stride-based reduction: O(1) per element instead of O(rank) coordinate conversion
        // View data as: [outer_size, dim_size, inner_size] where we find max over dim_size
        let outer_size: usize = shape[..dim].iter().product();
        let inner_size: usize = shape[dim + 1..].iter().product();

        // Find max over the target dimension using stride arithmetic
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let out_idx = outer * inner_size + inner;
                for k in 0..dim_size {
                    let in_idx = outer * dim_size * inner_size + k * inner_size + inner;
                    if data[in_idx] > result[out_idx] {
                        result[out_idx] = data[in_idx];
                        max_indices[out_idx] = in_idx; // store linear index of max element
                    }
                }
            }
        }

        let final_shape = if keepdim {
            out_shape.clone()
        } else {
            out_shape
                .iter()
                .enumerate()
                .filter(|(d, _)| *d != dim)
                .map(|(_, &sz)| sz)
                .collect()
        };

        let out = Self::new(result, &final_shape, req_grad);
        // Ensure the max result lives on the same logical device as the input.
        {
            let mut ob = out.borrow_mut();
            ob.data = ob.data.to_device(&device);
            ob.device = device;
        }

        if req_grad {
            out.borrow_mut().parents = vec![self_t.clone()];
            out.borrow_mut().grad_fn = Some(Box::new(MaxDimGradFn {
                input_shape: shape,
                max_indices,
                dim,
                keepdim,
            }));
        }
        out
    }

    pub fn softmax(self_t: &Tensor, dim: usize) -> Tensor {
        let max = Self::max_dim(self_t, dim, true);
        let shifted = self_t.sub(&max);
        let exp_x = shifted.exp();
        let sum_exp = Self::sum_dim(&exp_x, dim, true);
        exp_x.div(&sum_exp)
    }
    /// `LogSoftmax` along a specific axis (numerically stable)
    /// `log(exp(x_i)` / `sum(exp(x_j))`) = `x_i` - `log(sum(exp(x_j)))`
    /// Uses `LogSumExp` trick: log(sum(exp(x))) = m + log(sum(exp(x-m)))
    pub fn log_softmax(self_t: &Tensor, dim: usize) -> Tensor {
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
    pub fn mean_dim(self_t: &Tensor, dim: usize, keepdim: bool) -> Tensor {
        let (shape, device) = {
            let t = self_t.borrow();
            (t.shape.clone(), t.device.clone())
        };
        assert!(dim < shape.len(), "Dimension out of bounds");

        let n = shape[dim] as f32;
        let sum = Self::sum_dim(self_t, dim, keepdim);
        let div_tensor = Self::new(vec![n], &[1], false);

        let mean = sum.div(&div_tensor);
        // Keep mean result on same device as input
        Self::to_device(&mean, device)
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
        let loss = loss_fn(tensor);
        loss.backward();

        let analytical_grad = tensor.grad().expect("Tensor must have gradient");
        let mut numerical_grad = vec![0.0; analytical_grad.len()];

        let original_data = tensor.borrow().data.clone();
        let original_shape = tensor.borrow().shape.clone();
        let requires_grad = tensor.borrow().requires_grad;

        // Compute numerical gradient for each element
        for i in 0..original_data.len() {
            //f(x + epsilon)
            let mut data_plus = original_data.clone();
            data_plus[i] += epsilon;
            let tensor_plus = RawTensor::new(data_plus.to_vec(), &original_shape, requires_grad);
            let loss_plus = loss_fn(&tensor_plus);
            let val_plus = loss_plus.borrow().data[0];

            let mut data_minus = original_data.clone();
            data_minus[i] -= epsilon;
            let tensor_minus = RawTensor::new(data_minus.to_vec(), &original_shape, requires_grad);
            let loss_minus = loss_fn(&tensor_minus);
            let val_minus = loss_minus.borrow().data[0];
            //central diff
            numerical_grad[i] = (val_plus - val_minus) / (2.0 * epsilon);
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
                eprintln!(
                    "Gradient mismatch at index {}: analytical={:.6e}, numerical={:.6e}, error={:.6e}",
                    i, analytical, numerical, relative_error
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
            eprintln!(
                "Gradient check FAILED: max_error={:.6e}, mean_error={:.6e}",
                max_err, mean_err
            );
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

// ===== TRAIT-BASED API =====

/// Public trait for tensor operations
///
/// This provides a more ergonomic API: `tensor.add(&other)` instead of `RawTensor::add(&tensor, &other)`
pub trait TensorOps {
    //Binary ops
    fn add(&self, other: &Tensor) -> Tensor;
    fn sub(&self, other: &Tensor) -> Tensor;
    fn elem_mul(&self, other: &Tensor) -> Tensor;
    fn div(&self, other: &Tensor) -> Tensor;
    fn max_elem(&self, other: &Tensor) -> Tensor;
    fn modulo(&self, other: &Tensor) -> Tensor;
    fn cmplt(&self, other: &Tensor) -> Tensor;

    // Unary ops
    fn neg(&self) -> Tensor;
    fn recip(&self) -> Tensor;
    fn sqrt(&self) -> Tensor;
    fn exp2(&self) -> Tensor;
    fn log2(&self) -> Tensor;
    fn exp(&self) -> Tensor;
    fn log(&self) -> Tensor;
    fn sin(&self) -> Tensor;
    fn cos(&self) -> Tensor;
    fn tanh(&self) -> Tensor;
    fn sigmoid(&self) -> Tensor;
    fn relu(&self) -> Tensor;

    //Reduce ops
    fn sum(&self) -> Tensor;
    fn max_reduce(&self) -> Tensor;
    fn mean(&self) -> Tensor;

    //Ternary ops
    fn mulacc(&self, y: &Tensor, z: &Tensor) -> Tensor;
    fn where_op(&self, x: &Tensor, y: &Tensor) -> Tensor;

    // Movement ops
    fn reshape(&self, new_shape: &[usize]) -> Tensor;
    fn permute(&self, axes: &[usize]) -> Tensor;
    fn expand(&self, new_shape: &[usize]) -> Tensor;
    fn pad(&self, padding: &[(usize, usize)]) -> Tensor;
    fn shrink(&self, ranges: &[(usize, usize)]) -> Tensor;
    fn stride_op(&self, strides: &[usize]) -> Tensor;
    fn to_device(&self, device: Device) -> Tensor;

    //Matmul
    fn matmul(&self, other: &Tensor) -> Tensor;
    fn transpose(&self) -> Tensor;

    //Gradient ops
    fn backward(&self);
    fn grad(&self) -> Option<Vec<f32>>;
    fn zero_grad(&self);

    // Axis reductions
    fn sum_dim(&self, dim: usize, keepdim: bool) -> Tensor;
    fn max_dim(&self, dim: usize, keepdim: bool) -> Tensor;
    fn mean_dim(&self, dim: usize, keepdim: bool) -> Tensor;

    // Softmax
    fn softmax(&self, dim: usize) -> Tensor;
    fn log_softmax(&self, dim: usize) -> Tensor;
}

impl TensorOps for Tensor {
    fn add(&self, other: &Tensor) -> Tensor {
        RawTensor::add(self, other)
    }
    fn sub(&self, other: &Tensor) -> Tensor {
        RawTensor::sub(self, other)
    }
    fn elem_mul(&self, other: &Tensor) -> Tensor {
        RawTensor::elem_mul(self, other)
    }
    fn div(&self, other: &Tensor) -> Tensor {
        RawTensor::div(self, other)
    }
    fn max_elem(&self, other: &Tensor) -> Tensor {
        RawTensor::max_elem(self, other)
    }
    fn modulo(&self, other: &Tensor) -> Tensor {
        RawTensor::modulo(self, other)
    }
    fn cmplt(&self, other: &Tensor) -> Tensor {
        RawTensor::cmplt(self, other)
    }

    fn neg(&self) -> Tensor {
        RawTensor::neg(self)
    }
    fn recip(&self) -> Tensor {
        RawTensor::recip(self)
    }
    fn sqrt(&self) -> Tensor {
        RawTensor::sqrt(self)
    }
    fn exp2(&self) -> Tensor {
        RawTensor::exp2(self)
    }
    fn log2(&self) -> Tensor {
        RawTensor::log2(self)
    }
    fn exp(&self) -> Tensor {
        RawTensor::exp(self)
    }
    fn log(&self) -> Tensor {
        RawTensor::log(self)
    }
    fn sin(&self) -> Tensor {
        RawTensor::sin(self)
    }
    fn cos(&self) -> Tensor {
        RawTensor::cos(self)
    }
    fn tanh(&self) -> Tensor {
        RawTensor::tanh(self)
    }
    fn sigmoid(&self) -> Tensor {
        RawTensor::sigmoid(self)
    }
    fn relu(&self) -> Tensor {
        RawTensor::relu(self)
    }

    fn sum(&self) -> Tensor {
        RawTensor::sum(self)
    }
    fn max_reduce(&self) -> Tensor {
        RawTensor::max_reduce(self)
    }
    fn mean(&self) -> Tensor {
        RawTensor::mean(self)
    }

    fn mulacc(&self, y: &Tensor, z: &Tensor) -> Tensor {
        RawTensor::mulacc(self, y, z)
    }
    fn where_op(&self, x: &Tensor, y: &Tensor) -> Tensor {
        RawTensor::where_op(self, x, y)
    }

    fn reshape(&self, new_shape: &[usize]) -> Tensor {
        RawTensor::reshape(self, new_shape)
    }
    fn permute(&self, axes: &[usize]) -> Tensor {
        RawTensor::permute(self, axes)
    }
    fn expand(&self, new_shape: &[usize]) -> Tensor {
        RawTensor::expand(self, new_shape)
    }
    fn pad(&self, padding: &[(usize, usize)]) -> Tensor {
        RawTensor::pad(self, padding)
    }
    fn shrink(&self, ranges: &[(usize, usize)]) -> Tensor {
        RawTensor::shrink(self, ranges)
    }
    fn stride_op(&self, strides: &[usize]) -> Tensor {
        RawTensor::stride_op(self, strides)
    }

    fn to_device(&self, device: Device) -> Tensor {
        RawTensor::to_device(self, device)
    }

    fn matmul(&self, other: &Tensor) -> Tensor {
        RawTensor::matmul(self, other)
    }
    fn transpose(&self) -> Tensor {
        RawTensor::transpose(self)
    }

    fn backward(&self) {
        RawTensor::backward(self)
    }
    fn grad(&self) -> Option<Vec<f32>> {
        self.borrow().grad.as_ref().map(|g| g.to_vec())
    }
    fn zero_grad(&self) {
        RawTensor::zero_grad(self)
    }
    fn sum_dim(&self, dim: usize, keepdim: bool) -> Tensor {
        RawTensor::sum_dim(self, dim, keepdim)
    }
    fn max_dim(&self, dim: usize, keepdim: bool) -> Tensor {
        RawTensor::max_dim(self, dim, keepdim)
    }
    fn mean_dim(&self, dim: usize, keepdim: bool) -> Tensor {
        RawTensor::mean_dim(self, dim, keepdim)
    }
    fn softmax(&self, dim: usize) -> Tensor {
        RawTensor::softmax(self, dim)
    }
    fn log_softmax(&self, dim: usize) -> Tensor {
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

        DataLoader {
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
        let batch_indices = &self.indices[self.current..end];
        let actual_batch = batch_indices.len();

        let sample_size: usize = self.data_shape.iter().product();
        let target_size: usize = self.target_shape.iter().product();

        // Gather batch
        let mut batch_data = Vec::with_capacity(actual_batch * sample_size);
        let mut batch_targets = Vec::with_capacity(actual_batch * target_size);

        for &idx in batch_indices {
            let data_start = idx * sample_size;
            let target_start = idx * target_size;

            batch_data.extend_from_slice(&self.data[data_start..data_start + sample_size]);
            batch_targets
                .extend_from_slice(&self.targets[target_start..target_start + target_size]);
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

pub fn randn_like(tensor: &Tensor) -> Tensor {
    RawTensor::randn_like(tensor)
}

// Loss functions
pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    RawTensor::mse_loss(pred, target)
}

pub fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Tensor {
    RawTensor::cross_entropy_loss(logits, targets)
}

pub fn nll_loss(log_probs: &Tensor, targets: &Tensor) -> Tensor {
    RawTensor::nll_loss(log_probs, targets)
}

pub fn kl_divergence_gaussian(mu: &Tensor, logvar: &Tensor) -> Tensor {
    RawTensor::kl_divergence_gaussian(mu, logvar)
}

pub fn bce_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    RawTensor::bce_loss(pred, target)
}

pub fn bce_with_logits_loss(logits: &Tensor, target: &Tensor) -> Tensor {
    RawTensor::bce_with_logits_loss(logits, target)
}

// Axis reductions
pub fn sum_dim(t: &Tensor, dim: usize, keepdim: bool) -> Tensor {
    RawTensor::sum_dim(t, dim, keepdim)
}

pub fn max_dim(t: &Tensor, dim: usize, keepdim: bool) -> Tensor {
    RawTensor::max_dim(t, dim, keepdim)
}

pub fn softmax(t: &Tensor, dim: usize) -> Tensor {
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

    #[test]
    fn test_he_initialization_linear_shape() {
        let t = RawTensor::he_initialization(&[64, 32]);
        let b = t.borrow();
        assert_eq!(b.shape, vec![64, 32]);
        assert_eq!(b.data.len(), 64 * 32);
        assert!(b.data.iter().any(|v| v.abs() > 0.0));
    }

    #[test]
    fn test_he_initialization_conv_shape() {
        let t = RawTensor::he_initialization(&[16, 3, 3, 3]);
        let b = t.borrow();
        assert_eq!(b.shape, vec![16, 3, 3, 3]);
        assert_eq!(b.data.len(), 16 * 3 * 3 * 3);
        assert!(b.data.iter().any(|v| v.abs() > 0.0));
    }
}

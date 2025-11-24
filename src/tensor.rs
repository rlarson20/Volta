use crate::autograd::GradFn;
use crate::device::Device;
use rand::Rng;
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
/// - `data`: flat Vec<f32> of actual values (row-major order)
/// - `shape`: dimensions, e.g. [batch, channels, height, width]
/// - `grad`: accumulated gradient (Some if requires_grad, None otherwise)
/// - `requires_grad`: whether to track gradients for this tensor
/// - `grad_fn`: function to compute parent gradients during backward
/// - `parents`: input tensors that this tensor depends on
/// - `device`: where computation happens (CPU/GPU)
pub struct RawTensor {
    pub data: Vec<f32>,         // flat data vec, len = prod shape dims
    pub shape: Vec<usize>,      //tensor dims, eg [B,C,H,W]
    pub grad: Option<Vec<f32>>, //grad w.r.t tensor data, None if req_grad == false
    pub requires_grad: bool,
    pub grad_fn: Option<Box<dyn GradFn>>, //func to compute grad, if result of op
    pub parents: Vec<Tensor>,             //refs to parent tensor on graph
    pub device: Device,                   //cpu/gpu
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
    /// Panics if data.len() != shape.product()
    pub fn new(data: Vec<f32>, shape: &[usize], requires_grad: bool) -> Tensor {
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
            device: Device::CPU,
        };
        Rc::new(RefCell::new(raw))
    }
    /// Create a tensor filled with zeros
    pub fn zeros(shape: &[usize]) -> Tensor {
        let size = shape.iter().product();
        Self::new(vec![0.0; size], shape, false)
    }
    /// Create a tensor filled with ones
    pub fn ones(shape: &[usize]) -> Tensor {
        let size = shape.iter().product();
        Self::new(vec![1.0; size], shape, false)
    }
    /// Create a tensor with random values uniformly distributed in [0, 1)
    pub fn rand(shape: &[usize]) -> Tensor {
        let size = shape.iter().product();
        let mut rng = rand::rng();
        let data: Vec<f32> = (0..size).map(|_| rng.random::<f32>()).collect();
        Self::new(data, shape, false)
    }
    /// Create a tensor with values from standard normal distribution N(0, 1)
    pub fn randn(shape: &[usize]) -> Tensor {
        let size = shape.iter().product();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut rng = rand::rng();
        let data: Vec<f32> = (0..size).map(|_| normal.sample(&mut rng)).collect();
        Self::new(data, shape, false)
    }
    /// Xavier uniform initialization
    ///
    /// Samples weights uniformly from [-limit, limit] where
    /// limit = sqrt(6 / (fan_in + fan_out))
    ///
    /// This helps maintain gradient variance across layers.
    pub fn xavier_uniform(shape: &[usize]) -> Tensor {
        let fan_in = shape[0];
        let fan_out = shape[1];
        let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
        let data: Vec<f32> = (0..fan_in * fan_out)
            .map(|_| rand::rng().random_range(-limit..limit))
            .collect();
        Self::new(data, shape, false)
    }
    //TODO: implement ACTUAL He initialization
    //which is better for ReLU using networks
    //
    //whereas Xavier samples each element in W^(l) from the fan-in/out,
    //He uses a sample from N(0, 2/n_{l-1})
    pub fn he_initialization(_shape: &[usize]) -> Tensor {
        todo!("Need a deeper review before I add this.")
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
        let softmax = Self::softmax(logits, 1);
        let log_probs = softmax.log();
        // -sum(targets * log_probs, dim=1).mean()
        let prod = targets.elem_mul(&log_probs);
        let sum = Self::sum_dim(&prod, 1, false);
        sum.neg().mean()
    }
}

// ===== SOFTMAX & AXIS REDUCTIONS =====

/// Gradient for sum_dim: broadcast ones back to input shape
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

/// Gradient for max_dim: sparse gradient to max elements only
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
    /// let x = Tensor::new(vec![1,2,3,4,5,6], &[2,3], true);
    /// x.sum_dim(1, false) // -> [6, 15] shape [2]
    /// x.sum_dim(1, true)  // -> [[6], [15]] shape [2,1]
    pub fn sum_dim(self_t: &Tensor, dim: usize, keepdim: bool) -> Tensor {
        let (data, shape, req_grad) = {
            let s = self_t.borrow();
            assert!(
                dim < s.shape.len(),
                "dim {} out of bounds for shape {:?}",
                dim,
                s.shape
            );
            (s.data.clone(), s.shape.clone(), s.requires_grad)
        };

        let _dim_size = shape[dim];
        let mut out_shape = shape.clone();
        out_shape[dim] = 1; // intermediate shape before squeeze
        let out_size: usize = out_shape.iter().product();
        let mut result = vec![0.0; out_size];

        // Compute strides for indexing
        let _strides = Self::compute_strides(&shape);
        let out_strides = Self::compute_strides(&out_shape);

        // Sum over the target dimension
        #[allow(clippy::needless_range_loop)]
        for i in 0..data.len() {
            // Convert linear index to coordinates
            let mut coords = vec![0; shape.len()];
            let mut rem = i;
            for (d, &dim_sz) in shape.iter().enumerate().rev() {
                coords[d] = rem % dim_sz;
                rem /= dim_sz;
            }

            // Zero out the target dimension for output indexing
            let mut out_coords = coords.clone();
            out_coords[dim] = 0;

            // Convert output coords to linear index
            let out_idx: usize = out_coords
                .iter()
                .zip(&out_strides)
                .map(|(c, s)| c * s)
                .sum();
            result[out_idx] += data[i];
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

    /// Max along a specific axis
    ///
    /// Returns maximum value along dimension and stores indices for backward pass.
    pub fn max_dim(self_t: &Tensor, dim: usize, keepdim: bool) -> Tensor {
        let (data, shape, req_grad) = {
            let s = self_t.borrow();
            assert!(
                dim < s.shape.len(),
                "dim {} out of bounds for shape {:?}",
                dim,
                s.shape
            );
            (s.data.clone(), s.shape.clone(), s.requires_grad)
        };

        let _dim_size = shape[dim];
        let mut out_shape = shape.clone();
        out_shape[dim] = 1;
        let out_size: usize = out_shape.iter().product();

        let mut result = vec![f32::NEG_INFINITY; out_size];
        let mut max_indices = vec![0; out_size]; // track which index won

        let _strides = Self::compute_strides(&shape);
        let out_strides = Self::compute_strides(&out_shape);

        #[allow(clippy::needless_range_loop)]
        for i in 0..data.len() {
            let mut coords = vec![0; shape.len()];
            let mut rem = i;
            for (d, &dim_sz) in shape.iter().enumerate().rev() {
                coords[d] = rem % dim_sz;
                rem /= dim_sz;
            }

            let mut out_coords = coords.clone();
            out_coords[dim] = 0;
            let out_idx: usize = out_coords
                .iter()
                .zip(&out_strides)
                .map(|(c, s)| c * s)
                .sum();

            if data[i] > result[out_idx] {
                result[out_idx] = data[i];
                max_indices[out_idx] = i; // store linear index of max element
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

    /// Mean along a specific axis
    ///
    /// Implemented as sum_dim(dim) / size(dim)
    pub fn mean_dim(self_t: &Tensor, dim: usize, keepdim: bool) -> Tensor {
        let (shape, _req_grad) = {
            let t = self_t.borrow();
            (t.shape.clone(), t.requires_grad)
        };
        assert!(dim < shape.len(), "Dimension out of bounds");

        let n = shape[dim] as f32;
        let sum = Self::sum_dim(self_t, dim, keepdim);
        let div_tensor = Self::new(vec![n], &[1], false);

        sum.div(&div_tensor)
    }
}

// ===== NUMERICAL GRADIENT CHECKING =====

impl RawTensor {
    /// Check gradients numerically using finite differences
    ///
    /// For each parameter, we compute:
    ///
    /// Analytical gradient: What our backward() computes
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
    /// (max_error, mean_error, passed)
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
            let tensor_plus = RawTensor::new(data_plus, &original_shape, requires_grad);
            let loss_plus = loss_fn(&tensor_plus);
            let val_plus = loss_plus.borrow().data[0];

            let mut data_minus = original_data.clone();
            data_minus[i] -= epsilon;
            let tensor_minus = RawTensor::new(data_minus, &original_shape, requires_grad);
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
            let relative_error = if numerical.abs() > 1e-8 {
                error / numerical.abs()
            } else {
                error
            };

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

// OR WHERE TRAIT API GOES
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

    //Matmul
    fn matmul(&self, other: &Tensor) -> Tensor;
    fn transpose(&self) -> Tensor;

    //Gradient ops
    fn backward(&self);
    fn grad(&self) -> Option<Vec<f32>>;

    // Axis reductions
    fn sum_dim(&self, dim: usize, keepdim: bool) -> Tensor;
    fn max_dim(&self, dim: usize, keepdim: bool) -> Tensor;
    fn mean_dim(&self, dim: usize, keepdim: bool) -> Tensor;

    // Softmax
    fn softmax(&self, dim: usize) -> Tensor;
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
        self.borrow().grad.clone()
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
}

impl DataLoader {
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
        }
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

        Some((
            RawTensor::new(batch_data, &batch_shape, false),
            RawTensor::new(batch_targets, &target_batch_shape, false),
        ))
    }
}

// ===== PUBLIC API EXPORTS =====
pub use RawTensor as new_tensor;

// Tensor constructors
pub fn zeros(shape: &[usize]) -> Tensor {
    RawTensor::zeros(shape)
}

pub fn ones(shape: &[usize]) -> Tensor {
    RawTensor::ones(shape)
}

pub fn rand(shape: &[usize]) -> Tensor {
    RawTensor::rand(shape)
}

pub fn randn(shape: &[usize]) -> Tensor {
    RawTensor::randn(shape)
}

// Loss functions
pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    RawTensor::mse_loss(pred, target)
}

pub fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Tensor {
    RawTensor::cross_entropy_loss(logits, targets)
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

use crate::device::Device;
use crate::device_policy::{Pass, common_device, kernel_failed, on_gap};
use crate::error::{Result, VoltaError};
use crate::nn::layers::conv::{DirectConvGradFn, IGEMMGradFn, Im2colGradFn};
use crate::nn::layers::conv_transpose::ConvTranspose2dGradFn;
use crate::nn::layers::embedding::EmbeddingGradFn;
use crate::nn::layers::maxpool::MaxPool2dGradFn;
use crate::ops::binary::BinaryGradFn;
use crate::ops::matmul::{MatMulGradFn, TransposeGradFn};
use crate::ops::movement::MovementGradFn;
use crate::ops::reduce::{MaxReduceGradFn, MeanGradFn, SumGradFn};
use crate::ops::ternary::{MulAccGradFn, WhereGradFn};
use crate::ops::unary::UnaryGradFn;
use crate::shape::Shape;
use crate::storage::Storage;
use crate::tensor::{MaxDimGradFn, RawTensor, SumDimGradFn, Tensor};
use std::cell::Cell;
use std::collections::HashSet;

// ===== GRADIENT FUNCTIONS =====

/// How gradients flow backward through one operation.
///
/// A closed enum (ADDENDUM §2.5): each variant wraps the per-op struct that holds the
/// state `backward` needs. Because `wildcard_enum_match_arm` is denied, adding a variant
/// forces an arm in both matches below, so an op can't ship without a backward pass.
///
/// `backward` takes:
/// - `out_grad`: gradient of loss w.r.t. this operation's output
/// - `parents`: the input tensors to this operation
///
/// and returns one gradient per parent (`Some` if it needs one, `None` otherwise).
#[derive(Clone)]
pub(crate) enum GradFn {
    Unary(UnaryGradFn),
    Binary(BinaryGradFn),
    MatMul(MatMulGradFn),
    Transpose(TransposeGradFn),
    Sum(SumGradFn),
    MaxReduce(MaxReduceGradFn),
    Mean(MeanGradFn),
    SumDim(SumDimGradFn),
    MaxDim(MaxDimGradFn),
    MulAcc(MulAccGradFn),
    Where(WhereGradFn),
    Movement(MovementGradFn),
    Im2col(Im2colGradFn),
    DirectConv(DirectConvGradFn),
    Igemm(IGEMMGradFn),
    ConvTranspose2d(ConvTranspose2dGradFn),
    MaxPool2d(MaxPool2dGradFn),
    Embedding(EmbeddingGradFn),
}

impl GradFn {
    /// The op's name in device errors and fallback events: the variant, plus the inner
    /// op where there is one (`"Binary(Add)"`, `"Movement(Permute)"`, `"MatMul"`).
    pub(crate) fn op_name(&self) -> String {
        match self {
            Self::Unary(g) => g.op_name(),
            Self::Binary(g) => g.op_name(),
            Self::Movement(g) => g.op_name(),
            Self::MatMul(_) => "MatMul".to_string(),
            Self::Transpose(_) => "Transpose".to_string(),
            Self::Sum(_) => "Sum".to_string(),
            Self::MaxReduce(_) => "MaxReduce".to_string(),
            Self::Mean(_) => "Mean".to_string(),
            Self::SumDim(_) => "SumDim".to_string(),
            Self::MaxDim(_) => "MaxDim".to_string(),
            Self::MulAcc(_) => "MulAcc".to_string(),
            Self::Where(_) => "Where".to_string(),
            Self::Im2col(_) => "Im2col".to_string(),
            Self::DirectConv(_) => "DirectConv".to_string(),
            Self::Igemm(_) => "Igemm".to_string(),
            Self::ConvTranspose2d(_) => "ConvTranspose2d".to_string(),
            Self::MaxPool2d(_) => "MaxPool2d".to_string(),
            Self::Embedding(_) => "Embedding".to_string(),
        }
    }

    /// Compute gradients for parent tensors given output gradient.
    pub(crate) fn backward(&self, out_grad: &RawTensor, parents: &[Tensor]) -> Vec<Option<Tensor>> {
        match self {
            Self::Unary(g) => g.backward(out_grad, parents),
            Self::Binary(g) => g.backward(out_grad, parents),
            Self::MatMul(g) => g.backward(out_grad, parents),
            Self::Transpose(g) => g.backward(out_grad, parents),
            Self::Sum(g) => g.backward(out_grad, parents),
            Self::MaxReduce(g) => g.backward(out_grad, parents),
            Self::Mean(g) => g.backward(out_grad, parents),
            Self::SumDim(g) => g.backward(out_grad, parents),
            Self::MaxDim(g) => g.backward(out_grad, parents),
            Self::MulAcc(g) => g.backward(out_grad, parents),
            Self::Where(g) => g.backward(out_grad, parents),
            Self::Movement(g) => g.backward(out_grad, parents),
            Self::Im2col(g) => g.backward(out_grad, parents),
            Self::DirectConv(g) => g.backward(out_grad, parents),
            Self::Igemm(g) => g.backward(out_grad, parents),
            Self::ConvTranspose2d(g) => g.backward(out_grad, parents),
            Self::MaxPool2d(g) => g.backward(out_grad, parents),
            Self::Embedding(g) => g.backward(out_grad, parents),
        }
    }

    /// Whether this gradient function is a no-op (e.g. `Mod`/`Cmplt`'s comparison-like
    /// ops, which have no useful derivative). `edge()` never attaches a trivial
    /// `GradFn`, even if an input requires grad. Every variant is spelled out so a new
    /// one has to decide.
    pub(crate) fn is_trivial(&self) -> bool {
        match self {
            Self::Binary(g) => g.is_trivial(),
            Self::Unary(_)
            | Self::MatMul(_)
            | Self::Transpose(_)
            | Self::Sum(_)
            | Self::MaxReduce(_)
            | Self::Mean(_)
            | Self::SumDim(_)
            | Self::MaxDim(_)
            | Self::MulAcc(_)
            | Self::Where(_)
            | Self::Movement(_)
            | Self::Im2col(_)
            | Self::DirectConv(_)
            | Self::Igemm(_)
            | Self::ConvTranspose2d(_)
            | Self::MaxPool2d(_)
            | Self::Embedding(_) => false,
        }
    }
}

/// Edge creation function: the only code allowed to write `grad_fn`/`parents`.
///
/// Every op used to hand-write this "build output tensor, then conditionally attach
/// parents + `grad_fn`" ritual (see ADDENDUM.md §1.4). Centralizing it here means a new
/// op can't attach its grad fn under the wrong condition (the B1/B2/B3 bug class) —
/// "any input requires grad, `no_grad()` isn't active, and the grad isn't trivial" is
/// the only rule that exists, everywhere.
///
/// `shape` is a [`Shape`] (§1.2); `grad` is the closed [`GradFn`] enum (§2.5). Build
/// `shape` with `Shape::derived` when the op computed it from existing tensors. `device` isn't in the ADDENDUM sketch
/// either, but is required in practice: `Storage` doesn't uniquely determine a
/// `Device` value, and every real call site (GPU vs CPU op paths) already tracks
/// device separately from the data it produced.
///
/// It is also the forward device check (`crate::device_policy`): every input must be
/// on one device, and an output computed on the CPU for GPU inputs (an op with no GPU
/// kernel for this case) goes through the policy, which panics or moves it to the GPU.
///
/// # Panics
/// If the inputs are on different devices, under `DevicePolicy::Strict` when GPU inputs
/// produced a CPU output, or if CPU inputs produced a GPU output (an op bug).
pub(crate) fn edge(
    inputs: impl AsRef<[Tensor]>,
    mut data: Storage,
    shape: Shape,
    mut device: Device,
    grad: GradFn,
) -> Tensor {
    if let Some(in_device) = input_device(inputs.as_ref(), &grad) {
        match (in_device.is_gpu(), device.is_gpu()) {
            (true, false) => {
                on_gap(&grad.op_name(), Pass::Forward);
                data = data.to_device(&in_device);
                device = in_device;
            }
            (false, true) => panic!("{}: CPU inputs produced a {device} output", grad.op_name()),
            (true, true) | (false, false) => {}
        }
    }

    let requires_grad = is_grad_enabled()
        && inputs.as_ref().iter().any(Tensor::requires_grad)
        && !grad.is_trivial();

    let out = RawTensor::new_with_storage(data, shape, device, requires_grad);
    if requires_grad {
        let mut o = out.borrow_mut();
        o.parents = inputs.as_ref().to_vec();
        o.grad_fn = Some(grad);
    }
    out
}

/// The one device all `inputs` are on, or `None` for no inputs.
///
/// Compares first and names the op only on a mismatch, so the common case builds no
/// strings.
///
/// # Panics
/// If two inputs are on different devices.
fn input_device(inputs: &[Tensor], grad: &GradFn) -> Option<Device> {
    let first = inputs.first()?.borrow().device.clone();
    if inputs.iter().skip(1).any(|t| t.borrow().device != first) {
        let devices: Vec<Device> = inputs.iter().map(Tensor::device).collect();
        let _ = common_device(&grad.op_name(), &devices);
    }
    Some(first)
}

// Reentrant depth counter, not a bool: a plain on/off flag (as ADDENDUM.md §1.5
// sketches it) breaks the moment one `no_grad()` scope nests inside another — e.g.
// a layer's own `no_grad()` running inside `check_gradients`'s wrapping `no_grad()`.
// With a bool, the inner guard's `Drop` would reset the flag to `false` and
// incorrectly re-enable grad tracking while the outer scope is still active. A depth
// counter composes correctly: grad stays disabled until every nested guard has
// dropped.
//
// `thread_local!` rather than a bare `static`, matching the `GLOBAL_RNG` pattern
// already used in `src/tensor.rs`: a plain `static NO_GRAD_DEPTH: Cell<u32>` doesn't
// compile (`Cell` isn't `Sync`), and this crate is single-threaded by design anyway
// (`Rc<RefCell<_>>` throughout), so thread-local state is both correct and free.
thread_local! {
    static NO_GRAD_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// Whether `edge()` is currently allowed to attach graph edges.
pub(crate) fn is_grad_enabled() -> bool {
    NO_GRAD_DEPTH.with(|d| d.get() == 0)
}

/// Suppress graph attachment for the lifetime of the returned guard.
///
/// Use this to scope side-effecting mutations that shouldn't build graph — e.g.
/// `BatchNorm`'s running-stat update, or the perturbed forward passes inside
/// `check_gradients`'s finite-difference loop (ADDENDUM.md §1.5).
#[must_use]
pub fn no_grad() -> NoGradGuard {
    NO_GRAD_DEPTH.with(|d| d.set(d.get() + 1));
    NoGradGuard
}
pub struct NoGradGuard;
impl Drop for NoGradGuard {
    fn drop(&mut self) {
        NO_GRAD_DEPTH.with(|d| d.set(d.get() - 1));
    }
}

// ===== BACKPROPAGATION =====

impl RawTensor {
    /// Run backpropagation from a scalar loss.
    ///
    /// Only reachable through [`ScalarLoss::backward`], which has already checked
    /// that `tensor_ref` has one element and requires grad.
    ///
    /// 1. Seed this tensor's gradient with 1 (if not already set)
    /// 2. Traverse the computation graph backwards (topological sort via DFS)
    /// 3. For each node, call its `grad_fn` to compute parent gradients
    /// 4. Accumulate gradients in each parent tensor
    pub(crate) fn run_backward(tensor_ref: &Tensor) {
        enum Action {
            Visit(Tensor),
            PostVisit(Tensor),
        }

        // `into_loss` checked this, so it can only fail if another handle called
        // `set_requires_grad(false)` between `into_loss` and `backward`.
        debug_assert!(
            tensor_ref.borrow().requires_grad,
            "ScalarLoss::backward on a tensor that no longer requires grad"
        );
        // Initialize gradient if not already set
        {
            let mut tensor = tensor_ref.borrow_mut();
            if tensor.grad.is_none() {
                // Initialize grad on the same device as this tensor's data
                let base = Storage::cpu(vec![1.0]);
                let grad_storage = base.to_device(&tensor.device);
                tensor.grad = Some(grad_storage);
            }
        }

        // Use a topological sort to ensure we process nodes only after
        // ALL their consumers have contributed gradients.
        // A simple visited set in naive DFS is insufficient for "diamond" graphs
        // (nodes that are reachable via multiple paths).

        let mut topo_order = Vec::new();
        let mut visited = HashSet::new();

        // 1. Build topological order (post-order DFS)
        // We simulate recursion with a stack to avoid recursion limit issues on deep graphs
        let mut recursion_stack = vec![Action::Visit(tensor_ref.clone())];

        while let Some(action) = recursion_stack.pop() {
            match action {
                Action::Visit(t) => {
                    if visited.contains(&t.as_ptr()) {
                        continue;
                    }
                    visited.insert(t.as_ptr());
                    // Push post-visit marker
                    recursion_stack.push(Action::PostVisit(t.clone()));
                    // Push children (parents in backward graph) to visit
                    let parents = t.borrow().parents.clone();
                    for parent in parents {
                        recursion_stack.push(Action::Visit(parent));
                    }
                }
                Action::PostVisit(t) => {
                    topo_order.push(t);
                }
            }
        }

        // 2. Process in reverse topological order (consumers before producers)
        // topo_order has [leaf, ..., root]. We reverse to get [root, ..., leaf].

        for tensor in topo_order.into_iter().rev() {
            let (grad_fn, parents, grad_data, shape, device) = {
                let t = tensor.borrow();
                (
                    t.grad_fn.clone(),
                    t.parents.clone(),
                    t.grad.clone(),
                    t.shape.clone(),
                    t.device.clone(),
                )
            };
            // If this node has a gradient function and gradients, backpropagate
            if let Some(grad_fn) = grad_fn
                && let Some(grad_out_data) = grad_data
            {
                let grad_out = Self {
                    data: grad_out_data,
                    shape,
                    grad: None,
                    requires_grad: false,
                    grad_fn: None,
                    parents: vec![],
                    device,
                };

                // Compute gradients for parent tensors
                let parent_grads = grad_fn.backward(&grad_out, &parents);

                // Accumulate gradients in parents. `Tensor::accumulate_grad` is the
                // only code allowed to write `grad` (device-move + add-or-set), so
                // backward() just extracts the raw contribution and hands it over.
                for (parent_grad, parent_ref) in parent_grads.into_iter().zip(parents.iter()) {
                    if let Some(g) = parent_grad {
                        let grad_storage = g.borrow().data.clone();
                        parent_ref.accumulate_grad(grad_storage, &grad_fn);
                    }
                }
            }
        }
    }
}

/// A tensor that has been checked to be a single element that requires grad.
///
/// This is the only type with a `backward` method, so backward can't start from a
/// non-scalar or an untracked tensor. Get one with [`Tensor::into_loss`].
pub struct ScalarLoss(Tensor);

impl Tensor {
    /// Check that this tensor is a valid starting point for backward.
    ///
    /// # Errors
    /// [`VoltaError::NotScalar`] if it has more than one element, and
    /// [`VoltaError::NoGradRequired`] if it isn't tracked by autograd (for example
    /// it was built under `no_grad()` or from inputs that don't require grad).
    pub fn into_loss(self) -> Result<ScalarLoss> {
        let (numel, requires_grad) = {
            let t = self.borrow();
            (t.data.len(), t.requires_grad)
        };
        if numel != 1 {
            return Err(VoltaError::NotScalar {
                shape: self.shape().truncate_rank().to_vec(),
            });
        }
        if !requires_grad {
            return Err(VoltaError::NoGradRequired);
        }
        Ok(ScalarLoss(self))
    }
}

impl ScalarLoss {
    /// Backpropagate from this loss, accumulating into every reachable `grad`.
    ///
    /// Calling it again accumulates into the existing gradients.
    pub fn backward(&self) {
        RawTensor::run_backward(&self.0);
    }

    /// The loss value.
    ///
    /// # Panics
    /// Never in practice: `into_loss` guarantees exactly one element.
    #[must_use]
    pub fn item(&self) -> f32 {
        self.0
            .data()
            .first()
            .copied()
            .expect("ScalarLoss holds exactly one element")
    }

    /// The underlying tensor, for inspecting the loss or its gradient.
    #[must_use]
    pub fn tensor(&self) -> &Tensor {
        &self.0
    }
}

impl RawTensor {
    /// Add a gradient contribution to an existing gradient storage on the same device.
    ///
    /// # Panics
    /// If the two are on different devices, or the GPU add kernel returns no result.
    pub(crate) fn accumulate_grad(existing: &mut Storage, new_grad: &Storage) {
        assert_eq!(
            existing.is_gpu(),
            new_grad.is_gpu(),
            "accumulate_grad: gradient contributions on different devices"
        );
        if existing.is_gpu() {
            *existing = Self::gpu_add(existing, new_grad)
                .unwrap_or_else(|| kernel_failed("accumulate_grad"));
            return;
        }

        // CPU: element-wise addition on host data.
        let mut accum = existing.to_vec();
        let add = new_grad.to_vec();
        assert_eq!(
            accum.len(),
            add.len(),
            "Gradient size mismatch during accumulation"
        );
        for (a, b) in accum.iter_mut().zip(add.iter()) {
            *a += *b;
        }
        *existing = Storage::cpu(accum);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn into_loss_rejects_non_scalar() {
        let t = RawTensor::new(vec![1.0; 6], &[2, 3], true);
        match t.into_loss() {
            Err(VoltaError::NotScalar { shape }) => assert_eq!(shape, vec![2, 3]),
            Err(e) => panic!("wrong error: {e}"),
            Ok(_) => panic!("non-scalar accepted"),
        }
    }

    #[test]
    fn into_loss_rejects_untracked() {
        let t = RawTensor::new(vec![1.0], &[1], false);
        assert!(matches!(t.into_loss(), Err(VoltaError::NoGradRequired)));

        let x = RawTensor::new(vec![2.0], &[1], true);
        let y = {
            let _g = no_grad();
            x.elem_mul(&x)
        };
        assert!(matches!(y.into_loss(), Err(VoltaError::NoGradRequired)));
    }

    #[test]
    fn into_loss_accepts_any_single_element_shape() {
        for dims in [&[1][..], &[1, 1][..]] {
            let x = RawTensor::new(vec![3.0], dims, true);
            let loss = x.elem_mul(&x).into_loss().expect("single element");
            assert_eq!(loss.item(), 9.0);
            loss.backward();
            assert_eq!(x.grad().expect("grad"), vec![6.0]);
        }
    }

    #[test]
    fn backward_twice_accumulates() {
        let x = RawTensor::new(vec![3.0], &[1], true);
        let loss = x.elem_mul(&x).into_loss().expect("scalar");
        loss.backward();
        let first = x
            .grad()
            .expect("grad")
            .first()
            .copied()
            .expect("one element");
        loss.backward();
        let second = x
            .grad()
            .expect("grad")
            .first()
            .copied()
            .expect("one element");
        assert!(second > first, "second backward should add to the first");
    }

    /// A reshape shares its input's buffer (copy-on-write storage), so an in-place
    /// optimizer write to the input must copy first and leave the reshape's values alone.
    #[test]
    fn optimizer_write_does_not_leak_into_shared_reshape() {
        crate::tensor::manual_seed(7);
        let x = crate::tensor::randn(&[2, 3]);
        x.set_requires_grad(true);
        let y = x.reshape(&[3, 2]);
        assert_eq!(
            x.borrow().data.as_bytes().as_ptr(),
            y.borrow().data.as_bytes().as_ptr(),
            "reshape should share the buffer, or this test checks nothing"
        );
        let before = x.data();

        x.elem_mul(&x).sum().into_loss().expect("scalar").backward();
        let mut opt = crate::nn::optim::SGD::new(vec![x.clone()], 0.1, 0.0, 0.0);
        opt.step();

        assert_ne!(x.data(), before, "SGD step should have changed x");
        assert_eq!(y.data(), before);
    }

    /// `x` reaches the loss along two paths (`x * x` and `x`), so its gradient is the
    /// sum of both contributions: `2x + 1`.
    #[test]
    fn diamond_accumulates_both_paths() {
        crate::tensor::manual_seed(11);
        let x = crate::tensor::randn(&[5]);
        x.set_requires_grad(true);

        x.elem_mul(&x)
            .add(&x)
            .sum()
            .into_loss()
            .expect("scalar")
            .backward();

        let xs = x.data();
        let grad = x.grad().expect("x has a gradient");
        assert_eq!(grad.len(), xs.len());
        for (g, v) in grad.iter().zip(&xs) {
            assert!(
                (g - 2.0f32.mul_add(*v, 1.0)).abs() < 1e-6,
                "grad {g} at x={v}"
            );
        }
    }
}

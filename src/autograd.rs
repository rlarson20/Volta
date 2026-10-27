use crate::device::Device;
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
pub(crate) fn edge(
    inputs: impl AsRef<[Tensor]>,
    data: Storage,
    shape: Shape,
    device: Device,
    grad: GradFn,
) -> Tensor {
    let requires_grad = is_grad_enabled()
        && inputs.as_ref().iter().any(Tensor::requires_grad)
        && !grad.is_trivial();

    let out = RawTensor::new_with_storage(data, shape.truncate_rank(), device, requires_grad);
    if requires_grad {
        let mut o = out.borrow_mut();
        o.parents = inputs.as_ref().to_vec();
        o.grad_fn = Some(grad);
    }
    out
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
    /// Run backpropagation starting from this tensor
    ///
    /// This implements reverse-mode automatic differentiation:
    /// 1. Initialize this tensor's gradient to 1 (assumes it's a scalar loss)
    /// 2. Traverse the computation graph backwards (topological sort via DFS)
    /// 3. For each node, call its `grad_fn` to compute parent gradients
    /// 4. Accumulate gradients in each parent tensor
    ///
    /// Uses a `HashSet` to track visited nodes and avoid recomputation.
    /// # Panics
    /// Calling on tensor that doesn't need gradients
    pub fn backward(tensor_ref: &Tensor) {
        enum Action {
            Visit(Tensor),
            PostVisit(Tensor),
        }

        let tensor = tensor_ref.borrow();
        assert!(
            tensor.requires_grad,
            "Called backward on a tensor that doesn't require grad"
        );
        drop(tensor);
        // Initialize gradient if not already set
        {
            let mut tensor = tensor_ref.borrow_mut();
            if tensor.grad.is_none() {
                let grad_size = if tensor.shape.len() == 1 && tensor.shape.first() == Some(&1) {
                    1
                } else {
                    tensor.data.len()
                };
                // Initialize grad on the same device as this tensor's data
                let base = Storage::cpu(vec![1.0; grad_size]);
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
                        parent_ref.accumulate_grad(grad_storage);
                    }
                }
            }
        }
    }
}

impl RawTensor {
    /// Add a gradient contribution to an existing gradient storage, preferring GPU accumulation.
    pub(crate) fn accumulate_grad(existing: &mut Storage, new_grad: &Storage) {
        #[cfg(feature = "gpu")]
        {
            if existing.is_gpu() && new_grad.is_gpu() {
                if let Some(sum) = Self::gpu_add(existing, new_grad) {
                    *existing = sum;
                    return;
                }
                log::warn!("GPU gradient accumulation failed; falling back to CPU accumulation");
            }
        }

        // CPU fallback: both storages are synced to CPU for element-wise addition.
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

use crate::device::Device;
use crate::device_policy::{Pass, common_device, kernel_failed, on_gap};
use crate::dtype::DType;
use crate::error::{Result, VoltaError};
use crate::nn::layers::conv::{DirectConvGradFn, IGEMMGradFn, Im2colGradFn};
use crate::nn::layers::conv_transpose::ConvTranspose2dGradFn;
use crate::nn::layers::embedding::EmbeddingGradFn;
use crate::nn::layers::maxpool::MaxPool2dGradFn;
use crate::ops::binary::BinaryGradFn;
use crate::ops::loss::BceGradFn;
use crate::ops::matmul::{MatMulGradFn, TransposeGradFn};
use crate::ops::movement::MovementGradFn;
use crate::ops::reduce::{MaxReduceGradFn, MeanGradFn, SumGradFn};
use crate::ops::ternary::{MulAccGradFn, WhereGradFn};
use crate::ops::unary::UnaryGradFn;
use crate::shape::Shape;
use crate::storage::{Element, Storage};
use crate::tensor::{MaxDimGradFn, RawTensor, SumDimGradFn, Tensor};
use half::{bf16, f16};
use std::cell::Cell;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

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
    /// Elementwise unary ops (`exp`, `relu`, ...).
    Unary(UnaryGradFn),
    /// Elementwise binary ops with broadcasting.
    Binary(BinaryGradFn),
    /// Matrix product, in matrix and vector forms.
    MatMul(MatMulGradFn),
    /// 2D transpose.
    Transpose(TransposeGradFn),
    /// Sum over all elements.
    Sum(SumGradFn),
    /// Max over all elements.
    MaxReduce(MaxReduceGradFn),
    /// Mean over all elements.
    Mean(MeanGradFn),
    /// Sum along one axis.
    SumDim(SumDimGradFn),
    /// Max along one axis.
    MaxDim(MaxDimGradFn),
    /// Fused multiply-accumulate `x * y + z`.
    MulAcc(MulAccGradFn),
    /// Masked selection `condition ? x : y`.
    Where(WhereGradFn),
    /// Per-element binary cross entropy, with PyTorch's gradient.
    Bce(BceGradFn),
    /// Reshape, permute, expand, pad, shrink, stride.
    Movement(MovementGradFn),
    /// Convolution via `im2col` and a matmul.
    Im2col(Im2colGradFn),
    /// Direct (sliding-window) convolution.
    DirectConv(DirectConvGradFn),
    /// Convolution via implicit GEMM.
    Igemm(IGEMMGradFn),
    /// Transposed convolution.
    ConvTranspose2d(ConvTranspose2dGradFn),
    /// 2D max pooling.
    MaxPool2d(MaxPool2dGradFn),
    /// Embedding lookup.
    Embedding(EmbeddingGradFn),
    /// Device transfer (exempt from `edge()`'s input-to-output device check).
    ToDevice(ToDeviceGradFn),
    /// Dtype conversion (`to_dtype`).
    Cast(CastGradFn),
    /// Elementwise comparison. Never attached: its output is Bool.
    Compare(crate::ops::compare::CompareGradFn),
}

/// Backward of `to_dtype`: the gradient is cast back to the input's dtype, as PyTorch's
/// `.to(dtype)` backward does. Only built between float dtypes: a cast to an integer or
/// bool output carries no gradient (`edge()` never attaches one to a non-float output).
#[derive(Clone)]
pub(crate) struct CastGradFn {
    /// The input's dtype.
    pub(crate) source: DType,
}

impl CastGradFn {
    /// Casts `out_grad` to the source dtype.
    pub(crate) fn backward(
        &self,
        out_grad: &RawTensor,
        _parents: &[Tensor],
    ) -> Vec<Option<Tensor>> {
        vec![Some(RawTensor::new_with_storage(
            out_grad.data.cast(self.source),
            out_grad.shape.clone(),
            out_grad.device.clone(),
            false,
        ))]
    }
}

/// Backward of `to_device`: the gradient goes back to the device the tensor came from. A
/// deliberate transfer, so the device policy doesn't apply.
#[derive(Clone)]
pub(crate) struct ToDeviceGradFn {
    /// The input's device.
    pub(crate) source: Device,
}

impl ToDeviceGradFn {
    /// Moves `out_grad` to the source device.
    pub(crate) fn backward(
        &self,
        out_grad: &RawTensor,
        _parents: &[Tensor],
    ) -> Vec<Option<Tensor>> {
        vec![Some(RawTensor::new_with_storage(
            out_grad.data.to_device(&self.source),
            out_grad.shape.clone(),
            self.source.clone(),
            false,
        ))]
    }
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
            Self::Bce(_) => "Bce".to_string(),
            Self::Im2col(_) => "Im2col".to_string(),
            Self::DirectConv(_) => "DirectConv".to_string(),
            Self::Igemm(_) => "Igemm".to_string(),
            Self::ConvTranspose2d(_) => "ConvTranspose2d".to_string(),
            Self::MaxPool2d(_) => "MaxPool2d".to_string(),
            Self::Embedding(_) => "Embedding".to_string(),
            Self::ToDevice(_) => "ToDevice".to_string(),
            Self::Cast(_) => "Cast".to_string(),
            Self::Compare(g) => g.op_name(),
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
            Self::Bce(g) => g.backward(out_grad, parents),
            Self::Movement(g) => g.backward(out_grad, parents),
            Self::Im2col(g) => g.backward(out_grad, parents),
            Self::DirectConv(g) => g.backward(out_grad, parents),
            Self::Igemm(g) => g.backward(out_grad, parents),
            Self::ConvTranspose2d(g) => g.backward(out_grad, parents),
            Self::MaxPool2d(g) => g.backward(out_grad, parents),
            Self::Embedding(g) => g.backward(out_grad, parents),
            Self::ToDevice(g) => g.backward(out_grad, parents),
            Self::Cast(g) => g.backward(out_grad, parents),
            Self::Compare(g) => unreachable!("{}: a Bool output has no grad fn", g.op_name()),
        }
    }

    /// Whether this op's backward reads the values of parent `i` when it runs, so that an
    /// in-place write to it after the forward pass would change the gradient (S9). An op
    /// whose derivative doesn't depend on an input's values (`Add`, `Sum`, a reshape), or
    /// that saved what it needs in its own state (`MaxPool2d`'s indices, `Where`'s mask),
    /// returns `false`: PyTorch doesn't complain about those either. Every variant is
    /// spelled out so a new one has to decide.
    pub(crate) fn saves_input(&self, i: usize) -> bool {
        match self {
            Self::Unary(g) => g.saves_input(),
            Self::Binary(g) => g.saves_input(),
            // The winners are the input's elements equal to the maximum.
            Self::MaxReduce(_) => true,
            // The first two operands (x and y; input and weight). `MulAcc`'s addend and a
            // conv's bias (parent 2) are not read.
            Self::MulAcc(_)
            | Self::MatMul(_)
            | Self::Bce(_)
            | Self::DirectConv(_)
            | Self::Igemm(_)
            | Self::ConvTranspose2d(_) => i < 2,
            Self::Transpose(_)
            | Self::Sum(_)
            | Self::Mean(_)
            | Self::SumDim(_)
            | Self::MaxDim(_)
            | Self::Where(_)
            | Self::Movement(_)
            | Self::Im2col(_)
            | Self::MaxPool2d(_)
            | Self::Embedding(_)
            | Self::ToDevice(_)
            | Self::Cast(_)
            | Self::Compare(_) => false,
        }
    }

    /// Whether this gradient function is a no-op (`Mod`, which has no useful derivative,
    /// and comparisons). `edge()` never attaches a trivial
    /// `GradFn`, even if an input requires grad. Every variant is spelled out so a new
    /// one has to decide.
    pub(crate) fn is_trivial(&self) -> bool {
        match self {
            Self::Binary(g) => g.is_trivial(),
            Self::Compare(_) => true,
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
            | Self::Bce(_)
            | Self::Movement(_)
            | Self::Im2col(_)
            | Self::DirectConv(_)
            | Self::Igemm(_)
            | Self::ConvTranspose2d(_)
            | Self::MaxPool2d(_)
            | Self::Embedding(_)
            | Self::ToDevice(_)
            | Self::Cast(_) => false,
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
/// `GradFn::ToDevice` is exempt from the input-to-output check: a transfer is the one op
/// whose output is meant to be on another device.
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
    if let Some(first) = inputs.as_ref().first() {
        let first = first.borrow();
        let in_device = &first.device;
        check_one_device(inputs.as_ref(), in_device, &grad);
        let transfer = matches!(grad, GradFn::ToDevice(_));
        match (in_device.is_gpu(), device.is_gpu()) {
            _ if transfer => {}
            (true, false) => {
                on_gap(&grad.op_name(), Pass::Forward);
                data = data.to_device(in_device);
                device = in_device.clone();
            }
            (false, true) => panic!("{}: CPU inputs produced a {device} output", grad.op_name()),
            (true, true) | (false, false) => {}
        }
    }

    // Every op but a cast or a comparison keeps its inputs' dtype. A half op that forgot to round its f32
    // result (`Storage::from_compute`) would otherwise leak an f32 tensor.
    if let Some(first) = inputs.as_ref().first()
        && !matches!(grad, GradFn::Cast(_) | GradFn::Compare(_))
    {
        let want = first.borrow().data.dtype();
        assert!(
            data.dtype() == want,
            "{}: {want} inputs produced a {} output (an op keeps its dtype)",
            grad.op_name(),
            data.dtype()
        );
    }

    // A non-float output (an integer cast, later a comparison) has no gradient, whatever
    // its inputs: PyTorch's rule, and the reason integer tensors can't require grad.
    let requires_grad = is_grad_enabled()
        && data.dtype().is_float()
        && inputs.as_ref().iter().any(Tensor::requires_grad)
        && !grad.is_trivial();

    let out = RawTensor::new_with_storage(data, shape, device, requires_grad);
    if requires_grad {
        let mut o = out.borrow_mut();
        o.parents = inputs.as_ref().to_vec();
        // The versions of the inputs backward will read, to catch an in-place write in
        // between (S9).
        o.saved_versions = inputs
            .as_ref()
            .iter()
            .enumerate()
            .map(|(i, p)| grad.saves_input(i).then(|| p.borrow().version))
            .collect();
        o.grad_fn = Some(grad);
    }
    out
}

/// Check that every input is on `first`, the first input's device.
///
/// Compares borrowed devices (`==` on `Device` compares the GPU name too) and builds the
/// op name only on a mismatch, so the common case allocates nothing.
///
/// # Panics
/// If two inputs are on different devices.
fn check_one_device(inputs: &[Tensor], first: &Device, grad: &GradFn) {
    if inputs.iter().skip(1).any(|t| t.borrow().device != *first) {
        let devices: Vec<Device> = inputs.iter().map(Tensor::device).collect();
        let _ = common_device(&grad.op_name(), &devices);
    }
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
    NoGradGuard {
        _not_send: PhantomData,
    }
}
/// Ends a [`no_grad`] scope on drop.
///
/// `!Send`: the depth counter is thread-local, so a guard dropped on another thread would
/// end that thread's scope and leave this one's open. `tests/compile_fail/guards_not_send.rs`
/// pins it.
pub struct NoGradGuard {
    /// Makes the guard `!Send` (and `!Sync`).
    _not_send: PhantomData<*const ()>,
}
impl Drop for NoGradGuard {
    fn drop(&mut self) {
        NO_GRAD_DEPTH.with(|d| d.set(d.get() - 1));
    }
}

// ===== BACKPROPAGATION =====

impl RawTensor {
    /// Assert that `op`'s backward returned one gradient slot per parent.
    ///
    /// # Panics
    /// If `grads != parents`: an op bug, since `run_backward` pairs them by position.
    fn check_parent_grads(op: &GradFn, grads: usize, parents: usize) {
        assert_eq!(
            grads,
            parents,
            "{}: backward returned {grads} gradients for {parents} parents",
            op.op_name()
        );
    }

    /// Panics if an input that `grad_fn`'s backward reads was written in place after the
    /// forward pass recorded its version (S9): the gradient would silently use the new
    /// values, where PyTorch raises "modified by an inplace operation".
    #[track_caller]
    fn check_saved_versions(grad_fn: &GradFn, parents: &[Tensor], saved: &[Option<u64>]) {
        for (i, (parent, saved)) in parents.iter().zip(saved).enumerate() {
            if let Some(expected) = saved {
                let now = parent.borrow().version;
                assert!(
                    *expected == now,
                    "{}: input {i} was modified by an inplace operation after the forward \
                     pass saved it (version {expected} at forward, {now} now); an optimizer \
                     step, `load_state_dict`, or a dtype/device move before backward changes \
                     the gradient. Run backward before stepping, or redo the forward pass",
                    grad_fn.op_name()
                );
            }
        }
    }

    /// Run backpropagation from a scalar loss.
    ///
    /// Only reachable through [`ScalarLoss::backward`], which has already checked
    /// that `tensor_ref` has one element and requires grad.
    ///
    /// 1. Seed this tensor's gradient with 1, afresh on every call
    /// 2. Traverse the computation graph backwards (topological sort via DFS)
    /// 3. For each op output, call its `grad_fn` on the gradient this pass gave it
    /// 4. Add each parent's contribution: into `grad` for a leaf, into the pass's own
    ///    table for an op output (which keeps no `grad`, as in PyTorch)
    pub(crate) fn run_backward(tensor_ref: &Tensor) {
        /// A step of the iterative post-order DFS that builds the topological order.
        enum Action {
            /// First reach of a node: schedule its post-visit, then its parents.
            Visit(Tensor),
            /// All parents done: append the node to the order.
            PostVisit(Tensor),
        }

        // `into_loss` checked this, so it can only fail if another handle called
        // `set_requires_grad(false)` between `into_loss` and `backward`.
        debug_assert!(
            tensor_ref.borrow().requires_grad,
            "ScalarLoss::backward on a tensor that no longer requires grad"
        );
        // The gradient of an op output exists only for this pass (PyTorch's rule): it
        // waits in `pending` until the node's backward step consumes it. Only a leaf keeps
        // a `grad` across passes, so a second pass sends only its own contribution.
        let mut pending: HashMap<_, Storage> = HashMap::new();

        // Seed the root with 1, on the root's device and in its dtype (a gradient has its
        // tensor's dtype). Every pass seeds afresh.
        let (seed, root_is_leaf) = {
            let tensor = tensor_ref.borrow();
            let base = Storage::cpu(vec![1.0]).cast(tensor.data.dtype());
            (base.to_device(&tensor.device), tensor.grad_fn.is_none())
        };
        if root_is_leaf {
            tensor_ref.accumulate_named(seed, || "backward seed".to_string());
        } else {
            pending.insert(tensor_ref.as_ptr(), seed);
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
            let (grad_fn, parents, saved_versions, shape, device) = {
                let t = tensor.borrow();
                (
                    t.grad_fn.clone(),
                    t.parents.clone(),
                    t.saved_versions.clone(),
                    t.shape.clone(),
                    t.device.clone(),
                )
            };
            // A leaf has nothing to propagate: its gradient was added to `grad` as each
            // contribution arrived. An op output is propagated if anything reached it.
            if let Some(grad_fn) = grad_fn
                && let Some(grad_out_data) = pending.remove(&tensor.as_ptr())
            {
                let grad_out = Self {
                    data: grad_out_data,
                    shape,
                    grad: None,
                    requires_grad: false,
                    grad_fn: None,
                    parents: vec![],
                    device,
                    version: 0,
                    saved_versions: vec![],
                };

                Self::check_saved_versions(&grad_fn, &parents, &saved_versions);

                // Compute gradients for parent tensors
                let parent_grads = grad_fn.backward(&grad_out, &parents);
                // The zip below pairs by position: a short `backward` would leave a parent
                // without its gradient, silently freezing it (#169).
                Self::check_parent_grads(&grad_fn, parent_grads.len(), parents.len());

                // A leaf parent takes the contribution through `Tensor::accumulate_grad`,
                // the only code allowed to write `grad` (device-move + add-or-set). An
                // op-output parent gets the same checks and device move, but the sum waits
                // in `pending` for its own backward step.
                for (parent_grad, parent_ref) in parent_grads.into_iter().zip(parents.iter()) {
                    if let Some(g) = parent_grad {
                        let grad_storage = g.borrow().data.clone();
                        let parent_is_leaf = parent_ref.borrow().grad_fn.is_none();
                        if parent_is_leaf {
                            parent_ref.accumulate_grad(grad_storage, &grad_fn);
                        } else {
                            let routed = parent_ref.route_grad(grad_storage, &|| grad_fn.op_name());
                            match pending.entry(parent_ref.as_ptr()) {
                                Entry::Vacant(slot) => {
                                    slot.insert(routed);
                                }
                                Entry::Occupied(mut slot) => {
                                    Self::accumulate_grad(slot.get_mut(), &routed);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The gradients of every tensor reachable from a loss, saved so that
/// [`GradSnapshot::restore`] can put them back after a `backward()` that was only a
/// measurement (`check_gradients`).
///
/// CPU and GPU storage are copy-on-write, so holding a gradient here doesn't copy it,
/// and the accumulation in `backward` can't write through to the saved value.
pub(crate) struct GradSnapshot(Vec<(Tensor, Option<Storage>)>);

impl GradSnapshot {
    /// Saves the gradient (or its absence) of every tensor reachable from `root`
    /// through `parents`, `root` included, except `exempt`.
    pub(crate) fn take(root: &Tensor, exempt: &Tensor) -> Self {
        let mut seen = HashSet::new();
        let mut stack = vec![root.clone()];
        let mut saved = Vec::new();
        while let Some(t) = stack.pop() {
            if !seen.insert(t.as_ptr()) {
                continue;
            }
            stack.extend(t.borrow().parents.iter().cloned());
            if t.as_ptr() != exempt.as_ptr() {
                let grad = t.grad_storage();
                saved.push((t, grad));
            }
        }
        Self(saved)
    }

    /// Puts every saved gradient back: a tensor that had none has none again.
    pub(crate) fn restore(self) {
        for (t, grad) in self.0 {
            match grad {
                Some(grad) => t.set_grad(grad),
                None => t.zero_grad(),
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
    /// Backpropagate from this loss, accumulating into the `grad` of every reachable leaf.
    ///
    /// Calling it again adds that pass's gradient to each leaf's existing one. A tensor
    /// that is the output of an op keeps no `grad`.
    ///
    /// # Panics
    /// If a tensor an op saved for backward (a parameter it multiplied, say) was written in
    /// place after the forward pass: by an optimizer step, `load_state_dict`, or a dtype or
    /// device move. PyTorch raises the same "modified by an inplace operation" error. Ops
    /// whose gradient doesn't depend on an input's values (`add`, `sum`, `reshape`) don't
    /// check it.
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

        // CPU: element-wise addition in the gradient's own dtype. Half types add in f32
        // and round the sum (ADDENDUM §11), like every half-precision op.
        assert_eq!(
            existing.dtype(),
            new_grad.dtype(),
            "accumulate_grad: gradient contributions of different dtypes"
        );
        assert_eq!(
            existing.len(),
            new_grad.len(),
            "Gradient size mismatch during accumulation"
        );
        *existing = match existing.dtype() {
            DType::F32 => Storage::cpu(add_host::<f32>(existing, new_grad, |a, b| a + b)),
            DType::F64 => Storage::from_typed(add_host::<f64>(existing, new_grad, |a, b| a + b)),
            DType::F16 => Storage::from_typed(add_host::<f16>(existing, new_grad, |a, b| {
                f16::from_f32(a.to_f32() + b.to_f32())
            })),
            DType::BF16 => Storage::from_typed(add_host::<bf16>(existing, new_grad, |a, b| {
                bf16::from_f32(a.to_f32() + b.to_f32())
            })),
            other @ (DType::I32 | DType::I64 | DType::U8 | DType::Bool) => {
                panic!("accumulate_grad: a {other} gradient (only floats have gradients)")
            }
        };
    }
}

/// Elementwise `add(a, b)` of two CPU gradients of type `T`.
///
/// # Panics
/// If either is not CPU storage of `T` (callers check the dtype first).
fn add_host<T: Element>(a: &Storage, b: &Storage, add: impl Fn(T, T) -> T) -> Vec<T> {
    let a = a.as_host::<T>().expect("accumulate_grad: dtype checked");
    let b = b.as_host::<T>().expect("accumulate_grad: dtype checked");
    a.iter().zip(b).map(|(&x, &y)| add(x, y)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #169: a backward that returns fewer gradient slots than its op has parents is an
    /// op bug that panics, naming the op, instead of leaving a parent without a gradient.
    #[test]
    #[should_panic(expected = "backward returned 1 gradients for 2 parents")]
    fn parent_grad_count_mismatch_panics() {
        let x = RawTensor::new(vec![1.0, 2.0], &[2], true);
        let op = x
            .sum()
            .borrow()
            .grad_fn
            .clone()
            .expect("sum records a grad_fn");
        RawTensor::check_parent_grads(&op, 1, 2);
    }

    /// The matching count passes.
    #[test]
    fn parent_grad_count_match_passes() {
        let x = RawTensor::new(vec![1.0, 2.0], &[2], true);
        let op = x.sum().borrow().grad_fn.clone().expect("grad_fn");
        RawTensor::check_parent_grads(&op, 1, 1);
    }

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
        assert_eq!(x.grad().expect("grad"), vec![6.0]);
        loss.backward();
        assert_eq!(x.grad().expect("grad"), vec![12.0]);
    }

    /// Only a leaf keeps a gradient: `b = 2a` and the loss are op outputs, so after a
    /// backward they hold none (PyTorch's rule without `retain_grad`).
    #[test]
    fn op_outputs_keep_no_gradient() {
        let a = RawTensor::new(vec![1.0], &[1], true);
        let two = RawTensor::new(vec![2.0], &[1], false);
        let b = a.elem_mul(&two);
        let loss = b.sum().into_loss().expect("scalar");
        loss.backward();
        assert_eq!(a.grad().expect("leaf grad"), vec![2.0]);
        assert!(b.grad_storage().is_none(), "intermediate kept a grad");
        assert!(loss.tensor().grad_storage().is_none(), "root kept a grad");
    }

    /// A root that is itself a leaf is seeded with 1 on each pass and accumulates, as in
    /// PyTorch (`x.sum().backward()` on a one-element leaf twice gives 2).
    #[test]
    fn leaf_root_accumulates_its_seed() {
        let x = RawTensor::new(vec![5.0], &[1], true);
        let loss = x.clone().into_loss().expect("scalar");
        loss.backward();
        assert_eq!(x.grad().expect("grad"), vec![1.0]);
        loss.backward();
        assert_eq!(x.grad().expect("grad"), vec![2.0]);
    }

    /// An intermediate reached by two consumers sums both contributions within one pass,
    /// and a second pass adds the same total again rather than re-sending the first.
    #[test]
    fn shared_intermediate_sums_within_a_pass_and_adds_per_pass() {
        // b = 2a; loss = sum(b) + sum(3b); d loss / da = 2 + 6 = 8 per pass.
        let a = RawTensor::new(vec![1.0], &[1], true);
        let two = RawTensor::new(vec![2.0], &[1], false);
        let three = RawTensor::new(vec![3.0], &[1], false);
        let b = a.elem_mul(&two);
        let loss = b
            .sum()
            .add(&b.elem_mul(&three).sum())
            .into_loss()
            .expect("scalar");
        loss.backward();
        assert_eq!(a.grad().expect("grad"), vec![8.0]);
        loss.backward();
        assert_eq!(a.grad().expect("grad"), vec![16.0]);
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

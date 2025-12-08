use crate::device::Device;
use crate::storage::Storage;
use crate::tensor::{RawTensor, Tensor};
use std::collections::HashSet;

// ===== GRADIENT FUNCTION TRAIT =====

/// Trait for gradient computation functions.
///
/// Each operation type implements this to define how gradients flow backward.
/// The `backward` method takes:
/// - `out_grad`: gradient of loss w.r.t. this operation's output
/// - `parents`: the input tensors to this operation
///
/// Returns: vector of gradients w.r.t. each parent (Some if `requires_grad`, None otherwise)
pub trait GradFn {
    /// Compute gradients for parent tensors given output gradient
    fn backward(&self, out_grad: &RawTensor, parents: &[Tensor]) -> Vec<Option<Tensor>>;
    /// Clone this gradient function (needed for Rc/RefCell)
    fn clone_box(&self) -> Box<dyn GradFn>;
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
    pub fn backward(tensor_ref: &Tensor) {
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
                let grad_size = if tensor.shape.len() == 1 && tensor.shape[0] == 1 {
                    1
                } else {
                    tensor.data.len()
                };
                tensor.grad = Some(Storage::cpu(vec![1.0; grad_size]));
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
        enum Action {
            Visit(Tensor),
            PostVisit(Tensor),
        }
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
            let (grad_fn, parents, grad_data, shape) = {
                let t = tensor.borrow();
                (
                    t.grad_fn.as_ref().map(|gf| gf.clone_box()),
                    t.parents.clone(),
                    t.grad.clone(),
                    t.shape.clone(),
                )
            };

            // If this node has a gradient function and gradients, backpropagate
            if let Some(grad_fn) = grad_fn
                && let Some(grad_out_data) = grad_data
            {
                let grad_out = RawTensor {
                    data: grad_out_data,
                    shape,
                    grad: None,
                    requires_grad: false,
                    grad_fn: None,
                    parents: vec![],
                    device: Device::CPU,
                };

                // Compute gradients for parent tensors
                let parent_grads = grad_fn.backward(&grad_out, &parents);

                // Accumulate gradients in parents
                for (parent_grad, parent_ref) in parent_grads.into_iter().zip(parents.iter()) {
                    if let Some(g) = parent_grad {
                        let mut parent = parent_ref.borrow_mut();
                        let g_values = g.borrow().data.to_vec();
                        if parent.grad.is_none() {
                            parent.grad = Some(Storage::cpu(g_values));
                        } else {
                            let existing = parent.grad.as_mut().unwrap();
                            let slice = existing
                                .as_mut_slice()
                                .expect("Gradient accumulation only supported on CPU storage");
                            for (accum, &new) in slice.iter_mut().zip(&g_values) {
                                *accum += new;
                            }
                        }
                    }
                }
            }
        }
    }
}

use crate::device::Device;
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
/// Returns: vector of gradients w.r.t. each parent (Some if requires_grad, None otherwise)
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
    /// 3. For each node, call its grad_fn to compute parent gradients
    /// 4. Accumulate gradients in each parent tensor
    ///
    /// Uses a HashSet to track visited nodes and avoid recomputation.
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
                tensor.grad = Some(vec![1.0; grad_size]);
            }
        }

        // DFS-based topological traversal
        let mut stack = vec![tensor_ref.clone()];
        let mut visited = HashSet::new();

        while let Some(tensor) = stack.pop() {
            // Use raw pointer for HashSet (Rc doesn't impl Hash)
            if !visited.insert(tensor.as_ptr()) {
                continue;
            }
            let (grad_fn, parents, grad_data, shape) = {
                let t = tensor.borrow();
                (
                    t.grad_fn.as_ref().map(|gf| gf.clone_box()),
                    t.parents.clone(),
                    t.grad.clone(),
                    t.shape.clone(),
                )
            };
            // If this node has a gradient function, backpropagate
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
                        let g_data = g.borrow().data.clone();

                        // Initialize or accumulate gradient
                        if parent.grad.is_none() {
                            parent.grad = Some(g_data)
                        } else {
                            let existing = parent.grad.as_mut().unwrap();
                            for (accum, &new) in existing.iter_mut().zip(g_data.iter()) {
                                *accum += new;
                            }
                        }
                        // Add to stack if it has grad_fn (not a leaf)
                        if parent.grad_fn.is_some() {
                            drop(parent);
                            stack.push(parent_ref.clone());
                        }
                    }
                }
            }
        }
    }
}

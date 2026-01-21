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
    /// # Panics
    /// Calling on tensor taht doesn't need gradients
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
                    t.grad_fn.as_ref().map(|gf| gf.clone_box()),
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
                let grad_out = RawTensor {
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

                // Accumulate gradients in parents
                for (parent_grad, parent_ref) in parent_grads.into_iter().zip(parents.iter()) {
                    if let Some(g) = parent_grad {
                        let mut parent = parent_ref.borrow_mut();
                        let parent_device = parent.device.clone();

                        // Move this gradient contribution onto the parent's device.
                        let new_grad_storage = {
                            let g_borrow = g.borrow();
                            g_borrow.data.to_device(&parent_device)
                        };

                        match parent.grad {
                            None => {
                                // First contribution: just store it (already on correct device).
                                parent.grad = Some(new_grad_storage);
                            }
                            Some(ref mut existing) => {
                                RawTensor::accumulate_grad(existing, new_grad_storage);
                            }
                        }
                    }
                }
            }
        }
    }
}

impl RawTensor {
    /// Add a gradient contribution to an existing gradient storage, preferring GPU accumulation.
    fn accumulate_grad(existing: &mut Storage, new_grad: Storage) {
        #[cfg(feature = "gpu")]
        {
            if existing.is_gpu() && new_grad.is_gpu() {
                if let Some(sum) = RawTensor::gpu_add(existing, &new_grad) {
                    *existing = sum;
                    return;
                }
                eprintln!(
                    "Warning: GPU gradient accumulation failed; falling back to CPU accumulation"
                );
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

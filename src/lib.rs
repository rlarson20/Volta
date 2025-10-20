use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

pub type Tensor = Rc<RefCell<RawTensor>>;
//to allow shared mut refs to parent tensors in graph
//multiple child tensors can refer to parent
//during backprop can mutate parents through refs
//using Rc<RefCell<Tensor>> is simple approach for dyn graph
//for thread-safety, use Arc<Mutex<Tensor>>
//starting single-threaded

pub trait GradFn {
    fn backward(&self, out_grad: &RawTensor, parents: &[Tensor]) -> Vec<Option<Tensor>>;
    fn clone_box(&self) -> Box<dyn GradFn>;
}

struct AddGradFn;
impl GradFn for AddGradFn {
    fn backward(&self, out_grad: &RawTensor, _parents: &[Tensor]) -> Vec<Option<Tensor>> {
        //TODO: update to ensure it updates parents correctly
        let grad = RawTensor::new(out_grad.data.clone(), &out_grad.shape, false);
        vec![Some(grad.clone()), Some(grad)]
    }
    fn clone_box(&self) -> Box<dyn GradFn> {
        Box::new(AddGradFn)
    }
}

struct ElemMulGradFn;
impl GradFn for ElemMulGradFn {
    fn backward(&self, out_grad: &RawTensor, parents: &[Tensor]) -> Vec<Option<Tensor>> {
        let x_val = parents[0].borrow();
        let y_val = parents[1].borrow();
        let grad_x = if x_val.requires_grad {
            let data = out_grad
                .data
                .iter()
                .zip(y_val.data.iter())
                .map(|(g, &y)| g * y)
                .collect();
            Some(RawTensor::new(data, &out_grad.shape, false))
        } else {
            None
        };
        let grad_y = if y_val.requires_grad {
            let data = out_grad
                .data
                .iter()
                .zip(x_val.data.iter())
                .map(|(g, &x)| g * x)
                .collect();
            Some(RawTensor::new(data, &out_grad.shape, false))
        } else {
            None
        };
        vec![grad_x, grad_y]
    }
    fn clone_box(&self) -> Box<dyn GradFn> {
        Box::new(ElemMulGradFn)
    }
}

struct SubGradFn;
impl GradFn for SubGradFn {
    fn backward(&self, out_grad: &RawTensor, parents: &[Tensor]) -> Vec<Option<Tensor>> {
        let x_requires_grad = parents[0].borrow().requires_grad;
        let y_requires_grad = parents[1].borrow().requires_grad;
        let grad_x = if x_requires_grad {
            Some(RawTensor::new(
                out_grad.data.clone(),
                &out_grad.shape,
                false,
            ))
        } else {
            None
        };
        let grad_y = if y_requires_grad {
            let data = out_grad.data.iter().map(|&g| -g).collect();
            Some(RawTensor::new(data, &out_grad.shape, false))
        } else {
            None
        };
        vec![grad_x, grad_y]
    }
    fn clone_box(&self) -> Box<dyn GradFn> {
        Box::new(SubGradFn)
    }
}

struct SumGradFn {
    input_shape: Vec<usize>,
}

impl GradFn for SumGradFn {
    fn backward(&self, out_grad: &RawTensor, _parents: &[Tensor]) -> Vec<Option<Tensor>> {
        // Gradient of sum broadcasts output grad to input shape
        let size = self.input_shape.iter().product();
        let grad_val = out_grad.data[0]; // sum produces scalar
        vec![Some(RawTensor::new(
            vec![grad_val; size],
            &self.input_shape,
            false,
        ))]
    }

    fn clone_box(&self) -> Box<dyn GradFn> {
        Box::new(SumGradFn {
            input_shape: self.input_shape.clone(),
        })
    }
}

#[derive(Debug, Clone)]
pub enum Device {
    CPU,
    GPU,
    //TODO: possible Metal variant
}

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

impl RawTensor {
    //from data and shape
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
    //Tensor::zeros(shape: &[usize])
    pub fn zeros(shape: &[usize]) -> Tensor {
        let size = shape.iter().product();
        Self::new(vec![0.0; size], shape, false)
    }
    //Tensor::ones(shape: &[usize])
    pub fn ones(shape: &[usize]) -> Tensor {
        let size = shape.iter().product();
        Self::new(vec![1.0; size], shape, false)
    }
    //for something like
    //Tensor::rand(shape: &[usize])
    //integrate rand crate to fill data w normal dist
    //
    //impl debug, display trait
    //methods for props
    //tensor.shape()
    //tensor.num_elements()
    //etc
    //other methods to add:
    //reshape
    //to_device
    //
    //end goal:
    //getting basic funcs done
    //create tensor, print to verify shape and data align
    //requires_grad stays false for now
    //enable autograd in later steps
    //
    //next step: arithmetic ops on tensors
    //start with element wise
    //forward manner first
    //
    //for each, add:
    //num computation, eg adding 2 tensors element wise
    //building autograd graph:
    //new out tensor recording how it was computed from inputs
    //(eg, parents, grad_fn)
    //
    //shape compat:
    //first require exact shape matches for bin op
    //later add broadcasting a la numpy
    pub fn add(self_t: &Tensor, other: &Tensor) -> Tensor {
        let (data_a, shape_a, req_a) = {
            let s = self_t.borrow();
            (s.data.clone(), s.shape.clone(), s.requires_grad)
        };
        let (data_b, shape_b, req_b) = {
            let o = other.borrow();
            (o.data.clone(), o.shape.clone(), o.requires_grad)
        };
        assert_eq!(shape_a, shape_b);

        let result_data: Vec<f32> = data_a.iter().zip(&data_b).map(|(a, b)| a + b).collect();
        let out = Self::new(result_data, &shape_a, req_a || req_b);

        if out.borrow().requires_grad {
            out.borrow_mut().parents = vec![self_t.clone(), other.clone()];
            out.borrow_mut().grad_fn = Some(Box::new(AddGradFn));
        }
        out
    }
    pub fn sub(self_t: &Tensor, other: &Tensor) -> Tensor {
        let (data_a, shape_a, req_a) = {
            let s = self_t.borrow();
            (s.data.clone(), s.shape.clone(), s.requires_grad)
        };
        let (data_b, shape_b, req_b) = {
            let o = other.borrow();
            (o.data.clone(), o.shape.clone(), o.requires_grad)
        };
        assert_eq!(shape_a, shape_b);

        let result_data: Vec<f32> = data_a
            .iter()
            .zip(data_b.iter())
            .map(|(a, b)| a - b)
            .collect();
        let out = Self::new(result_data, &shape_a, req_a || req_b);
        if out.borrow().requires_grad {
            out.borrow_mut().parents = vec![self_t.clone(), other.clone()];
            out.borrow_mut().grad_fn = Some(Box::new(SubGradFn));
        }
        out
    }
    pub fn elem_mul(self_t: &Tensor, other: &Tensor) -> Tensor {
        let (data_a, shape_a, req_a) = {
            let s = self_t.borrow();
            (s.data.clone(), s.shape.clone(), s.requires_grad)
        };
        let (data_b, shape_b, req_b) = {
            let o = other.borrow();
            (o.data.clone(), o.shape.clone(), o.requires_grad)
        };
        assert_eq!(shape_a, shape_b);

        let result_data: Vec<f32> = data_a
            .iter()
            .zip(data_b.iter())
            .map(|(a, b)| a * b)
            .collect();
        let out = Self::new(result_data, &shape_a, req_a || req_b);
        if out.borrow().requires_grad {
            out.borrow_mut().parents = vec![self_t.clone(), other.clone()];
            out.borrow_mut().grad_fn = Some(Box::new(ElemMulGradFn));
        }
        out
    }
    //iterating through self/other.data to compute val
    //make new out tensor, requires_grad if either needs grad
    //if grad needed, store refs in parents
    //also assign grad_fn, obj knows how to prop grad for op
    //make OpGradFn type (struct/enum var) impl trait GradFn
    //other ops:
    //negation, flips sign with simple grad func
    //relu: max(0,x) elementwise, adds non-lin
    //grad 1 for +, 0 for - during backprop
    //
    //matmul/dotprod: crucial
    //basic matmul for 2D tensor:
    //A.shape (m,n)
    //B.shape (n,p)
    //res.shape(m,p)
    //
    //triple nested loop to compute
    //C_{i,j} = sum_k A_{i,k} * B_{k,j}
    //to optimize later w/ BLAS or parallel loop
    //
    //if needed, impl tensor indexing, slicing, reshaping
    //eg: flattening tensor, reshape for ops
    //each op producing new tensor should have appropriate grad_fn
    //
    //eg:
    //MatMulGradFn: more complex: grad involves transposes:
    //if z = X dot W, grad w.rt X = grad_out dot W.transpose,
    //and grad w.rt W = X.transpose dot grad.out
    //
    //
    //with forward pass creating comp graph
    //via parents links and grad_fn
    //impl back pass to perform autodiff
    //
    //add method Tensor::backward(&mut self)
    //computes grad w.rt tensor
    //usually called on final loss scalar
    //if tensor not scalar
    //enforce user calls .backward() only on scalar loss
    //or define what it means:
    //eg sum of grads
    //
    //backward should traverse graph from curr tensor back to all deps:
    //start by setting grad of final tensor (self) to appropriate val:
    //typically tensor of ones w same shape (del loss/ del loss = 1)
    //for scalar loss, grad = 1.0
    //use topological order traversal (acyclic graph)
    //propogate grads
    //collect all tensors in graph by traversing parents recursively
    //dfs or iterative stack
    //alternatively
    //during forward op creation
    //you can keep a global list or assign each tensor an incremental id,
    //sort by creation order (topo sort)
    //iterate in rev topo order: use its grad_fn.backward to get grad for parents, accumulate into
    //parents .grad
    //mark tensors as visited to avoid repeating backprop
    //also handle case of leaves (no parents, req_grad false)

    pub fn sum(self_t: &Tensor) -> Tensor {
        let (data, shape, req_grad) = {
            let s = self_t.borrow();
            (s.data.clone(), s.shape.clone(), s.requires_grad)
        };
        let sum_val: f32 = data.iter().sum();
        let out = Self::new(vec![sum_val], &[1], req_grad);

        if out.borrow().requires_grad {
            out.borrow_mut().parents = vec![self_t.clone()];
            out.borrow_mut().grad_fn = Some(Box::new(SumGradFn { input_shape: shape }));
        }
        out
    }

    pub fn backward(tensor_ref: &Tensor) {
        let tensor = tensor_ref.borrow();
        assert!(
            tensor.requires_grad,
            "Called backward on a tensor that doesn't require grad"
        );
        drop(tensor);
        //init gradient for starting tensor
        {
            let mut tensor = tensor_ref.borrow_mut();
            if tensor.grad.is_none() {
                tensor.grad = Some(vec![1.0; tensor.data.len()]);
            }
        }

        //use TensorRef stack instead of mut refs
        let mut stack = vec![tensor_ref.clone()];
        let mut visited = HashSet::new();

        while let Some(tensor) = stack.pop() {
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
                let parent_grads = grad_fn.backward(&grad_out, &parents);

                for (parent_grad, parent_ref) in parent_grads.into_iter().zip(parents.iter()) {
                    if let Some(g) = parent_grad {
                        let mut parent = parent_ref.borrow_mut();
                        let g_data = g.borrow().data.clone();
                        if parent.grad.is_none() {
                            parent.grad = Some(g_data)
                        } else {
                            let existing = parent.grad.as_mut().unwrap();
                            for (accum, new) in existing.iter_mut().zip(g_data.iter()) {
                                *accum += new;
                            }
                        }
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

pub trait TensorOps {
    fn add(&self, other: &Tensor) -> Tensor;
    fn sub(&self, other: &Tensor) -> Tensor;
    fn elem_mul(&self, other: &Tensor) -> Tensor;
    fn sum(&self) -> Tensor;
    fn backward(&self);
    fn grad(&self) -> Option<Vec<f32>>;
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
    fn sum(&self) -> Tensor {
        RawTensor::sum(self)
    }
    fn backward(&self) {
        RawTensor::backward(self)
    }
    fn grad(&self) -> Option<Vec<f32>> {
        self.borrow().grad.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_add_backward() {
        let a = RawTensor::new(vec![2.0], &[1], true);
        let b = RawTensor::new(vec![3.0], &[1], true);
        let c = a.add(&b);
        c.backward();

        assert_eq!(a.grad(), Some(vec![1.0]));
        assert_eq!(b.grad(), Some(vec![1.0]));
    }

    #[test]
    fn test_multiply_backward() {
        let a = RawTensor::new(vec![3.0], &[1], true);
        let b = RawTensor::new(vec![4.0], &[1], true);
        let c = a.elem_mul(&b);
        c.backward();

        assert_eq!(a.grad(), Some(vec![4.0]));
        assert_eq!(b.grad(), Some(vec![3.0]));
    }

    #[test]
    fn test_chain_rule() {
        let a = RawTensor::new(vec![2.0], &[1], true);
        let b = RawTensor::new(vec![3.0], &[1], true);
        let c = a.add(&b);
        let d = c.elem_mul(&a);
        d.backward();

        assert_eq!(a.grad(), Some(vec![7.0]));
        assert_eq!(b.grad(), Some(vec![2.0]));
    }

    #[test]
    fn test_sum_backward() {
        let a = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], true);
        let loss = a.sum();
        loss.backward();

        assert_eq!(a.grad(), Some(vec![1.0, 1.0, 1.0]));
    }

    #[test]
    fn test_multidim_ops() {
        let a = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        let b = RawTensor::new(vec![0.5, 0.5, 0.5, 0.5], &[2, 2], true);
        let c = a.elem_mul(&b);
        let loss = c.sum();
        loss.backward();

        assert_eq!(a.grad(), Some(vec![0.5, 0.5, 0.5, 0.5]));
        assert_eq!(b.grad(), Some(vec![1.0, 2.0, 3.0, 4.0]));
    }
}

type TensorRef = std::rc::Rc<std::cell::RefCell<Tensor>>;
//to allow shared mut refs to parent tensors in graph
//multiple child tensors can refer to parent
//during backprop can mutate parents through refs
//using Rc<RefCell<Tensor>> is simple approach for dyn graph
//for thread-safety, use Arc<Mutex<Tensor>>
//starting single-threaded

trait GradFn {
    fn backward(&self, out_grad: &Tensor, parents: &[TensorRef]) -> Vec<Option<Tensor>>;
}

struct AddGradFn;
impl GradFn for AddGradFn {
    fn backward(&self, out_grad: &Tensor, parents: &[TensorRef]) -> Vec<Option<Tensor>> {
        vec![
            Some(Tensor::new(out_grad.data.clone(), &out_grad.shape, false)),
            Some(Tensor::new(out_grad.data.clone(), &out_grad.shape, false)),
        ]
    }
}

struct MulGradFn;
impl GradFn for MulGradFn {
    fn backward(&self, out_grad: &Tensor, parents: &[TensorRef]) -> Vec<Option<Tensor>> {
        let x_val = parents[0].borrow();
        let y_val = parents[1].borrow();
        let grad_x = if x_val.requires_grad {
            let data = out_grad
                .data
                .iter()
                .zip(y_val.data.iter())
                .map(|(g, &y)| g * y)
                .collect();
            Some(Tensor::new(data, &out_grad.shape, false))
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
            Some(Tensor::new(data, &out_grad.shape, false))
        } else {
            None
        };
        vec![grad_x, grad_y]
    }
}
//Note:
//above use parents as slite of TensorRef
//assume like Rc<RefCell<Tensor>>
//call borrow to get actual Tensor
//in practice, design GradFn to carry necessary info
//maybe store copy of operand if needed for back
//to avoid borroiwng parents
//to keep conceptual, directly access parents here
//each GradFn::backward returns vec of grads aligned w parents vec
//using none for parent w no grad req

pub enum Device {
    CPU,
    GPU,
    //TODO: possible Metal variant
}

impl Tensor {
    //from data and shape
    pub fn new(data: Vec<f32>, shape: &[usize], requires_grad: bool) -> Tensor {
        assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "Data length must match shape"
        );
        Tensor {
            data,
            shape: shape.to_vec(),
            grad: None,
            requires_grad,
            grad_fn: None,
            parents: vec![],
            device: Device::CPU,
        }
    }
    //Tensor::zeros(shape: &[usize])
    pub fn zeros(shape: &[usize]) -> Tensor {
        let size = shape.iter().product();
        Tensor::new(vec![0.0; size], shape, false)
    }
    //Tensor::ones(shape: &[usize])
    pub fn ones(shape: &[usize]) -> Tensor {
        let size = shape.iter().product();
        Tensor::new(vec![1.0; size], shape, false)
    }
    //ref to data or copy of data
    pub fn data(&self) -> &[f32] {
        &self.data
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
    pub fn add(&self, other: &Tensor) -> Tensor {
        assert!(self.shape == other.shape, "Shapes must be equal to add");
        let result_data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        let mut out = Tensor::new(
            result_data,
            &self.shape,
            self.requires_grad || other.requires_grad,
        );
        if out.requires_grad {
            let self_ref = TensorRef::new(RefCell::new(self.clone()));
            let other_ref = TensorRef::new(RefCell::new(other.clone()));
            out.parents = vec![self_ref, other_ref];
            out.grad_fn = Some(Box::new(AddGradFn));
        }
        out
    }
    pub fn sub(&self, other: &Tensor) -> Tensor {
        assert!(self.shape == other.shape, "Shapes must be equal to add");
        let result_data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        let mut out = Tensor::new(
            result_data,
            &self.shape,
            self.requires_grad || other.requires_grad,
        );
        if out.requires_grad {
            let self_ref = TensorRef::new(RefCell::new(self.clone()));
            let other_ref = TensorRef::new(RefCell::new(other.clone()));
            out.parents = vec![self_ref, other_ref];
            out.grad_fn = Some(Box::new(SubGradFn));
        }
        out
    }
    pub fn elem_mul(&self, other: &Tensor) -> Tensor {
        assert!(self.shape == other.shape, "Shapes must be equal to add");
        let result_data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        let mut out = Tensor::new(
            result_data,
            &self.shape,
            self.requires_grad || other.requires_grad,
        );
        if out.requires_grad {
            let self_ref = TensorRef::new(RefCell::new(self.clone()));
            let other_ref = TensorRef::new(RefCell::new(other.clone()));
            out.parents = vec![self_ref, other_ref];
            out.grad_fn = Some(Box::new(ElemMulGradFn));
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

    pub fn backward(&mut self) {
        assert!(
            self.requires_grad,
            "Called backward on a tensor that doesn't require grad"
        );
        if self.grad.is_none() {
            self.grad = Some(vec![1.0; self.data.len()]);
        }
        let mut stack = vec![self];
        while let Some(tensor) = stack.pop() {
            //if has grad_fn and parents, prop grads
            if let Some(ref grad_fn) = tensor.grad_fn {
                //ensure grad is present
                //should be if set for output
                let grad_out_data = tensor.grad.as_ref().unwrap();
                let grad_out = Tensor::new(grad_out_data.clone(), &tensor.shape, false);
                //make parent grads
                let parent_grads = grad_fn.backward(&grad_out, &tensor.parents);
                for (parent_grad, parent_ref) in parent_grads.into_iter().zip(tensor.parents.iter())
                {
                    if let Some(g) = parent_grad {
                        let mut parent = parent_ref.borrow_mut();
                        if parent.grad.is_none() {
                            parent.grad = Some(g.data)
                        } else {
                            let existing = parent.grad.as_mut().unwrap();
                            for (accum, new) in existing.iter_mut().zip(g.data.iter()) {
                                *accum += new;
                            }
                        }
                    }
                    if parent_ref.borrow().grad_fn.is_some() {
                        //push parent to stack for further backprop
                        stack.push(&mut *parent_ref.borrow_mut());
                    }
                }
            }
        }
    }
}

pub struct Tensor {
    pub data: Vec<f32>,         // flat data vec, len = prod shape dims
    pub shape: Vec<usize>,      //tensor dims, eg [B,C,H,W]
    pub grad: Option<Vec<f32>>, //grad w.r.t tensor data, None if req_grad == false
    pub requires_grad: bool,
    pub grad_fn: Option<Box<dyn GradFn>>, //func to compute grad, if result of op
    pub parents: Vec<TensorRef>,          //refs to parent tensor on graph
    pub device: Device,                   //cpu/gpu
}

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}

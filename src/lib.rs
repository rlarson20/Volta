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
    //AddGradFn: in back, take grad of out, pass to parents
    //since delta(a+b)/delta(a) = 1, delta(a+b)/delta(b) = 1,
    //grad flows unchanged to in
    //MulGradFn: z = x * y, delta(z)/delta(x) = y, back mul incoming grad by other parent val
    //MatMulGradFn: more complex: grad involves transposes:
    //if z = X dot W, grad w.rt X = grad_out dot W.transpose,
    //and grad w.rt W = X.transpose dot grad.out
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

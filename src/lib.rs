pub trait GradFn {}
pub struct TensorRef {
    //may end up as:
    //type TensorRef = std::rc::Rc<std::cell::RefCell<Tensor>>
    //to allow shared mut refs to parent tensors in graph
    //multiple child tensors can refer to parent
    //during backprop can mutate parents through refs
    //using Rc<RefCell<Tensor>> is simple approach for dyn graph
    //for thread-safety, use Arc<Mutex<Tensor>>
    //starting single-threaded
}

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
    //for something like
    //Tensor::rand(shape: &[usize])
    //integrate rand crate to fill data w normal dist
    //
    //impl debug, display trait
    //methods for props
    //tensor.shape()
    //tensor.num_elements()
    //etc
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

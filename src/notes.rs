// 1. Add dtype support (f32/f64) to improve numerical stability for grad checks and heavy ops; enables using f64 in gradcheck even if model tensors are f32.
// 2. Expand matmul gradients to fully support batched matmul (3D+) and broadcasting semantics; current implementation only covers 1D/2D.
// 3. Introduce keepdim reductions and axis-specific reduce ops; ensure backward sums with correct unsqueeze/expand.
//
//
// - **Blockers:**
//
// - **Risky Areas:**
//   - **Performance:** All operations currently use naive `Vec<f32>` loops and extensive cloning. While correct, this is slow. The next performance bottleneck will be `matmul`. Integrating a BLAS library (like Apple's Accelerate for macOS) is a high-leverage optimization but requires `unsafe` FFI calls that can be a source of bugs.
//   - **API Ergonomics:** The `Rc<RefCell<RawTensor>>` pattern is simple but verbose. As the API grows, managing borrows and `_mut` calls can become cumbersome. A future refactor might explore a more ergonomic API, but this risks significant breaking changes. For now, correctness is paramount.

// TODO:
// 3. **Improve optimizer with weight decay and other SGD variants** - More training options
//
// **Risky areas:**
// - Memory management in the autograd graph (potential cycles)
// - Numerical stability for certain operations (e.g., softmax)
//
//end goal:
//getting basic funcs done
//create tensor, print to verify shape and data align
//requires_grad stays false for now
//enable autograd in later steps
//building autograd graph:
//new out tensor recording how it was computed from inputs
//(eg, parents, grad_fn)

//make OpGradFn type (struct/enum var) impl trait GradFn
//if needed, impl tensor indexing, slicing, reshaping
//eg: flattening tensor
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

//need to finish:
//basic operations
//autograd
//broadcasting: allowing ops on tensors of diff shapes, see line 465 for explanation
//forward/back for each
//
//
//
//avoid in place ops
//use pure funcs returning new tensors
//if needed,
//add in place later
//
//make sure to test for each to ensure grad impl correct
//
//when autograd in place, make higher abs for NN layers and models
//
//define structs for layers:
//Linear, fully connected
//Conv2d, conv if ambitious
//hold tensor params
//
//eg
//
//pub struct Linear {
//  pub weight: Tensor, //shape: (in_feat, out_feat)
//  pub bias: Tensor, //shape: (out_feat,)
//}
//
//impl Linear {
//  pub fn new(in_feat: usize, out_feat: usize) -> Linear {
//      //init weight w small rand vals, bias w 0s
//      let w = Tensor::new(randow_vector(in_feat * out_feat), &[in_feat, out_feat], true);
//      let b = Tensor::zeros(&[out_feat])
//      Linear { weight: w, bias: b }
//  }
//  pub fn forward(&self, input: &Tensor) -> Tensor {
//      //in shape: batch size, in_feat
//      //out = in dot weight plus bias
//      let out = input.matmul(&self.weight); //batchsize, out_feat
//      out.add(&self.bias) //bc bias over batch
//  }
//}
//
//Linear::forward use matmul impled, then add bias
//
//bc across batch dim
//
//bc weight, bias have requires_grad=true,
//all ops w them w track grads
//
//make trait Module if polymorphic API for layers (like in PyTorch, all layers are modules)
//
//eg
//
//pub trait Module {
//  fn forward(&self, input: &Tensor) -> Tensor;
//  fn parameters(&self) -> Vec<TensorRef>; //collect train params, fix bc no more tensor-ref
//}
//
//impl Module for Linear {
// fn forward(&self, input: &Tensor) -> Tensor { self.forward(input)}
// fn parameters(&self) -> Vec<TensorRef> { //again, fix no more tensorref
// vec![Rc::new(RefCell::new(self.weight.clone())),Rc::new(RefCell::new(self.bias.clone()))]
// }
//}
//
//params method returns ref to internal tensors
//so an optim can update
//if building larger network (struct w multi layers)
//
//can impl Module for it returning concat of all sub-layers params
//
//can add other layers:
//
//ReLU as layer (although it's stateless/pure, so can use tensor op)
//Sequential to stack layers in order
//depends on how full featured i want
//
//to train models:
//
//need to compute loss
//update weights
//
//loss: common loss includes MSELoss/MeanSquaredErr for regression
//Cross-entropy for classifcation
//
//impl funcs take pred and target Tensors and return Tensor loss
//
//eg
//
//fn mse_loss(pred: &Tensor, target: &Tensor) -> Tensor {
//  let diff = pred.add(&target.neg()) //pred - target
//  let sq = diff * &diff; //elem wise square
//  sq.mean() //ret avg of sq err
//}
//
//autograd will handle bw pass for loss
//composed of basic ops defined
//
//optims:
//create optim struct
//begin with SGD
//needs to adjust model params based on grads
//
//eg
//
//pub struct SGD {
// lr: f32,
//}
//
//impl SGD {
//  pub fn new(lr: f32) -> Self { SGD { lr } }
//  pub fn step(&self, params: &[TensorRef]) {//FIX TR EXISTENCE
//      for param_ref in params {
//          let mut param = param_ref.borrow_mut();
//          if !param.requires_grad {
//              continue;
//          }
//          if let Some(ref grad) = param.grad {
//              //update params: param = param - lr * grad
//              for (val, grad_val) in param.data.iter_mut().zip(grad.iter()) {
//                  *val -= self.lr * grad_val;
//              }
//          }
//          //typically reset grads to 0 after update
//          param.grad = None;
//      }
//  }
//}
//
//usage:
//after comp loss.backward()
//call optim.step(params) where params is list of all train params
//eg gathered from layers
//will subtract lr times grad from each params data in place
//after step, set each param.grad = None
//or 0 out grad vec
//so that next iter back starts fresh
//
//training loop:
//loop over epoch, batches
//
//do pred = model.forward(input)
//loss = loss_fn(pred, target)
//loss.backward()
//optim.step()
//outside framework code, good to verify things work
//
//to make fully featured, need to add HW accel
//involves abstracting ops over diff backends
//
//eg enum Device { CPU, GPU } w/ vars like GPU(Metal) or more specific
//
//add fields in Tensor  like device
//include to_device(&self, device: Device) -> Tensor
//to transfer data btwn CPU/GPU
//eg
//CPU tensor but want to run GPU op
//need to copy to GPU mem
//
//for each op: branch on device:
//
//impl Tensor {
//  pub fn add(&self, other: &Tensor) -> Tensor {
//      asset!(self.shape == other.shape);
//      match self.device {
//          Device::CPU => {
//              //CPU impl
//          }
//          Device::GPU => {
//              //GPU impl
//          }
//      }
//  }
//  //similar device specific impl
//}
//
//on CPU, already have impl
//
//for GPU, need to impl using Apple framework
//
//can design backend trait instead
//eg
//trait Backend { fn add(x:&Tensor, y&Tensor) -> Tensor; //other ops}
//have a CpuBackend struct
//MetalBackend struct
//impl Backend trait
//
//can add complexity
//
//simpler approach is match on device
//but i do still want to learn rust better w this
//
//start w few GPU ops
//to test concept
//eg add, matmul
//default to CPU for any NYI
//or raise error if attempted
//
//since silicon doesnt support CUDA
//use apple compute APIs to accel ops on GPU
//
//few options
//
//Apple Metal API:
//for ML
//MPS: collection of high-performance GPU routine
//eg
//optimized kernels for malmul
//MPSMatrixMultiply
//convolution
//etc
//call these from rust using Obj-C runtime
//crates like metal for low-level Metal
//objc for apple frameworks
//using MPS
//can offload heavy ops
//eg
//call MPS matmul on GPU
//setting up MTLBuffers for input tensors
//getting result
//
//custom Metal compute shaders
//write Metal compute function in Apple shading lang, similar to C
//perform element-wise ops or custom kernels
//in rust,
//load shader code
//execute on GPU w metal crate
//to do vec add on gpu
//write shader adding 2 arrays element wise
//rust code would make buffers for 2 input arrays
//one output array on GPU
//encode compute command
//run it
//
//direct Metal gives bess perf
//complexity tradeoff
//manage GPU and mem transfers
//
//init: might only impl key op or 2
//matmul, conv
//to see speedup
//
//accel/BLAS
//
//not GPU
//accel framework
//espec BLAS, vDSP libs
//optim for Apple CPUs (neon vectorization)
//can use Neural Engine for tasks under hood
//can use accel for matmuls,
//FFTs by calling C interface
//
//eg
//cblas_sgemm for matmul == easy cpu perf boost
//
//add FFI binding:
//
//extern "C" {
//  fn cblas_sgemm(order: i3d, transA: i32, transB: i32, m: i32, n: i32, k: i32, alpha: f32, A: *const f32, lda: i32, B: *const f32, ldb: i32, beta: f32, C: *mut f32, ldc: i32);
//}
//
//then in Tensor::matmul for CPU
//call func w appropriate params to mul A by B into C
//offloads heavy math to apples optimed impl
//using all cores and vec units
//quick perf win on CPU
//doesnt use GPU
//
//if want non-apple specific soln
//use wgpu crate
//rust impl of webgpu
//
//wgpu can target mac metal
//vulkan on other
//directx/opengl where appropriate
//can write shater/compute programs in WGSL or SPIR-V
//
//more general, can add matmul shader, WGPU run on metal for m1, CUDA for nvidia w vulkan, etc
//one api for all
//can be overkill
//
//for my framework:
//
//start w key ops using MPS
//provides high level GPU ops
//to do GPU matmul:
//
//create metal device and cmd queue w metal crate
//create mps_matrix_descriptor for input/output, allocate mps_matrix objs backed by MTLBuffers
//create MPSMatMul kernel w appropriate dims
//encode kernel on cmd buffer w in/out matrices, commit cmd buffer
//after exec, read back result from output mtlbuffer in Rust Tensor
//keep it on GPU if chaining GPU ops
//
//procedure wrapped inside Tensor::matmul when device == Device::GPU
//bit code involved
//conceptually straightforward
//allocate GPU mem
//call GPU kernel
//sync results
//
//write some unsafe
//use Obj-C runtime
//
//elem-wise ops like add, relu
//
//could use MPS: MPSVector, MPSUnaryImageKernel for certain ops
//or just quickly write simple Metal compute shader string
//compile it at runtime
//for isntance, shader that does output[i] = input1[i] + input2[i] in parallel
//metal crate == make library from src and compute pipeline
//
//provide way to move data btwn CPU and GPU
//to_device
//
//eg
//
//tensor on CPU, call GPU op
//should internally copy to GPU buffer
//perform op
//keep res on GPU perhaps
//managing when to bring data back
//sync is important
//might not copy back until needed or do immediate for simplicity
//
//ANE:
//not publicly accessible for custom code
//used via coreml for inference
//stick w metal and gpu methods
//
//impl and debug GPU harder than CPU
//get CPU solid
//progressively add GPU support
//feature gate GPU code
//eg
//only compile metal-code on macOS
//use rust conditional compilation
//
//start with 1 op
//test it
//
//check that CPU vs GPU ive same res for some input
//proceed to add more
//
//build iteratively
//
//first get core autograd done on CPU
//add layers and training loop
//train simple model: 2-layer neural net on small DS
//entirely on CPU
//ensures APU and autograd are correct
//add hw accel for comp heavy parts
//matmul, conv
//using apple GPU
//can fallback when not impl
//learn internals at each stage, verify right

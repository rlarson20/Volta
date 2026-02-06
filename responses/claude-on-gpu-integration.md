Looking at your Volta codebase, I can see you've built a solid foundation for GPU support with the storage abstraction, GPU context, and basic kernels. However, there are several architectural blockers preventing full GPU integration. Let me walk you through each one systematically, explain why it's blocking you, and provide concrete solutions.

## Critical Architectural Blockers

### Blocker 1: The Autograd Engine is Hardwired to CPU

Your biggest challenge is that the entire automatic differentiation system in `src/autograd.rs` assumes CPU storage. When you look at the backward pass, gradients are accumulated directly into CPU vectors. The line `tensor.grad = Some(Storage::cpu(vec![1.0; grad_size]))` initializes gradients on CPU regardless of where the forward computation happened.

**Why this matters:** Even if you run the forward pass on GPU, the moment you call `backward()`, everything gets pulled back to CPU for gradient computation. This creates a severe performance bottleneck because you're constantly transferring data between devices.

**Solution approach:** You need to make gradient storage device-aware. When initializing gradients, they should live on the same device as the tensor data. In the backward pass, gradient accumulation should happen on the appropriate device. This means modifying the accumulation code around line 80 in `autograd.rs` to handle both CPU and GPU storage paths.

### Blocker 2: No Automatic Operation Dispatch

Currently, your tensor operations like `add`, `mul`, and `matmul` in `src/ops/` directly execute CPU code. The GPU implementations in `gpu_ops.rs` exist but are never automatically called. There's no dispatch mechanism that says "if this tensor is on GPU, use the GPU kernel."

**Why this matters:** You can manually call GPU functions, but the ergonomic API through `TensorOps` trait always uses CPU. This means your high-level code (neural networks, training loops) can't benefit from GPU acceleration without major rewrites.

**Solution approach:** Implement operation dispatch at the point where operations are called. For example, in `src/ops/binary.rs`, the `binary_op` function should check the device of input tensors and route to either CPU or GPU implementations. You'll need a pattern like checking `if both tensors are on GPU && GPU is available, use GPU kernel, else fall back to CPU`.

### Blocker 3: Gradient Functions Only Know About CPU

Every `GradFn` implementation throughout your `src/ops/` directory works exclusively with CPU data. Look at `BinaryGradFn::backward` in `binary.rs` - it directly accesses `data` as CPU vectors and performs computations on them.

**Why this matters:** Even if you get the forward pass running on GPU, the backward pass through each `GradFn` will force data back to CPU. You lose all GPU acceleration benefits during backpropagation, which is typically the most expensive part of training.

**Solution approach:** You have two architectural options. First option: make gradient functions device-polymorphic by having them inspect storage type and dispatch to GPU kernels for gradient computation. Second option (cleaner but more work): create GPU-specific gradient function variants that operate on GPU buffers. The second approach gives you better type safety and performance but requires duplicating gradient logic.

### Blocker 4: Movement Operations Have No GPU Path

Operations in `src/ops/movement.rs` like `reshape`, `permute`, and `pad` all manipulate CPU vectors directly. There's no GPU implementation for these critical operations.

**Why this matters:** Neural networks constantly reshape and permute tensors (think of batch normalization or attention mechanisms). If these operations force GPU-to-CPU transfers, you'll spend more time moving data than computing.

**Solution approach:** Movement operations on GPU are tricky because they often involve non-contiguous memory access patterns. For `reshape`, you can often avoid actual data movement by just updating metadata (shape, strides). For `permute` and `pad`, you need GPU kernels. Consider implementing these as compute shaders in `src/gpu/shaders/` following the pattern of your existing element-wise operations.

## Secondary Integration Blockers

### Blocker 5: Neural Network Layers Are CPU-Locked

Your `Conv2d`, `Linear`, and `BatchNorm2d` implementations in `src/nn/layers/` all operate on CPU. The `im2col` implementation in `conv.rs` builds CPU matrices. Layer parameters are stored as CPU tensors.

**Why this matters:** Even with GPU-enabled operations, if your layer forward passes extract data to CPU, process it, and push back to GPU, you've destroyed any performance gains.

**Solution approach:** Layer implementations need to be device-agnostic. Parameters should be created on and stay on the device where computation happens. The `im2col` approach for convolution needs a GPU variant - either implement im2col as a GPU kernel or switch to a direct convolution approach using GPU compute shaders.

### Blocker 6: Optimizer Updates Are CPU-Only

Your `Adam` and `SGD` implementations in `src/nn/optim/` directly manipulate CPU parameter data during the `step()` function. Look at line 40 in `adam.rs` where it iterates over `p.data` assuming CPU storage.

**Why this matters:** After computing gradients on GPU, you'd have to transfer all parameters and gradients to CPU for the optimizer update, then transfer updated parameters back to GPU. For large models, this transfer time can dominate training time.

**Solution approach:** Optimizer step functions need GPU kernel implementations. For Adam, you need kernels that perform the momentum updates and parameter updates on GPU. The optimizer should detect parameter device and dispatch to the appropriate kernel.

### Blocker 7: Incomplete GPU Operation Coverage

You've implemented GPU kernels for basic operations (add, mul, matmul) but are missing many operations that neural networks need: reduction operations beyond sum, advanced activation functions, and specialized operations like batch normalization primitives.

**Why this matters:** The moment your computation graph uses an unsupported operation, you're forced back to CPU, negating all GPU benefits.

**Solution approach:** Systematically go through operations in `src/ops/` and identify which ones are frequently used in neural networks. Prioritize implementing GPU kernels for reductions (sum, mean along dimensions), advanced activations (LayerNorm, GELU), and specialized ops (softmax, cross-entropy).

## Testing and Validation Strategy

To ensure your GPU implementation is correct and well-integrated, you need a comprehensive testing approach across multiple levels:

**Unit testing approach:** For each GPU kernel, write tests that compare GPU results against CPU results for the same operation with various input sizes and edge cases. Your current tests in `tests/gpu_test.rs` are a good start but need expansion. Test not just correctness but also proper error handling when GPU is unavailable.

**Gradient checking on GPU:** Extend your numerical gradient checking infrastructure to work with GPU tensors. The `check_gradients` function should support checking gradients for tensors on any device. This is crucial because GPU floating-point arithmetic can have subtle differences from CPU.

**Integration testing strategy:** Write end-to-end tests that train small networks entirely on GPU and verify that loss decreases correctly. Compare final model weights between CPU-trained and GPU-trained models to ensure numerical consistency (within acceptable floating-point tolerance).

**Performance validation:** Create benchmarks that measure throughput (samples per second) for various operations and layer types on both CPU and GPU. This helps you identify which operations actually benefit from GPU acceleration and where bottlenecks remain. Include profiling to measure time spent in computation vs. data transfer.

## Recommended Implementation Order

Based on the architectural dependencies, here's how I'd suggest tackling this:

**Phase 1 - Foundation (2-3 weeks):** Make the autograd engine device-aware. Modify gradient initialization and accumulation in `autograd.rs` to respect tensor device. Implement device-aware operation dispatch in the core binary, unary, and matmul operations. This creates the foundation for everything else.

**Phase 2 - Core Operations (1-2 weeks):** Implement GPU kernels for missing critical operations, particularly reductions along dimensions and movement operations. Focus on operations heavily used by your neural network layers. Update all `GradFn` implementations to support GPU backward passes.

**Phase 3 - Layer Support (1-2 weeks):** Refactor `Linear` and `Conv2d` layers to be device-agnostic. Implement GPU-accelerated convolution (either via GPU im2col or direct convolution kernels). Make layer parameters device-aware.

**Phase 4 - Training Infrastructure (1 week):** Update optimizers to support GPU parameter updates. Ensure DataLoader can feed data to the appropriate device. Test complete training loops on GPU.

**Phase 5 - Validation and Optimization (ongoing):** Write comprehensive tests for gradient correctness. Benchmark performance. Profile to identify remaining bottlenecks. Optimize memory transfer patterns to minimize GPU-CPU round trips.

The key architectural insight is that GPU support isn't just about writing GPU kernels - it's about making device awareness pervasive throughout your system while maintaining the clean API you've built. The storage abstraction you've created is excellent, but you need to thread device-aware logic through every layer of the stack: operations, gradients, layers, and optimizers.

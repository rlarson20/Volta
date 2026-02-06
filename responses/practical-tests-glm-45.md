# Volta GPU Integration Plan

To integrate the GPU backend into Volta's core tensor operations and autograd system, we need to:

1. **Enable GPU tensor creation and movement**
2. **Wire GPU kernels into core operations**
3. **Extend autograd for GPU tensors**
4. **Update IO for GPU serialization**

Here's the implementation approach:

## 1. Enable GPU Tensor Creation and Movement

### Storage Enhancement

```rust
// src/storage.rs
impl Storage {
    pub fn new_on_device(data: Vec<f32>, device: &Device) -> Self {
        match device {
            Device::CPU => Storage::cpu(data),
            Device::GPU(_) | Device::Metal(_) => {
                #[cfg(feature = "gpu")]
                {
                    if is_gpu_available() {
                        Storage::gpu(data)
                    } else {
                        eprintln!("Warning: GPU requested but not available, using CPU");
                        Storage::cpu(data)
                    }
                }
                #[cfg(not(feature = "gpu"))]
                {
                    eprintln!("Warning: GPU feature not enabled, using CPU");
                    Storage::cpu(data)
                }
            }
        }
    }
}
```

### Tensor Creation on Device

```rust
// src/tensor.rs
impl RawTensor {
    pub fn new_on_device(data: Vec<f32>, shape: &[usize], requires_grad: bool, device: Device) -> Tensor {
        assert_eq!(data.len(), shape.iter().product::<usize>());
        assert!(data.len() > 0 && data.len() <= 100_000_000, "Invalid tensor size");

        let storage = Storage::new_on_device(data, &device);
        let raw = RawTensor {
            data: storage,
            shape: shape.to_vec(),
            grad: None,
            requires_grad,
            grad_fn: None,
            parents: vec![],
            device,
        };
        Rc::new(RefCell::new(raw))
    }

    pub fn to_device(self_t: &Tensor, device: Device) -> Tensor {
        // Fast path: already on requested device
        if self_t.borrow().device == device {
            return self_t.clone();
        }

        let t = self_t.borrow();
        let data = t.data.to_vec(); // Copy data to CPU first
        Self::new_on_device(data, &t.shape, t.requires_grad, device)
    }
}
```

## 2. Wire GPU Kernels into Core Operations

### Binary Operations with GPU Support

```rust
// src/ops/binary.rs
impl RawTensor {
    pub fn binary_op(self_t: &Tensor, other: &Tensor, op: BinaryOp) -> Tensor {
        let self_device = self_t.borrow().device.clone();
        let other_device = other.borrow().device.clone();

        // Handle device mismatch
        if self_device != other_device {
            // Move both tensors to CPU for the operation
            let self_on_cpu = self_t.to_device(Device::CPU);
            let other_on_cpu = other.to_device(Device::CPU);
            return RawTensor::binary_op(&self_on_cpu, &other_on_cpu, op);
        }

        let (data_a, shape_a, req_a) = {
            let s = self_t.borrow();
            (s.data.clone(), s.shape.clone(), s.requires_grad)
        };
        let (data_b, shape_b, req_b) = {
            let o = other.borrow();
            (o.data.clone(), o.shape.clone(), o.requires_grad)
        };

        let out_shape = Self::broadcast_shape(&shape_a, &shape_b);
        let bc_data_a = Self::broadcast_to(&data_a, &shape_a, &out_shape);
        let bc_data_b = Self::broadcast_to(&data_b, &shape_b, &out_shape);

        // GPU path
        if self_device.is_gpu() {
            let storage_a = Storage::new_on_device(bc_data_a, &self_device);
            let storage_b = Storage::new_on_device(bc_data_b, &self_device);

            let result_storage = match op {
                BinaryOp::Add => RawTensor::gpu_add(&storage_a, &storage_b),
                BinaryOp::Sub => RawTensor::gpu_sub(&storage_a, &storage_b),
                BinaryOp::Mul => RawTensor::gpu_mul(&storage_a, &storage_b),
                BinaryOp::Div => RawTensor::gpu_div(&storage_a, &storage_b),
                // Fallback to CPU for non-differentiable ops
                _ => {
                    let result_data = Self::binary_op_core(&bc_data_a, &bc_data_b, op);
                    Storage::cpu(result_data)
                }
            };

            // Fall back to CPU if GPU operation fails
            let result_storage = result_storage.unwrap_or_else(|| {
                let result_data = Self::binary_op_core(&bc_data_a, &bc_data_b, op);
                Storage::cpu(result_data)
            });

            let out = Self::new(result_storage.to_vec(), &out_shape, req_a || req_b);

            if out.borrow().requires_grad {
                out.borrow_mut().parents = vec![self_t.clone(), other.clone()];
                out.borrow_mut().grad_fn = Some(Box::new(BinaryGradFn { op }));
            }
            return out;
        }

        // CPU path (existing implementation)
        // ... [existing CPU binary_op implementation] ...
    }
}
```

### Matrix Multiplication with GPU Support

```rust
// src/ops/matmul.rs
impl RawTensor {
    pub fn matmul(self_t: &Tensor, other: &Tensor) -> Tensor {
        let self_device = self_t.borrow().device.clone();
        let other_device = other.borrow().device.clone();

        // Handle device mismatch
        if self_device != other_device {
            let self_on_cpu = self_t.to_device(Device::CPU);
            let other_on_cpu = other.to_device(Device::CPU);
            return RawTensor::matmul(&self_on_cpu, &other_on_cpu);
        }

        // ... [existing shape validation and data extraction] ...

        // GPU path
        if self_device.is_gpu() {
            let storage_a = Storage::new_on_device(data_a, &self_device);
            let storage_b = Storage::new_on_device(data_b, &self_device);

            let result_storage = match (shape_a.len(), shape_b.len()) {
                (2, 2) => {
                    let (m, n) = (shape_a[0], shape_a[1]);
                    let (n2, p) = (shape_b[0], shape_b[1]);
                    assert_eq!(n, n2);
                    RawTensor::gpu_matmul(&storage_a, &storage_b, m, n, p)
                }
                // ... [other cases] ...
                _ => None, // Fallback to CPU
            };

            let result_storage = result_storage.unwrap_or_else(|| {
                // Fallback to CPU matmul
                Self::matmul_raw(&data_a, &data_b, m, k, p)
            });

            let out = Self::new(result_storage.to_vec(), &[m, p], req_a || req_b);

            if out.borrow().requires_grad {
                out.borrow_mut().parents = vec![self_t.clone(), other.clone()];
                out.borrow_mut().grad_fn = Some(Box::new(MatMulGradFn));
            }
            return out;
        }

        // CPU path (existing implementation)
        // ... [existing CPU matmul implementation] ...
    }
}
```

## 3. Extend Autograd for GPU Tensors

### Backward Pass with GPU Support

```rust
// src/autograd.rs
impl RawTensor {
    pub fn backward(tensor_ref: &Tensor) {
        // ... [existing backward computation] ...

        // Move gradients to GPU if original tensor was on GPU
        {
            let mut tensor = tensor_ref.borrow_mut();
            if tensor.device.is_gpu() && tensor.grad.is_some() {
                let grad_storage = tensor.grad.take().unwrap();
                tensor.grad = Some(grad_storage.to_device(&tensor.device));
            }
        }
    }
}
```

## 4. Update IO for GPU Serialization

### TensorData with GPU Support

```rust
// src/io.rs
impl TensorData {
    pub fn from_tensor(t: &Tensor) -> Self {
        let borrowed = t.borrow();
        TensorData {
            data: borrowed.data.to_vec(), // Automatically handles GPU→CPU copy
            shape: borrowed.shape.clone(),
        }
    }
}
```

## Implementation Steps

1. **Add `new_on_device` and `to_device` to `RawTensor`**
2. **Modify binary operations to support GPU**
3. **Update matrix multiplication for GPU**
4. **Extend autograd for GPU tensors**
5. **Update IO to handle GPU tensors**
6. **Add comprehensive tests**

## Testing Strategy

1. **Unit tests for GPU tensor creation/movement**
2. **Operation tests comparing CPU vs GPU outputs**
3. **Gradient validation tests on GPU**
4. **End-to-end training tests with GPU**
5. **Performance benchmarks for GPU operations**

This implementation will enable Volta to perform computations on GPU while maintaining compatibility with existing CPU code and ensuring correct gradient propagation through the computation graph.

use volta::gpu;
#[cfg(all(test, feature = "gpu"))]
mod gpu_tests {
    use super::*;
    use volta::{Device, RawTensor, TensorOps};

    #[test]
    fn test_gpu_available() {
        // This test just checks that GPU initialization doesn't panic
        let available = gpu::is_gpu_available();
        println!("GPU available: {available}");
    }

    #[test]
    fn test_gpu_buffer_roundtrip() {
        if !gpu::is_gpu_available() {
            println!("Skipping GPU test - no GPU available");
            return;
        }

        let data = vec![1.0, 2.0, 3.0, 4.0];
        let buffer = gpu::GpuBuffer::from_slice(&data).unwrap();
        let result = buffer.to_vec();

        assert_eq!(data, result);
    }

    #[test]
    fn test_gpu_add() {
        if !gpu::is_gpu_available() {
            println!("Skipping GPU test - no GPU available");
            return;
        }

        let a = gpu::GpuBuffer::from_slice(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = gpu::GpuBuffer::from_slice(&[5.0, 6.0, 7.0, 8.0]).unwrap();

        let c = gpu::GpuKernels::binary_op(&a, &b, "add").unwrap();
        let result = c.to_vec();

        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_gpu_matmul() {
        if !gpu::is_gpu_available() {
            println!("Skipping GPU test - no GPU available");
            return;
        }

        // 2x2 @ 2x2
        let a = gpu::GpuBuffer::from_slice(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = gpu::GpuBuffer::from_slice(&[5.0, 6.0, 7.0, 8.0]).unwrap();

        let c = gpu::GpuKernels::matmul(&a, &b, 2, 2, 2).unwrap();
        let result = c.to_vec();

        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        //         = [[19, 22], [43, 50]]
        assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_gpu_autograd_root_grad_on_gpu() {
        if !gpu::is_gpu_available() {
            println!("Skipping GPU autograd test - no GPU available");
            return;
        }

        let x = RawTensor::new(vec![2.0], &[1], true);
        let x_gpu = x.to_device(Device::GPU("TestDevice".to_string()));

        x_gpu.backward();

        let xb = x_gpu.borrow();
        let grad_storage = xb.grad.as_ref().expect("Gradient should be set");
        assert!(
            grad_storage.is_gpu(),
            "Expected root gradient storage to live on GPU"
        );
        assert_eq!(grad_storage.to_vec(), vec![1.0]);
    }

    #[test]
    fn test_gpu_autograd_accumulates_on_gpu() {
        if !gpu::is_gpu_available() {
            println!("Skipping GPU autograd accumulation test - no GPU available");
            return;
        }

        let device = Device::GPU("TestDevice".to_string());

        // a and b live on GPU
        let a = RawTensor::new(vec![1.0], &[1], true).to_device(device.clone());
        let b = RawTensor::new(vec![2.0], &[1], true).to_device(device.clone());

        // Build a small graph where 'a' is used twice:
        // c1 = a + b
        // c2 = a + b
        // loss = c1 + c2
        // d(loss)/d(a) = 2
        let c1 = a.add(&b).to_device(device.clone());
        let c2 = a.add(&b).to_device(device.clone());
        let loss = c1.add(&c2).to_device(device.clone());

        loss.backward();

        let ab = a.borrow();
        let grad_storage = ab.grad.as_ref().expect("Gradient for 'a' missing");
        assert!(
            grad_storage.is_gpu(),
            "Expected accumulated gradient for 'a' to live on GPU"
        );
        let grad_vals = grad_storage.to_vec();
        assert_eq!(grad_vals, vec![2.0]);
    }

    #[test]
    fn test_gpu_movement_ops() {
        if !gpu::is_gpu_available() {
            println!("Skipping GPU movement ops test - no GPU available");
            return;
        }

        let device = Device::GPU("TestDevice".to_string());

        // Note: GPU movement kernels are in progress (shader debugging needed)
        // For now, these work via CPU fallback which is correct (memory-bound ops)

        // Test permute (transpose 2D) - uses CPU fallback for now
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false).to_device(device.clone());
        let x_t = x.permute(&[1, 0]);
        assert_eq!(x_t.borrow().shape, vec![2, 2]);
        assert_eq!(x_t.borrow().data.to_vec(), vec![1.0, 3.0, 2.0, 4.0]);

        // Test expand (broadcast)
        let y = RawTensor::new(vec![1.0, 2.0], &[1, 2], false).to_device(device.clone());
        let y_expanded = y.expand(&[3, 2]);
        assert_eq!(y_expanded.borrow().shape, vec![3, 2]);
        assert_eq!(
            y_expanded.borrow().data.to_vec(),
            vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        );

        // Test shrink (slice)
        let z = RawTensor::new((0..16).map(|i| i as f32).collect(), &[4, 4], false)
            .to_device(device.clone());
        let z_shrunk = z.shrink(&[(1, 3), (1, 3)]);
        assert_eq!(z_shrunk.borrow().shape, vec![2, 2]);
        assert_eq!(z_shrunk.borrow().data.to_vec(), vec![5.0, 6.0, 9.0, 10.0]);

        // Test stride (subsample)
        let w = RawTensor::new((0..16).map(|i| i as f32).collect(), &[4, 4], false)
            .to_device(device.clone());
        let w_strided = w.stride_op(&[2, 2]);
        assert_eq!(w_strided.borrow().shape, vec![2, 2]);
        assert_eq!(w_strided.borrow().data.to_vec(), vec![0.0, 2.0, 8.0, 10.0]);
    }
}

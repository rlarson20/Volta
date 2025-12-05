use volta::gpu;
#[cfg(all(test, feature = "gpu"))]
mod gpu_tests {
    use super::*;

    #[test]
    fn test_gpu_available() {
        // This test just checks that GPU initialization doesn't panic
        let available = gpu::is_gpu_available();
        println!("GPU available: {}", available);
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
}

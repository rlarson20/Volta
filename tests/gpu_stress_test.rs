use volta::gpu;

#[cfg(all(test, feature = "gpu"))]
mod gpu_extended_tests {
    use super::*;

    // Helper to run comprehensive binary op tests
    fn run_binary_op_test(op: &str, expected: &[f32]) {
        if !gpu::is_gpu_available() {
            println!("Skipping GPU test (no device)");
            return;
        }

        let a = gpu::GpuBuffer::from_slice(&[10.0, 20.0, 30.0, 40.0]).unwrap();
        let b = gpu::GpuBuffer::from_slice(&[2.0, 4.0, 5.0, 8.0]).unwrap();

        let c = gpu::GpuKernels::binary_op(&a, &b, op)
            .unwrap_or_else(|| panic!("Failed to run binary op {}", op));

        let result = c.to_vec();
        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-5,
                "Op {} failed at index {}: got {}, expected {}",
                op,
                i,
                r,
                e
            );
        }
    }

    #[test]
    fn test_gpu_binary_ops_comprehensive() {
        // Sub: 10-2, 20-4, 30-5, 40-8
        run_binary_op_test("sub", &[8.0, 16.0, 25.0, 32.0]);
        // Mul: 10*2, 20*4, 30*5, 40*8
        run_binary_op_test("mul", &[20.0, 80.0, 150.0, 320.0]);
        // Div: 10/2, 20/4, 30/5, 40/8
        run_binary_op_test("div", &[5.0, 5.0, 6.0, 5.0]);
    }

    #[test]
    fn test_gpu_unary_ops_comprehensive() {
        if !gpu::is_gpu_available() {
            return;
        }

        let input = [-1.0, 0.0, 1.0, 4.0];
        let buf = gpu::GpuBuffer::from_slice(&input).unwrap();

        // Test ReLU
        let res = gpu::GpuKernels::unary_op(&buf, "relu").unwrap().to_vec();
        assert_eq!(res, vec![0.0, 0.0, 1.0, 4.0]);

        // Test Sqrt (using positive inputs)
        let buf_pos = gpu::GpuBuffer::from_slice(&[4.0, 9.0, 16.0]).unwrap();
        let res_sqrt = gpu::GpuKernels::unary_op(&buf_pos, "sqrt")
            .unwrap()
            .to_vec();
        assert_eq!(res_sqrt, vec![2.0, 3.0, 4.0]);

        // Test Neg
        let res_neg = gpu::GpuKernels::unary_op(&buf, "neg").unwrap().to_vec();
        assert_eq!(res_neg, vec![1.0, -0.0, -1.0, -4.0]);

        // Test Recip (using positive inputs to avoid division by zero)
        let buf_recip = gpu::GpuBuffer::from_slice(&[1.0, 2.0, 4.0, 8.0]).unwrap();
        let res_recip = gpu::GpuKernels::unary_op(&buf_recip, "recip")
            .unwrap()
            .to_vec();
        for (i, &val) in res_recip.iter().enumerate() {
            let expected = 1.0 / [1.0, 2.0, 4.0, 8.0][i];
            assert!(
                (val - expected).abs() < 1e-5,
                "Recip failed at index {}: got {}, expected {}",
                i,
                val,
                expected
            );
        }

        // Test Exp2
        let res_exp2 = gpu::GpuKernels::unary_op(&buf, "exp2").unwrap().to_vec();
        for (i, &val) in res_exp2.iter().enumerate() {
            let expected = 2_f32.powf([-1.0, 0.0, 1.0, 4.0][i]);
            assert!(
                (val - expected).abs() < 1e-4,
                "Exp2 failed at index {}: got {}, expected {}",
                i,
                val,
                expected
            );
        }

        // Test Log2 (using positive inputs)
        let buf_log2 = gpu::GpuBuffer::from_slice(&[1.0, 2.0, 4.0, 8.0]).unwrap();
        let res_log2 = gpu::GpuKernels::unary_op(&buf_log2, "log2")
            .unwrap()
            .to_vec();
        for (i, &val) in res_log2.iter().enumerate() {
            let expected = [0.0, 1.0, 2.0, 3.0][i];
            assert!(
                (val - expected).abs() < 1e-5,
                "Log2 failed at index {}: got {}, expected {}",
                i,
                val,
                expected
            );
        }

        // Test Sin
        let res_sin = gpu::GpuKernels::unary_op(&buf, "sin").unwrap().to_vec();
        for (i, &val) in res_sin.iter().enumerate() {
            let expected = [-1.0_f32.sin(), 0.0_f32.sin(), 1.0_f32.sin(), 4.0_f32.sin()][i];
            assert!(
                (val - expected).abs() < 1e-5,
                "Sin failed at index {}: got {}, expected {}",
                i,
                val,
                expected
            );
        }

        // Test Cos
        let res_cos = gpu::GpuKernels::unary_op(&buf, "cos").unwrap().to_vec();
        let expected_vals = [0.5403023, 1.0, 0.5403023, -0.6536436]; // cos([-1, 0, 1, 4])
        for (i, &val) in res_cos.iter().enumerate() {
            assert!(
                (val - expected_vals[i]).abs() < 1e-5,
                "Cos failed at index {}: got {}, expected {}",
                i,
                val,
                expected_vals[i]
            );
        }
    }
}

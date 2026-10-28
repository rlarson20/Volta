#![allow(missing_docs)] // not public API
#[cfg(feature = "gpu")]
use volta::gpu;
#[cfg(feature = "gpu")]
use volta::ops::{BinaryOp, UnaryOp};

#[cfg(all(test, feature = "gpu"))]
mod gpu_extended_tests {
    use super::*;

    // Helper to run comprehensive binary op tests
    fn run_binary_op_test(op: BinaryOp, expected: &[f32]) {
        if !gpu::is_gpu_available() {
            println!("Skipping GPU test (no device)");
            return;
        }

        let a = gpu::GpuBuffer::from_slice(&[10.0, 20.0, 30.0, 40.0]).unwrap();
        let b = gpu::GpuBuffer::from_slice(&[2.0, 4.0, 5.0, 8.0]).unwrap();

        let c = gpu::GpuKernels::binary_op(&a, &b, op)
            .unwrap_or_else(|| panic!("Failed to run binary op {op:?}"));

        let result = c.to_vec();
        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-5,
                "Op {op:?} failed at index {i}: got {r}, expected {e}"
            );
        }
    }

    #[test]
    fn test_gpu_binary_ops_comprehensive() {
        // Sub: 10-2, 20-4, 30-5, 40-8
        run_binary_op_test(BinaryOp::Sub, &[8.0, 16.0, 25.0, 32.0]);
        // Mul: 10*2, 20*4, 30*5, 40*8
        run_binary_op_test(BinaryOp::Mul, &[20.0, 80.0, 150.0, 320.0]);
        // Div: 10/2, 20/4, 30/5, 40/8
        run_binary_op_test(BinaryOp::Div, &[5.0, 5.0, 6.0, 5.0]);
    }

    #[test]
    fn test_gpu_unary_ops_comprehensive() {
        if !gpu::is_gpu_available() {
            return;
        }

        let input = [-1.0, 0.0, 1.0, 4.0];
        let buf = gpu::GpuBuffer::from_slice(&input).unwrap();

        // Test ReLU
        let res = gpu::GpuKernels::unary_op(&buf, UnaryOp::ReLU)
            .unwrap()
            .to_vec();
        assert_eq!(res, vec![0.0, 0.0, 1.0, 4.0]);

        // Test Sqrt (using positive inputs)
        let buf_pos = gpu::GpuBuffer::from_slice(&[4.0, 9.0, 16.0]).unwrap();
        let res_sqrt = gpu::GpuKernels::unary_op(&buf_pos, UnaryOp::Sqrt)
            .unwrap()
            .to_vec();
        assert_eq!(res_sqrt, vec![2.0, 3.0, 4.0]);

        // Test Neg
        let res_neg = gpu::GpuKernels::unary_op(&buf, UnaryOp::Neg)
            .unwrap()
            .to_vec();
        assert_eq!(res_neg, vec![1.0, -0.0, -1.0, -4.0]);

        // Test Recip (using positive inputs to avoid division by zero)
        let buf_recip = gpu::GpuBuffer::from_slice(&[1.0, 2.0, 4.0, 8.0]).unwrap();
        let res_recip = gpu::GpuKernels::unary_op(&buf_recip, UnaryOp::Recip)
            .unwrap()
            .to_vec();
        for (i, &val) in res_recip.iter().enumerate() {
            let expected_vals = [1.0, 2.0, 4.0, 8.0];
            let expected = 1.0 / expected_vals.get(i).copied().unwrap_or(f32::NAN);
            assert!(
                (val - expected).abs() < 1e-5,
                "Recip failed at index {i}: got {val}, expected {expected}"
            );
        }

        // Test Exp2
        let res_exp2 = gpu::GpuKernels::unary_op(&buf, UnaryOp::Exp2)
            .unwrap()
            .to_vec();
        for (i, &val) in res_exp2.iter().enumerate() {
            let exp_vals = [-1.0, 0.0, 1.0, 4.0];
            let expected = 2_f32.powf(exp_vals.get(i).copied().unwrap_or(f32::NAN));
            assert!(
                (val - expected).abs() < 1e-4,
                "Exp2 failed at index {i}: got {val}, expected {expected}"
            );
        }

        // Test Log2 (using positive inputs)
        let buf_log2 = gpu::GpuBuffer::from_slice(&[1.0, 2.0, 4.0, 8.0]).unwrap();
        let res_log2 = gpu::GpuKernels::unary_op(&buf_log2, UnaryOp::Log2)
            .unwrap()
            .to_vec();
        for (i, &val) in res_log2.iter().enumerate() {
            let expected_vals = [0.0, 1.0, 2.0, 3.0];
            let expected = expected_vals.get(i).copied().unwrap_or(f32::NAN);
            assert!(
                (val - expected).abs() < 1e-5,
                "Log2 failed at index {i}: got {val}, expected {expected}"
            );
        }

        // Test Sin
        let res_sin = gpu::GpuKernels::unary_op(&buf, UnaryOp::Sin)
            .unwrap()
            .to_vec();
        for (i, &val) in res_sin.iter().enumerate() {
            let expected_vals = [-1.0_f32.sin(), 0.0_f32.sin(), 1.0_f32.sin(), 4.0_f32.sin()];
            let expected = expected_vals.get(i).copied().unwrap_or(f32::NAN);
            assert!(
                (val - expected).abs() < 1e-5,
                "Sin failed at index {i}: got {val}, expected {expected}"
            );
        }

        // Test Cos
        let res_cos = gpu::GpuKernels::unary_op(&buf, UnaryOp::Cos)
            .unwrap()
            .to_vec();
        let expected_vals = [0.540_302_3, 1.0, 0.540_302_3, -0.653_643_6]; // cos([-1, 0, 1, 4])
        for (i, &val) in res_cos.iter().enumerate() {
            assert!(
                (val - expected_vals.get(i).copied().unwrap_or(f32::NAN)).abs() < 1e-5,
                "Cos failed at index {}: got {}, expected {}",
                i,
                val,
                expected_vals.get(i).copied().unwrap_or(f32::NAN)
            );
        }
    }
}

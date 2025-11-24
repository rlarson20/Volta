//! # Volta
//!
//! A minimal automatic differentiation library implementing PyTorch-like tensor operations
//! from scratch in pure Rust. This library provides:
//! - Dynamic computation graphs for automatic differentiation
//! - Broadcasting support for tensor operations
//! - Common neural network operations (matmul, activations, etc.)
//! - Numerical gradient checking for validation
//!
//! ## Architecture
//!
//! The library uses reference-counted interior mutability (`Rc<RefCell<RawTensor>>`) to build
//! dynamic computation graphs. Each tensor operation creates new tensors and stores gradient
//! functions that know how to backpropagate through that operation.

pub mod autograd;
pub mod device;
pub mod io;
pub mod nn;
pub mod ops;
pub mod tensor;

// Re-export main types for easy access
pub use autograd::GradFn;
pub use device::Device;
pub use nn::layers::flatten::Flatten;
pub use nn::{Adam, BatchNorm2d, Conv2d, Linear, MaxPool2d, Module, ReLU, SGD, Sequential};
pub use tensor::{RawTensor, Tensor, TensorOps};

// Main entry points

pub use tensor::{
    check_gradients, check_gradients_simple, cross_entropy_loss, max_dim, mse_loss, new_tensor,
    ones, rand, randn, softmax, sum_dim, zeros,
};

pub use ops::{
    BinaryGradFn, BinaryOp, MatMulGradFn, MaxReduceGradFn, MeanGradFn, MovementGradFn, MovementOp,
    MulAccGradFn, ReduceOp, SumGradFn, TernaryOp, UnaryGradFn, UnaryOp, WhereGradFn,
};

pub use tensor::DataLoader;

// ===== TESTS =====
//
// The test suite validates:
// - Basic operations (add, mul, etc.)
// - Gradient correctness (chain rule, broadcasting)
// - Complex scenarios (neural networks, matmul variants)
// - Numerical gradient checking (validates all gradients)

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_add_backward() {
        let a = RawTensor::new(vec![2.0], &[1], true);
        let b = RawTensor::new(vec![3.0], &[1], true);
        let c = a.add(&b);
        c.backward();

        assert_eq!(a.grad(), Some(vec![1.0]));
        assert_eq!(b.grad(), Some(vec![1.0]));
    }

    #[test]
    fn test_multiply_backward() {
        let a = RawTensor::new(vec![3.0], &[1], true);
        let b = RawTensor::new(vec![4.0], &[1], true);
        let c = a.elem_mul(&b);
        c.backward();

        assert_eq!(a.grad(), Some(vec![4.0]));
        assert_eq!(b.grad(), Some(vec![3.0]));
    }

    #[test]
    fn test_chain_rule() {
        let a = RawTensor::new(vec![2.0], &[1], true);
        let b = RawTensor::new(vec![3.0], &[1], true);
        let c = a.add(&b);
        let d = c.elem_mul(&a);
        d.backward();

        assert_eq!(a.grad(), Some(vec![7.0]));
        assert_eq!(b.grad(), Some(vec![2.0]));
    }

    #[test]
    fn test_sum_backward() {
        let a = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], true);
        let loss = a.sum();
        loss.backward();

        assert_eq!(a.grad(), Some(vec![1.0, 1.0, 1.0]));
    }

    #[test]
    fn test_multidim_ops() {
        let a = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        let b = RawTensor::new(vec![0.5, 0.5, 0.5, 0.5], &[2, 2], true);
        let c = a.elem_mul(&b);
        let loss = c.sum();
        loss.backward();

        assert_eq!(a.grad(), Some(vec![0.5, 0.5, 0.5, 0.5]));
        assert_eq!(b.grad(), Some(vec![1.0, 2.0, 3.0, 4.0]));
    }
}

#[cfg(test)]
mod unary_tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_neg_forward_backward() {
        let x = RawTensor::new(vec![2.0, -3.0], &[2], true);
        let y = x.neg();

        // Forward
        assert_eq!(y.borrow().data, vec![-2.0, 3.0]);

        // Backward: ∂(-x)/∂x = -1
        y.backward();
        assert_eq!(x.grad(), Some(vec![-1.0, -1.0]));
    }

    #[test]
    fn test_sqrt_chain() {
        let x = RawTensor::new(vec![4.0], &[1], true);
        let y = x.sqrt(); // y = 2.0
        let z = y.elem_mul(&y); // z = 4.0
        z.backward();

        // ∂z/∂x = ∂z/∂y * ∂y/∂x = 2y * 1/(2√x) = 2*2 * 1/4 = 1.0
        assert_relative_eq!(x.grad().unwrap()[0], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_exp2_log2_inverse() {
        let x = RawTensor::new(vec![2.0], &[1], true);
        let y = x.exp2().log2(); // should recover x
        y.backward();

        assert_relative_eq!(y.borrow().data[0], 2.0, epsilon = 1e-6);
        // Chain rule: ∂(log2(2^x))/∂x = 1
        assert_relative_eq!(x.grad().unwrap()[0], 1.0, epsilon = 1e-6);
    }
}

#[cfg(test)]
mod binary_tests {
    use super::*;

    #[test]
    fn test_div_backward() {
        let x = RawTensor::new(vec![6.0], &[1], true);
        let y = RawTensor::new(vec![2.0], &[1], true);
        let z = x.div(&y); // z = 3.0
        z.backward();

        // ∂(x/y)/∂x = 1/y = 0.5
        assert_eq!(x.grad(), Some(vec![0.5]));
        // ∂(x/y)/∂y = -x/y² = -6/4 = -1.5
        assert_eq!(y.grad(), Some(vec![-1.5]));
    }

    #[test]
    fn test_max_backward() {
        let x = RawTensor::new(vec![3.0, 1.0], &[2], true);
        let y = RawTensor::new(vec![2.0, 4.0], &[2], true);
        let z = x.max_elem(&y);
        let loss = z.sum();
        loss.backward();

        // max picks [3.0, 4.0], so grads flow to x[0] and y[1]
        assert_eq!(x.grad(), Some(vec![1.0, 0.0]));
        assert_eq!(y.grad(), Some(vec![0.0, 1.0]));
    }
}

#[cfg(test)]
mod reduce_tests {
    use super::*;

    #[test]
    fn test_reduce_max_backward() {
        let x = RawTensor::new(vec![1.0, 5.0, 3.0], &[3], true);
        let y = x.max_reduce(); // finds 5.0 at index 1
        y.backward();

        // Only max element gets gradient
        assert_eq!(x.grad(), Some(vec![0.0, 1.0, 0.0]));
    }
}

#[cfg(test)]
mod ternary_tests {
    use super::*;

    #[test]
    fn test_mulacc_backward() {
        // z = x*y + w
        let x = RawTensor::new(vec![2.0], &[1], true);
        let y = RawTensor::new(vec![3.0], &[1], true);
        let w = RawTensor::new(vec![1.0], &[1], true);
        let z = x.mulacc(&y, &w); // z = 7.0
        z.backward();

        assert_eq!(x.grad(), Some(vec![3.0])); // ∂z/∂x = y
        assert_eq!(y.grad(), Some(vec![2.0])); // ∂z/∂y = x
        assert_eq!(w.grad(), Some(vec![1.0])); // ∂z/∂w = 1
    }

    #[test]
    fn test_where_backward() {
        let cond = RawTensor::new(vec![1.0, 0.0], &[2], false);
        let x = RawTensor::new(vec![10.0, 20.0], &[2], true);
        let y = RawTensor::new(vec![30.0, 40.0], &[2], true);
        let z = cond.where_op(&x, &y); // picks [10.0, 40.0]
        z.backward();

        assert_eq!(x.grad(), Some(vec![1.0, 0.0])); // grad flows where cond=1
        assert_eq!(y.grad(), Some(vec![0.0, 1.0])); // grad flows where cond=0
    }
}

#[cfg(test)]
mod movement_tests {
    use super::*;

    #[test]
    fn test_reshape_backward() {
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4], true);
        let y = x.reshape(&[2, 2]);
        let loss = y.sum();
        loss.backward();

        // Gradient reshapes back to [4]
        assert_eq!(x.grad(), Some(vec![1.0, 1.0, 1.0, 1.0]));
    }

    #[test]
    fn test_permute_backward() {
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        let y = x.permute(&[1, 0]); // transpose
        let loss = y.sum();
        loss.backward();

        // Gradient permutes back
        assert_eq!(x.grad(), Some(vec![1.0, 1.0, 1.0, 1.0]));
    }
}

#[cfg(test)]
mod misc_tests {
    use super::*;
    // ===== NEURAL NETWORK LAYER TEST =====

    #[test]
    fn test_linear_layer() {
        // Simple linear layer: y = xW + b
        let x = RawTensor::new(vec![1.0, 2.0, 3.0], &[1, 3], true);
        let w = RawTensor::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[3, 2], true);
        let b = RawTensor::new(vec![0.1, 0.2], &[1, 2], true);

        let y = x.matmul(&w); // [1,3] @ [3,2] = [1,2]
        let out = y.add(&b);
        let loss = out.sum();

        loss.backward();

        // All should have gradients
        assert!(x.grad().is_some());
        assert!(w.grad().is_some());
        assert!(b.grad().is_some());

        // b gradient should be ones (direct path from sum)
        assert_eq!(b.grad(), Some(vec![1.0, 1.0]));
    }

    // ===== BROADCASTING TESTS =====

    #[test]
    fn test_broadcast_shape() {
        // (3, 1) and (1, 4) -> (3, 4)
        let shape = RawTensor::broadcast_shape(&[3, 1], &[1, 4]);
        assert_eq!(shape, vec![3, 4]);

        // (5, 3, 1) and (1, 4) -> (5, 3, 4)
        let shape = RawTensor::broadcast_shape(&[5, 3, 1], &[1, 4]);
        assert_eq!(shape, vec![5, 3, 4]);

        // (1,) and (3, 4) -> (3, 4)
        let shape = RawTensor::broadcast_shape(&[1], &[3, 4]);
        assert_eq!(shape, vec![3, 4]);

        // (3, 4) and (4,) -> (3, 4)
        let shape = RawTensor::broadcast_shape(&[3, 4], &[4]);
        assert_eq!(shape, vec![3, 4]);
    }

    #[test]
    #[should_panic(expected = "Cannot broadcast")]
    fn test_broadcast_incompatible() {
        RawTensor::broadcast_shape(&[3, 2], &[4, 3]);
    }

    #[test]
    fn test_broadcast_add_scalar() {
        // (2, 3) + scalar -> (2, 3)
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let scalar = RawTensor::new(vec![10.0], &[1], true);
        let y = x.add(&scalar);

        assert_eq!(y.borrow().shape, vec![2, 3]);
        assert_eq!(y.borrow().data, vec![11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);

        y.backward();

        // x gradient: all ones
        assert_eq!(x.grad(), Some(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]));
        // scalar gradient: sum of all gradients = 6.0
        assert_eq!(scalar.grad(), Some(vec![6.0]));
    }

    #[test]
    fn test_broadcast_mul_vector() {
        // (2, 3) * (3,) -> (2, 3)
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let v = RawTensor::new(vec![2.0, 3.0, 4.0], &[3], true);
        let y = x.elem_mul(&v);

        assert_eq!(y.borrow().shape, vec![2, 3]);
        assert_eq!(y.borrow().data, vec![2.0, 6.0, 12.0, 8.0, 15.0, 24.0]);

        y.backward();

        // x gradient: broadcast v
        assert_eq!(x.grad(), Some(vec![2.0, 3.0, 4.0, 2.0, 3.0, 4.0]));
        // v gradient: sum over rows
        assert_eq!(v.grad(), Some(vec![5.0, 7.0, 9.0])); // [1+4, 2+5, 3+6]
    }

    #[test]
    fn test_broadcast_add_matrix() {
        // (2, 1) + (1, 3) -> (2, 3)
        let x = RawTensor::new(vec![1.0, 2.0], &[2, 1], true);
        let y = RawTensor::new(vec![10.0, 20.0, 30.0], &[1, 3], true);
        let z = x.add(&y);

        assert_eq!(z.borrow().shape, vec![2, 3]);
        assert_eq!(z.borrow().data, vec![11.0, 21.0, 31.0, 12.0, 22.0, 32.0]);

        z.backward();

        // x gradient: sum over columns -> [3.0, 3.0]
        assert_eq!(x.grad(), Some(vec![3.0, 3.0]));
        // y gradient: sum over rows -> [2.0, 2.0, 2.0]
        assert_eq!(y.grad(), Some(vec![2.0, 2.0, 2.0]));
    }

    #[test]
    fn test_broadcast_batch_bias() {
        // Simulate batch with bias: (batch=3, features=2) + (features=2,)
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], true);
        let bias = RawTensor::new(vec![0.5, 1.0], &[2], true);
        let y = x.add(&bias);

        assert_eq!(y.borrow().shape, vec![3, 2]);
        assert_eq!(y.borrow().data, vec![1.5, 3.0, 3.5, 5.0, 5.5, 7.0]);

        let loss = y.sum();
        loss.backward();

        // x gradient: all ones
        assert_eq!(x.grad(), Some(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]));
        // bias gradient: sum over batch -> [3.0, 3.0]
        assert_eq!(bias.grad(), Some(vec![3.0, 3.0]));
    }

    #[test]
    fn test_broadcast_div() {
        // (2, 3) / (1, 3) -> (2, 3)
        let x = RawTensor::new(vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0], &[2, 3], true);
        let y = RawTensor::new(vec![2.0, 2.0, 2.0], &[1, 3], true);
        let z = x.div(&y);

        assert_eq!(z.borrow().shape, vec![2, 3]);
        assert_eq!(z.borrow().data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        z.backward();

        // x gradient: 1/y broadcast
        assert_eq!(x.grad(), Some(vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5]));
        // y gradient: sum(-x/y²) over rows
        // -2/4=-0.5, -4/4=-1.0, -6/4=-1.5 (row 1)
        // -8/4=-2.0, -10/4=-2.5, -12/4=-3.0 (row 2)
        // sum: [-2.5, -3.5, -4.5]
        assert_eq!(y.grad(), Some(vec![-2.5, -3.5, -4.5]));
    }

    #[test]
    fn test_broadcast_3d() {
        // (1, 2, 3) + (2, 1) -> (1, 2, 3) but will broadcast to match
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3], true);
        let y = RawTensor::new(vec![10.0, 20.0], &[2, 1], true);
        let z = x.add(&y);

        assert_eq!(z.borrow().shape, vec![1, 2, 3]);
        // Row 0: [1,2,3] + 10 = [11,12,13]
        // Row 1: [4,5,6] + 20 = [24,25,26]
        assert_eq!(z.borrow().data, vec![11.0, 12.0, 13.0, 24.0, 25.0, 26.0]);

        z.backward();

        // x gradient: all ones
        assert_eq!(x.grad(), Some(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]));
        // y gradient: sum over last dimension -> [3.0, 3.0]
        assert_eq!(y.grad(), Some(vec![3.0, 3.0]));
    }

    #[test]
    fn test_broadcast_max() {
        // (2, 3) max (3,) -> (2, 3)
        let x = RawTensor::new(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3], true);
        let y = RawTensor::new(vec![2.0, 3.0, 4.0], &[3], true);
        let z = x.max_elem(&y);

        assert_eq!(z.borrow().shape, vec![2, 3]);
        // [max(1,2), max(5,3), max(3,4)] = [2,5,4]
        // [max(4,2), max(2,3), max(6,4)] = [4,3,6]
        assert_eq!(z.borrow().data, vec![2.0, 5.0, 4.0, 4.0, 3.0, 6.0]);

        z.backward();

        // Gradient flows to max elements
        // x: [0, 1, 0, 1, 0, 1] (x wins at indices 1, 3, 5)
        assert_eq!(x.grad(), Some(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]));
        // y: sum over rows where y wins
        // y[0] wins at (0,0): 1
        // y[1] wins at (1,1): 1
        // y[2] wins at (0,2): 1
        // Total: [1, 1, 1]
        assert_eq!(y.grad(), Some(vec![1.0, 1.0, 1.0]));
    }

    #[test]
    fn test_broadcast_bias_add() {
        // Common pattern: batch matmul + bias
        // (batch=2, in=3) @ (3, 4) + (4,) -> (2, 4)
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let w = RawTensor::new(
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
            &[3, 4],
            true,
        );
        let b = RawTensor::new(vec![0.01, 0.02, 0.03, 0.04], &[4], true);

        let y = x.matmul(&w);
        let z = y.add(&b); // Broadcasting happens here
        let loss = z.sum();

        loss.backward();

        // All should have gradients
        assert!(x.grad().is_some());
        assert!(w.grad().is_some());
        assert!(b.grad().is_some());

        // Bias gradient should be [batch_size, batch_size, ...]
        // Sum over batch dimension -> [2, 2, 2, 2]
        assert_eq!(b.grad(), Some(vec![2.0, 2.0, 2.0, 2.0]));
    }

    #[test]
    fn test_batched_matmul() {
        // (2, 2, 3) @ (2, 3, 2) -> (2, 2, 2)
        let x = RawTensor::ones(&[2, 2, 3]);
        let y = RawTensor::ones(&[2, 3, 2]);
        let z = x.matmul(&y);

        assert_eq!(z.borrow().shape, vec![2, 2, 2]);
        // dot product of two [1,1,1] vecs is 3.0
        assert_eq!(z.borrow().data[0], 3.0);
        assert_eq!(z.borrow().data[7], 3.0);
    }
    #[test]
    #[allow(clippy::identity_op)]
    //Allow identity op to make the example clearer
    fn test_batched_matmul_broadcasting() {
        // (2, 1, 2, 3) @ (1, 2, 3, 1) -> (2, 2, 2, 1)
        // Checks if batch dims [2, 1] and [1, 2] broadcast to [2, 2]

        let a_data = vec![1.0; 2 * 1 * 2 * 3]; // 12 elements, all 1s
        let b_data = vec![2.0; 1 * 2 * 3 * 1]; // 6 elements, all 2s

        let a = RawTensor::new(a_data, &[2, 1, 2, 3], true);
        let b = RawTensor::new(b_data, &[1, 2, 3, 1], true);

        let c = a.matmul(&b);

        // Output shape should be [2, 2, 2, 1]
        assert_eq!(c.borrow().shape, vec![2, 2, 2, 1]);

        // Values: Row (1,1,1) dot Col (2,2,2) = 3*2 = 6
        assert_eq!(c.borrow().data[0], 6.0);

        let loss = c.sum();
        loss.backward();

        // Gradient check
        // C sum is 8 elements * 6.0 = 48.0
        // A grad should capture broadcasted dims
        assert!(a.grad().is_some());
        assert!(b.grad().is_some());
    }

    #[test]
    fn test_matmul_matrix_vector_backward() {
        // (m,n) @ (n,) -> (m,)
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], true);
        let v = RawTensor::new(vec![0.5, -1.0], &[2], true);

        // z = X @ v
        let z = x.matmul(&v);
        // loss = sum(z) => ∂L/∂z = 1
        let loss = z.sum();
        loss.backward();

        // ∂L/∂X = outer(ones(m), v) = repeat v on each row
        assert_eq!(x.grad(), Some(vec![0.5, -1.0, 0.5, -1.0, 0.5, -1.0]));
        // ∂L/∂v = X^T @ ones(m) = column sums of X
        // sums: col0 = 1+3+5 = 9, col1 = 2+4+6 = 12
        assert_eq!(v.grad(), Some(vec![9.0, 12.0]));
    }

    #[test]
    fn test_dot_backward() {
        // (n,) @ (n,) -> scalar
        let a = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], true);
        let b = RawTensor::new(vec![4.0, 5.0, 6.0], &[3], true);

        // loss = a · b = 1*4 + 2*5 + 3*6 = 32
        let loss = a.matmul(&b);
        loss.backward();

        // ∂L/∂a = b
        assert_eq!(a.grad(), Some(vec![4.0, 5.0, 6.0]));
        // ∂L/∂b = a
        assert_eq!(b.grad(), Some(vec![1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_gradcheck_matrix_vector_matmul() {
        // Check gradients numerically for X in (m,n) @ (n,)
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        let v = RawTensor::new(vec![0.3, -0.7], &[2], false);
        let passed = RawTensor::check_gradients_simple(&x, |t| t.matmul(&v).sum());
        assert!(passed, "Matrix-vector matmul gradient check failed");
    }

    #[test]
    fn test_broadcast_sub() {
        // Test that sub also broadcasts correctly
        let x = RawTensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2], true);
        let y = RawTensor::new(vec![1.0, 2.0], &[2], true);
        let z = x.sub(&y);

        assert_eq!(z.borrow().shape, vec![2, 2]);
        assert_eq!(z.borrow().data, vec![4.0, 4.0, 6.0, 6.0]);

        z.backward();

        assert_eq!(x.grad(), Some(vec![1.0, 1.0, 1.0, 1.0]));
        // Sub gradient for y is negative and summed
        assert_eq!(y.grad(), Some(vec![-2.0, -2.0]));
    }

    // ===== NUMERICAL GRADIENT CHECKING TESTS =====

    #[test]
    fn test_gradcheck_unary_ops() {
        // Test sqrt gradient
        let x = RawTensor::new(vec![4.0, 9.0, 16.0], &[3], true);
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let y = t.sqrt();
            y.sum()
        });
        assert!(passed, "Sqrt gradient check failed");

        // Test sin gradient
        let x = RawTensor::new(vec![0.5, 1.0, 1.5], &[3], true);
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let y = t.sin();
            y.sum()
        });
        assert!(passed, "Sin gradient check failed");

        // Test sigmoid gradient
        let x = RawTensor::new(vec![0.0, 1.0, -1.0], &[3], true);
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let y = t.sigmoid();
            y.sum()
        });
        assert!(passed, "Sigmoid gradient check failed");
    }

    #[test]
    fn test_gradcheck_binary_ops() {
        // Test add gradient
        let x = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], true);
        let y = RawTensor::new(vec![4.0, 5.0, 6.0], &[3], false);
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let z = t.add(&y);
            z.sum()
        });
        assert!(passed, "Add gradient check failed");

        // Test mul gradient
        let x = RawTensor::new(vec![2.0, 3.0], &[2], true);
        let y = RawTensor::new(vec![4.0, 5.0], &[2], false);
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let z = t.elem_mul(&y);
            z.sum()
        });
        assert!(passed, "Mul gradient check failed");

        // Test div gradient
        let x = RawTensor::new(vec![6.0, 8.0], &[2], true);
        let y = RawTensor::new(vec![2.0, 4.0], &[2], false);
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let z = t.div(&y);
            z.sum()
        });
        assert!(passed, "Div gradient check failed");
    }

    #[test]
    fn test_gradcheck_matmul() {
        // Test matmul gradient for first operand
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let w = RawTensor::new(
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            &[3, 3],
            false,
        );
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let y = t.matmul(&w);
            y.sum()
        });
        assert!(passed, "Matmul gradient check failed");
    }

    #[test]
    fn test_gradcheck_broadcast() {
        // Test broadcasting gradient
        let x = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], true);
        let y = RawTensor::new(vec![0.5], &[1], false);
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let z = t.elem_mul(&y);
            z.sum()
        });
        assert!(passed, "Broadcast gradient check failed");

        // Test with matrix broadcast
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        let y = RawTensor::new(vec![0.5, 1.0], &[2], false);
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let z = t.add(&y);
            z.sum()
        });
        assert!(passed, "Matrix broadcast gradient check failed");
    }

    #[test]
    fn test_gradcheck_movement_ops() {
        // Test reshape gradient
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4], true);
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let y = t.reshape(&[2, 2]);
            y.sum()
        });
        assert!(passed, "Reshape gradient check failed");

        // Test permute gradient
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let y = t.permute(&[1, 0]);
            y.sum()
        });
        assert!(passed, "Permute gradient check failed");

        // Test pad gradient
        let x = RawTensor::new(vec![1.0, 2.0], &[2], true);
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let y = t.pad(&[(1, 1)]);
            y.sum()
        });
        assert!(passed, "Pad gradient check failed");

        // Test shrink gradient
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4], true);
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let y = t.shrink(&[(1, 3)]);
            y.sum()
        });
        assert!(passed, "Shrink gradient check failed");
    }

    #[test]
    fn test_gradcheck_reduce_ops() {
        // Test mean gradient
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4], true);
        let passed = RawTensor::check_gradients_simple(&x, |t| t.mean());
        assert!(passed, "Mean gradient check failed");

        // Test max gradient (more challenging due to discontinuity)
        let x = RawTensor::new(vec![1.0, 5.0, 3.0, 2.0], &[4], true);
        let passed = RawTensor::check_gradients_simple(&x, |t| t.max_reduce());
        assert!(passed, "Max gradient check failed");
    }

    #[test]
    fn test_gradcheck_ternary_ops() {
        // Test mulacc gradient
        let x = RawTensor::new(vec![1.0, 2.0], &[2], true);
        let y = RawTensor::new(vec![3.0, 4.0], &[2], false);
        let z = RawTensor::new(vec![0.5, 1.0], &[2], false);
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let out = t.mulacc(&y, &z);
            out.sum()
        });
        assert!(passed, "MulAcc gradient check failed");
    }

    #[test]
    fn test_gradcheck_complex_chain() {
        // Test complex computation graph
        let x = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], true);
        let w = RawTensor::new(vec![0.5, 1.0, 1.5], &[3], false);

        let passed = RawTensor::check_gradients_simple(&x, |t| {
            // y = sigmoid(x * w)
            let prod = t.elem_mul(&w);
            let y = prod.sigmoid();
            y.sum()
        });
        assert!(passed, "Complex chain gradient check failed");
    }

    #[test]
    fn test_gradcheck_neural_network_layer() {
        // Test full linear layer: y = xW + b
        let x = RawTensor::new(vec![1.0, 2.0, 3.0], &[1, 3], true);
        let w = RawTensor::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[3, 2], false);
        let b = RawTensor::new(vec![0.1, 0.2], &[2], false);

        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let y = t.matmul(&w);
            let z = y.add(&b);
            z.sum()
        });
        assert!(passed, "Neural network layer gradient check failed");
    }

    #[test]
    fn test_gradcheck_with_tolerance() {
        // Test with custom epsilon and tolerance
        let x = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], true);

        let (max_err, mean_err, passed) = RawTensor::check_gradients(
            &x,
            |t| {
                let y = t.relu();
                y.sum()
            },
            1e-5, // smaller epsilon
            1e-2, // larger tolerance (ReLU has discontinuity at 0)
        );

        assert!(passed, "Custom tolerance gradient check failed");
        println!(
            "ReLU gradcheck: max_err={:.6e}, mean_err={:.6e}",
            max_err, mean_err
        );
    }

    #[test]
    fn test_gradcheck_multidim() {
        // Test with 2D tensors
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);

        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let y = t.sqrt();
            let z = y.elem_mul(t);
            z.sum()
        });
        assert!(passed, "Multidim gradient check failed");
    }

    #[test]
    fn test_gradcheck_expand() {
        let x = RawTensor::new(vec![1.0, 2.0], &[2, 1], true);
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let y = t.expand(&[2, 3]);
            y.sum()
        });
        assert!(passed, "Expand gradient check failed");
    }

    #[test]
    fn test_gradcheck_transpose() {
        // Test standalone transpose gradient
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let y = t.transpose();
            // Multiply by some weights to make gradient non-uniform
            let w = RawTensor::new(vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0], &[3, 2], false);
            y.elem_mul(&w).sum()
        });
        assert!(passed, "Transpose gradient check failed");
    }

    #[test]
    fn test_gradcheck_pad() {
        let x = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], true);
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let y = t.pad(&[(1, 1)]);
            y.sum()
        });
        assert!(passed, "Pad gradient check failed");
    }

    #[test]
    fn test_gradcheck_shrink() {
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5], true);
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let y = t.shrink(&[(1, 4)]);
            y.sum()
        });
        assert!(passed, "Shrink gradient check failed");
    }

    #[test]
    fn test_gradcheck_stride() {
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], true);
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let y = t.stride_op(&[2]);
            y.sum()
        });
        assert!(passed, "Stride gradient check failed");
    }

    #[test]
    fn test_gradcheck_matmul_vec() {
        // vec-mat: (n,) @ (n,p) -> (p,)
        let x = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], true);
        let w = RawTensor::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[3, 2], false);
        let passed = RawTensor::check_gradients_simple(&x, |t| {
            let y = t.matmul(&w);
            y.sum()
        });
        assert!(passed, "Vec-mat matmul gradient check failed");
    }
    #[test]
    fn test_broadcast_3d_fix() {
        // (2,1) broadcasted with (1,2,3) -> (1,2,3)
        let x = RawTensor::new(vec![10.0, 20.0], &[2, 1], true);
        let y = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3], true);
        let z = x.add(&y);

        assert_eq!(z.borrow().shape, vec![1, 2, 3]);
        // Row 0: [1,2,3] + 10 = [11,12,13]
        // Row 1: [4,5,6] + 20 = [24,25,26]
        assert_eq!(z.borrow().data, vec![11.0, 12.0, 13.0, 24.0, 25.0, 26.0]);

        z.backward();
        assert_eq!(x.grad(), Some(vec![3.0, 3.0])); // sum over last dim
        assert_eq!(y.grad(), Some(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]));
    }

    #[test]
    fn test_broadcast_batch_channels() {
        // Typical conv bias: (B,C,H,W) + (C,1,1) -> (B,C,H,W)
        let x = RawTensor::new((0..16).map(|i| i as f32).collect(), &[2, 2, 2, 2], true);
        let bias = RawTensor::new(vec![0.1, 0.2], &[2, 1, 1], true);
        let z = x.add(&bias);

        assert_eq!(z.borrow().shape, vec![2, 2, 2, 2]);
        let loss = z.sum();
        loss.backward();

        // Bias grad should sum over B,H,W -> [8.0, 8.0]
        assert_eq!(bias.grad(), Some(vec![8.0, 8.0]));
    }

    #[test]
    fn test_gradcheck_broadcast_3d() {
        let x = RawTensor::new(vec![1.0, 2.0], &[2, 1], true);
        let y = RawTensor::new(vec![0.5; 6], &[1, 2, 3], false);
        let passed = RawTensor::check_gradients_simple(&x, |t| t.add(&y).sum());
        assert!(passed, "3D broadcast gradcheck failed");
    }
    #[test]
    fn test_sequential_forward() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(3, 4, true)),
            Box::new(ReLU),
            Box::new(Linear::new(4, 2, true)),
        ]);

        let x = RawTensor::new(vec![1.0, 2.0, 3.0], &[1, 3], true);
        let y = model.forward(&x);

        assert_eq!(y.borrow().shape, vec![1, 2]);

        let loss = y.sum();
        loss.backward();

        // All layer params should have gradients
        for param in model.parameters() {
            assert!(param.grad().is_some(), "Missing gradient");
        }
    }

    #[test]
    fn test_sequential_zero_grad() {
        let mut model = Sequential::new(vec![Box::new(Linear::new(2, 3, true))]);

        let x = RawTensor::new(vec![1.0, 2.0], &[1, 2], true);
        model.forward(&x).sum().backward();

        // Params have grads
        assert!(model.parameters()[0].grad().is_some());

        model.zero_grad();

        // Grads cleared
        for p in model.parameters() {
            assert!(p.grad().is_none());
        }
    }
    #[test]
    fn test_adam_converges_faster() {
        // Robust test: Learn identity y=x with badly scaled gradients
        // Problem: y = 2*x.
        // SGD struggles with scaling differences if not tuned perfectly.
        // Adam adapts per-parameter learning rates.

        let x_data: Vec<f32> = (0..10).map(|i| i as f32 * 0.1).collect(); // 10 samples
        let y_data: Vec<f32> = x_data.iter().map(|v| v * 2.0).collect();

        let x = RawTensor::new(x_data.clone(), &[10, 1], false);
        let y = RawTensor::new(y_data.clone(), &[10, 1], false);

        // Simple Linear model 1->1
        // Initialize deliberately far from solution (w=0)
        let layer = Linear::new(1, 1, false);
        // Force weight to 0.0
        layer.weight.borrow_mut().data[0] = 0.0;

        let model = Sequential::new(vec![Box::new(layer)]);

        let params = model.parameters();
        // more aggressive learning rate
        let mut opt = Adam::new(params, 0.5, (0.9, 0.999), 1e-8);

        let mut losses = vec![];
        for _ in 0..50 {
            opt.zero_grad();

            let pred = model.forward(&x);
            let loss = RawTensor::mse_loss(&pred, &y);
            loss.backward();
            opt.step();

            losses.push(loss.borrow().data[0]);
        }

        let final_loss = *losses.last().unwrap();
        assert!(
            final_loss < 0.01,
            "Adam failed simple regression convergence: {:.6}",
            final_loss
        );
    }
    #[test]
    fn test_adam_vs_sgd() {
        // Same setup, train two models
        fn train_model(use_adam: bool) -> f32 {
            let x_data = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
            let x = RawTensor::new(x_data, &[4, 2], false);
            let y_data = vec![0.0, 1.0, 1.0, 0.0];
            let y = RawTensor::new(y_data, &[4], false);

            let model = Sequential::new(vec![
                Box::new(Linear::new(2, 8, true)),
                Box::new(ReLU),
                Box::new(Linear::new(8, 1, true)),
            ]);

            let params = model.parameters();

            if use_adam {
                let mut opt = Adam::new(params, 0.05, (0.9, 0.999), 1e-8);
                for _ in 0..50 {
                    opt.zero_grad();
                    let pred = model.forward(&x).reshape(&[4]);
                    let loss = RawTensor::mse_loss(&pred, &y);
                    loss.backward();
                    opt.step();
                }
            } else {
                let mut opt = SGD::new(params, 0.01, 0.0);
                for _ in 0..50 {
                    opt.zero_grad();
                    let pred = model.forward(&x).reshape(&[4]);
                    let loss = RawTensor::mse_loss(&pred, &y);
                    loss.backward();
                    opt.step();
                }
            }

            // Return final loss
            let pred = model.forward(&x).reshape(&[4]);
            RawTensor::mse_loss(&pred, &y).borrow().data[0]
        }

        let adam_loss = train_model(true);
        let sgd_loss = train_model(false);

        println!(
            "Adam final loss: {:.6}, SGD final loss: {:.6}",
            adam_loss, sgd_loss
        );

        // Adam should be significantly better
        assert!(adam_loss < sgd_loss * 0.75, "Adam not outperforming SGD");
    }
    #[test]
    fn test_dataloader_iteration() {
        // 8 samples, 2 features each
        let data = (0..16).map(|i| i as f32).collect();
        let targets = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];

        let mut loader = DataLoader::new(
            data,
            targets,
            &[2],  // 2 features per sample
            &[1],  // 1 target per sample
            3,     // batch_size
            false, // no shuffle for deterministic test
        );

        // First batch: samples 0,1,2
        let (x, y) = loader.next().unwrap();
        assert_eq!(x.borrow().shape, vec![3, 2]);
        assert_eq!(x.borrow().data, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(y.borrow().shape, vec![3, 1]);

        // Second batch: samples 3,4,5
        let (x, _y) = loader.next().unwrap();
        assert_eq!(x.borrow().shape, vec![3, 2]);

        // Third batch: samples 6,7 (partial)
        let (x, _y) = loader.next().unwrap();
        assert_eq!(x.borrow().shape, vec![2, 2]);

        // Done
        assert!(loader.next().is_none());

        // Reset
        loader.reset();
        let (x, _) = loader.next().unwrap();
        assert_eq!(x.borrow().shape, vec![3, 2]);
    }

    #[test]
    fn test_dataloader_in_training_loop() {
        let data = vec![0.0; 40]; // 10 samples, 4 features
        let targets = vec![1.0; 10];

        let model = Sequential::new(vec![Box::new(Linear::new(4, 2, true))]);

        let mut opt = SGD::new(model.parameters(), 0.1, 0.0);

        for epoch in 0..2 {
            let loader = DataLoader::new(data.clone(), targets.clone(), &[4], &[1], 3, false);

            for (batch_x, _batch_y) in loader {
                opt.zero_grad();
                let pred = model.forward(&batch_x);
                // Dummy loss
                let loss = pred.sum();
                loss.backward();
                opt.step();
            }

            println!("Epoch {} complete", epoch);
        }
    }
    #[test]
    fn bench_matmul_speedup() {
        use std::time::Instant;

        let a = vec![1.0; 256 * 256];
        let b = vec![1.0; 256 * 256];

        let start = Instant::now();
        let _ = RawTensor::matmul_raw(&a, &b, 256, 256, 256);
        let duration = start.elapsed();

        println!("256x256 matmul: {:?}", duration);

        #[cfg(all(feature = "accelerate", target_os = "macos"))]
        assert!(
            duration.as_millis() < 10,
            "BLAS should be <10ms, got {:?}",
            duration
        );

        #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
        assert!(
            duration.as_millis() < 55,
            "Optimized fallback should be <55ms, got {:?}",
            duration
        );
    }
    #[test]
    fn test_batchnorm_working() {
        let mut bn = BatchNorm2d::new(3);
        let x = RawTensor::randn(&[2, 3, 4, 4]);

        // Training mode
        bn.train(true);
        let y = bn.forward(&x);
        assert_eq!(y.borrow().shape, vec![2, 3, 4, 4]);

        // Eval mode
        bn.eval();
        let y2 = bn.forward(&x);
        assert_eq!(y2.borrow().shape, vec![2, 3, 4, 4]);
    }
}

#[cfg(test)]
mod axis_reduce_tests {
    use super::*;

    #[test]
    fn test_sum_dim_basic() {
        // [2,3] sum along dim=1 -> [2]
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let y = RawTensor::sum_dim(&x, 1, false);

        assert_eq!(y.borrow().shape, vec![2]);
        assert_eq!(y.borrow().data, vec![6.0, 15.0]); // [1+2+3, 4+5+6]
    }

    #[test]
    fn test_sum_dim_keepdim() {
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false);
        let y = RawTensor::sum_dim(&x, 0, true);

        assert_eq!(y.borrow().shape, vec![1, 2]);
        assert_eq!(y.borrow().data, vec![4.0, 6.0]); // [1+3, 2+4]
    }

    #[test]
    fn test_sum_dim_backward() {
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let y = RawTensor::sum_dim(&x, 1, false); // [6, 15]
        y.backward();

        // Gradient broadcasts back: each element contributed once
        assert_eq!(x.grad(), Some(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]));
    }

    #[test]
    fn test_max_dim_basic() {
        let x = RawTensor::new(vec![1.0, 5.0, 3.0, 2.0, 8.0, 4.0], &[2, 3], false);
        let y = RawTensor::max_dim(&x, 1, false);

        assert_eq!(y.borrow().shape, vec![2]);
        assert_eq!(y.borrow().data, vec![5.0, 8.0]); // max of each row
    }

    #[test]
    fn test_max_dim_backward() {
        let x = RawTensor::new(vec![1.0, 5.0, 3.0, 2.0, 8.0, 4.0], &[2, 3], true);
        let y = RawTensor::max_dim(&x, 1, false);
        y.backward();

        // Only max elements get gradient
        assert_eq!(x.grad(), Some(vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0]));
    }

    #[test]
    fn test_gradcheck_sum_dim() {
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        let passed =
            RawTensor::check_gradients_simple(&x, |t| RawTensor::sum_dim(t, 0, false).sum());
        assert!(passed, "sum_dim gradient check failed");
    }

    #[test]
    fn test_gradcheck_max_dim() {
        let x = RawTensor::new(vec![1.0, 5.0, 3.0, 2.0], &[2, 2], true);
        let passed =
            RawTensor::check_gradients_simple(&x, |t| RawTensor::max_dim(t, 1, false).sum());
        assert!(passed, "max_dim gradient check failed");
    }

    #[test]
    fn test_softmax_forward() {
        // Test softmax computation
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let y = RawTensor::softmax(&x, 1);

        // Each row should sum to 1.0
        let row0_sum: f32 = y.borrow().data[0..3].iter().sum();
        let row1_sum: f32 = y.borrow().data[3..6].iter().sum();

        approx::assert_relative_eq!(row0_sum, 1.0, epsilon = 1e-6);
        approx::assert_relative_eq!(row1_sum, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_gradcheck_softmax() {
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        let passed = RawTensor::check_gradients_simple(&x, |t| RawTensor::softmax(t, 1).sum());
        assert!(passed, "Softmax gradient check failed");
    }

    #[test]
    fn test_cross_entropy_loss() {
        // Simple 2-class, 2-sample batch
        let logits = RawTensor::new(vec![2.0, 1.0, 0.5, 2.5], &[2, 2], true);
        let targets = RawTensor::new(vec![1.0, 0.0, 0.0, 1.0], &[2, 2], false);

        let loss = RawTensor::cross_entropy_loss(&logits, &targets);
        loss.backward();

        // Loss should be positive scalar
        assert_eq!(loss.borrow().shape, vec![1]);
        assert!(loss.borrow().data[0] > 0.0);

        // Gradients should exist and have correct shape
        assert_eq!(logits.grad().unwrap().len(), 4);
    }
    #[test]
    fn test_mean_dim() {
        // [2, 3]
        // [[1, 2, 3],
        //  [4, 5, 6]]
        let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);

        // mean(dim=1) -> [2, 5]
        let m = x.mean_dim(1, false);
        assert_eq!(m.borrow().shape, vec![2]);
        assert!((m.borrow().data[0] - 2.0).abs() < 1e-6);
        assert!((m.borrow().data[1] - 5.0).abs() < 1e-6);

        // Check gradient
        m.sum().backward();
        // d(mean)/dx = 1/N. Here N=3. Grad should be 1/3 for all elements.
        let grads = x.grad().unwrap();
        for g in grads {
            assert!((g - 1.0 / 3.0).abs() < 1e-6);
        }
    }
}

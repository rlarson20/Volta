//! Broadcasting, gradient checks across ops, batched matmul, and a spread of layers (`LayerNorm`, `PixelShuffle`, `Embedding`, `BatchNorm1d`, `Sequential`), plus `DataLoader` and `Adam`.

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

    loss.into_loss().expect("scalar loss").backward();

    // All should have gradients
    assert!(x.grad().is_some());
    assert!(w.grad().is_some());
    assert!(b.grad().is_some());

    // b gradient should be ones (direct path from sum)
    assert_eq!(b.grad(), Some(vec![1.0, 1.0]));
}

#[test]
fn test_tensor_zero_grad() {
    let x = RawTensor::new(vec![1.0, 2.0, 3.0], &[3], true);
    let loss = x.sum();
    loss.into_loss().expect("scalar loss").backward();

    assert!(x.grad().is_some());

    x.zero_grad();
    assert!(x.grad().is_none());

    let rewind = x.sum();
    rewind.into_loss().expect("scalar loss").backward();
    assert_eq!(x.grad(), Some(vec![1.0, 1.0, 1.0]));
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
    let _ = RawTensor::broadcast_shape(&[3, 2], &[4, 3]);
}

#[test]
fn test_broadcast_add_scalar() {
    // (2, 3) + scalar -> (2, 3)
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
    let scalar = RawTensor::new(vec![10.0], &[1], true);
    let y = x.add(&scalar);

    assert_eq!(y.borrow().shape, vec![2, 3]);
    assert_eq!(y.borrow().data, vec![11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);

    y.sum().into_loss().expect("scalar loss").backward();

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

    y.sum().into_loss().expect("scalar loss").backward();

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

    z.sum().into_loss().expect("scalar loss").backward();

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
    loss.into_loss().expect("scalar loss").backward();

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

    z.sum().into_loss().expect("scalar loss").backward();

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

    z.sum().into_loss().expect("scalar loss").backward();

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

    z.sum().into_loss().expect("scalar loss").backward();

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

    loss.into_loss().expect("scalar loss").backward();

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
    // (2, 2, 3) @ (2, 3, 2) -> (2, 2, 2). Distinct, signed values per batch, so a
    // swapped batch, a transposed operand, or a dropped batch offset changes the result
    // (all-ones inputs gave 3.0 everywhere and could see none of them).
    let x = RawTensor::new((1..=12).map(|v| v as f32).collect(), &[2, 2, 3], false);
    let y = RawTensor::new(
        vec![
            1.0, -2.0, 3.0, 4.0, -5.0, 6.0, // batch 0
            2.0, 0.0, -1.0, 3.0, 4.0, -2.0, // batch 1
        ],
        &[2, 3, 2],
        false,
    );
    let z = x.matmul(&y);

    assert_eq!(z.shape(), vec![2, 2, 2]);
    // Computed independently (Python triple loop).
    assert_eq!(
        z.data(),
        vec![-8.0, 24.0, -11.0, 48.0, 42.0, 6.0, 57.0, 9.0]
    );
}
#[test]
#[expect(
    clippy::identity_op,
    reason = "spelled-out batch dims keep the example clearer"
)]
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
    assert_eq!(
        c.borrow()
            .data
            .to_host_f32()
            .first()
            .copied()
            .expect("tensor data should exist"),
        6.0
    );

    let loss = c.sum();
    loss.into_loss().expect("scalar loss").backward();

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
    loss.into_loss().expect("scalar loss").backward();

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
    loss.into_loss().expect("scalar loss").backward();

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

    z.sum().into_loss().expect("scalar loss").backward();

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

    // Test recip gradient
    let x = RawTensor::new(vec![1.0, 2.0, 4.0], &[3], true);
    let passed = RawTensor::check_gradients_simple(&x, |t| {
        let y = t.recip();
        y.sum()
    });
    assert!(passed, "Recip gradient check failed");

    // Test exp2 gradient
    let x = RawTensor::new(vec![0.0, 1.0, 2.0], &[3], true);
    let passed = RawTensor::check_gradients_simple(&x, |t| {
        let y = t.exp2();
        y.sum()
    });
    assert!(passed, "Exp2 gradient check failed");

    // Test log2 gradient
    let x = RawTensor::new(vec![1.0, 2.0, 4.0], &[3], true);
    let passed = RawTensor::check_gradients_simple(&x, |t| {
        let y = t.log2();
        y.sum()
    });
    assert!(passed, "Log2 gradient check failed");

    // Test cos gradient
    let x = RawTensor::new(vec![0.0, 0.5, 1.0], &[3], true);
    let passed = RawTensor::check_gradients_simple(&x, |t| {
        let y = t.cos();
        y.sum()
    });
    assert!(passed, "Cos gradient check failed");

    // Test tanh gradient
    let x = RawTensor::new(vec![0.0, 1.0, -1.0], &[3], true);
    let passed = RawTensor::check_gradients_simple(&x, |t| {
        let y = t.tanh();
        y.sum()
    });
    assert!(passed, "Tanh gradient check failed");
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
        let y = t.permute(&[Axis(1), Axis(0)]);
        y.sum()
    });
    assert!(passed, "Permute gradient check failed");

    // Test pad gradient
    let x = RawTensor::new(vec![1.0, 2.0], &[2], true);
    let passed = RawTensor::check_gradients_simple(&x, |t| {
        let y = t.pad_with(&[(1, 1)], PadValue::ZERO);
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
    let passed = RawTensor::check_gradients_simple(&x, Tensor::mean);
    assert!(passed, "Mean gradient check failed");

    // Test max gradient (more challenging due to discontinuity)
    let x = RawTensor::new(vec![1.0, 5.0, 3.0, 2.0], &[4], true);
    let passed = RawTensor::check_gradients_simple(&x, Tensor::max_reduce);
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
    println!("ReLU gradcheck: max_err={max_err:.6e}, mean_err={mean_err:.6e}");
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
        let y = t.pad_with(&[(1, 1)], PadValue::ZERO);
        y.sum()
    });
    assert!(passed, "Pad gradient check failed");
}

#[test]
fn test_pad_with_neg_inf_fills_border_only() {
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false);
    let y = x.pad_with(&[(1, 0), (0, 1)], PadValue::NEG_INF);
    assert_eq!(y.shape(), vec![3, 3]);
    let ninf = f32::NEG_INFINITY;
    assert_eq!(
        y.data(),
        vec![ninf, ninf, ninf, 1.0, 2.0, ninf, 3.0, 4.0, ninf]
    );
}

#[test]
fn test_pad_with_neg_inf_gradient_is_finite_and_ones() {
    // Backward shrinks the gradient back, so the fill never reaches it.
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true);
    let y = x.pad_with(&[(1, 1), (2, 0)], PadValue::NEG_INF);
    // Sum over the interior only (a -inf in the loss would poison it).
    let interior = y.shrink(&[(1, 3), (2, 4)]);
    interior.sum().into_loss().expect("scalar loss").backward();
    assert_eq!(x.grad().expect("grad"), vec![1.0; 4]);
}

#[test]
fn test_try_pad_with_reports_errors_instead_of_panicking() {
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false);
    let rank = RawTensor::try_pad_with(&x, &[(1, 1)], PadValue::ZERO);
    assert!(
        matches!(&rank, Err(VoltaError::InvalidParameter(m)) if m.contains("must match tensor rank")),
        "{rank:?}"
    );
    let huge = RawTensor::try_pad_with(&x, &[(50_000, 50_000), (50_000, 50_000)], PadValue::ZERO);
    assert!(
        matches!(&huge, Err(VoltaError::InvalidParameter(m)) if m.contains("memory issues")),
        "{huge:?}"
    );
    assert!(RawTensor::try_pad_with(&x, &[(1, 1), (0, 0)], PadValue::ZERO).is_ok());
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

    z.sum().into_loss().expect("scalar loss").backward();
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
    loss.into_loss().expect("scalar loss").backward();

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
    loss.into_loss().expect("scalar loss").backward();

    // All layer params should have gradients
    for param in model.parameters() {
        assert!(param.grad().is_some(), "Missing gradient");
    }
}

#[test]
fn test_sequential_zero_grad() {
    let mut model = Sequential::new(vec![Box::new(Linear::new(2, 3, true))]);

    let x = RawTensor::new(vec![1.0, 2.0], &[1, 2], true);
    model
        .forward(&x)
        .sum()
        .into_loss()
        .expect("scalar loss")
        .backward();

    // Params have grads
    assert!(model.parameters().first().unwrap().grad().is_some());

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

    let x = RawTensor::new(x_data, &[10, 1], false);
    let y = RawTensor::new(y_data, &[10, 1], false);

    // Simple Linear model 1->1
    // Initialize deliberately far from solution (w=0)
    let layer = Linear::new(1, 1, false);
    // Force weight to 0.0
    *layer
        .weight
        .borrow_mut()
        .data
        .as_host_f32_mut()
        .expect("test weight is CPU F32")
        .first_mut()
        .unwrap() = 0.0;

    let model = Sequential::new(vec![Box::new(layer)]);

    let params = model.parameters();
    // more aggressive learning rate, no weight decay
    let mut opt = Adam::new(params, 0.5, (0.9, 0.999), 1e-8, 0.0);

    let mut losses = vec![];
    for _ in 0..50 {
        opt.zero_grad();

        let pred = model.forward(&x);
        let loss = RawTensor::mse_loss(&pred, &y);
        loss.clone().into_loss().expect("scalar loss").backward();
        opt.step();

        losses.push(
            loss.borrow()
                .data
                .to_host_f32()
                .first()
                .copied()
                .expect("tensor data should exist"),
        );
    }

    let final_loss = *losses.last().unwrap();
    assert!(
        final_loss < 0.01,
        "Adam failed simple regression convergence: {final_loss:.6}"
    );
}
// Same setup, train two models
// Used for test below it
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
        let mut opt = Adam::new(params, 0.05, (0.9, 0.999), 1e-8, 0.0);
        for _ in 0..50 {
            opt.zero_grad();
            let pred = model.forward(&x).reshape(&[4]);
            let loss = RawTensor::mse_loss(&pred, &y);
            loss.into_loss().expect("scalar loss").backward();
            opt.step();
        }
    } else {
        let mut opt = SGD::new(params, 0.01, 0.0, 0.0);
        for _ in 0..50 {
            opt.zero_grad();
            let pred = model.forward(&x).reshape(&[4]);
            let loss = RawTensor::mse_loss(&pred, &y);
            loss.into_loss().expect("scalar loss").backward();
            opt.step();
        }
    }

    // Return final loss
    let pred = model.forward(&x).reshape(&[4]);
    RawTensor::mse_loss(&pred, &y)
        .borrow()
        .data
        .to_host_f32()
        .first()
        .copied()
        .expect("tensor data should exist")
}

#[test]
fn test_adam_vs_sgd() {
    crate::manual_seed(42); //set for repro

    let adam_loss = train_model(true);
    let sgd_loss = train_model(false);

    println!("Adam final loss: {adam_loss:.6}, SGD final loss: {sgd_loss:.6}");

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

    let mut opt = SGD::new(model.parameters(), 0.1, 0.0, 0.0);

    for epoch in 0..2 {
        let loader = DataLoader::new(data.clone(), targets.clone(), &[4], &[1], 3, false);

        for (batch_x, _batch_y) in loader {
            opt.zero_grad();
            let pred = model.forward(&batch_x);
            // Dummy loss
            let loss = pred.sum();
            loss.into_loss().expect("scalar loss").backward();
            opt.step();
        }

        println!("Epoch {epoch} complete");
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

    // `just bench-matmul` parses this line; keep the unit fixed at ms.
    println!("256x256 matmul: {:.3}ms", duration.as_secs_f64() * 1e3);
    let max_duration_ms: u128 = if cfg!(all(feature = "accelerate", target_os = "macos")) {
        if cfg!(debug_assertions) { 50 } else { 10 }
    } else if cfg!(debug_assertions) {
        250
    } else {
        100
    };

    assert!(
        duration.as_millis() < max_duration_ms,
        "Matmul took {duration:?} (> {max_duration_ms}ms threshold for this build configuration)"
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

    // Test mode
    bn.train(false);
    let y2 = bn.forward(&x);
    assert_eq!(y2.borrow().shape, vec![2, 3, 4, 4]);
}

#[test]
fn test_batchnorm1d_forward_shape() {
    let bn = BatchNorm1d::new(32);
    let x = RawTensor::randn(&[8, 32]); // batch=8, features=32
    let y = bn.forward(&x);
    assert_eq!(y.borrow().shape, vec![8, 32]);
}

#[test]
fn test_batchnorm1d_train_vs_test_mode() {
    let mut bn = BatchNorm1d::new(4);

    // Training mode - run a few batches to populate running stats
    bn.train(true);
    for _ in 0..5 {
        let x = RawTensor::randn(&[16, 4]);
        let _ = bn.forward(&x);
    }

    // Now test that test mode uses different stats
    let test_input = RawTensor::randn(&[8, 4]);
    bn.train(true);
    let y_train = bn.forward(&test_input);

    bn.train(false);
    let y_test = bn.forward(&test_input);

    // Outputs should differ because train mode uses batch stats
    // while test mode uses running stats
    let train_data = &y_train.borrow().data;
    let test_data = &y_test.borrow().data;
    let differs = train_data
        .to_host_f32()
        .iter()
        .zip(test_data.to_host_f32().iter())
        .any(|(a, b)| (a - b).abs() > 1e-5);
    assert!(differs, "Train and test outputs should differ");
}

#[test]
fn test_batchnorm1d_parameters() {
    let bn = BatchNorm1d::new(16);
    let params = bn.parameters();
    // Should have gamma and beta
    assert_eq!(params.len(), 2);
    // gamma shape [16]
    assert_eq!(params.first().unwrap().borrow().shape, vec![16]);
    // beta shape [16]
    assert_eq!(params.get(1).unwrap().borrow().shape, vec![16]);
}

// ===== LAYERNORM TESTS =====

#[test]
fn test_layernorm_1d_forward_shape() {
    // Test basic 2D input (B, C)
    let ln = LayerNorm::new(vec![32]); // normalize over last dimension (C)
    let x = RawTensor::randn(&[8, 32]); // batch=8, features=32
    let y = ln.forward(&x);
    assert_eq!(y.borrow().shape, vec![8, 32]);
}

#[test]
fn test_layernorm_2d_forward_shape() {
    // Test 3D input (B, T, C) for transformer-style inputs
    let ln = LayerNorm::new(vec![64]); // normalize over last dimension (C)
    let x = RawTensor::randn(&[4, 10, 64]); // batch=4, seq_len=10, features=64
    let y = ln.forward(&x);
    assert_eq!(y.borrow().shape, vec![4, 10, 64]);
}

#[test]
fn test_layernorm_normalization_stats() {
    // Verify that normalization produces mean≈0 and std≈1
    let ln = LayerNorm::new_with_eps(vec![16], 1e-5);

    // Create input with non-zero mean and non-unit variance
    let data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.1 + 5.0).collect();
    let x = RawTensor::new(data, &[8, 16], false);

    let y = ln.forward(&x);

    // Check each sample has mean≈0 and std≈1
    let y_data = y.data();
    for i in 0..8 {
        let start = i * 16;
        let end = start + 16;
        let sample = y_data
            .get(start..end)
            .expect("Sample slice should be valid");

        let mean: f32 = sample.iter().sum::<f32>() / 16.0;
        let variance: f32 = sample.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 16.0;
        let std = variance.sqrt();

        // Mean should be very close to 0
        assert!(mean.abs() < 1e-4, "Sample {i} mean = {mean}, expected ≈0");
        // Std should be very close to 1.0 (allow some tolerance for numerical errors)
        assert!(
            (std - 1.0).abs() < 1e-3,
            "Sample {i} std = {std}, expected ≈1"
        );
    }
}

#[test]
fn test_layernorm_parameters() {
    // Check gamma and beta shapes and requires_grad
    let ln = LayerNorm::new(vec![32]);
    let params = ln.parameters();

    // Should have gamma and beta
    assert_eq!(params.len(), 2);
    // gamma shape [32]
    assert_eq!(params.first().unwrap().borrow().shape, vec![32]);
    // beta shape [32]
    assert_eq!(params.get(1).unwrap().borrow().shape, vec![32]);

    // Both should require grad
    assert!(params.first().unwrap().borrow().requires_grad);
    assert!(params.get(1).unwrap().borrow().requires_grad);
}

#[test]
fn test_layernorm_gradient() {
    // Numerical gradient check with deterministic data
    let ln = LayerNorm::new_with_eps(vec![4], 1e-5);

    // Use deterministic data (not random) for stable gradients
    let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
    let x = RawTensor::new(data, &[8, 4], true);

    // Simple forward pass: sum of output
    let y = ln.forward(&x);
    let loss = y.sum();
    loss.into_loss().expect("scalar loss").backward();

    // Check that gradients exist and are finite
    let x_grad = x.grad();
    assert!(x_grad.is_some(), "Input should have gradient");

    let grad_data = x_grad.unwrap();
    assert_eq!(
        grad_data.len(),
        32,
        "Gradient should have same size as input"
    );

    // Check gradients are finite (not NaN or Inf)
    for &g in &grad_data {
        assert!(g.is_finite(), "Gradient should be finite, found {g}");
    }

    // Check gamma and beta have gradients
    let params = ln.parameters();
    let gamma_grad = params.first().unwrap().grad();
    let beta_grad = params.get(1).unwrap().grad();

    assert!(gamma_grad.is_some(), "Gamma should have gradient");
    assert!(beta_grad.is_some(), "Beta should have gradient");

    // Gradients should be finite
    for g in &gamma_grad.unwrap() {
        assert!(g.is_finite(), "Gamma gradient should be finite");
    }
    for g in &beta_grad.unwrap() {
        assert!(g.is_finite(), "Beta gradient should be finite");
    }
}

#[test]
fn test_layernorm_state_dict() {
    // Test state dict save/load roundtrip
    let ln1 = LayerNorm::new_with_eps(vec![8], 1e-5);

    // Forward pass to ensure parameters are initialized
    let x = RawTensor::randn(&[4, 8]);
    let _ = ln1.forward(&x);

    // Save state dict
    let state = ln1.state_dict();

    // Create new layer and load state
    let mut ln2 = LayerNorm::new(vec![8]);
    ln2.load_state_dict(&state).expect("state loads");

    // Check that parameters match
    let params1 = ln1.parameters();
    let params2 = ln2.parameters();

    // Gamma should match
    let gamma1_data = &params1.first().unwrap().borrow().data;
    let gamma2_data = &params2.first().unwrap().borrow().data;
    assert_eq!(
        gamma1_data, gamma2_data,
        "Gamma should match after state dict roundtrip"
    );

    // Beta should match
    let beta1_data = &params1.get(1).unwrap().borrow().data;
    let beta2_data = &params2.get(1).unwrap().borrow().data;
    assert_eq!(
        beta1_data, beta2_data,
        "Beta should match after state dict roundtrip"
    );
}

#[test]
fn test_pixelshuffle_forward_shape() {
    // Test with upscale_factor=3: [2, 36, 4, 4] -> [2, 4, 12, 12]
    let layer = PixelShuffle::new(3);
    let x = RawTensor::randn(&[2, 36, 4, 4]); // 4 channels * 9
    let y = layer.forward(&x);
    assert_eq!(y.borrow().shape, vec![2, 4, 12, 12]);

    // Test with upscale_factor=2: [1, 12, 8, 8] -> [1, 3, 16, 16]
    let layer2 = PixelShuffle::new(2);
    let x2 = RawTensor::randn(&[1, 12, 8, 8]); // 3 channels * 4
    let y2 = layer2.forward(&x2);
    assert_eq!(y2.borrow().shape, vec![1, 3, 16, 16]);
}

#[test]
fn test_pixelshuffle_backward_flow() {
    let layer = PixelShuffle::new(2);
    let x = RawTensor::randn(&[2, 4, 3, 3]); // 1 channel * 4
    x.borrow_mut().requires_grad = true;

    let y = layer.forward(&x);
    assert_eq!(y.borrow().shape, vec![2, 1, 6, 6]);

    let loss = y.sum();
    loss.into_loss().expect("scalar loss").backward();

    let grad = x.grad();
    assert!(
        grad.is_some(),
        "Gradient should flow back through PixelShuffle"
    );
    // Check gradient has correct number of elements: 2 * 4 * 3 * 3 = 72
    assert_eq!(grad.unwrap().len(), 72);
}

#[test]
fn test_pixelshuffle_values() {
    // Small manual test with known values
    // Input: [1, 4, 2, 2] with upscale_factor=2
    // This should rearrange 4 channels of 2x2 into 1 channel of 4x4
    let layer = PixelShuffle::new(2);
    #[rustfmt::skip]
        let data = vec![
            // Channel 0
            1.0, 2.0,
            3.0, 4.0,
            // Channel 1
            5.0, 6.0,
            7.0, 8.0,
            // Channel 2
            9.0, 10.0,
            11.0, 12.0,
            // Channel 3
            13.0, 14.0,
            15.0, 16.0,
        ];
    let x = RawTensor::new(data, &[1, 4, 2, 2], false);
    let y = layer.forward(&x);

    assert_eq!(y.borrow().shape, vec![1, 1, 4, 4]);

    // After PixelShuffle, the output should interleave values from the 4 input channels
    // The exact pattern depends on the reshape/permute order
    let out_data = &y.borrow().data;
    assert_eq!(out_data.len(), 16);

    // Verify the transformation preserved all values (just check sum as a sanity check)
    let input_sum: f32 = (1..=16).map(|x| x as f32).sum();
    let output_sum: f32 = out_data.to_host_f32().iter().sum();
    assert!((input_sum - output_sum).abs() < 1e-5);
}

#[test]
fn test_embedding_forward_shape() {
    let embedding = Embedding::new(100, 32);
    let indices = RawTensor::from_typed(vec![5i64, 12, 7, 99], &[4]);
    let output = embedding.forward(&indices);
    assert_eq!(output.borrow().shape, vec![4, 32]);
}

#[test]
fn test_embedding_backward_flow() {
    let embedding = Embedding::new(50, 16);
    let indices = RawTensor::from_typed(vec![3i64, 10, 3], &[3]); // Note: repeated index 3
    let output = embedding.forward(&indices);

    // Sum and backward
    let loss = output.sum();
    loss.into_loss().expect("scalar loss").backward();

    // Check that weight has gradients
    let grad = embedding.weight.grad();
    assert!(grad.is_some(), "Weight should have gradients");

    let grad_data = grad.unwrap();
    // Each embedding contributes 1.0 per dimension from sum
    // Index 3 appears twice, so should have accumulated grad of 2.0 per dim
    let grad_at_idx3_sum: f32 = (0..16)
        .map(|d| {
            grad_data
                .get(3 * 16 + d)
                .copied()
                .expect("tensor data should exist")
        })
        .sum();
    let expected_sum = 2.0 * 16.0; // 2 occurrences * 16 dimensions
    assert!(
        (grad_at_idx3_sum - expected_sum).abs() < 1e-4,
        "Expected accumulated grad sum {expected_sum}, got {grad_at_idx3_sum}"
    );
}

#[test]
fn test_embedding_gradient_accumulation() {
    let embedding = Embedding::new(10, 4);
    let indices = RawTensor::from_typed(vec![2i64, 5, 2, 2], &[4]); // Index 2 appears 3 times
    let output = embedding.forward(&indices);

    let loss = output.sum();
    loss.into_loss().expect("scalar loss").backward();

    let grad = embedding.weight.grad().unwrap();
    // Index 2 should have grad of 3.0 per dimension (appears 3 times)
    for d in 0..4 {
        let grad_val = grad
            .get(2 * 4 + d)
            .copied()
            .expect("tensor data should exist");
        assert!(
            (grad_val - 3.0).abs() < 1e-5,
            "Expected grad 3.0 for index 2, got {grad_val}"
        );
    }

    // Index 5 should have grad of 1.0 per dimension (appears once)
    for d in 0..4 {
        let grad_val = grad
            .get(5 * 4 + d)
            .copied()
            .expect("tensor data should exist");
        assert!(
            (grad_val - 1.0).abs() < 1e-5,
            "Expected grad 1.0 for index 5, got {grad_val}"
        );
    }
}

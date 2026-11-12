//! In-place writes after a forward pass (S9): backward refuses a graph whose saved inputs
//! were rewritten, as PyTorch's version counter does, and leaves alone a graph whose ops
//! don't read the rewritten tensor.

use volta::{Adam, DType, Linear, Module, Muon, RawTensor, SGD, Storage, Tensor};

/// Panic text shared by every stale-graph case.
const STALE: &str = "modified by an inplace operation";

/// A 2x2 leaf that requires grad, with a gradient set so an optimizer will write it.
fn param_with_grad() -> Tensor {
    let w = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true);
    w.set_grad(Storage::cpu(vec![0.5, -1.0, 0.25, 2.0]));
    w
}

/// A 2x2 input with distinct values, so a gradient through it is not degenerate.
fn input() -> Tensor {
    RawTensor::new(vec![0.5, -1.5, 2.0, 3.0], &[2, 2], true)
}

#[test]
#[should_panic(expected = "modified by an inplace operation")]
fn adam_step_invalidates_a_saved_weight() {
    let w = param_with_grad();
    let loss = w.elem_mul(&input()).sum().into_loss().expect("scalar");
    Adam::new(vec![w], 0.1, (0.9, 0.999), 1e-8, 0.0).step();
    loss.backward();
}

#[test]
#[should_panic(expected = "modified by an inplace operation")]
fn muon_step_invalidates_a_saved_weight() {
    let w = param_with_grad();
    let loss = w.elem_mul(&input()).sum().into_loss().expect("scalar");
    Muon::new(vec![w], 0.1, 0.9, false, 5).step();
    loss.backward();
}

#[test]
#[should_panic(expected = "modified by an inplace operation")]
fn sgd_step_invalidates_a_matmul_operand() {
    let w = param_with_grad();
    let loss = input().matmul(&w).sum().into_loss().expect("scalar");
    SGD::new(vec![w], 0.1, 0.0, 0.0).step();
    loss.backward();
}

#[test]
#[should_panic(expected = "modified by an inplace operation")]
fn sgd_step_invalidates_an_exp_input() {
    let w = param_with_grad();
    let loss = w.exp().sum().into_loss().expect("scalar");
    SGD::new(vec![w], 0.1, 0.0, 0.0).step();
    loss.backward();
}

#[test]
fn the_message_names_the_op_and_the_input() {
    let w = param_with_grad();
    let loss = input().elem_mul(&w).sum().into_loss().expect("scalar");
    SGD::new(vec![w], 0.1, 0.0, 0.0).step();
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| loss.backward()))
        .expect_err("backward must refuse the stale graph");
    let text = caught
        .downcast_ref::<String>()
        .cloned()
        .expect("a formatted panic message");
    assert!(
        text.contains("Binary(Mul): input 1") && text.contains(STALE),
        "{text}"
    );
}

#[test]
fn an_add_does_not_read_its_inputs_so_a_step_is_allowed() {
    // d(sum(w + x))/dx = 1 whatever `w` is, so torch doesn't complain either.
    let w = param_with_grad();
    let x = input();
    let loss = w.add(&x).sum().into_loss().expect("scalar");
    SGD::new(vec![w], 0.1, 0.0, 0.0).step();
    loss.backward();
    assert_eq!(x.grad().expect("grad"), vec![1.0; 4]);
}

#[test]
fn a_reshape_and_a_sum_do_not_read_their_inputs() {
    let w = param_with_grad();
    let loss = w.reshape(&[4]).sum().into_loss().expect("scalar");
    SGD::new(vec![w.clone()], 0.1, 0.0, 0.0).step();
    loss.backward();
    // The gradient set before the step, plus 1 per element from this backward.
    assert_eq!(w.grad().expect("grad"), vec![1.5, 0.0, 1.25, 3.0]);
}

#[test]
#[should_panic(expected = "modified by an inplace operation")]
fn load_state_dict_invalidates_a_saved_weight() {
    let mut layer = Linear::new(2, 2, false);
    let x = input();
    let loss = layer.forward(&x).sum().into_loss().expect("scalar");
    let state = layer.state_dict();
    layer.load_state_dict(&state).expect("reload");
    loss.backward();
}

#[test]
#[should_panic(expected = "modified by an inplace operation")]
fn a_dtype_move_invalidates_a_saved_weight() {
    let mut layer = Linear::new(2, 2, false);
    let loss = layer.forward(&input()).sum().into_loss().expect("scalar");
    layer.to_dtype(DType::F64);
    loss.backward();
}

#[test]
fn the_training_loop_order_is_unaffected() {
    // forward, backward, step, zero_grad, then a fresh forward: nothing is stale.
    let w = RawTensor::new(vec![2.0], &[1], true);
    let x = RawTensor::new(vec![3.0], &[1], true);
    let mut sgd = SGD::new(vec![w.clone()], 0.5, 0.0, 0.0);
    for _ in 0..3 {
        sgd.zero_grad();
        w.elem_mul(&x).sum().into_loss().expect("scalar").backward();
        sgd.step();
    }
    // Each step moves w by -lr * x = -1.5.
    assert_eq!(w.data(), vec![-2.5]);
}

#[test]
fn a_second_backward_without_a_write_still_works() {
    let w = param_with_grad();
    let x = input();
    let loss = w.elem_mul(&x).sum().into_loss().expect("scalar");
    loss.backward();
    loss.backward();
    assert_eq!(x.grad().expect("grad"), vec![2.0, 4.0, 6.0, 8.0]);
}

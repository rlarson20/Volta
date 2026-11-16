//! Finite-difference gradient checks for the transformer layers that had only an
//! `assert_trainable!` test (#123): `FeedForward`, `TransformerBlock` and the learned
//! `PositionalEncoding`. `MultiHeadAttention` and `ScaledDotProductAttention` have theirs in
//! their own modules. Everything runs in f64 with a random weighted loss, so a gradient
//! that is wrong, zero, or permuted fails.

use volta::dtype::DType;
use volta::io::TensorData;
use volta::nn::{
    FeedForward, Layer, Module, PositionalEncoding, PositionalEncodingType, TransformerBlock,
};
use volta::{RawTensor, Tensor, manual_seed, randn};

/// A random f64 tensor; a leaf that requires grad if `grad`.
fn operand(dims: &[usize], grad: bool) -> Tensor {
    let vals: Vec<f64> = randn(dims).data().iter().map(|&v| f64::from(v)).collect();
    let t = RawTensor::from_typed(vals, dims);
    t.set_requires_grad(grad);
    t
}

/// `check_gradients` of `loss` with respect to `x`, at f64 precision.
fn assert_input_grad(name: &str, x: &Tensor, loss: impl Fn(&Tensor) -> Tensor) {
    let (max, mean, ok) = RawTensor::check_gradients(x, loss, 1e-6, 1e-5);
    assert!(
        ok,
        "{name}: grad wrt input failed: max={max:.3e} mean={mean:.3e}"
    );
}

#[test]
fn feed_forward_input_gradient_matches_finite_differences() {
    manual_seed(0x123F);
    let mut ffn = FeedForward::new(4, 2, 0.0);
    ffn.to_dtype(DType::F64);
    let x = operand(&[2, 3, 4], true);
    let w = operand(&[2, 3, 4], false);
    assert_input_grad("FeedForward", &x, |t| ffn.forward(t).elem_mul(&w).sum());
}

#[test]
fn transformer_block_input_gradient_matches_finite_differences() {
    for causal in [false, true] {
        manual_seed(0x123B);
        let mut block = TransformerBlock::new(4, 2, 2, 0.0);
        block.to_dtype(DType::F64);
        let x = operand(&[2, 3, 4], true);
        let w = operand(&[2, 3, 4], false);
        assert_input_grad(&format!("TransformerBlock causal={causal}"), &x, |t| {
            block.forward(t, causal).elem_mul(&w).sum()
        });
    }
}

/// The learned table's gradient, by central differences through `load_state_dict`
/// (`check_gradients` perturbs only its input). The layer is linear in the table, so the
/// difference is exact up to rounding.
#[test]
fn learned_positional_encoding_gradients_match_finite_differences() {
    manual_seed(0x123E);
    let (seq, dim, max_len) = (3, 4, 5);
    let mut pe = PositionalEncoding::new(dim, max_len, PositionalEncodingType::Learned);
    pe.to_dtype(DType::F64);
    let x = operand(&[2, seq, dim], true);
    let w = operand(&[2, seq, dim], false);
    let loss = |pe: &PositionalEncoding, x: &Tensor| pe.forward(x).elem_mul(&w).sum();

    assert_input_grad("PositionalEncoding", &x, |t| loss(&pe, t));

    let key = "learned_embedding.weight";
    let table = pe.state_dict()[key].to_vec_as::<f64>().expect("f64 table");
    let params = pe.parameters();
    let [weight] = params.as_slice() else {
        panic!("a learned encoding has one parameter");
    };
    loss(&pe, &x.detach())
        .into_loss()
        .expect("scalar")
        .backward();
    let grad = weight.grad_storage().expect("table grad");
    assert_eq!(grad.dtype(), DType::F64);
    let analytic: Vec<f64> = grad.as_host::<f64>().expect("a host f64 gradient").to_vec();
    assert_eq!(analytic.len(), table.len());

    let eps = 1e-6;
    let shape = vec![max_len, dim];
    for i in 0..table.len() {
        let at = |delta: f64, pe: &mut PositionalEncoding| {
            let mut t = table.clone();
            t[i] += delta;
            let mut state = pe.state_dict();
            state.insert(key.to_string(), TensorData::from_typed(t, shape.clone()));
            pe.load_state_dict(&state).expect("same shape");
            loss(pe, &x.detach()).to_vec_as::<f64>().expect("f64 loss")[0]
        };
        let numeric = (at(eps, &mut pe) - at(-eps, &mut pe)) / (2.0 * eps);
        assert!(
            (numeric - analytic[i]).abs() < 1e-6,
            "table element {i}: numeric {numeric} vs analytic {}",
            analytic[i]
        );
        // Rows past the sequence length get no gradient; check they really are unused.
        if i >= seq * dim {
            assert_eq!(analytic[i], 0.0, "row {} is past the sequence", i / dim);
        }
    }
}

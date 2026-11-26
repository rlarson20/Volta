//! `Linear`: shapes, bias, forward value, input gradient, and trainability.

use super::*;
use crate::nn::Linear;
use crate::nn::Module;
use crate::storage::Storage;

#[test]
fn weight_shape_and_bias_present() {
    let layer = Linear::new(4, 3, true);
    assert_eq!(layer.weight.borrow().shape, vec![4, 3]);
    assert!(layer.bias.is_some());
    let b = layer.bias.as_ref().unwrap();
    assert_eq!(b.borrow().shape, vec![3]);
}

#[test]
fn no_bias_when_use_bias_false() {
    let layer = Linear::new(4, 3, false);
    assert!(layer.bias.is_none());
    assert_eq!(layer.parameters().len(), 1);
}

#[test]
fn forward_shape_with_bias() {
    let layer = Linear::new(5, 2, true);
    let x = RawTensor::new(vec![1.0; 3 * 5], &[3, 5], true);
    let y = layer.forward(&x);
    assert_eq!(y.borrow().shape, vec![3, 2]);
}

#[test]
fn forward_value_with_known_weights() {
    // Build a deterministic Linear: y = xW + b with shape (1,2) input,
    // (2,3) weight, (3,) bias.
    let layer = Linear::new(2, 3, true);
    // Override randomly initialized weights with known values.
    layer
        .weight
        .replace_data(Storage::cpu(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    layer
        .bias
        .as_ref()
        .unwrap()
        .replace_data(Storage::cpu(vec![10.0, 20.0, 30.0]));
    let x = RawTensor::new(vec![1.0, 1.0], &[1, 2], false);
    let y = layer.forward(&x);
    // x @ W = [1*1+1*4, 1*2+1*5, 1*3+1*6] = [5, 7, 9]; + bias = [15, 27, 39]
    assert_eq!(y.borrow().data.to_vec(), vec![15.0, 27.0, 39.0]);
}

#[test]
fn backward_gradcheck_input() {
    // Seeded, with `eps = 0.5`: unseeded, about 1 in 130 weight draws failed at
    // `check_gradients_simple`'s `eps = 1e-2`, and none of the same 400 seeds fail at
    // 0.5. `x @ w + b` is linear in `x`, so the step only affects f32 roundoff.
    crate::tensor::manual_seed(0x11E);
    let layer = Linear::new(3, 2, true);
    // Snapshot the (random) weight so the closure captures stable values.
    let w_snapshot = layer.weight.borrow().data.to_vec();
    let b_snapshot = layer
        .bias
        .as_ref()
        .map(|b| b.borrow().data.to_vec())
        .unwrap_or_default();
    let w_shape = layer.weight.shape();
    let b_shape = layer
        .bias
        .as_ref()
        .expect("Linear::new(.., true) has a bias")
        .shape();
    let x = RawTensor::new(vec![0.5, -1.0, 1.5, 0.25, 0.0, 2.0], &[2, 3], true);

    let (max_err, mean_err, passed) = RawTensor::check_gradients(
        &x,
        |t| {
            let w = RawTensor::from_shape(w_snapshot.clone(), w_shape.clone(), false);
            let b = RawTensor::from_shape(b_snapshot.clone(), b_shape.clone(), false);
            t.matmul(&w).add(&b).sum()
        },
        0.5,
        1e-3,
    );
    assert!(
        passed,
        "Linear input gradient check failed: max_error={max_err:.6e}, mean_error={mean_err:.6e}"
    );
}

#[test]
fn assert_trainable_all_params_get_gradient() {
    let layer = Linear::new(4, 3, true);
    let x = crate::macros::trainable_input(&[2, 4]);
    crate::macros::assert_trainable!(layer, x);
}

/// f64 leaf of `dims` with seeded random values.
fn f64_leaf(dims: &[usize]) -> Tensor {
    let x = RawTensor::randn(dims).to_dtype(crate::DType::F64);
    x.set_requires_grad(true);
    x
}

/// `y[r, j] = b[j] + sum_k x[r, k] * w[k, j]` over the flattened leading dims `r`, in f64.
/// This is the definition of PyTorch's `nn.Linear` on `[.., in]` (with Volta's `[in, out]`
/// weight layout), written out without `matmul` or `reshape`.
fn reference_linear(x: &[f64], w: &[f64], b: &[f64], in_f: usize, out_f: usize) -> Vec<f64> {
    assert_eq!(x.len() % in_f, 0);
    assert_eq!(w.len(), in_f * out_f);
    assert_eq!(b.len(), out_f);
    let rows = x.len() / in_f;
    let mut y = vec![0.0; rows * out_f];
    for r in 0..rows {
        for j in 0..out_f {
            let mut acc = b[j];
            for k in 0..in_f {
                acc += x[r * in_f + k] * w[k * out_f + j];
            }
            y[r * out_f + j] = acc;
        }
    }
    y
}

/// #486: a `[.., in]` input of rank 3 or 4 maps the last dim and keeps the leading ones,
/// with the values of the per-row definition.
#[test]
fn forward_rank3_and_rank4_match_the_per_row_definition() {
    crate::tensor::manual_seed(0x486);
    let (in_f, out_f) = (4, 3);
    let mut layer = Linear::new(in_f, out_f, true);
    layer.to_dtype(crate::DType::F64);
    let w = layer.weight.borrow().data.host::<f64>().to_vec();
    let b = layer
        .bias
        .as_ref()
        .expect("Linear::new(.., true) has a bias")
        .borrow()
        .data
        .host::<f64>()
        .to_vec();
    // Make the bias non-zero, so a dropped bias add would show.
    let b: Vec<f64> = b
        .iter()
        .enumerate()
        .map(|(j, v)| v + 0.25 * (j as f64 + 1.0))
        .collect();
    layer
        .bias
        .as_ref()
        .expect("Linear::new(.., true) has a bias")
        .replace_data(Storage::from_typed(b.clone()));

    for dims in [vec![2, 5, in_f], vec![2, 3, 2, in_f]] {
        let x = f64_leaf(&dims);
        let y = layer.forward(&x);
        let mut want_dims = dims.clone();
        *want_dims.last_mut().expect("rank >= 3") = out_f;
        assert_eq!(y.shape(), want_dims, "input {dims:?}");
        let want = reference_linear(&x.borrow().data.host::<f64>(), &w, &b, in_f, out_f);
        let got = y.borrow().data.host::<f64>().to_vec();
        assert_eq!(got.len(), want.len());
        for (i, (g, e)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - e).abs() < 1e-12,
                "input {dims:?}, element {i}: {g} vs {e}"
            );
        }
    }
}

/// #486: gradients through a rank-3 input reach the input, the weight and the bias.
#[test]
fn backward_gradcheck_rank3_input_weight_and_bias() {
    crate::tensor::manual_seed(0x4861);
    let mut layer = Linear::new(4, 3, true);
    layer.to_dtype(crate::DType::F64);
    let x = f64_leaf(&[2, 5, 4]);
    // A random weighted sum, so no gradient is degenerate.
    let r = RawTensor::randn(&[2, 5, 3]).to_dtype(crate::DType::F64);
    let loss = |y: Tensor| y.elem_mul(&r).sum();

    let (max_e, _, passed) = RawTensor::check_gradients(&x, |t| loss(layer.forward(t)), 1e-6, 1e-6);
    assert!(passed, "input gradient: max_error={max_e:.3e}");

    // `check_gradients` hands the closure a perturbed copy, so build the layer from it.
    let bias = layer.bias.clone();
    let with_weight = |w: &Tensor| Linear {
        weight: w.clone(),
        bias: bias.clone(),
    };
    let (max_e, _, passed) = RawTensor::check_gradients(
        &layer.weight,
        |w| loss(with_weight(w).forward(&x)),
        1e-6,
        1e-6,
    );
    assert!(passed, "weight gradient: max_error={max_e:.3e}");

    let weight = layer.weight.clone();
    let with_bias = |b: &Tensor| Linear {
        weight: weight.clone(),
        bias: Some(b.clone()),
    };
    let (max_e, _, passed) = RawTensor::check_gradients(
        layer
            .bias
            .as_ref()
            .expect("Linear::new(.., true) has a bias"),
        |b| loss(with_bias(b).forward(&x)),
        1e-6,
        1e-6,
    );
    assert!(passed, "bias gradient: max_error={max_e:.3e}");
}

#[test]
fn assert_trainable_rank3_input() {
    let layer = Linear::new(4, 3, true);
    let x = crate::macros::trainable_input(&[2, 5, 4]);
    crate::macros::assert_trainable!(layer, x);
}

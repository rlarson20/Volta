#![allow(missing_docs)] // not public API
use volta::io::{StateDict, TensorData};
use volta::{Embedding, Linear, Module};

fn assert_same(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(x.to_bits() == y.to_bits(), "elem {i}: {x} != {y}");
    }
}

/// H7: a state dict round-trips into a fresh layer with distinct, non-degenerate values.
#[test]
fn test_linear_load_state_dict_round_trip() {
    let src = Linear::new(4, 3, true);
    let mut dst = Linear::new(4, 3, true);
    assert_ne!(src.weight.data(), dst.weight.data(), "seeds collided");

    dst.load_state_dict(&src.state_dict());

    assert_same(&dst.weight.data(), &src.weight.data());
    assert_eq!(dst.weight.shape(), vec![4, 3]);
    if let (Some(d), Some(s)) = (dst.bias.as_ref(), src.bias.as_ref()) {
        assert_same(&d.data(), &s.data());
    } else {
        panic!("both layers were built with bias");
    }
}

/// H7: loading must not change the destination's shape, even if the data has the right numel.
#[test]
#[should_panic(expected = "weight")]
fn test_linear_load_state_dict_rejects_transposed_shape() {
    let mut layer = Linear::new(4, 3, false);
    let mut state = StateDict::new();
    state.insert(
        "weight".to_string(),
        TensorData {
            data: vec![0.5; 12],
            shape: vec![3, 4], // same numel, wrong layout
        },
    );
    layer.load_state_dict(&state);
}

/// H7: data shorter than the declared shape must not be accepted.
#[test]
#[should_panic(expected = "weight")]
fn test_linear_load_state_dict_rejects_short_data() {
    let mut layer = Linear::new(4, 3, false);
    let mut state = StateDict::new();
    state.insert(
        "weight".to_string(),
        TensorData {
            data: vec![0.5; 6],
            shape: vec![4, 3],
        },
    );
    layer.load_state_dict(&state);
}

/// Embedding already validated shape before H7; keep it that way.
#[test]
#[should_panic(expected = "weight")]
fn test_embedding_load_state_dict_rejects_wrong_shape() {
    let mut layer = Embedding::new(5, 3);
    let mut state = StateDict::new();
    state.insert(
        "weight".to_string(),
        TensorData {
            data: vec![0.5; 6],
            shape: vec![2, 3],
        },
    );
    layer.load_state_dict(&state);
}

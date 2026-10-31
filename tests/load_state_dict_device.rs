//! H7: `load_state_dict` goes through `Tensor::load_weight`. A `Linear` state dict round-trips into a fresh layer with distinct, non-degenerate values and keeps its shape; a transposed shape or short data is rejected for `Linear`, and a wrong shape is rejected for `Embedding`. A rejected entry is an `Err` naming the key (a checkpoint is data from outside, not a programming error), also through a `Sequential`.
//!
//! Device preservation (a GPU destination stays on the GPU) is covered by `tensor::load_weight_tests` in `src/`.
use volta::io::{StateDict, TensorData};
use volta::{Embedding, Linear, Module, ReLU, Sequential, VoltaError};

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

    dst.load_state_dict(&src.state_dict())
        .expect("matching state dict loads");

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
fn test_linear_load_state_dict_rejects_transposed_shape() {
    let mut layer = Linear::new(4, 3, false);
    let before = layer.weight.data();
    let mut state = StateDict::new();
    state.insert(
        "weight".to_string(),
        TensorData {
            data: vec![0.5; 12],
            shape: vec![3, 4], // same numel, wrong layout
        },
    );
    match layer.load_state_dict(&state) {
        Err(VoltaError::StateShapeMismatch {
            key,
            expected,
            actual,
        }) => {
            assert_eq!(key, "weight");
            assert_eq!(expected, vec![4, 3]);
            assert_eq!(actual, vec![3, 4]);
        }
        other => panic!("expected StateShapeMismatch, got {other:?}"),
    }
    assert_same(&layer.weight.data(), &before);
    assert_eq!(layer.weight.shape(), vec![4, 3]);
}

/// H7: data shorter than the declared shape must not be accepted.
#[test]
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
    match layer.load_state_dict(&state) {
        Err(VoltaError::StateDataLength { key, expected, len }) => {
            assert_eq!(key, "weight");
            assert_eq!((expected, len), (12, 6));
        }
        other => panic!("expected StateDataLength, got {other:?}"),
    }
}

/// Embedding already validated shape before H7; keep it that way.
#[test]
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
    assert!(matches!(
        layer.load_state_dict(&state),
        Err(VoltaError::StateShapeMismatch { ref key, .. }) if key == "weight"
    ));
}

/// The error names the entry as the caller wrote it in the state dict: through a
/// `Sequential` that is the prefixed key, which is what a UI would show.
#[test]
fn test_sequential_load_state_dict_names_the_prefixed_key() {
    let mut model = Sequential::new(vec![
        Box::new(Linear::new(2, 3, true)),
        Box::new(ReLU),
        Box::new(Linear::new(3, 1, true)),
    ]);
    let mut state = model.state_dict();
    state.insert(
        "2.weight".to_string(),
        TensorData {
            data: vec![0.5; 6],
            shape: vec![1, 3], // layer 2 expects [3, 1]
        },
    );
    match model.load_state_dict(&state) {
        Err(VoltaError::StateShapeMismatch { key, .. }) => assert_eq!(key, "2.weight"),
        other => panic!("expected StateShapeMismatch, got {other:?}"),
    }
}

//! State-dict loads are atomic: an entry that does not fit anywhere in a model leaves
//! every layer unchanged, through the strict, partial and checked loads (#65).

use volta::VoltaError;
use volta::io::{StateDict, TensorData, load_state_dict_checked};
use volta::nn::{Linear, Module, ReLU, Sequential, TransformerBlock};

/// A small model whose state dict has entries in two layers.
fn model() -> Sequential {
    Sequential::new(vec![
        Box::new(Linear::new(3, 4, true)),
        Box::new(ReLU),
        Box::new(Linear::new(4, 2, true)),
    ])
}

/// `state` with every entry's values replaced by `v`, keeping shapes and dtypes.
fn filled(state: &StateDict, v: f32) -> StateDict {
    state
        .iter()
        .map(|(k, e)| {
            let n = e.shape.iter().product();
            (k.clone(), TensorData::from_f32(vec![v; n], e.shape.clone()))
        })
        .collect()
}

/// An f32 entry of `shape` holding only `bytes` bytes, too few for its shape.
fn short(shape: Vec<usize>, bytes: usize) -> TensorData {
    let mut e = TensorData::from_f32(vec![0.0; shape.iter().product()], shape);
    e.data.truncate(bytes);
    e
}

/// An entry in the last layer with too few bytes for its shape fails the strict load,
/// and the first layer, whose entries were fine, keeps its values.
#[test]
fn a_short_entry_in_the_last_layer_leaves_the_first_unchanged() {
    let mut m = model();
    let before = m.state_dict();
    let mut state = filled(&before, 7.0);
    state.insert("2.bias".to_string(), short(vec![2], 3));
    let err = m.load_state_dict(&state).unwrap_err();
    assert!(
        matches!(&err, VoltaError::StateDataLength { key, .. } if key == "2.bias"),
        "{err:?}"
    );
    assert_eq!(m.state_dict(), before);
}

/// A wrong shape deep inside a nested module names the full key and writes nothing, in
/// the partial load too.
#[test]
fn a_nested_shape_mismatch_names_the_full_key_and_writes_nothing() {
    let mut block = TransformerBlock::new(8, 2, 2, 0.0);
    let before = block.state_dict();
    let mut state = filled(&before, 0.5);
    let key = "attention.out_proj.weight".to_string();
    state.insert(key.clone(), TensorData::from_f32(vec![0.5; 8], vec![8, 1]));
    let err = block.load_state_dict_partial(&state).unwrap_err();
    assert!(
        matches!(&err, VoltaError::StateShapeMismatch { key: k, .. } if *k == key),
        "{err:?}"
    );
    assert_eq!(block.state_dict(), before);
}

/// The same state without the bad entry loads every value.
#[test]
fn a_valid_state_loads_every_entry() {
    let mut m = model();
    let state = filled(&m.state_dict(), 7.0);
    m.load_state_dict(&state).expect("every entry fits");
    assert_eq!(m.state_dict(), state);
}

/// `load_state_dict_checked` skips shape mismatches (they are in the diff) but is atomic
/// for the rest: a short entry fails it with nothing written.
#[test]
fn load_state_dict_checked_is_atomic_too() {
    let mut m = model();
    let before = m.state_dict();
    let mut state = filled(&before, 7.0);
    state.insert(
        "0.weight".to_string(),
        TensorData::from_f32(vec![7.0; 4], vec![2, 2]),
    );
    let diff = load_state_dict_checked(&mut m, &state).expect("a shape mismatch is in the diff");
    assert_eq!(diff.shape_mismatches.len(), 1);
    assert_ne!(m.state_dict(), before, "the fitting entries loaded");

    let mut m = model();
    let before = m.state_dict();
    let mut state = filled(&before, 7.0);
    state.insert("2.weight".to_string(), short(vec![4, 2], 5));
    let err = load_state_dict_checked(&mut m, &state).unwrap_err();
    assert!(
        matches!(&err, VoltaError::StateDataLength { key, .. } if key == "2.weight"),
        "{err:?}"
    );
    assert_eq!(m.state_dict(), before);
}

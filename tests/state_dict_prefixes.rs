//! Containers save each child's entries under a prefix and load them back from it (#119):
//! `Sequential` by name or index, the transformer layers by field name, nested.

use volta::io::{StateDict, TensorData};
use volta::nn::TransformerBlock;
use volta::{Linear, Module, ReLU, Sequential, VoltaError};

/// The values of the `key` entry, which must be f32.
fn values(state: &StateDict, key: &str) -> Vec<f32> {
    state
        .get(key)
        .unwrap_or_else(|| panic!("no entry {key}"))
        .to_tensor(false)
        .data()
}

/// A fresh `Linear` of the same shape loads only from the entries meant for it.
#[test]
fn sequential_named_layer_prefers_its_name_and_falls_back_to_its_index() {
    let source = Linear::new(2, 3, true);
    let weights = source.state_dict();
    let prefixed = |prefix: &str| -> StateDict {
        weights
            .iter()
            .map(|(k, v)| (format!("{prefix}{k}"), v.clone()))
            .collect()
    };
    let build = || {
        Sequential::builder()
            .add_named("fc", Box::new(Linear::new(2, 3, true)))
            .build()
    };

    // Saved under the index (a model built without names): the named layer reads it.
    let mut model = build();
    model
        .load_state_dict(&prefixed("0."))
        .expect("load by index");
    assert_eq!(
        values(&model.state_dict(), "fc.weight"),
        values(&weights, "weight")
    );

    // Both present: the name wins, and the index entries are not read.
    let mut model = build();
    let mut state = prefixed("fc.");
    state.insert(
        "0.weight".to_string(),
        TensorData::from_f32(vec![0.0; 2], vec![1, 2]),
    );
    model
        .load_state_dict(&state)
        .expect("the name's entries fit");
    assert_eq!(
        values(&model.state_dict(), "fc.weight"),
        values(&weights, "weight")
    );
}

#[test]
fn sequential_state_dict_round_trips_and_skips_stateless_layers() {
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 3, true)),
        Box::new(ReLU),
        Box::new(Linear::new(3, 1, false)),
    ]);
    let state = model.state_dict();
    let keys: Vec<&str> = state.keys().map(String::as_str).collect();
    assert_eq!(keys, ["0.bias", "0.weight", "2.weight"]);

    let mut fresh = Sequential::new(vec![
        Box::new(Linear::new(2, 3, true)),
        Box::new(ReLU),
        Box::new(Linear::new(3, 1, false)),
    ]);
    fresh.load_state_dict(&state).expect("same architecture");
    for key in keys {
        assert_eq!(
            values(&fresh.state_dict(), key),
            values(&state, key),
            "{key}"
        );
    }
}

#[test]
fn transformer_block_round_trips_nested_prefixes() {
    let block = TransformerBlock::new(4, 2, 2, 0.0);
    let state = block.state_dict();
    for key in [
        "norm1.gamma",
        "norm2.beta",
        "ffn.expand.weight",
        "ffn.contract.bias",
        "attention.q_proj.weight",
        "attention.out_proj.bias",
    ] {
        assert!(
            state.contains_key(key),
            "missing {key} in {:?}",
            state.keys()
        );
    }

    let mut fresh = TransformerBlock::new(4, 2, 2, 0.0);
    fresh.load_state_dict(&state).expect("same architecture");
    let loaded = fresh.state_dict();
    assert_eq!(loaded.len(), state.len());
    for key in state.keys() {
        assert_eq!(values(&loaded, key), values(&state, key), "{key}");
    }
}

/// An error two containers deep names the entry as it is in the caller's state dict.
#[test]
fn transformer_block_load_error_names_the_full_key() {
    let mut block = TransformerBlock::new(4, 2, 2, 0.0);
    let mut state = block.state_dict();
    state.insert(
        "ffn.expand.weight".to_string(),
        TensorData::from_f32(vec![0.0; 3], vec![1, 3]),
    );
    match block.load_state_dict(&state) {
        Err(VoltaError::StateShapeMismatch { key, .. }) => assert_eq!(key, "ffn.expand.weight"),
        other => panic!("expected StateShapeMismatch, got {other:?}"),
    }
}

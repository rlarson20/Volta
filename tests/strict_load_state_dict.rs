//! `Module::load_state_dict` is strict, `load_state_dict_partial` is lenient (#344).

use volta::VoltaError;
use volta::io::{StateDict, TensorData};
use volta::nn::{Linear, Module, ReLU, Sequential};

/// The `(missing, unexpected)` keys of a `StateKeyMismatch`, or a panic naming `result`.
fn key_mismatch(result: volta::Result<()>) -> (Vec<String>, Vec<String>) {
    match result {
        Err(VoltaError::StateKeyMismatch {
            missing,
            unexpected,
        }) => (missing, unexpected),
        other => panic!("expected StateKeyMismatch, got {other:?}"),
    }
}

fn strings(keys: &[&str]) -> Vec<String> {
    keys.iter().map(|k| (*k).to_string()).collect()
}

/// The issue's repro: an empty state dict used to load nothing and return `Ok`.
#[test]
fn empty_state_dict_is_an_error_naming_every_parameter() {
    let mut layer = Linear::new(2, 2, true);
    let before = layer.state_dict();
    let (missing, unexpected) = key_mismatch(layer.load_state_dict(&StateDict::new()));
    assert_eq!(missing, strings(&["bias", "weight"]));
    assert!(unexpected.is_empty());
    assert_eq!(layer.state_dict(), before);
    layer
        .load_state_dict_partial(&StateDict::new())
        .expect("the partial form loads nothing and succeeds");
}

/// A typo in a key prefix names both sides, and nothing is written, not even the
/// entries that do match.
#[test]
fn prefix_typo_is_an_error_and_loads_nothing() {
    let build = || {
        Sequential::builder()
            .add_named("enc", Box::new(Linear::new(2, 3, true)))
            .add_unnamed(Box::new(ReLU))
            .add_named("dec", Box::new(Linear::new(3, 2, true)))
            .build()
    };
    let source = build();
    let mut state = source.state_dict();
    let w = state.remove("dec.weight").expect("dec.weight");
    state.insert("decc.weight".to_string(), w);

    let mut model = build();
    let before = model.state_dict();
    let (missing, unexpected) = key_mismatch(model.load_state_dict(&state));
    assert_eq!(missing, strings(&["dec.weight"]));
    assert_eq!(unexpected, strings(&["decc.weight"]));
    assert_eq!(
        model.state_dict(),
        before,
        "a failed strict load writes nothing"
    );
}

/// The matching state loads, values and all.
#[test]
fn exact_state_dict_loads() {
    let source = Linear::new(3, 2, true);
    let mut dest = Linear::new(3, 2, true);
    dest.load_state_dict(&source.state_dict())
        .expect("same keys");
    assert_eq!(dest.state_dict(), source.state_dict());
}

/// An extra entry is refused by the strict form and skipped by the partial one.
#[test]
fn unexpected_key_is_refused_strictly_and_skipped_partially() {
    let source = Linear::new(2, 2, false);
    let mut state = source.state_dict();
    state.insert(
        "running_mean".to_string(),
        TensorData::from_f32(vec![0.0; 2], vec![2]),
    );
    let mut dest = Linear::new(2, 2, false);
    let (missing, unexpected) = key_mismatch(dest.load_state_dict(&state));
    assert!(missing.is_empty());
    assert_eq!(unexpected, strings(&["running_mean"]));
    dest.load_state_dict_partial(&state)
        .expect("extra key skipped");
    assert_eq!(dest.state_dict(), source.state_dict());
}

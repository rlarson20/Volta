//! Serialization round-trip tests for `BatchNorm2d`, `LSTMCell`, and `ConvTranspose2d`.
//!
//! Each layer round-trips through safetensors, the only state-dict file format.
//! Layers with optional bias are also tested in no-bias mode.
use volta::io::{load_safetensors, save_safetensors};
use volta::nn::{Layer, Module};
use volta::{BatchNorm2d, ConvTranspose2d, LSTMCell, RawTensor};

/// Helper: assert two state dicts have exactly the same keys and values.
fn assert_state_dicts_equal(
    original: &volta::io::StateDict,
    loaded: &volta::io::StateDict,
    label: &str,
) {
    assert_eq!(
        original.len(),
        loaded.len(),
        "{label}: key count mismatch (original={}, loaded={})",
        original.len(),
        loaded.len()
    );
    for (key, orig) in original {
        let loaded_val = loaded
            .get(key)
            .unwrap_or_else(|| panic!("{label}: missing key '{key}' in loaded state dict"));
        assert_eq!(
            orig.shape, loaded_val.shape,
            "{label}: shape mismatch for key '{key}'"
        );
        assert_eq!(
            orig.dtype, loaded_val.dtype,
            "{label}: dtype mismatch for key '{key}'"
        );
        // Both formats store each entry in its own dtype, so the round trip is bit-exact.
        assert_eq!(
            orig.data, loaded_val.data,
            "{label}: value mismatch for key '{key}'"
        );
    }
}

// ─── BatchNorm2d ────────────────────────────────────────────────────────────

#[test]
fn test_batchnorm2d_roundtrip_safetensors() {
    let bn = BatchNorm2d::new(4);
    let x = RawTensor::randn(&[2, 4, 6, 6]);
    let _ = bn.forward(&x);

    let sd = bn.state_dict();
    assert_eq!(sd.len(), 4, "BN state dict should have 4 keys");
    assert!(sd.contains_key("gamma"), "missing gamma");
    assert!(sd.contains_key("beta"), "missing beta");
    assert!(sd.contains_key("running_mean"), "missing running_mean");
    assert!(sd.contains_key("running_var"), "missing running_var");

    let dir = std::env::temp_dir().join("volta_test_bn_safetensors");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.safetensors");

    save_safetensors(&sd, &path).unwrap();
    let loaded = load_safetensors(&path).unwrap();

    assert_state_dicts_equal(&sd, &loaded, "bn_safetensors");

    // The forward pass updated the running stats, so they are not all zeros.
    let rm = loaded.get("running_mean").unwrap();
    let has_nonzero_mean = rm
        .to_vec_as::<f32>()
        .expect("f32")
        .iter()
        .any(|v| v.abs() > 1e-8);
    assert!(
        has_nonzero_mean,
        "running_mean should have non-zero values after forward pass"
    );

    std::fs::remove_dir_all(&dir).ok();
}

// ─── LSTMCell ───────────────────────────────────────────────────────────────

#[test]
fn test_lstm_roundtrip_safetensors() {
    let lstm = LSTMCell::new(8, 16, true);

    let sd = lstm.state_dict();
    assert_eq!(sd.len(), 4, "LSTM with bias should have 4 keys");
    assert!(sd.contains_key("weight_ih"));
    assert!(sd.contains_key("weight_hh"));
    assert!(sd.contains_key("bias_ih"));
    assert!(sd.contains_key("bias_hh"));

    let dir = std::env::temp_dir().join("volta_test_lstm_safetensors");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.safetensors");

    save_safetensors(&sd, &path).unwrap();
    let loaded = load_safetensors(&path).unwrap();

    assert_state_dicts_equal(&sd, &loaded, "lstm_safetensors");

    // weight_ih = [4*hidden, input], weight_hh = [4*hidden, hidden]
    assert_eq!(
        loaded.get("weight_ih").unwrap().shape,
        vec![64, 8],
        "weight_ih shape"
    );
    assert_eq!(
        loaded.get("weight_hh").unwrap().shape,
        vec![64, 16],
        "weight_hh shape"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_lstm_no_bias_roundtrip() {
    let lstm = LSTMCell::new(5, 12, false);

    let sd = lstm.state_dict();
    assert_eq!(
        sd.len(),
        2,
        "LSTM without bias should have only 2 keys (weight_ih, weight_hh)"
    );
    assert!(sd.contains_key("weight_ih"));
    assert!(sd.contains_key("weight_hh"));
    assert!(!sd.contains_key("bias_ih"), "should not have bias_ih");
    assert!(!sd.contains_key("bias_hh"), "should not have bias_hh");

    // Round-trip via safetensors
    let dir = std::env::temp_dir().join("volta_test_lstm_nobias");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.safetensors");

    save_safetensors(&sd, &path).unwrap();
    let loaded = load_safetensors(&path).unwrap();

    assert_state_dicts_equal(&sd, &loaded, "lstm_nobias");

    // Load into a fresh model and verify shapes are correct
    let mut lstm2 = LSTMCell::new(5, 12, false);
    lstm2.load_state_dict(&loaded).expect("state loads");
    let sd2 = lstm2.state_dict();
    assert_state_dicts_equal(&sd, &sd2, "lstm_nobias_reload");

    std::fs::remove_dir_all(&dir).ok();
}

// ─── ConvTranspose2d ────────────────────────────────────────────────────────

#[test]
fn test_conv_transpose2d_roundtrip_safetensors() {
    let ct = ConvTranspose2d::new(4, 8, 3, 1, 0, true);

    let sd = ct.state_dict();
    assert_eq!(sd.len(), 2, "ConvTranspose2d with bias should have 2 keys");
    let w = sd.get("weight").unwrap();
    assert_eq!(w.shape, vec![4, 8, 3, 3], "weight shape [in, out, kH, kW]");
    let b = sd.get("bias").unwrap();
    assert_eq!(b.shape, vec![8], "bias shape [out_channels]");

    let dir = std::env::temp_dir().join("volta_test_ct_safetensors");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.safetensors");

    save_safetensors(&sd, &path).unwrap();
    let loaded = load_safetensors(&path).unwrap();

    assert_state_dicts_equal(&sd, &loaded, "ct_safetensors");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_conv_transpose2d_no_bias_roundtrip() {
    let ct = ConvTranspose2d::new(2, 4, 3, 1, 0, false);

    let sd = ct.state_dict();
    assert_eq!(
        sd.len(),
        1,
        "ConvTranspose2d without bias should have only 1 key (weight)"
    );
    assert!(sd.contains_key("weight"));
    assert!(!sd.contains_key("bias"), "should not have bias key");

    // Round-trip via safetensors
    let dir = std::env::temp_dir().join("volta_test_ct_nobias");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.safetensors");

    save_safetensors(&sd, &path).unwrap();
    let loaded = load_safetensors(&path).unwrap();

    assert_state_dicts_equal(&sd, &loaded, "ct_nobias");

    // Load into a fresh model and verify
    let mut ct2 = ConvTranspose2d::new(2, 4, 3, 1, 0, false);
    ct2.load_state_dict(&loaded).expect("state loads");
    let sd2 = ct2.state_dict();
    assert_state_dicts_equal(&sd, &sd2, "ct_nobias_reload");

    std::fs::remove_dir_all(&dir).ok();
}

/// Serialization round-trip tests for `BatchNorm2d`, `LSTMCell`, and `ConvTranspose2d`.
///
/// Each layer is tested with both bincode and safetensors formats.
/// Layers with optional bias are also tested in no-bias mode.
use volta::io::{load_safetensors, load_state_dict, save_safetensors, save_state_dict};
use volta::nn::Module;
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
        for (i, (a, b)) in orig.data.iter().zip(loaded_val.data.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "{label}: value mismatch for key '{key}' at index {i}: {a} vs {b}"
            );
        }
    }
}

// ─── BatchNorm2d ────────────────────────────────────────────────────────────

#[test]
fn test_batchnorm2d_roundtrip_bincode() {
    let bn = BatchNorm2d::new(3);
    // Run a forward pass so running_mean/running_var get updated
    let x = RawTensor::randn(&[4, 3, 8, 8]);
    let _ = bn.forward(&x);

    let sd = bn.state_dict();
    // Verify expected keys
    assert!(sd.contains_key("gamma"), "missing gamma");
    assert!(sd.contains_key("beta"), "missing beta");
    assert!(sd.contains_key("running_mean"), "missing running_mean");
    assert!(sd.contains_key("running_var"), "missing running_var");

    let dir = std::env::temp_dir().join("volta_test_bn_bincode");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.bin");
    let path_str = path.to_str().unwrap();

    save_state_dict(&sd, path_str).unwrap();
    let loaded = load_state_dict(path_str).unwrap();

    assert_state_dicts_equal(&sd, &loaded, "bn_bincode");

    // Also verify running stats are non-trivial (not all zeros / all ones)
    let rm = loaded.get("running_mean").unwrap();
    let has_nonzero_mean = rm.data.iter().any(|v| v.abs() > 1e-8);
    assert!(
        has_nonzero_mean,
        "running_mean should have non-zero values after forward pass"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_batchnorm2d_roundtrip_safetensors() {
    let bn = BatchNorm2d::new(4);
    let x = RawTensor::randn(&[2, 4, 6, 6]);
    let _ = bn.forward(&x);

    let sd = bn.state_dict();
    assert_eq!(sd.len(), 4, "BN state dict should have 4 keys");

    let dir = std::env::temp_dir().join("volta_test_bn_safetensors");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.safetensors");

    save_safetensors(&sd, &path).unwrap();
    let loaded = load_safetensors(&path).unwrap();

    assert_state_dicts_equal(&sd, &loaded, "bn_safetensors");

    std::fs::remove_dir_all(&dir).ok();
}

// ─── LSTMCell ───────────────────────────────────────────────────────────────

#[test]
fn test_lstm_roundtrip_bincode() {
    let lstm = LSTMCell::new(10, 20, true);

    let sd = lstm.state_dict();
    assert_eq!(sd.len(), 4, "LSTM with bias should have 4 keys");
    assert!(sd.contains_key("weight_ih"));
    assert!(sd.contains_key("weight_hh"));
    assert!(sd.contains_key("bias_ih"));
    assert!(sd.contains_key("bias_hh"));

    let dir = std::env::temp_dir().join("volta_test_lstm_bincode");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.bin");
    let path_str = path.to_str().unwrap();

    save_state_dict(&sd, path_str).unwrap();
    let loaded = load_state_dict(path_str).unwrap();

    assert_state_dicts_equal(&sd, &loaded, "lstm_bincode");

    // Verify shapes: weight_ih = [4*hidden, input], weight_hh = [4*hidden, hidden]
    let wih = loaded.get("weight_ih").unwrap();
    assert_eq!(wih.shape, vec![80, 10], "weight_ih shape");
    let whh = loaded.get("weight_hh").unwrap();
    assert_eq!(whh.shape, vec![80, 20], "weight_hh shape");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_lstm_roundtrip_safetensors() {
    let lstm = LSTMCell::new(8, 16, true);

    let sd = lstm.state_dict();
    assert_eq!(sd.len(), 4);

    let dir = std::env::temp_dir().join("volta_test_lstm_safetensors");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.safetensors");

    save_safetensors(&sd, &path).unwrap();
    let loaded = load_safetensors(&path).unwrap();

    assert_state_dicts_equal(&sd, &loaded, "lstm_safetensors");

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

    // Round-trip via bincode
    let dir = std::env::temp_dir().join("volta_test_lstm_nobias");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.bin");
    let path_str = path.to_str().unwrap();

    save_state_dict(&sd, path_str).unwrap();
    let loaded = load_state_dict(path_str).unwrap();

    assert_state_dicts_equal(&sd, &loaded, "lstm_nobias");

    // Load into a fresh model and verify shapes are correct
    let mut lstm2 = LSTMCell::new(5, 12, false);
    lstm2.load_state_dict(&loaded);
    let sd2 = lstm2.state_dict();
    assert_state_dicts_equal(&sd, &sd2, "lstm_nobias_reload");

    std::fs::remove_dir_all(&dir).ok();
}

// ─── ConvTranspose2d ────────────────────────────────────────────────────────

#[test]
fn test_conv_transpose2d_roundtrip_bincode() {
    let ct = ConvTranspose2d::new(3, 6, 4, 2, 1, true);

    let sd = ct.state_dict();
    assert_eq!(sd.len(), 2, "ConvTranspose2d with bias should have 2 keys");
    assert!(sd.contains_key("weight"));
    assert!(sd.contains_key("bias"));

    // Verify shapes
    let w = sd.get("weight").unwrap();
    assert_eq!(w.shape, vec![3, 6, 4, 4], "weight shape [in, out, kH, kW]");
    let b = sd.get("bias").unwrap();
    assert_eq!(b.shape, vec![6], "bias shape [out_channels]");

    let dir = std::env::temp_dir().join("volta_test_ct_bincode");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.bin");
    let path_str = path.to_str().unwrap();

    save_state_dict(&sd, path_str).unwrap();
    let loaded = load_state_dict(path_str).unwrap();

    assert_state_dicts_equal(&sd, &loaded, "ct_bincode");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_conv_transpose2d_roundtrip_safetensors() {
    let ct = ConvTranspose2d::new(4, 8, 3, 1, 0, true);

    let sd = ct.state_dict();
    assert_eq!(sd.len(), 2);

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

    // Round-trip via bincode
    let dir = std::env::temp_dir().join("volta_test_ct_nobias");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.bin");
    let path_str = path.to_str().unwrap();

    save_state_dict(&sd, path_str).unwrap();
    let loaded = load_state_dict(path_str).unwrap();

    assert_state_dicts_equal(&sd, &loaded, "ct_nobias");

    // Load into a fresh model and verify
    let mut ct2 = ConvTranspose2d::new(2, 4, 3, 1, 0, false);
    ct2.load_state_dict(&loaded);
    let sd2 = ct2.state_dict();
    assert_state_dicts_equal(&sd, &sd2, "ct_nobias_reload");

    std::fs::remove_dir_all(&dir).ok();
}

//! Repros for defects in `docs/audits/2026-11-sweep.md` whose correct answer is known
//! without PyTorch fixtures (a hand calculation, or PyTorch's documented behaviour).
//!
//! Each test asserts the correct behaviour and carries `#[ignore = "S<n>: ..."]` while
//! the defect is open. Run them with
//! `cargo test --no-default-features --test sweep_repros -- --ignored`.

use std::collections::BTreeMap;
use volta::io::mapping::StateDictMapper;
use volta::io::{TensorData, load_safetensors, load_safetensors_raw};
use volta::nn::{FeedForward, MultiHeadAttention, ScaledDotProductAttention, TransformerBlock};
use volta::{BatchNorm1d, DataLoader, Layer, Module, RawTensor, SGD, Storage};

/// Zero every tensor in `m` whose state-dict key contains one of `needles`.
fn zero_matching(m: &mut impl Module, needles: &[&str]) {
    let mut state = m.state_dict();
    for (key, value) in &mut state {
        if needles.iter().any(|n| key.contains(n)) {
            value.data.iter_mut().for_each(|v| *v = 0.0);
        }
    }
    m.load_state_dict(&state).expect("load zeroed weights");
}

#[test]
fn second_backward_on_a_graph_with_no_intermediate_node_is_exact() {
    // y = x * x at x = 3 has no intermediate node (the root is only seeded, never
    // accumulated into), so two backward calls give 6 + 6 = 12, as torch does.
    let x = RawTensor::new(vec![3.0], &[1], true);
    let loss = x.elem_mul(&x).into_loss().expect("scalar");
    loss.backward();
    loss.backward();
    assert_eq!(x.grad().expect("grad"), vec![12.0]);
}

#[test]
fn second_backward_through_an_intermediate_adds_only_the_new_gradient() {
    // l = sum(2a). d/da = 2 per backward, so retain_graph=True twice gives 4.
    let a = RawTensor::new(vec![1.0], &[1], true);
    let two = RawTensor::new(vec![2.0], &[1], false);
    let loss = a.elem_mul(&two).sum().into_loss().expect("scalar");
    loss.backward();
    loss.backward();
    assert_eq!(a.grad().expect("grad"), vec![4.0]);
}

#[test]
fn two_losses_sharing_an_intermediate_sum_their_gradients() {
    // b = 2a, l1 = sum(b), l2 = sum(3b). a.grad = 2*1 + 2*3 = 8 in torch.
    let a = RawTensor::new(vec![1.0], &[1], true);
    let two = RawTensor::new(vec![2.0], &[1], false);
    let three = RawTensor::new(vec![3.0], &[1], false);
    let b = a.elem_mul(&two);
    b.sum().into_loss().expect("scalar").backward();
    b.elem_mul(&three)
        .sum()
        .into_loss()
        .expect("scalar")
        .backward();
    assert_eq!(a.grad().expect("grad"), vec![8.0]);
}

#[test]
fn transformer_block_with_zero_sublayers_is_the_identity() {
    // Pre-LN block: y = x + Attn(LN(x)); z = y + FFN(LN(y)). With out_proj and the FFN's
    // contract layer zeroed, both sublayers output 0, so z must equal x.
    let mut block = TransformerBlock::new(4, 1, 1, 0.0);
    zero_matching(&mut block, &["out_proj", "contract"]);
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 4], false);
    let y = block.forward(&x, false);
    let got = y.data();
    for (i, (g, e)) in got.iter().zip([1.0f32, 2.0, 3.0, 4.0]).enumerate() {
        assert!(
            (g - e).abs() < 1e-5,
            "y[{i}] = {g}, expected {e}; got {got:?}"
        );
    }
}

#[test]
fn mha_adds_no_residual() {
    // S11. Like torch.nn.MultiheadAttention, the layer returns out_proj(Attn(x)) and
    // leaves the residual to the block. With out_proj zeroed (weight and bias), the
    // output is exactly zero for any input.
    let mut mha = MultiHeadAttention::new(4, 1, 0.0);
    zero_matching(&mut mha, &["out_proj"]);
    let x = RawTensor::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 4], false);
    let got = mha.forward(&x, false, None).data();
    assert_eq!(got, vec![0.0; 4], "MultiHeadAttention added its input back");
}

#[test]
#[ignore = "S15: intermediate_dim() returns the expansion factor (4), not embed_dim * factor (64)"]
fn feed_forward_intermediate_dim_is_the_hidden_width() {
    assert_eq!(FeedForward::new(16, 4, 0.0).intermediate_dim(), 64);
}

#[test]
#[ignore = "S19: add_prefix removes each old key as it inserts the new one, so a key that is another's new name is lost"]
fn add_prefix_keeps_every_tensor() {
    let td = |v: f32| TensorData {
        data: vec![v],
        shape: vec![1],
    };
    let mut state = BTreeMap::new();
    state.insert("w".to_string(), td(1.0));
    state.insert("xw".to_string(), td(2.0));
    let out = StateDictMapper::new().add_prefix("x").map(state);
    assert_eq!(
        out.len(),
        2,
        "keys after add_prefix: {:?}",
        out.keys().collect::<Vec<_>>()
    );
    assert_eq!(out["xw"].data, vec![1.0]);
    assert_eq!(out["xxw"].data, vec![2.0]);
}

/// Write a one-tensor safetensors file with the given dtype, shape, and raw bytes.
fn write_raw(
    name: &str,
    dtype: safetensors::Dtype,
    shape: Vec<usize>,
    bytes: &[u8],
) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "volta_sweep_{name}_{}.safetensors",
        std::process::id()
    ));
    let view = safetensors::tensor::TensorView::new(dtype, shape, bytes).expect("view");
    safetensors::serialize_to_file([("t", view)], None, &path).expect("write");
    path
}

/// S18: an unsupported dtype is rejected by name, by both loaders, rather than read as
/// f32 bit patterns. Converting by value was the alternative; it is lossy for U32/U64.
fn assert_rejected(name: &str, dtype: safetensors::Dtype, shape: Vec<usize>, bytes: &[u8]) {
    let path = write_raw(name, dtype, shape, bytes);
    let f32_load = load_safetensors(&path).map(|_| ());
    let raw_load = load_safetensors_raw(&path).map(|_| ());
    let _ = std::fs::remove_file(&path);
    for (loader, result) in [
        ("load_safetensors", f32_load),
        ("load_safetensors_raw", raw_load),
    ] {
        let err = result.expect_err(&format!("{loader} accepted a {dtype:?} tensor"));
        assert_eq!(
            err.kind(),
            std::io::ErrorKind::InvalidData,
            "{loader}: {err}"
        );
        assert!(
            err.to_string().contains(&format!("{dtype:?}")),
            "{loader}'s error should name the dtype: {err}"
        );
    }
}

/// U32 has no Volta dtype; it used to load as bit patterns (1 became 1.4e-45).
#[test]
fn u32_safetensors_is_rejected() {
    let bytes: Vec<u8> = [1u32, 2u32].iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_rejected("u32", safetensors::Dtype::U32, vec![2], &bytes);
}

/// I8 has no Volta dtype; it used to be read as 4-byte f32.
#[test]
fn i8_safetensors_is_rejected() {
    // Four I8 values: the byte length a 4-byte f32 read used to accept as one value.
    assert_rejected("i8", safetensors::Dtype::I8, vec![4], &[1, 2, 3, 4]);
    // Three I8 values: a length an f32 read would have panicked on.
    assert_rejected("i8_odd", safetensors::Dtype::I8, vec![3], &[1, 2, 3]);
}

#[test]
fn batchnorm_with_one_sample_per_channel_never_poisons_running_stats() {
    let mut bn = BatchNorm1d::new(2);
    let x = RawTensor::new(vec![1.0, 2.0], &[1, 2], false);
    // Rejecting the batch loudly (a panic here, like torch's ValueError) is acceptable.
    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| bn.forward(&x))).is_err() {
        return;
    }
    bn.train(false);
    let y = bn
        .forward(&RawTensor::new(vec![0.5, 0.5], &[1, 2], false))
        .data();
    assert!(
        y.iter().all(|v| v.is_finite()),
        "eval output after a 1-sample train step: {y:?}"
    );
}

#[test]
#[ignore = "S14: FeedForward stores `dropout` but never applies it; in training mode torch applies dropout, so two forwards at p=0.9 differ"]
fn feed_forward_dropout_is_applied_in_training_mode() {
    volta::manual_seed(11);
    let mut ffn = FeedForward::new(8, 4, 0.9);
    ffn.train(true);
    let x = volta::randn(&[1, 2, 8]);
    assert_ne!(ffn.forward(&x).data(), ffn.forward(&x).data());
}

#[test]
#[ignore = "S14: MultiHeadAttention stores `dropout` but never applies it"]
fn mha_dropout_is_applied_in_training_mode() {
    volta::manual_seed(12);
    let mut mha = MultiHeadAttention::new(8, 2, 0.9);
    mha.train(true);
    let x = volta::randn(&[1, 3, 8]);
    assert_ne!(
        mha.forward(&x, false, None).data(),
        mha.forward(&x, false, None).data()
    );
}

#[test]
#[ignore = "S16: ScaledDotProductAttention asserts key length == query length; cross-attention (L=2 queries, S=3 keys) works in torch"]
fn attention_supports_cross_attention_lengths() {
    let q = volta::randn(&[1, 2, 4]);
    let k = volta::randn(&[1, 3, 4]);
    let v = volta::randn(&[1, 3, 4]);
    let out = ScaledDotProductAttention::new().forward(&q, &k, &v, false, None);
    assert_eq!(out.shape(), vec![1, 2, 4]);
}

#[test]
#[ignore = "S9: an optimizer step rewrites a parameter in place after the forward pass saved it; backward then uses the new value (torch raises 'modified by an inplace operation')"]
fn backward_after_an_optimizer_step_does_not_silently_use_the_new_weights() {
    // y = sum(w * x), w = 2, x = 3, so dy/dx = w = 2 at forward time.
    let w = RawTensor::new(vec![2.0], &[1], true);
    let x = RawTensor::new(vec![3.0], &[1], true);
    let loss = w.elem_mul(&x).sum().into_loss().expect("scalar");
    w.set_grad(Storage::cpu(vec![1.0]));
    SGD::new(vec![w], 1.0, 0.0, 0.0).step(); // w becomes 1
    // Either backward refuses (a panic), or it uses the forward-time w = 2.
    let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| loss.backward()));
    if ran.is_ok() {
        assert_eq!(
            x.grad().expect("grad"),
            vec![2.0],
            "x.grad used the post-step weight"
        );
    }
}

#[test]
#[ignore = "S20: DataLoader::new does not check that targets cover every sample; the mismatch surfaces as a slice panic partway through an epoch"]
fn dataloader_rejects_targets_shorter_than_data_up_front() {
    // 4 samples of 2 values, but targets for only 2 samples.
    let built = std::panic::catch_unwind(|| {
        DataLoader::new(vec![0.0; 8], vec![0.0; 2], &[2], &[1], 2, false)
    });
    assert!(
        built.is_err(),
        "DataLoader::new accepted 4 samples with only 2 targets"
    );
}

#[test]
#[ignore = "S23: the README lists example commands (char_lm, lstm_time_series, polynomial_regression) that do not exist"]
fn readme_example_commands_name_real_examples() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let readme = std::fs::read_to_string(root.join("README.md")).expect("README.md");
    let cargo = std::fs::read_to_string(root.join("Cargo.toml")).expect("Cargo.toml");
    let missing: Vec<String> = readme
        .lines()
        .filter_map(|l| l.split("cargo run").nth(1))
        .filter_map(|rest| rest.split("--example").nth(1))
        .filter_map(|rest| rest.split_whitespace().next())
        .map(str::to_string)
        .filter(|name| {
            !root.join("examples").join(format!("{name}.rs")).exists()
                && !cargo.contains(&format!("name = \"{name}\""))
        })
        .collect();
    assert!(
        missing.is_empty(),
        "README names examples that do not exist: {missing:?}"
    );
}

---
paths:
  - "src/io.rs"
  - "src/io/**"
  - "tests/test_serialization.rs"
  - "examples/load_external_*.rs"
---

# Serialization (state dicts)

- `StateDict` is a `BTreeMap<String, TensorData>`. `TensorData` holds `data: Vec<f32>` and `shape: Vec<usize>`. `StateDictDiff` compares expected and loaded dicts when debugging.
- Formats: bincode (Volta-native: `save_state_dict` / `load_state_dict`) and SafeTensors (`save_safetensors` / `load_safetensors`, plus `load_safetensors_with_mapping` for key and weight transforms via `StateDictMapper`).
- `Sequential` supports named layers. `examples/load_external_mnist.rs` is the end-to-end PyTorch → Volta reference.
- Loading must preserve the destination's device and validate numel (H7).
- Stored weight layouts differ by layer: `Linear` is `[in, out]`, `LSTMCell` is `[4H, in]`, `Embedding` is `[vocab, dim]`, and PyTorch `nn.Linear` is `[out, in]`. `load_weight`'s exact shape check cannot catch a wrong transpose on a square matrix, so any loader test must use a square, asymmetric weight and compare values, not shapes. Only `Linear` weights need a transpose, and key names can't tell them from `Embedding` weights: transpose by layer name (`StateDictMapper::transpose_linear`, `load_pytorch_linear_weights(state, &["fc1", ..])`), never by key pattern.
- Never add a loader that can execute code (pickle-based `.pth`, for example) without an explicit allowlist or safe-unpickler design. Raise it with the user first.

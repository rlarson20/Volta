//! CPU autocast in bf16 and f16 against PyTorch (`tests/fixtures/torch/autocast.safetensors`,
//! generator group `autocast`): an MLP (`Linear`, `ReLU`, `Linear`, class-index cross-entropy)
//! with its loss and every parameter's f32 gradient, `Conv2d` and `ConvTranspose2d` with
//! their low-dtype outputs and the f32 gradients of input, weight and bias, and broadcast
//! batched matmul. Parameters and inputs are f32 on both sides; only the forward runs
//! under autocast.
//!
//! Errors are in units of the low dtype's epsilon relative to `max(|torch|, rms(torch))`,
//! as in `tests/torch_ref_half.rs`. Volta's `Linear` and convolutions round the product
//! and the bias add separately where torch's fused kernels round once, and gradients pass
//! through the low dtype, so each case has a bound: about twice the error measured.

use std::collections::BTreeMap;
use volta::amp::autocast;
use volta::io::{StateDict, TensorData, TypedTensorData, load_safetensors_raw};
use volta::storage::Element;
use volta::{
    BatchNorm2d, Conv2d, ConvAlgo, ConvTranspose2d, DType, Layer, LayerNorm, Linear, Module,
    RawTensor, ReLU, Sequential, Tensor,
};

/// The low dtypes under test: fixture prefix, dtype, machine epsilon.
const LOW: [(&str, DType, f64); 2] = [
    ("bf16", DType::BF16, 1.0 / 128.0),
    ("f16", DType::F16, 1.0 / 1024.0),
];

/// Bounds in epsilons per case-name prefix (first match wins), about twice the largest
/// error measured over both dtypes. The MLP's gradients are the loosest (4.55 f16 eps,
/// 0.44% relative, on `w1`): Volta's `Linear` rounds the product and the bias add apart,
/// torch's `addmm` once, and the difference grows through two layers of backward. In
/// bf16 the same cases stay within 1.01 eps. The norms after a low-precision op match
/// within 0.99 eps (`LayerNorm`) and 0.001 eps (`BatchNorm2d`): one f32 normalization and
/// one rounding, as torch's kernels do.
const BOUNDS: &[(&str, f64)] = &[
    ("mlp.loss", 2.0),
    ("mlp.", 9.0),
    ("conv.", 2.0),
    ("convt.", 2.0),
    ("matmul.", 2.0),
    ("ln.", 2.0),
    ("bn.", 1.0),
];

/// The `autocast` fixture group.
struct Fixture(BTreeMap<String, TypedTensorData>);

impl Fixture {
    /// Load the fixture, or panic with the regeneration command.
    fn load() -> Self {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/torch/autocast.safetensors"
        );
        Self(
            load_safetensors_raw(path)
                .unwrap_or_else(|e| panic!("{path}: {e}; regenerate with `just torch-fixtures`")),
        )
    }

    /// The storage and shape under `name`.
    fn get(&self, name: &str) -> (volta::Storage, Vec<usize>) {
        let t = self
            .0
            .get(name)
            .unwrap_or_else(|| panic!("autocast fixture: no tensor {name:?}"));
        (t.to_storage().expect("valid bytes"), t.shape.clone())
    }

    /// The values under `name` decoded to f64, with the stored dtype.
    fn values(&self, name: &str) -> (Vec<f64>, DType) {
        let (s, _) = self.get(name);
        let d = s.dtype();
        (
            s.cast(DType::F64).as_host::<f64>().expect("f64").to_vec(),
            d,
        )
    }

    /// The f32 values under `name`.
    fn f32s(&self, name: &str) -> Vec<f32> {
        let (s, _) = self.get(name);
        s.as_host::<f32>().expect("stored as f32").to_vec()
    }

    /// A tensor of `T` holding `name`.
    fn tensor<T: Element>(&self, name: &str, requires_grad: bool) -> Tensor {
        let (s, shape) = self.get(name);
        let t = RawTensor::from_typed(s.as_host::<T>().expect("dtype").to_vec(), &shape);
        if requires_grad {
            t.set_requires_grad(true);
        }
        t
    }
}

/// The bound for `label`.
fn bound(label: &str) -> f64 {
    BOUNDS
        .iter()
        .find(|(p, _)| label.starts_with(p))
        .unwrap_or_else(|| panic!("no bound for {label}"))
        .1
}

/// Asserts `got` matches torch's `name` (prefixed by `low`) in dtype and within the bound.
fn assert_close(f: &Fixture, low: (&str, DType, f64), name: &str, got: &Tensor) {
    let (prefix, _, eps) = low;
    let (want, want_dtype) = f.values(&format!("{prefix}.{name}"));
    assert_eq!(got.dtype(), want_dtype, "{prefix}.{name}: dtype");
    let got = got.to_dtype(DType::F64).to_vec_as::<f64>().expect("f64");
    assert_eq!(got.len(), want.len(), "{prefix}.{name}: length");
    #[expect(clippy::cast_precision_loss, reason = "small test tensors")]
    let rms = (want.iter().map(|w| w * w).sum::<f64>() / want.len() as f64).sqrt();
    let b = bound(name);
    for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
        let k = (g - w).abs() / (eps * w.abs().max(rms).max(f64::MIN_POSITIVE));
        assert!(
            k <= b,
            "{prefix}.{name}[{i}]: volta {g:e}, torch {w:e} ({k:.2} eps > {b})"
        );
    }
}

/// The gradient of `t` as a tensor (for `assert_close`), which must be f32.
fn grad(t: &Tensor) -> Tensor {
    let g = t.grad_storage().expect("a gradient");
    assert_eq!(g.dtype(), DType::F32, "f32 parameters get f32 gradients");
    RawTensor::new(g.as_host::<f32>().expect("f32").to_vec(), &[g.len()], false)
}

/// `values` row-major `rows × cols`, transposed.
fn transpose(values: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(values.len(), rows * cols);
    (0..cols)
        .flat_map(|c| values.iter().skip(c).step_by(cols).copied())
        .collect()
}

/// A state dict from `(key, values, shape)` entries.
fn state(entries: Vec<(&str, Vec<f32>, Vec<usize>)>) -> StateDict {
    entries
        .into_iter()
        .map(|(k, data, shape)| (k.to_string(), TensorData { data, shape }))
        .collect()
}

/// Backprop `sum(out * w)` outside autocast, as the fixture does.
fn backprop_weighted(f: &Fixture, prefix: &str, name: &str, out: &Tensor) {
    let w = f.tensor::<f32>(&format!("{prefix}.{name}.w"), false);
    out.elem_mul(&w)
        .sum()
        .into_loss()
        .expect("scalar")
        .backward();
}

/// The MLP: loss and the f32 gradients of the input and every parameter.
#[test]
fn mlp_under_autocast_matches_torch() {
    let f = Fixture::load();
    for low in LOW {
        let (p, dtype, _) = low;
        let mut l1 = Linear::new(8, 16, true);
        let mut l2 = Linear::new(16, 3, true);
        l1.load_state_dict(&state(vec![
            (
                "weight",
                transpose(&f.f32s(&format!("{p}.mlp.w1")), 16, 8),
                vec![8, 16],
            ),
            ("bias", f.f32s(&format!("{p}.mlp.b1")), vec![16]),
        ]))
        .expect("load");
        l2.load_state_dict(&state(vec![
            (
                "weight",
                transpose(&f.f32s(&format!("{p}.mlp.w2")), 3, 16),
                vec![16, 3],
            ),
            ("bias", f.f32s(&format!("{p}.mlp.b2")), vec![3]),
        ]))
        .expect("load");
        let model = Sequential::new(vec![Box::new(l1), Box::new(ReLU), Box::new(l2)]);
        let x = f.tensor::<f32>(&format!("{p}.mlp.x"), true);
        let y = f.tensor::<i64>(&format!("{p}.mlp.y"), false);
        let loss = {
            let _ac = autocast(dtype);
            RawTensor::cross_entropy_loss(&model.forward(&x), &y)
        };
        assert_close(&f, low, "mlp.loss", &loss);
        loss.into_loss().expect("scalar").backward();
        assert_close(&f, low, "mlp.grad_x", &grad(&x));
        let params = model.parameters();
        let [w1, b1, w2, b2] = params.as_slice() else {
            panic!("four parameters")
        };
        let t = |g: &Tensor, rows, cols| {
            RawTensor::new(
                transpose(&g.grad().expect("grad"), rows, cols),
                &[rows * cols],
                false,
            )
        };
        assert_eq!(grad(w1).dtype(), DType::F32);
        assert_close(&f, low, "mlp.grad_w1", &t(w1, 8, 16));
        assert_close(&f, low, "mlp.grad_b1", &grad(b1));
        assert_close(&f, low, "mlp.grad_w2", &t(w2, 16, 3));
        assert_close(&f, low, "mlp.grad_b2", &grad(b2));
    }
}

/// Conv2d under autocast with each algorithm: the low-dtype output and the f32 gradients.
#[test]
fn conv2d_under_autocast_matches_torch() {
    let f = Fixture::load();
    for low in LOW {
        let (p, dtype, _) = low;
        for algo in [ConvAlgo::Direct, ConvAlgo::Im2col, ConvAlgo::IGEMM] {
            let mut c = Conv2d::new(2, 3, 3, 2, 1, true);
            c.set_algo(algo);
            c.load_state_dict(&state(vec![
                (
                    "weight",
                    f.f32s(&format!("{p}.conv.weight")),
                    vec![3, 2, 3, 3],
                ),
                ("bias", f.f32s(&format!("{p}.conv.bias")), vec![3]),
            ]))
            .expect("load");
            let x = f.tensor::<f32>(&format!("{p}.conv.x"), true);
            let out = {
                let _ac = autocast(dtype);
                c.forward(&x)
            };
            assert_close(&f, low, "conv.out", &out);
            backprop_weighted(&f, p, "conv", &out);
            let params = c.parameters();
            let [w, b] = params.as_slice() else {
                panic!("two parameters")
            };
            assert_close(&f, low, "conv.grad_x", &grad(&x));
            assert_close(&f, low, "conv.grad_weight", &grad(w));
            assert_close(&f, low, "conv.grad_bias", &grad(b));
        }
    }
}

/// `ConvTranspose2d` under autocast.
#[test]
fn conv_transpose2d_under_autocast_matches_torch() {
    let f = Fixture::load();
    for low in LOW {
        let (p, dtype, _) = low;
        let mut c = ConvTranspose2d::new(2, 3, 3, 2, 1, true);
        c.load_state_dict(&state(vec![
            (
                "weight",
                f.f32s(&format!("{p}.convt.weight")),
                vec![2, 3, 3, 3],
            ),
            ("bias", f.f32s(&format!("{p}.convt.bias")), vec![3]),
        ]))
        .expect("load");
        let x = f.tensor::<f32>(&format!("{p}.convt.x"), true);
        let out = {
            let _ac = autocast(dtype);
            c.forward(&x)
        };
        assert_close(&f, low, "convt.out", &out);
        backprop_weighted(&f, p, "convt", &out);
        let params = c.parameters();
        let [w, b] = params.as_slice() else {
            panic!("two parameters")
        };
        assert_close(&f, low, "convt.grad_x", &grad(&x));
        assert_close(&f, low, "convt.grad_weight", &grad(w));
        assert_close(&f, low, "convt.grad_bias", &grad(b));
    }
}

/// Broadcast batched matmul of f32 operands under autocast.
#[test]
fn matmul_under_autocast_matches_torch() {
    let f = Fixture::load();
    for low in LOW {
        let (p, dtype, _) = low;
        let a = f.tensor::<f32>(&format!("{p}.matmul.a"), true);
        let b = f.tensor::<f32>(&format!("{p}.matmul.b"), true);
        let out = {
            let _ac = autocast(dtype);
            a.matmul(&b)
        };
        assert_close(&f, low, "matmul.out", &out);
        backprop_weighted(&f, p, "matmul", &out);
        assert_close(&f, low, "matmul.grad_a", &grad(&a));
        assert_close(&f, low, "matmul.grad_b", &grad(&b));
    }
}

/// `LayerNorm` after a `Linear` under autocast: the norm takes the half activation with
/// f32 parameters, normalizes in f32 and returns the half dtype, as torch does (audit A5);
/// the parameters get f32 gradients.
#[test]
fn layernorm_after_linear_under_autocast_matches_torch() {
    let f = Fixture::load();
    for low in LOW {
        let (p, dtype, _) = low;
        let mut lin = Linear::new(6, 5, true);
        lin.load_state_dict(&state(vec![
            (
                "weight",
                transpose(&f.f32s(&format!("{p}.ln.weight")), 5, 6),
                vec![6, 5],
            ),
            ("bias", f.f32s(&format!("{p}.ln.bias")), vec![5]),
        ]))
        .expect("load");
        let mut ln = LayerNorm::new(vec![5]);
        ln.load_state_dict(&state(vec![
            ("gamma", f.f32s(&format!("{p}.ln.gamma")), vec![5]),
            ("beta", f.f32s(&format!("{p}.ln.beta")), vec![5]),
        ]))
        .expect("load");
        let x = f.tensor::<f32>(&format!("{p}.ln.x"), true);
        let out = {
            let _ac = autocast(dtype);
            ln.forward(&lin.forward(&x))
        };
        assert_close(&f, low, "ln.out", &out);
        backprop_weighted(&f, p, "ln", &out);
        let lp = lin.parameters();
        let [w, b] = lp.as_slice() else {
            panic!("two parameters")
        };
        let np = ln.parameters();
        let [g, be] = np.as_slice() else {
            panic!("two parameters")
        };
        let wt = RawTensor::new(transpose(&w.grad().expect("grad"), 6, 5), &[30], false);
        assert_close(&f, low, "ln.grad_x", &grad(&x));
        assert_eq!(grad(w).dtype(), DType::F32);
        assert_close(&f, low, "ln.grad_weight", &wt);
        assert_close(&f, low, "ln.grad_bias", &grad(b));
        assert_close(&f, low, "ln.grad_gamma", &grad(g));
        assert_close(&f, low, "ln.grad_beta", &grad(be));
    }
}

/// `BatchNorm2d` (training) after a `Conv2d` under autocast, likewise.
#[test]
fn batchnorm_after_conv_under_autocast_matches_torch() {
    let f = Fixture::load();
    for low in LOW {
        let (p, dtype, _) = low;
        let mut c = Conv2d::new(2, 2, 3, 1, 1, false);
        c.load_state_dict(&state(vec![(
            "weight",
            f.f32s(&format!("{p}.bn.weight")),
            vec![2, 2, 3, 3],
        )]))
        .expect("load");
        let mut bn = BatchNorm2d::new(2);
        bn.load_state_dict(&state(vec![
            ("gamma", f.f32s(&format!("{p}.bn.gamma")), vec![2]),
            ("beta", f.f32s(&format!("{p}.bn.beta")), vec![2]),
        ]))
        .expect("load");
        let x = f.tensor::<f32>(&format!("{p}.bn.x"), true);
        let out = {
            let _ac = autocast(dtype);
            bn.forward(&c.forward(&x))
        };
        assert_close(&f, low, "bn.out", &out);
        backprop_weighted(&f, p, "bn", &out);
        let cp = c.parameters();
        let [w] = cp.as_slice() else {
            panic!("one parameter")
        };
        let np = bn.parameters();
        let [g, be] = np.as_slice() else {
            panic!("two parameters")
        };
        assert_close(&f, low, "bn.grad_x", &grad(&x));
        assert_close(&f, low, "bn.grad_weight", &grad(w));
        assert_close(&f, low, "bn.grad_gamma", &grad(g));
        assert_close(&f, low, "bn.grad_beta", &grad(be));
    }
}

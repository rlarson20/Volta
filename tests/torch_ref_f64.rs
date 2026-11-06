//! Ops, every layer kind, and Adam in f64 against PyTorch computing in float64
//! (`tests/fixtures/torch/f64.safetensors`, generator group `f64`). Values are compared at a
//! relative tolerance of 1e-10, where the f32 reference tests use 1e-5 to 1e-3.
//!
//! The fixture's inputs and weights are f32 values widened to f64, so the f32 state dicts
//! that `load_state_dict` takes carry them exactly; every computation after that is f64.

use std::collections::BTreeMap;
use volta::io::{StateDict, TensorData, TypedTensorData, load_safetensors_raw};
use volta::nn::IndexedLayer;
use volta::{
    Adam, Axis, BatchNorm2d, Conv2d, ConvAlgo, ConvTranspose2d, DType, Embedding, Layer, LayerNorm,
    Linear, MaxPool2d, Module, RawTensor, Storage, Tensor,
};

/// Relative tolerance, with an absolute floor of 1 on the scale.
const TOL: f64 = 1e-10;

/// The `f64` fixture group.
struct Fixture(BTreeMap<String, TypedTensorData>);

impl Fixture {
    /// Load the fixture, or panic with the regeneration command.
    fn load() -> Self {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/torch/f64.safetensors"
        );
        Self(
            load_safetensors_raw(path)
                .unwrap_or_else(|e| panic!("{path}: {e}; regenerate with `just torch-fixtures`")),
        )
    }

    /// The f64 values and shape stored under `name`.
    fn get(&self, name: &str) -> (Vec<f64>, Vec<usize>) {
        let t = self
            .0
            .get(name)
            .unwrap_or_else(|| panic!("f64 fixture: no tensor {name:?}"));
        assert_eq!(t.dtype, DType::F64, "{name} is stored as f64");
        let s = t.to_storage().expect("valid f64 bytes");
        (s.as_host::<f64>().expect("f64").to_vec(), t.shape.clone())
    }

    /// The values under `name`.
    fn values(&self, name: &str) -> Vec<f64> {
        self.get(name).0
    }

    /// An f64 leaf tensor holding `name`.
    fn tensor(&self, name: &str, requires_grad: bool) -> Tensor {
        let (v, shape) = self.get(name);
        let t = RawTensor::from_typed(v, &shape);
        t.set_requires_grad(requires_grad);
        t
    }

    /// `name` narrowed to f32, which must be exact (the generator draws f32 values).
    fn f32_exact(&self, name: &str) -> Vec<f32> {
        self.values(name)
            .iter()
            .map(|&v| {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "checked exact on the next line"
                )]
                let n = v as f32;
                assert_eq!(f64::from(n), v, "{name}: value not representable in f32");
                n
            })
            .collect()
    }
}

/// `values` row-major `rows × cols`, transposed.
fn transpose<T: Copy>(values: &[T], rows: usize, cols: usize) -> Vec<T> {
    assert_eq!(values.len(), rows * cols);
    (0..cols)
        .flat_map(|c| values.iter().skip(c).step_by(cols).copied())
        .collect()
}

/// A state dict from `(key, f32 values, shape)` entries.
fn state(entries: &[(&str, Vec<f32>, Vec<usize>)]) -> StateDict {
    entries
        .iter()
        .map(|(k, data, shape)| {
            (
                (*k).to_string(),
                TensorData {
                    data: data.clone(),
                    shape: shape.clone(),
                },
            )
        })
        .collect()
}

/// Asserts `got` matches `want` to `TOL`, relative to `max(|want|, 1)`.
fn assert_close(label: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= TOL * w.abs().max(1.0),
            "{label}[{i}]: volta {g:e}, torch {w:e} (diff {:e})",
            (g - w).abs()
        );
    }
}

/// The gradient of `t` as f64.
fn grad(t: &Tensor) -> Vec<f64> {
    let g = t.grad_storage().expect("a gradient");
    g.as_host::<f64>().expect("an f64 gradient").to_vec()
}

/// A gradient to compare, with an optional fix-up for a transposed layout.
type Check<'a> = (
    &'a str,
    &'a Tensor,
    Option<&'a dyn Fn(Vec<f64>) -> Vec<f64>>,
);

/// Compare `out` with `<name>.out`, backprop `sum(out * <name>.w)`, and compare each
/// named gradient with `<name>.grad_<g>`.
fn check(f: &Fixture, name: &str, out: &Tensor, grads: &[Check<'_>]) {
    assert_eq!(out.dtype(), DType::F64, "{name}: output dtype");
    assert_close(
        &format!("{name}.out"),
        &out.to_vec_as::<f64>().expect("f64"),
        &f.values(&format!("{name}.out")),
    );
    let w = f.tensor(&format!("{name}.w"), false);
    out.elem_mul(&w)
        .sum()
        .into_loss()
        .expect("scalar")
        .backward();
    for (g, t, fix) in grads {
        let mut got = grad(t);
        if let Some(fix) = fix {
            got = fix(got);
        }
        assert_close(
            &format!("{name}.grad_{g}"),
            &got,
            &f.values(&format!("{name}.grad_{g}")),
        );
    }
}

/// A unary op under test.
type Op = fn(&Tensor) -> Tensor;

/// Elementwise ops, the softmaxes, and batched matmul with broadcast batch dims.
#[test]
fn ops_match_torch_in_f64() {
    let f = Fixture::load();
    let ops: [(&str, Op); 6] = [
        ("exp", Tensor::exp),
        ("tanh", Tensor::tanh),
        ("sigmoid", Tensor::sigmoid),
        ("erf", Tensor::erf),
        ("softmax", |t| t.softmax(Axis(1))),
        ("log_softmax", |t| t.log_softmax(Axis(0))),
    ];
    for (name, op) in ops {
        let x = f.tensor("ops.x", true);
        check(&f, &format!("ops.{name}"), &op(&x), &[("x", &x, None)]);
    }
    let (a, b) = (f.tensor("matmul.a", true), f.tensor("matmul.b", true));
    check(
        &f,
        "matmul",
        &a.matmul(&b),
        &[("a", &a, None), ("b", &b, None)],
    );
}

/// `Linear`, whose weight Volta stores transposed (`[in, out]`).
#[test]
fn linear_matches_torch_in_f64() {
    let f = Fixture::load();
    let mut l = Linear::new(2, 3, true);
    l.to_dtype(DType::F64);
    l.load_state_dict(&state(&[
        (
            "weight",
            transpose(&f.f32_exact("linear.weight"), 3, 2),
            vec![2, 3],
        ),
        ("bias", f.f32_exact("linear.bias"), vec![3]),
    ]))
    .expect("load");
    let x = f.tensor("linear.x", true);
    let out = l.forward(&x);
    let p = l.parameters();
    let back = |g: Vec<f64>| transpose(&g, 2, 3);
    check(
        &f,
        "linear",
        &out,
        &[
            ("x", &x, None),
            ("weight", &p[0], Some(&back)),
            ("bias", &p[1], None),
        ],
    );
}

/// `Conv2d` with each of its three algorithms against one torch fixture.
#[test]
fn conv2d_matches_torch_in_f64_for_every_algorithm() {
    let f = Fixture::load();
    for algo in [ConvAlgo::Direct, ConvAlgo::Im2col, ConvAlgo::IGEMM] {
        let mut c = Conv2d::new(2, 3, 3, 2, 1, true);
        c.set_algo(algo);
        c.to_dtype(DType::F64);
        c.load_state_dict(&state(&[
            ("weight", f.f32_exact("conv.weight"), vec![3, 2, 3, 3]),
            ("bias", f.f32_exact("conv.bias"), vec![3]),
        ]))
        .expect("load");
        let x = f.tensor("conv.x", true);
        let out = c.forward(&x);
        let p = c.parameters();
        check(
            &f,
            "conv",
            &out,
            &[
                ("x", &x, None),
                ("weight", &p[0], None),
                ("bias", &p[1], None),
            ],
        );
    }
}

/// `ConvTranspose2d`, stride 2, padding 1.
#[test]
fn conv_transpose2d_matches_torch_in_f64() {
    let f = Fixture::load();
    let mut c = ConvTranspose2d::new(2, 3, 3, 2, 1, true);
    c.to_dtype(DType::F64);
    c.load_state_dict(&state(&[
        ("weight", f.f32_exact("convt.weight"), vec![2, 3, 3, 3]),
        ("bias", f.f32_exact("convt.bias"), vec![3]),
    ]))
    .expect("load");
    let x = f.tensor("convt.x", true);
    let out = c.forward(&x);
    let p = c.parameters();
    check(
        &f,
        "convt",
        &out,
        &[
            ("x", &x, None),
            ("weight", &p[0], None),
            ("bias", &p[1], None),
        ],
    );
}

/// `MaxPool2d`, kernel 2, stride 2.
#[test]
fn maxpool2d_matches_torch_in_f64() {
    let f = Fixture::load();
    let x = f.tensor("maxpool.x", true);
    let out = MaxPool2d::new(2, 2, 0).forward(&x);
    check(&f, "maxpool", &out, &[("x", &x, None)]);
}

/// `BatchNorm2d` in training mode (batch statistics).
#[test]
fn batchnorm2d_matches_torch_in_f64() {
    let f = Fixture::load();
    let mut bn = BatchNorm2d::new(2);
    bn.to_dtype(DType::F64);
    bn.load_state_dict(&state(&[
        ("gamma", f.f32_exact("batchnorm.gamma"), vec![2]),
        ("beta", f.f32_exact("batchnorm.beta"), vec![2]),
    ]))
    .expect("load");
    let x = f.tensor("batchnorm.x", true);
    let out = bn.forward(&x);
    let p = bn.parameters();
    check(
        &f,
        "batchnorm",
        &out,
        &[
            ("x", &x, None),
            ("gamma", &p[0], None),
            ("beta", &p[1], None),
        ],
    );
}

/// `LayerNorm` over the last dim.
#[test]
fn layernorm_matches_torch_in_f64() {
    let f = Fixture::load();
    let mut l = LayerNorm::new(vec![4]);
    l.to_dtype(DType::F64);
    l.load_state_dict(&state(&[
        ("gamma", f.f32_exact("layernorm.gamma"), vec![4]),
        ("beta", f.f32_exact("layernorm.beta"), vec![4]),
    ]))
    .expect("load");
    let x = f.tensor("layernorm.x", true);
    let out = l.forward(&x);
    let p = l.parameters();
    check(
        &f,
        "layernorm",
        &out,
        &[
            ("x", &x, None),
            ("gamma", &p[0], None),
            ("beta", &p[1], None),
        ],
    );
}

/// `Embedding` with a repeated index, so the weight gradient accumulates.
#[test]
fn embedding_matches_torch_in_f64() {
    let f = Fixture::load();
    let mut e = Embedding::new(5, 3);
    e.to_dtype(DType::F64);
    e.load_state_dict(&state(&[(
        "weight",
        f.f32_exact("embedding.weight"),
        vec![5, 3],
    )]))
    .expect("load");
    let out = e.forward_indices(&[1, 3, 1]);
    let p = e.parameters();
    check(&f, "embedding", &out, &[("weight", &p[0], None)]);
}

/// Three Adam steps with L2 weight decay under fixed f64 gradients.
#[test]
fn adam_matches_torch_in_f64() {
    let f = Fixture::load();
    let p = f.tensor("adam.init", true);
    let mut opt = Adam::new(
        vec![p.clone()],
        2f32.powi(-7),
        (0.875, 1.0 - 2f32.powi(-10)),
        2f32.powi(-27),
        2f32.powi(-7),
    );
    for k in 0..3 {
        p.set_grad(Storage::from_typed(f.values(&format!("adam.g{k}"))));
        opt.step();
        assert_close(
            &format!("adam.after{k}"),
            &p.to_vec_as::<f64>().expect("f64"),
            &f.values(&format!("adam.after{k}")),
        );
    }
}

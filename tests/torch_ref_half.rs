//! Every op family, every layer kind, SGD and Adam in f16 and bf16 against PyTorch
//! computing natively in those dtypes on the CPU (`tests/fixtures/torch/half.safetensors`,
//! generator group `half`).
//!
//! Both sides decode each op's inputs to f32, compute, and round the output, so an
//! elementwise op agrees to the last bit or one rounding step. Composite ops (softmax,
//! norms, losses, conv backward) round their intermediates where torch's fused kernels
//! may not, and summation order differs, so each family has its own bound, in units of
//! the dtype's machine epsilon relative to `max(|torch|, rms(torch))`: the rms floor
//! keeps a gradient entry that cancels to near zero from demanding more relative
//! precision than its tensor carries. `BOUNDS` lists them; many are zero (bit-exact).

use half::{bf16, f16};
use std::collections::BTreeMap;
use volta::io::{StateDict, TensorData, TypedTensorData, load_safetensors_raw};
use volta::storage::Element;
use volta::{
    Adam, Axis, BatchNorm2d, Conv2d, ConvAlgo, ConvTranspose2d, DType, Embedding, Layer, LayerNorm,
    Linear, MaxPool2d, Module, RawTensor, SGD, Storage, Tensor,
};

/// A half type under test.
trait Half: Element + PartialEq + std::fmt::Debug {
    /// Key prefix in the fixture.
    const NAME: &'static str;
    /// Machine epsilon (2^-10 for f16, 2^-7 for bf16).
    const EPS: f64;
    /// Exact widening.
    fn widen(self) -> f64;
}

impl Half for f16 {
    const NAME: &'static str = "f16";
    const EPS: f64 = 1.0 / 1024.0;
    fn widen(self) -> f64 {
        self.to_f64()
    }
}

impl Half for bf16 {
    const NAME: &'static str = "bf16";
    const EPS: f64 = 1.0 / 128.0;
    fn widen(self) -> f64 {
        self.to_f64()
    }
}

/// Bounds in epsilons, per case-name prefix (first match wins): `(prefix, forward,
/// gradient)`. An optimizer's parameter trajectory counts as forward.
///
/// Forward results of elementwise ops, data movement and pooling are bit-exact: both
/// sides compute one f32 op and round once. A gradient differs where torch's backward
/// formula reads its rounded output (`exp`'s backward is `g * out`) or rounds a product
/// before a broadcast sum; reductions, norms and losses round intermediates differently.
/// The non-zero bounds are about twice the largest error measured on this fixture
/// (1.83 eps, f16 cross-entropy's gradient).
const BOUNDS: &[(&str, f64, f64)] = &[
    ("unary.", 0.0, 1.5),
    ("binary.", 0.0, 2.0),
    ("ternary.", 0.0, 1.5),
    ("move.", 0.0, 0.0),
    ("maxpool", 0.0, 0.0),
    ("embedding", 0.0, 1.0),
    ("reduce.", 1.5, 2.5),
    ("matmul", 1.5, 2.0),
    ("linear", 1.5, 2.0),
    ("loss.", 1.5, 3.5),
    ("conv", 2.0, 2.0),
    ("batchnorm", 1.5, 3.0),
    ("layernorm", 1.5, 2.0),
    ("sgd", 1.0, 0.0),
    ("adam", 1.0, 0.0),
];

/// The bound for `label` (`<case>.out`, `<case>.grad_<x>` or `<optimizer>.after<k>`).
fn bound(label: &str) -> f64 {
    let &(_, fwd, grad) = BOUNDS
        .iter()
        .find(|(p, _, _)| label.starts_with(p))
        .unwrap_or_else(|| panic!("no bound for {label}"));
    if label.contains(".grad_") { grad } else { fwd }
}

/// The `half` fixture group, read for one dtype.
struct Fixture<H> {
    /// Every tensor in the file.
    all: BTreeMap<String, TypedTensorData>,
    /// The dtype under test.
    _h: std::marker::PhantomData<H>,
}

impl<H: Half> Fixture<H> {
    /// Load the fixture, or panic with the regeneration command.
    fn load() -> Self {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/torch/half.safetensors"
        );
        Self {
            all: load_safetensors_raw(path)
                .unwrap_or_else(|e| panic!("{path}: {e}; regenerate with `just torch-fixtures`")),
            _h: std::marker::PhantomData,
        }
    }

    /// The `H` values and shape stored under `<dtype>.name`.
    fn get(&self, name: &str) -> (Vec<H>, Vec<usize>) {
        let key = format!("{}.{name}", H::NAME);
        let t = self
            .all
            .get(&key)
            .unwrap_or_else(|| panic!("half fixture: no tensor {key:?}"));
        assert_eq!(t.dtype, H::DTYPE, "{key}: stored dtype");
        let s = t.to_storage().expect("valid half bytes");
        (s.as_host::<H>().expect("half").to_vec(), t.shape.clone())
    }

    /// The values under `name`, widened.
    fn values(&self, name: &str) -> Vec<f64> {
        self.get(name).0.into_iter().map(H::widen).collect()
    }

    /// A leaf tensor of `H` holding `name`.
    fn tensor(&self, name: &str, requires_grad: bool) -> Tensor {
        let (v, shape) = self.get(name);
        let t = RawTensor::from_typed(v, &shape);
        t.set_requires_grad(requires_grad);
        t
    }

    /// `name` as f32 (exact), for a state dict.
    fn f32s(&self, name: &str) -> Vec<f32> {
        self.values(name)
            .iter()
            .map(|&v| {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "half values are exact in f32"
                )]
                let n = v as f32;
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

/// Asserts `got` matches `want` within `bound(label)` epsilons of `H` (module docs).
fn assert_close<H: Half>(label: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    #[expect(clippy::cast_precision_loss, reason = "small test tensors")]
    let rms = (want.iter().map(|w| w * w).sum::<f64>() / want.len() as f64).sqrt();
    let bound = bound(label);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let k = (g - w).abs() / (H::EPS * w.abs().max(rms).max(f64::MIN_POSITIVE));
        assert!(
            k <= bound,
            "{}.{label}[{i}]: volta {g:e}, torch {w:e} ({k:.2} eps > {bound})",
            H::NAME
        );
    }
}

/// The values of `t`, which must be an `H` tensor, widened.
fn read<H: Half>(t: &Tensor) -> Vec<f64> {
    assert_eq!(t.dtype(), H::DTYPE, "output dtype");
    t.to_vec_as::<H>()
        .expect("half")
        .into_iter()
        .map(H::widen)
        .collect()
}

/// The gradient of `t`, which must have `H`'s dtype, widened.
fn grad<H: Half>(t: &Tensor) -> Vec<f64> {
    let g = t.grad_storage().expect("a gradient");
    assert_eq!(g.dtype(), H::DTYPE, "gradient dtype");
    g.as_host::<H>()
        .expect("half gradient")
        .iter()
        .map(|&x| x.widen())
        .collect()
}

/// A gradient to compare, with an optional fix-up for a transposed layout.
type Check<'a> = (
    &'a str,
    &'a Tensor,
    Option<&'a dyn Fn(Vec<f64>) -> Vec<f64>>,
);

/// Compare `out` with `<name>.out`, backprop `sum(out * <name>.w)` in `H`, and compare each
/// named gradient with `<name>.grad_<g>`.
fn check<H: Half>(f: &Fixture<H>, name: &str, out: &Tensor, grads: &[Check<'_>]) {
    assert_close::<H>(
        &format!("{name}.out"),
        &read::<H>(out),
        &f.values(&format!("{name}.out")),
    );
    let w = f.tensor(&format!("{name}.w"), false);
    out.elem_mul(&w)
        .sum()
        .into_loss()
        .expect("scalar")
        .backward();
    for (g, t, fix) in grads {
        let mut got = grad::<H>(t);
        if let Some(fix) = fix {
            got = fix(got);
        }
        assert_close::<H>(
            &format!("{name}.grad_{g}"),
            &got,
            &f.values(&format!("{name}.grad_{g}")),
        );
    }
}

/// A unary op under test.
type Op = fn(&Tensor) -> Tensor;

/// A binary op under test.
type BinOp = fn(&Tensor, &Tensor) -> Tensor;

/// A layer's two parameters (weight and bias, or gamma and beta).
fn pair(p: &[Tensor]) -> (&Tensor, &Tensor) {
    let [a, b] = p else {
        panic!("expected two parameters, got {}", p.len())
    };
    (a, b)
}

/// Every `UnaryOp` row; recip, sqrt and the logs on a positive input.
fn unary<H: Half>() {
    let f = Fixture::<H>::load();
    let ops: [(&str, &str, Op); 13] = [
        ("neg", "unary.x", Tensor::neg),
        ("exp", "unary.x", Tensor::exp),
        ("exp2", "unary.x", Tensor::exp2),
        ("sin", "unary.x", Tensor::sin),
        ("cos", "unary.x", Tensor::cos),
        ("tanh", "unary.x", Tensor::tanh),
        ("sigmoid", "unary.x", Tensor::sigmoid),
        ("relu", "unary.x", Tensor::relu),
        ("erf", "unary.x", Tensor::erf),
        ("recip", "unary.xp", Tensor::recip),
        ("sqrt", "unary.xp", Tensor::sqrt),
        ("log", "unary.xp", Tensor::log),
        ("log2", "unary.xp", Tensor::log2),
    ];
    for (name, input, op) in ops {
        let x = f.tensor(input, true);
        check(&f, &format!("unary.{name}"), &op(&x), &[("x", &x, None)]);
    }
}

/// The differentiable `BinaryOp` rows, with `b` broadcast along the rows.
fn binary<H: Half>() {
    let f = Fixture::<H>::load();
    let ops: [(&str, &str, BinOp); 5] = [
        ("add", "binary.b", Tensor::add),
        ("sub", "binary.b", Tensor::sub),
        ("mul", "binary.b", Tensor::elem_mul),
        ("max", "binary.b", Tensor::max_elem),
        ("div", "binary.bp", Tensor::div),
    ];
    for (name, rhs, op) in ops {
        let (a, b) = (f.tensor("binary.a", true), f.tensor(rhs, true));
        check(
            &f,
            &format!("binary.{name}"),
            &op(&a, &b),
            &[("a", &a, None), ("b", &b, None)],
        );
    }
}

/// Full and per-axis reductions, softmax and log-softmax.
fn reductions<H: Half>() {
    let f = Fixture::<H>::load();
    let ops: [(&str, Op); 8] = [
        ("sum", Tensor::sum),
        ("mean", Tensor::mean),
        ("max", Tensor::max_reduce),
        ("sum_dim", |t| t.sum_dim(Axis(1), false)),
        ("mean_dim", |t| t.mean_dim(Axis(0), true)),
        ("max_dim", |t| t.max_dim(Axis(1), false)),
        ("softmax", |t| t.softmax(Axis(1))),
        ("log_softmax", |t| t.log_softmax(Axis(0))),
    ];
    for (name, op) in ops {
        let x = f.tensor("reduce.x", true);
        check(&f, &format!("reduce.{name}"), &op(&x), &[("x", &x, None)]);
    }
}

/// Movement ops: exact on both sides (no arithmetic forward; backward sums at most the
/// expanded copies, which `expand`'s case does in f32 before one rounding).
fn movement<H: Half>() {
    let f = Fixture::<H>::load();
    let ops: [(&str, Op); 4] = [
        ("permute", |t| t.permute(&[Axis(2), Axis(0), Axis(1)])),
        ("reshape", |t| t.reshape(&[6, 4])),
        ("shrink", |t| t.shrink(&[(0, 2), (1, 3), (0, 2)])),
        ("expand", |t| {
            t.shrink(&[(0, 2), (0, 1), (0, 4)]).expand(&[2, 5, 4])
        }),
    ];
    for (name, op) in ops {
        let x = f.tensor("move.x", true);
        check(&f, &format!("move.{name}"), &op(&x), &[("x", &x, None)]);
    }
    let m = f.tensor("move.m", true);
    check(&f, "move.transpose", &m.transpose(), &[("x", &m, None)]);
}

/// Batched matmul with broadcast batch dims.
fn matmul<H: Half>() {
    let f = Fixture::<H>::load();
    let (a, b) = (f.tensor("matmul.a", true), f.tensor("matmul.b", true));
    check(
        &f,
        "matmul",
        &a.matmul(&b),
        &[("a", &a, None), ("b", &b, None)],
    );
}

/// `mulacc` (torch's `addcmul`) and `where`.
fn ternary<H: Half>() {
    let f = Fixture::<H>::load();
    let (x, y, z) = (
        f.tensor("ternary.x", true),
        f.tensor("ternary.y", true),
        f.tensor("ternary.z", true),
    );
    check(
        &f,
        "ternary.mulacc",
        &x.mulacc(&y, &z),
        &[("x", &x, None), ("y", &y, None), ("z", &z, None)],
    );
    let (y, z) = (f.tensor("ternary.y", true), f.tensor("ternary.z", true));
    let cond = f.tensor("ternary.cond", false).to_dtype(DType::Bool);
    check(
        &f,
        "ternary.where",
        &cond.where_op(&y, &z),
        &[("y", &y, None), ("z", &z, None)],
    );
}

/// MSE and cross-entropy (one-hot targets).
fn losses<H: Half>() {
    let f = Fixture::<H>::load();
    let target = f.tensor("loss.target", false);
    let p = f.tensor("loss.pred", true);
    check(
        &f,
        "loss.mse",
        &RawTensor::mse_loss(&p, &target),
        &[("pred", &p, None)],
    );
    let onehot = f.tensor("loss.onehot", false);
    let p = f.tensor("loss.pred", true);
    check(
        &f,
        "loss.cross_entropy",
        &RawTensor::cross_entropy_loss(&p, &onehot),
        &[("pred", &p, None)],
    );
}

/// `Linear`, whose weight Volta stores transposed (`[in, out]`).
fn linear<H: Half>() {
    let f = Fixture::<H>::load();
    let mut l = Linear::new(2, 3, true);
    l.to_dtype(H::DTYPE);
    l.load_state_dict(&state(&[
        (
            "weight",
            transpose(&f.f32s("linear.weight"), 3, 2),
            vec![2, 3],
        ),
        ("bias", f.f32s("linear.bias"), vec![3]),
    ]))
    .expect("load");
    let x = f.tensor("linear.x", true);
    let out = l.forward(&x);
    let params = l.parameters();
    let p = pair(&params);
    let back = |g: Vec<f64>| transpose(&g, 2, 3);
    check(
        &f,
        "linear",
        &out,
        &[
            ("x", &x, None),
            ("weight", p.0, Some(&back)),
            ("bias", p.1, None),
        ],
    );
}

/// `Conv2d` with each of its three algorithms.
fn conv2d<H: Half>() {
    let f = Fixture::<H>::load();
    for algo in [ConvAlgo::Direct, ConvAlgo::Im2col, ConvAlgo::IGEMM] {
        let mut c = Conv2d::new(2, 3, 3, 2, 1, true);
        c.set_algo(algo);
        c.to_dtype(H::DTYPE);
        c.load_state_dict(&state(&[
            ("weight", f.f32s("conv.weight"), vec![3, 2, 3, 3]),
            ("bias", f.f32s("conv.bias"), vec![3]),
        ]))
        .expect("load");
        let x = f.tensor("conv.x", true);
        let out = c.forward(&x);
        let params = c.parameters();
        let p = pair(&params);
        check(
            &f,
            "conv",
            &out,
            &[("x", &x, None), ("weight", p.0, None), ("bias", p.1, None)],
        );
    }
}

/// `ConvTranspose2d`, stride 2, padding 1.
fn conv_transpose2d<H: Half>() {
    let f = Fixture::<H>::load();
    let mut c = ConvTranspose2d::new(2, 3, 3, 2, 1, true);
    c.to_dtype(H::DTYPE);
    c.load_state_dict(&state(&[
        ("weight", f.f32s("convt.weight"), vec![2, 3, 3, 3]),
        ("bias", f.f32s("convt.bias"), vec![3]),
    ]))
    .expect("load");
    let x = f.tensor("convt.x", true);
    let out = c.forward(&x);
    let params = c.parameters();
    let p = pair(&params);
    check(
        &f,
        "convt",
        &out,
        &[("x", &x, None), ("weight", p.0, None), ("bias", p.1, None)],
    );
}

/// `MaxPool2d`, `BatchNorm2d` (training mode), `LayerNorm` and `Embedding`.
fn other_layers<H: Half>() {
    let f = Fixture::<H>::load();
    let x = f.tensor("maxpool.x", true);
    let out = MaxPool2d::new(2, 2, 0).forward(&x);
    check(&f, "maxpool", &out, &[("x", &x, None)]);

    let mut bn = BatchNorm2d::new(2);
    bn.to_dtype(H::DTYPE);
    bn.load_state_dict(&state(&[
        ("gamma", f.f32s("batchnorm.gamma"), vec![2]),
        ("beta", f.f32s("batchnorm.beta"), vec![2]),
    ]))
    .expect("load");
    let x = f.tensor("batchnorm.x", true);
    let out = bn.forward(&x);
    let params = bn.parameters();
    let p = pair(&params);
    check(
        &f,
        "batchnorm",
        &out,
        &[("x", &x, None), ("gamma", p.0, None), ("beta", p.1, None)],
    );

    let mut l = LayerNorm::new(vec![4]);
    l.to_dtype(H::DTYPE);
    l.load_state_dict(&state(&[
        ("gamma", f.f32s("layernorm.gamma"), vec![4]),
        ("beta", f.f32s("layernorm.beta"), vec![4]),
    ]))
    .expect("load");
    let x = f.tensor("layernorm.x", true);
    let out = l.forward(&x);
    let params = l.parameters();
    let p = pair(&params);
    check(
        &f,
        "layernorm",
        &out,
        &[("x", &x, None), ("gamma", p.0, None), ("beta", p.1, None)],
    );

    let mut e = Embedding::new(5, 3);
    e.to_dtype(H::DTYPE);
    e.load_state_dict(&state(&[(
        "weight",
        f.f32s("embedding.weight"),
        vec![5, 3],
    )]))
    .expect("load");
    let out = e.forward(&RawTensor::from_typed(vec![1i64, 3, 1], &[3]));
    let params = e.parameters();
    let [w] = params.as_slice() else {
        panic!("Embedding has one parameter")
    };
    check(&f, "embedding", &out, &[("weight", w, None)]);
}

/// Three steps of SGD with momentum and of Adam with L2 weight decay under fixed
/// gradients. Volta steps in f32 and rounds parameter and state once per step; torch
/// rounds after each of its in-place ops (f32 master weights are P7).
fn optimizers<H: Half>() {
    let f = Fixture::<H>::load();
    for name in ["sgd", "adam"] {
        let p = f.tensor(&format!("{name}.init"), true);
        let mut opt: Box<dyn FnMut()> = if name == "sgd" {
            let mut o = SGD::new(vec![p.clone()], 2f32.powi(-4), 0.875, 0.0);
            Box::new(move || o.step())
        } else {
            let mut o = Adam::new(
                vec![p.clone()],
                2f32.powi(-7),
                (0.875, 1.0 - 2f32.powi(-10)),
                2f32.powi(-14),
                2f32.powi(-7),
            );
            Box::new(move || o.step())
        };
        for k in 0..3 {
            p.set_grad(Storage::from_typed(f.get(&format!("{name}.g{k}")).0));
            opt();
            assert_close::<H>(
                &format!("{name}.after{k}"),
                &read::<H>(&p),
                &f.values(&format!("{name}.after{k}")),
            );
        }
    }
}

/// One `#[test]` per family and dtype.
macro_rules! families {
    ($($fam:ident),* $(,)?) => {
        mod f16_tests {
            $(#[test] fn $fam() { super::$fam::<half::f16>(); })*
        }
        mod bf16_tests {
            $(#[test] fn $fam() { super::$fam::<half::bf16>(); })*
        }
    };
}

families!(
    unary,
    binary,
    reductions,
    movement,
    matmul,
    ternary,
    losses,
    linear,
    conv2d,
    conv_transpose2d,
    other_layers,
    optimizers,
);

/// The half fixture really holds half data: a smoke check that the loader would catch a
/// regenerated file stored as f32.
#[test]
fn fixture_is_stored_in_half_dtypes() {
    let f = Fixture::<f16>::load();
    assert_eq!(f.all["f16.unary.x"].dtype, DType::F16);
    assert_eq!(f.all["bf16.unary.x"].dtype, DType::BF16);
}

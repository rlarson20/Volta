//! I64 and Bool tensors as data, against PyTorch (`tests/fixtures/torch/intbool.safetensors`,
//! generator group `intbool`): the six comparisons on floats (ties and a NaN) and on i64
//! beyond 2^40, `where` with a bool condition (float branches with gradients, i64 branches),
//! cross-entropy and NLL with i64 class targets, `Embedding` with 2-D i64 indices, and the
//! movement ops on i64 and bool. Bool, integer and movement results must be exact; float
//! values match at 1e-6 relative (f32, one rounding order apart at most).

use std::collections::BTreeMap;
use volta::io::{TypedTensorData, load_safetensors_raw};
use volta::storage::Element;
use volta::{Axis, DType, Embedding, Layer, Module, PadValue, RawTensor, Tensor};

/// The `intbool` fixture group.
struct Fixture(BTreeMap<String, TypedTensorData>);

impl Fixture {
    /// Load the fixture, or panic with the regeneration command.
    fn load() -> Self {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/torch/intbool.safetensors"
        );
        Self(
            load_safetensors_raw(path)
                .unwrap_or_else(|e| panic!("{path}: {e}; regenerate with `just torch-fixtures`")),
        )
    }

    /// The `T` values and shape stored under `name`.
    fn get<T: Element>(&self, name: &str) -> (Vec<T>, Vec<usize>) {
        let t = self
            .0
            .get(name)
            .unwrap_or_else(|| panic!("intbool fixture: no tensor {name:?}"));
        assert_eq!(t.dtype, T::DTYPE, "{name}: stored dtype");
        let s = t.to_storage().expect("valid bytes");
        (
            s.as_host::<T>().expect("dtype checked").to_vec(),
            t.shape.clone(),
        )
    }

    /// The values under `name`.
    fn values<T: Element>(&self, name: &str) -> Vec<T> {
        self.get(name).0
    }

    /// A leaf tensor holding `name`, in its stored dtype.
    fn tensor<T: Element>(&self, name: &str) -> Tensor {
        let (v, shape) = self.get::<T>(name);
        RawTensor::from_typed(v, &shape)
    }

    /// An f32 leaf that requires grad.
    fn leaf(&self, name: &str) -> Tensor {
        let t = self.tensor::<f32>(name);
        t.set_requires_grad(true);
        t
    }
}

/// Asserts `got` matches `want` to 1e-6 relative (floor 1).
fn assert_close(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-6 * w.abs().max(1.0),
            "{label}[{i}]: volta {g}, torch {w}"
        );
    }
}

/// `t` as `T`, which must be its dtype.
fn read<T: Element>(t: &Tensor) -> Vec<T> {
    assert_eq!(t.dtype(), T::DTYPE, "dtype");
    t.to_vec_as::<T>().expect("dtype checked")
}

/// A binary op under test.
type BinOp = fn(&Tensor, &Tensor) -> Tensor;

/// All six comparisons, on f32 (with ties and a NaN) and on i64, with broadcasting.
#[test]
fn comparisons_match_torch() {
    let f = Fixture::load();
    let ops: [(&str, BinOp); 6] = [
        ("lt", Tensor::lt),
        ("le", Tensor::le),
        ("gt", Tensor::gt),
        ("ge", Tensor::ge),
        ("eq", Tensor::elem_eq),
        ("ne", Tensor::elem_ne),
    ];
    let (a, b) = (f.tensor::<f32>("cmp.a"), f.tensor::<f32>("cmp.b"));
    let (ia, ib) = (f.tensor::<i64>("cmp.ia"), f.tensor::<i64>("cmp.ib"));
    for (name, op) in ops {
        assert_eq!(
            read::<bool>(&op(&a, &b)),
            f.values::<bool>(&format!("cmp.{name}")),
            "{name}"
        );
        assert_eq!(
            read::<bool>(&op(&ia, &ib)),
            f.values::<bool>(&format!("cmp.int_{name}")),
            "int {name}"
        );
    }
}

/// `where` with a broadcast bool condition: float branches with their gradients, and i64
/// branches (exact).
#[test]
fn where_matches_torch() {
    let f = Fixture::load();
    let cond = f.tensor::<bool>("where.cond");
    let (y, z) = (f.leaf("where.y"), f.leaf("where.z"));
    let out = cond.where_op(&y, &z);
    assert_close(
        "where.out",
        &read::<f32>(&out),
        &f.values::<f32>("where.out"),
    );
    out.elem_mul(&f.tensor::<f32>("where.w"))
        .sum()
        .into_loss()
        .expect("scalar")
        .backward();
    assert_close(
        "where.grad_y",
        &y.grad().expect("grad"),
        &f.values::<f32>("where.grad_y"),
    );
    assert_close(
        "where.grad_z",
        &z.grad().expect("grad"),
        &f.values::<f32>("where.grad_z"),
    );

    let iy = f.tensor::<i64>("where.iy");
    let minus_one = RawTensor::from_typed(vec![-1i64], &[1]);
    assert_eq!(
        read::<i64>(&cond.where_op(&iy, &minus_one)),
        f.values::<i64>("where.int_out")
    );
}

/// Cross-entropy and NLL with i64 class targets: the mean loss and its gradient.
#[test]
fn class_index_losses_match_torch() {
    let f = Fixture::load();
    let target = f.tensor::<i64>("ce.target");
    for (name, input, loss) in [
        ("ce", "logits", RawTensor::cross_entropy_loss as BinOp),
        ("nll", "logp", RawTensor::nll_loss as BinOp),
    ] {
        let x = f.leaf(&format!("{name}.{input}"));
        let out = loss(&x, &target);
        assert_close(
            &format!("{name}.out"),
            &[out.item()],
            &f.values::<f32>(&format!("{name}.out")),
        );
        out.into_loss().expect("scalar").backward();
        assert_close(
            &format!("{name}.grad_{input}"),
            &x.grad().expect("grad"),
            &f.values::<f32>(&format!("{name}.grad_{input}")),
        );
    }
}

/// `Embedding` as a `Layer` on 2-D i64 indices with repeats: output `[2, 3, dim]` and the
/// accumulated weight gradient.
#[test]
fn embedding_with_2d_indices_matches_torch() {
    let f = Fixture::load();
    let mut e = Embedding::new(6, 3);
    let (w, _) = f.get::<f32>("embedding.weight");
    e.load_state_dict(
        &[(
            "weight".to_string(),
            volta::io::TensorData::from_f32(w, vec![6, 3]),
        )]
        .into_iter()
        .collect(),
    )
    .expect("load");
    let out = e.forward(&f.tensor::<i64>("embedding.idx"));
    assert_eq!(out.shape(), vec![2, 3, 3]);
    assert_close(
        "embedding.out",
        &read::<f32>(&out),
        &f.values::<f32>("embedding.out"),
    );
    out.elem_mul(&f.tensor::<f32>("embedding.w"))
        .sum()
        .into_loss()
        .expect("scalar")
        .backward();
    let params = e.parameters();
    let [weight] = params.as_slice() else {
        panic!("one parameter")
    };
    assert_close(
        "embedding.grad_weight",
        &weight.grad().expect("grad"),
        &f.values::<f32>("embedding.grad_weight"),
    );
}

/// The movement ops copy i64 (beyond 2^53) and bool exactly, as torch does.
#[test]
fn int_and_bool_movement_matches_torch() {
    /// Every case on one tensor, against `move.<name>.<case>`.
    fn cases<T: Element + PartialEq + std::fmt::Debug>(f: &Fixture, name: &str) {
        let x = f.tensor::<T>(&format!("move.{name}"));
        let first = x.shrink(&[(0, 1), (0, 3), (0, 4)]).reshape(&[3, 4]);
        let got = [
            ("permute", x.permute(&[Axis(2), Axis(0), Axis(1)])),
            ("shrink", x.shrink(&[(0, 2), (1, 3), (0, 2)])),
            (
                "expand",
                x.shrink(&[(0, 2), (0, 1), (0, 4)]).expand(&[2, 5, 4]),
            ),
            ("transpose", first.transpose()),
            ("pad", first.pad_with(&[(0, 0), (1, 2)], PadValue::ZERO)),
        ];
        for (case, t) in got {
            assert_eq!(
                read::<T>(&t),
                f.values::<T>(&format!("move.{name}.{case}")),
                "{name}.{case}"
            );
        }
    }
    let f = Fixture::load();
    cases::<i64>(&f, "x");
    cases::<bool>(&f, "m");
    assert_eq!(f.tensor::<bool>("move.m").dtype(), DType::Bool);
}

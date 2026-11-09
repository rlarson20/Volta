//! Implicit promotion against PyTorch (`tests/fixtures/torch/promotion.safetensors`,
//! generator group `promotion`): for each of the 64 ordered pairs of Volta's dtypes,
//! `add`, `mul`, `div`, `lt` and `where` give torch's result dtype and values, and for
//! float pairs the gradients of `sum(a * b * w)` come back in each operand's own dtype
//! with torch's values. All comparisons are exact: the operands are small integers,
//! representable in every dtype.
//!
//! Where torch's result is an integer or bool dtype for `add`/`mul` (`i64 + i64`,
//! `bool + bool`), Volta has no integer arithmetic kernels (issue #75): the test
//! checks that the op panics naming the missing kernel, rather than skipping the pair.

use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use volta::io::{TypedTensorData, load_safetensors_raw};
use volta::{DType, RawTensor, Storage, Tensor};

/// Volta's dtypes, in the generator's order and naming.
const DTYPES: [(&str, DType); 8] = [
    ("f16", DType::F16),
    ("bf16", DType::BF16),
    ("f32", DType::F32),
    ("f64", DType::F64),
    ("i32", DType::I32),
    ("i64", DType::I64),
    ("u8", DType::U8),
    ("bool", DType::Bool),
];

/// The `promotion` fixture group.
struct Fixture(BTreeMap<String, TypedTensorData>);

impl Fixture {
    /// Load the fixture, or panic with the regeneration command.
    fn load() -> Self {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/torch/promotion.safetensors"
        );
        Self(
            load_safetensors_raw(path)
                .unwrap_or_else(|e| panic!("{path}: {e}; regenerate with `just torch-fixtures`")),
        )
    }

    /// The storage stored under `name`.
    fn storage(&self, name: &str) -> Storage {
        self.0
            .get(name)
            .unwrap_or_else(|| panic!("promotion fixture: no tensor {name:?}"))
            .to_storage()
            .expect("valid bytes")
    }

    /// The f32 tensor stored under `name`.
    fn f32_tensor(&self, name: &str) -> Tensor {
        let s = self.storage(name);
        RawTensor::new(s.as_host::<f32>().expect("f32").to_vec(), &[s.len()], false)
    }
}

/// `s` decoded exactly to f64 (every value in this fixture is exact in f64).
fn wide(s: &Storage) -> Vec<f64> {
    s.cast(DType::F64).as_host::<f64>().expect("f64").to_vec()
}

/// Asserts `got` has `want`'s dtype and values.
fn assert_same(label: &str, got: &Tensor, want: &Storage) {
    assert_eq!(got.dtype(), want.dtype(), "{label}: result dtype");
    let got = got.to_vec_as::<f64>().unwrap_or_else(|_| {
        let d = got.to_dtype(DType::F64);
        d.to_vec_as::<f64>().expect("f64")
    });
    assert_eq!(got, wide(want), "{label}: values");
}

/// A binary op under test.
type BinOp = fn(&Tensor, &Tensor) -> Tensor;

/// Every op on every ordered dtype pair.
#[test]
fn every_dtype_pair_matches_torch() {
    let f = Fixture::load();
    let (a, b) = (f.f32_tensor("a"), f.f32_tensor("b"));
    let cond = {
        let s = f.storage("cond");
        RawTensor::from_typed(s.as_host::<bool>().expect("bool").to_vec(), &[s.len()])
    };
    let ops: [(&str, BinOp); 4] = [
        ("add", Tensor::add),
        ("mul", Tensor::elem_mul),
        ("div", Tensor::div),
        ("lt", Tensor::lt),
    ];
    for (na, da) in DTYPES {
        for (nb, db) in DTYPES {
            let key = format!("{na}.{nb}");
            let (x, y) = (a.to_dtype(da), b.to_dtype(db));
            for (op, f_op) in ops {
                let want = f.storage(&format!("{op}.{key}"));
                let label = format!("{op}({na}, {nb})");
                if want.dtype().is_float() || want.dtype() == DType::Bool && op == "lt" {
                    assert_same(&label, &f_op(&x, &y), &want);
                } else {
                    let err = catch_unwind(AssertUnwindSafe(|| f_op(&x, &y)))
                        .expect_err(&format!("{label}: integer arithmetic has no kernel"));
                    let msg = err.downcast_ref::<String>().map_or("", String::as_str);
                    assert!(
                        msg.contains(&format!("no {} kernel", want.dtype())),
                        "{label}: {msg}"
                    );
                }
            }
            assert_same(
                &format!("where({na}, {nb})"),
                &cond.where_op(&x, &y),
                &f.storage(&format!("where.{key}")),
            );
        }
    }
}

/// For every float pair, the gradients of `sum(a * b * w)` have each operand's dtype and
/// torch's values: promotion's casts send each gradient back through `to_dtype`.
#[test]
fn float_pair_gradients_keep_each_operand_dtype() {
    let f = Fixture::load();
    let (a, b, w) = (f.f32_tensor("a"), f.f32_tensor("b"), f.f32_tensor("w"));
    for (na, da) in DTYPES.into_iter().filter(|(_, d)| d.is_float()) {
        for (nb, db) in DTYPES.into_iter().filter(|(_, d)| d.is_float()) {
            let key = format!("{na}.{nb}");
            let (x, y) = (a.to_dtype(da).detach(), b.to_dtype(db).detach());
            x.set_requires_grad(true);
            y.set_requires_grad(true);
            let wp = w.to_dtype(DType::promote(da, db));
            let prod = x.elem_mul(&y);
            assert_eq!(prod.dtype(), DType::promote(da, db), "{na} * {nb}: dtype");
            prod.elem_mul(&wp)
                .sum()
                .into_loss()
                .expect("scalar")
                .backward();
            for (g, t) in [("grad_a", &x), ("grad_b", &y)] {
                let got = t.grad_storage().expect("a gradient");
                let want = f.storage(&format!("{g}.{key}"));
                assert_eq!(got.dtype(), want.dtype(), "{g}({na}, {nb}): dtype");
                assert_eq!(wide(&got), wide(&want), "{g}({na}, {nb}): values");
            }
        }
    }
}

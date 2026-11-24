//! Shared helpers for the `torch_ref_*` tests: load a PyTorch reference fixture and compare
//! Volta's results against it.
//!
//! The fixtures in `tests/fixtures/torch/` are written by `scripts/torch_ref/gen_fixtures.py`
//! (`just torch-fixtures`). They hold f32 inputs, PyTorch's forward outputs, and PyTorch's
//! gradients, so a test here checks Volta against an outside library, which the CPU/GPU
//! parity tests cannot do.

#![allow(dead_code)] // each test file uses only a subset

use std::path::PathBuf;
use volta::io::{StateDict, load_safetensors};
use volta::{RawTensor, Tensor};

/// Tolerance for forward values that are one or two f32 operations from the input.
pub const TOL_FWD: f32 = 1e-5;
/// Tolerance for gradients.
pub const TOL_BWD: f32 = 1e-4;

/// One fixture group, loaded from `tests/fixtures/torch/<group>.safetensors`.
pub struct Fixture {
    /// Group name, for failure messages.
    group: String,
    /// The tensors, by name.
    tensors: StateDict,
}

impl Fixture {
    /// Load the fixture group `group`. Panics with the regeneration command if missing.
    #[must_use]
    pub fn load(group: &str) -> Self {
        let path: PathBuf = [
            env!("CARGO_MANIFEST_DIR"),
            "tests",
            "fixtures",
            "torch",
            &format!("{group}.safetensors"),
        ]
        .iter()
        .collect();
        let tensors = load_safetensors(&path).unwrap_or_else(|e| {
            panic!("fixture {path:?}: {e}; regenerate with `just torch-fixtures`")
        });
        Self {
            group: group.to_string(),
            tensors,
        }
    }

    /// The values and shape stored under `name`.
    #[must_use]
    pub fn get(&self, name: &str) -> (Vec<f32>, Vec<usize>) {
        let t = self
            .tensors
            .get(name)
            .unwrap_or_else(|| panic!("fixture {}: no tensor {name:?}", self.group));
        // Fixtures are stored in their own dtype; this reader decodes to f32.
        let values = t
            .to_storage(name)
            .unwrap_or_else(|e| panic!("fixture {}: {e}", self.group))
            .to_f32_vec();
        (values, t.shape.clone())
    }

    /// Whether the fixture holds a tensor called `name`.
    #[must_use]
    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// The values stored under `name`.
    #[must_use]
    pub fn values(&self, name: &str) -> Vec<f32> {
        self.get(name).0
    }

    /// A leaf tensor holding the fixture tensor `name`.
    #[must_use]
    pub fn tensor(&self, name: &str, requires_grad: bool) -> Tensor {
        let (data, shape) = self.get(name);
        RawTensor::new(data, &shape, requires_grad)
    }
}

/// Assert that `actual` matches `expected` elementwise within `tol` (absolute, or relative
/// for magnitudes above 1). A NaN matches only a NaN, and an infinity only the same
/// infinity.
///
/// # Panics
/// On a length mismatch or the first element outside the tolerance, naming `label` and
/// both values.
pub fn assert_close(label: &str, actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length {} (volta) vs {} (torch)",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let ok = if e.is_nan() || a.is_nan() {
            e.is_nan() && a.is_nan()
        } else if e.is_infinite() || a.is_infinite() {
            a == e
        } else {
            (a - e).abs() <= tol * e.abs().max(1.0)
        };
        assert!(
            ok,
            "{label}[{i}]: volta {a} vs torch {e} (tol {tol})\n  volta: {actual:?}\n  torch: {expected:?}"
        );
    }
}

/// Volta's gradient of `t`, or a panic naming `label` if none was computed.
#[must_use]
pub fn grad_of(label: &str, t: &Tensor) -> Vec<f32> {
    t.grad()
        .unwrap_or_else(|| panic!("{label}: no gradient on the input"))
}

/// Backpropagate `sum(out * weights)` from `out` and return nothing; read gradients from
/// the leaves afterwards. `weights` is `None` for a plain sum.
pub fn backward_weighted(out: &Tensor, weights: Option<&Tensor>) {
    let scalar = weights.map_or_else(|| out.sum(), |w| out.elem_mul(w).sum());
    scalar.into_loss().expect("scalar loss").backward();
}

/// Transpose a row-major `[rows, cols]` matrix.
#[must_use]
pub fn transpose(v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(
        v.len(),
        rows * cols,
        "transpose: {} values for {rows}x{cols}",
        v.len()
    );
    (0..cols)
        .flat_map(|c| {
            (0..rows).map(move |r| {
                v.get(r * cols + c)
                    .copied()
                    .expect("index within rows * cols")
            })
        })
        .collect()
}

/// A state dict built from `(key, values, shape)` triples, for loading torch's weights into a
/// layer with `Module::load_state_dict`.
#[must_use]
pub fn state_of(entries: &[(&str, Vec<f32>, Vec<usize>)]) -> StateDict {
    entries
        .iter()
        .map(|(k, data, shape)| {
            (
                (*k).to_string(),
                volta::io::TensorData::from_f32(data.clone(), shape.clone()),
            )
        })
        .collect()
}

/// A gradient to compare in [`check_case`]: the fixture's gradient name, the Volta tensor
/// holding it, and an optional fix-up applied to Volta's gradient first (for a transposed layout).
pub type GradCheck<'a> = (
    &'a str,
    &'a Tensor,
    Option<&'a dyn Fn(Vec<f32>) -> Vec<f32>>,
);

/// Compare a case recorded by the generator's `case()` helper: backpropagate
/// `sum(out * w)` and compare the output and each named input's gradient against torch.
///
/// `grads` pairs the fixture's gradient name with the Volta tensor holding it and an optional
/// fix-up applied to Volta's gradient before comparing (for a transposed weight layout).
pub fn check_case(
    f: &Fixture,
    name: &str,
    out: &Tensor,
    grads: &[GradCheck<'_>],
    tol_fwd: f32,
    tol_bwd: f32,
) {
    assert_close(
        &format!("{name}.out"),
        &out.data(),
        &f.values(&format!("{name}.out")),
        tol_fwd,
    );
    let (w, shape) = f.get(&format!("{name}.w"));
    let w = RawTensor::new(w, &shape, false);
    backward_weighted(out, Some(&w));
    for (gname, tensor, fix) in grads {
        let mut got = grad_of(&format!("{name}.{gname}"), tensor);
        if let Some(fix) = fix {
            got = fix(got);
        }
        assert_close(
            &format!("{name}.grad_{gname}"),
            &got,
            &f.values(&format!("{name}.grad_{gname}")),
            tol_bwd,
        );
    }
}

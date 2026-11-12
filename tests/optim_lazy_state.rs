//! S17: optimizer state is allocated on a parameter's first step, so a model converted with
//! `to_dtype` after its optimizer was built is stepped exactly like one converted before.
//! The device half is in `tests/gpu_sweep.rs`.
//!
//! Adam also counts steps per parameter; that trajectory is checked against PyTorch in
//! `tests/torch_ref_optim.rs`.

use volta::{Adam, DType, Linear, Module, Muon, Optimizer, SGD, Storage};

/// Builds an optimizer over a layer's parameters.
type Build = fn(Vec<volta::Tensor>) -> Box<dyn Optimizer>;

/// Each optimizer that keeps state, with fixed hyperparameters.
const BUILDERS: [(&str, Build); 3] = [
    ("SGD", |p| Box::new(SGD::new(p, 0.1, 0.9, 0.0))),
    ("Adam", |p| {
        Box::new(Adam::new(p, 0.01, (0.9, 0.999), 1e-8, 0.0))
    }),
    ("Muon", |p| Box::new(Muon::new(p, 0.02, 0.95, true, 5))),
];

/// A distinct, non-degenerate f64 gradient for `p`.
fn grad_f64(len: usize) -> Storage {
    let g: Vec<f32> = (0..len)
        .map(|i| 0.3 + 0.17 * (i % 5) as f32 - 0.05 * i as f32)
        .collect();
    Storage::cpu(g).cast(DType::F64)
}

/// The layer's weights as f64.
fn weights_f64(layer: &Linear) -> Vec<f64> {
    layer.weight.to_vec_as::<f64>().expect("f64 weights")
}

/// Two steps of `opt` with the same f64 gradients; returns the layer's weights afterwards.
fn run(layer: &Linear, opt: &mut dyn Optimizer) -> Vec<f64> {
    let len = layer.weight.data().len();
    for _ in 0..2 {
        layer.weight.set_grad(grad_f64(len));
        opt.step();
    }
    weights_f64(layer)
}

#[test]
fn optimizer_built_before_to_dtype_steps_like_one_built_after() {
    for (name, build) in BUILDERS {
        let src = Linear::new(4, 3, false);

        let mut before = Linear::new(4, 3, false);
        before.load_state_dict(&src.state_dict()).expect("load");
        let mut opt_before = build(before.parameters());
        before.to_dtype(DType::F64);
        let got = run(&before, opt_before.as_mut());

        let mut after = Linear::new(4, 3, false);
        after.load_state_dict(&src.state_dict()).expect("load");
        after.to_dtype(DType::F64);
        let mut opt_after = build(after.parameters());
        let want = run(&after, opt_after.as_mut());

        assert_eq!(got.len(), want.len(), "{name}");
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "{name} weight {i}: {g} != {w}");
        }
        // Not a vacuous pass: the optimizer moved the weights.
        let start = src.weight.to_vec_as::<f32>().expect("f32 weights");
        assert_eq!(got.len(), start.len(), "{name}");
        assert!(
            got.iter()
                .zip(&start)
                .any(|(g, s)| (g - f64::from(*s)).abs() > 1e-4),
            "{name}: weights did not move"
        );
    }
}

/// Converting after the first step would need the state converted too, which the optimizer
/// does not do (it would reset momentum); the mismatch stays a loud panic.
#[test]
#[should_panic(expected = "first step")]
fn to_dtype_after_the_first_step_still_panics() {
    let mut layer = Linear::new(4, 3, false);
    let mut opt = SGD::new(layer.parameters(), 0.1, 0.9, 0.0);
    layer.weight.set_grad(Storage::cpu(vec![1.0; 12]));
    opt.step();
    layer.to_dtype(DType::F64);
    layer.weight.set_grad(grad_f64(12));
    opt.step();
}

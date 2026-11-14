//! `LSTMCell` against PyTorch's `torch.nn.LSTMCell`, with torch's weights loaded
//! (`tests/fixtures/torch/lstm.safetensors`, #55).
//!
//! Both libraries store the weights as `[4H, in]` / `[4H, H]` with the gates in the order
//! i, f, g, o, so torch's state dict loads unchanged. The cell runs three steps from a
//! random, non-zero `(h0, c0)`. The loss is `sum_t sum(h_t * w_t) + sum(c_T * w_c)`, so the
//! test compares every step's output, the final cell state, and the gradients of every
//! step's input, the initial state and every parameter, all of which flow through the
//! carried state.

#[path = "torch_ref/mod.rs"]
mod torch_ref;

use torch_ref::{Fixture, TOL_BWD, TOL_FWD, assert_close, grad_of};
use volta::io::StateDict;
use volta::nn::{LSTMCell, RecurrentLayer};
use volta::{Module, RawTensor};

/// Input width of the fixture cell.
const I: usize = 3;
/// Hidden width of the fixture cell.
const H: usize = 4;
/// Timesteps the fixture runs.
const T: usize = 3;
/// The parameter names, shared by torch and Volta, in `Module::parameters` order.
const PARAMS: [&str; 4] = ["weight_ih", "weight_hh", "bias_ih", "bias_hh"];

/// torch's weights as a Volta state dict.
fn torch_state(f: &Fixture) -> StateDict {
    let entries: Vec<(&str, Vec<f32>, Vec<usize>)> = PARAMS
        .iter()
        .map(|&n| {
            let (v, shape) = f.get(&format!("lstm.{n}"));
            (n, v, shape)
        })
        .collect();
    torch_ref::state_of(&entries)
}

#[test]
fn lstm_cell_matches_torch_over_three_steps() {
    let f = Fixture::load("lstm");
    let mut cell = LSTMCell::new(I, H, true);
    cell.load_state_dict(&torch_state(&f))
        .expect("torch weights load");
    let params = cell.parameters();
    assert_eq!(params.len(), PARAMS.len(), "parameter count");

    let xs: Vec<_> = (0..T)
        .map(|k| f.tensor(&format!("lstm.x{k}"), true))
        .collect();
    let h0 = f.tensor("lstm.h_init", true);
    let c0 = f.tensor("lstm.c_init", true);
    let mut state = (h0.clone(), c0.clone());

    let mut loss: Option<volta::Tensor> = None;
    for (k, x) in xs.iter().enumerate() {
        let h = cell.step(x, &mut state);
        assert_close(
            &format!("lstm.out{k}"),
            &h.data(),
            &f.values(&format!("lstm.out{k}")),
            TOL_FWD,
        );
        let (w, shape) = f.get(&format!("lstm.w{k}"));
        let term = h.elem_mul(&RawTensor::new(w, &shape, false)).sum();
        loss = Some(match loss {
            Some(l) => l.add(&term),
            None => term,
        });
    }
    let c_last = &state.1;
    assert_close(
        "lstm.c_last",
        &c_last.data(),
        &f.values("lstm.c_last"),
        TOL_FWD,
    );
    let (wc, shape) = f.get("lstm.w_c");
    let loss = loss
        .expect("at least one step")
        .add(&c_last.elem_mul(&RawTensor::new(wc, &shape, false)).sum());
    loss.into_loss().expect("scalar loss").backward();

    for (k, x) in xs.iter().enumerate() {
        let name = format!("lstm.grad_x{k}");
        assert_close(&name, &grad_of(&name, x), &f.values(&name), TOL_BWD);
    }
    for (name, t) in [("lstm.grad_h_init", &h0), ("lstm.grad_c_init", &c0)] {
        assert_close(name, &grad_of(name, t), &f.values(name), TOL_BWD);
    }
    for (n, p) in PARAMS.iter().zip(&params) {
        let name = format!("lstm.grad_{n}");
        assert_close(&name, &grad_of(&name, p), &f.values(&name), TOL_BWD);
    }
}

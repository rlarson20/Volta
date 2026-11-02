//! `SGD`, `Adam`, and `Muon` trajectories against PyTorch: five steps with fixed
//! gradients, comparing every parameter after every step
//! (`tests/fixtures/torch/optim.safetensors`).

#[path = "torch_ref/mod.rs"]
mod torch_ref;

use torch_ref::{Fixture, assert_close};
use volta::{Adam, Muon, SGD, Storage, Tensor};

/// Tolerance for plain arithmetic trajectories.
const TOL: f32 = 1e-5;

#[test]
fn sgd_plain_matches_torch() {
    let f = Fixture::load("optim");
    let p = f.tensor("sgd_plain.p0.init", true);
    let mut opt = SGD::new(vec![p.clone()], 0.1, 0.0, 0.0);
    drive(&f, "sgd_plain", &[p], || opt.step(), TOL);
}

#[test]
fn sgd_momentum_and_weight_decay_match_torch() {
    let f = Fixture::load("optim");
    let p = f.tensor("sgd_mom_wd.p0.init", true);
    let mut opt = SGD::new(vec![p.clone()], 0.1, 0.9, 0.01);
    drive(&f, "sgd_mom_wd", &[p], || opt.step(), TOL);
}

#[test]
fn adam_with_weight_decay_matches_torch() {
    let f = Fixture::load("optim");
    let p = f.tensor("adam.p0.init", true);
    let mut opt = Adam::new(vec![p.clone()], 0.01, (0.9, 0.999), 1e-8, 0.01);
    drive(&f, "adam", &[p], || opt.step(), TOL);
}

#[test]
#[ignore = "S17: Adam keeps one global step count, so a parameter whose first gradient arrives at step 3 is bias-corrected with t=3; torch uses t=1"]
fn adam_counts_steps_per_parameter_like_torch() {
    let f = Fixture::load("optim");
    let ps = vec![
        f.tensor("adam_late.p0.init", true),
        f.tensor("adam_late.p1.init", true),
    ];
    let mut opt = Adam::new(ps.clone(), 0.01, (0.9, 0.999), 1e-8, 0.0);
    drive(&f, "adam_late", &ps, || opt.step(), TOL);
}

#[test]
fn muon_wide_matches_torch() {
    let f = Fixture::load("optim");
    let p = f.tensor("muon_wide.p0.init", true);
    let mut opt = Muon::new(vec![p.clone()], 0.02, 0.95, true, 5);
    drive(&f, "muon_wide", &[p], || opt.step(), 1e-4);
}

#[test]
fn muon_tall_matches_torch() {
    let f = Fixture::load("optim");
    let p = f.tensor("muon_tall.p0.init", true);
    let mut opt = Muon::new(vec![p.clone()], 0.02, 0.95, true, 5);
    drive(&f, "muon_tall", &[p], || opt.step(), 1e-4);
}

/// For five steps, set each parameter's fixture gradient (none if the fixture has none for
/// that step), call `step`, and compare every parameter against torch.
fn drive(f: &Fixture, name: &str, params: &[Tensor], mut step: impl FnMut(), tol: f32) {
    for k in 0..5 {
        for (i, p) in params.iter().enumerate() {
            let key = format!("{name}.p{i}.g{k}");
            p.zero_grad();
            if f.has(&key) {
                p.set_grad(Storage::cpu(f.values(&key)));
            }
        }
        step();
        for (i, p) in params.iter().enumerate() {
            let want = f.values(&format!("{name}.p{i}.after{k}"));
            assert_close(
                &format!("{name}.p{i} after step {k}"),
                &p.data(),
                &want,
                tol,
            );
        }
    }
}

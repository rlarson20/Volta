//! `MultiHeadAttention` and `TransformerBlock` against PyTorch, with torch's weights loaded
//! (`tests/fixtures/torch/transformer.safetensors`, S10/S11).
//!
//! The references are `torch.nn.MultiheadAttention` and a Pre-LN
//! `torch.nn.TransformerEncoderLayer` (`norm_first=True`, GELU, no dropout). Neither adds a
//! residual inside attention, so these tests pin that Volta's attention returns only
//! `out_proj(Attn(x))` and the block computes `x + Attn(LN1(x))`, then `+ FFN(LN2(.))`.
//! torch's fused `in_proj_bias` is zeroed, because Volta's q/k/v projections have no bias.
//! Each case compares the output and the gradients of `sum(out * w)` for a random `w`, for
//! the input and for every parameter.

#[path = "torch_ref/mod.rs"]
mod torch_ref;

use torch_ref::{Fixture, check_case, state_of, transpose};
use volta::io::StateDict;
use volta::nn::{MultiHeadAttention, TransformerBlock};
use volta::{Module, Tensor};

/// Model width of every fixture case.
const E: usize = 8;
/// Attention heads of every fixture case.
const H: usize = 2;
/// Hidden width of the block's feed-forward network (`expansion_factor` 2).
const FF: usize = 2 * E;
/// Forward tolerance: softmax, two layer norms and several matmuls deep.
const FWD: f32 = 1e-5;
/// Gradient tolerance.
const BWD: f32 = 1e-4;

/// A Volta parameter and how it maps to the fixture: the Volta state-dict key, the torch
/// name (fixture key `<case>.<torch>` and `<case>.grad_<torch>`), and, for a `Linear`
/// weight, its torch shape `[out, in]` (Volta stores `[in, out]`).
type Param = (&'static str, &'static str, Option<(usize, usize)>);

/// A gradient fix-up: Volta's gradient, rewritten to torch's layout.
type Fix = Box<dyn Fn(Vec<f32>) -> Vec<f32>>;

/// `MultiHeadAttention` parameters in `Module::parameters` order.
const MHA_PARAMS: [Param; 5] = [
    ("q_proj.weight", "q_proj.weight", Some((E, E))),
    ("k_proj.weight", "k_proj.weight", Some((E, E))),
    ("v_proj.weight", "v_proj.weight", Some((E, E))),
    ("out_proj.weight", "out_proj.weight", Some((E, E))),
    ("out_proj.bias", "out_proj.bias", None),
];

/// `TransformerBlock` parameters in `Module::parameters` order: norm1, norm2, ffn, attention.
const BLOCK_PARAMS: [Param; 13] = [
    ("norm1.gamma", "norm1.weight", None),
    ("norm1.beta", "norm1.bias", None),
    ("norm2.gamma", "norm2.weight", None),
    ("norm2.beta", "norm2.bias", None),
    ("ffn.expand.weight", "linear1.weight", Some((FF, E))),
    ("ffn.expand.bias", "linear1.bias", None),
    ("ffn.contract.weight", "linear2.weight", Some((E, FF))),
    ("ffn.contract.bias", "linear2.bias", None),
    (
        "attention.q_proj.weight",
        "self_attn.q_proj.weight",
        Some((E, E)),
    ),
    (
        "attention.k_proj.weight",
        "self_attn.k_proj.weight",
        Some((E, E)),
    ),
    (
        "attention.v_proj.weight",
        "self_attn.v_proj.weight",
        Some((E, E)),
    ),
    (
        "attention.out_proj.weight",
        "self_attn.out_proj.weight",
        Some((E, E)),
    ),
    ("attention.out_proj.bias", "self_attn.out_proj.bias", None),
];

/// torch's weights for `case`, as a Volta state dict.
fn state_for(f: &Fixture, case: &str, params: &[Param]) -> StateDict {
    let entries: Vec<(&str, Vec<f32>, Vec<usize>)> = params
        .iter()
        .map(|&(volta, torch, linear)| {
            let (v, shape) = f.get(&format!("{case}.{torch}"));
            match linear {
                Some((out, inp)) => (volta, transpose(&v, out, inp), vec![inp, out]),
                None => (volta, v, shape),
            }
        })
        .collect();
    state_of(&entries)
}

/// Load torch's weights into `m`, run `forward` on the fixture input, and compare the output
/// and every gradient with torch's.
fn check_module<M: Module>(
    f: &Fixture,
    case: &str,
    m: &mut M,
    params: &[Param],
    forward: impl Fn(&M, &Tensor) -> Tensor,
) {
    m.load_state_dict(&state_for(f, case, params))
        .expect("torch weights load");
    let p = m.parameters();
    assert_eq!(p.len(), params.len(), "{case}: parameter count");
    let x = f.tensor(&format!("{case}.x"), true);
    let out = forward(m, &x);

    // Volta's `[in, out]` weight gradients, transposed to torch's `[out, in]`.
    let fixes: Vec<Option<Fix>> = params
        .iter()
        .map(|&(_, _, linear)| {
            linear.map(|(out, inp)| Box::new(move |g: Vec<f32>| transpose(&g, inp, out)) as Fix)
        })
        .collect();
    let mut grads: Vec<torch_ref::GradCheck<'_>> = vec![("x", &x, None)];
    for ((&(_, torch, _), tensor), fix) in params.iter().zip(&p).zip(&fixes) {
        grads.push((torch, tensor, fix.as_deref()));
    }
    check_case(f, case, &out, &grads, FWD, BWD);
}

#[test]
fn mha_matches_torch() {
    let f = Fixture::load("transformer");
    let mut mha = MultiHeadAttention::new(E, H, 0.0);
    check_module(&f, "mha", &mut mha, &MHA_PARAMS, |m, x| {
        m.forward(x, false, None)
    });
}

#[test]
fn mha_causal_matches_torch() {
    let f = Fixture::load("transformer");
    let mut mha = MultiHeadAttention::new(E, H, 0.0);
    check_module(&f, "mha_causal", &mut mha, &MHA_PARAMS, |m, x| {
        m.forward(x, true, None)
    });
}

#[test]
fn transformer_block_matches_torch_pre_ln_encoder_layer() {
    let f = Fixture::load("transformer");
    let mut block = TransformerBlock::new(E, H, FF / E, 0.0);
    check_module(&f, "block", &mut block, &BLOCK_PARAMS, |m, x| {
        m.forward(x, false)
    });
}

#[test]
fn transformer_block_causal_matches_torch_pre_ln_encoder_layer() {
    let f = Fixture::load("transformer");
    let mut block = TransformerBlock::new(E, H, FF / E, 0.0);
    check_module(&f, "block_causal", &mut block, &BLOCK_PARAMS, |m, x| {
        m.forward(x, true)
    });
}

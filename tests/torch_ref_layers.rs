//! Layers, matmul, and movement ops against PyTorch, with torch's weights loaded into the
//! Volta layer (`tests/fixtures/torch/layers.safetensors`). Each case compares the forward
//! output and the gradients of `sum(out * w)` for a random `w`.
//!
//! Volta stores `Linear` weights as `[in, out]` (torch: `[out, in]`), so those are
//! transposed on the way in and out; every other layer here shares torch's layout.

#[path = "torch_ref/mod.rs"]
mod torch_ref;

use torch_ref::{Fixture, check_case, state_of, transpose};
use volta::{
    Axis, BatchNorm2d, Conv2d, ConvTranspose2d, GELU, Layer, LayerNorm, Linear, MaxPool2d, Module,
    PixelShuffle,
};

const FWD: f32 = 1e-5;
const BWD: f32 = 1e-4;
const CONV_FWD: f32 = 1e-4;
const CONV_BWD: f32 = 1e-3;

#[test]
fn linear_matches_torch() {
    let f = Fixture::load("layers");
    let mut l = Linear::new(2, 3, true);
    l.load_state_dict(&state_of(&[
        (
            "weight",
            transpose(&f.values("linear.weight"), 3, 2),
            vec![2, 3],
        ),
        ("bias", f.values("linear.bias"), vec![3]),
    ]))
    .expect("load");
    let x = f.tensor("linear.x", true);
    let out = l.forward(&x);
    let p = l.parameters();
    let back = |g: Vec<f32>| transpose(&g, 2, 3);
    check_case(
        &f,
        "linear",
        &out,
        &[
            ("x", &x, None),
            ("weight", &p[0], Some(&back)),
            ("bias", &p[1], None),
        ],
        FWD,
        BWD,
    );
}

#[test]
fn layernorm_matches_torch() {
    let f = Fixture::load("layers");
    let mut l = LayerNorm::new(vec![4]);
    l.load_state_dict(&state_of(&[
        ("gamma", f.values("layernorm.gamma"), vec![4]),
        ("beta", f.values("layernorm.beta"), vec![4]),
    ]))
    .expect("load");
    let x = f.tensor("layernorm.x", true);
    let out = l.forward(&x);
    let p = l.parameters();
    check_case(
        &f,
        "layernorm",
        &out,
        &[
            ("x", &x, None),
            ("gamma", &p[0], None),
            ("beta", &p[1], None),
        ],
        FWD,
        BWD,
    );
}

#[test]
fn gelu_matches_torch() {
    let f = Fixture::load("layers");
    let x = f.tensor("gelu.x", true);
    let out = GELU::new().forward(&x);
    check_case(&f, "gelu", &out, &[("x", &x, None)], FWD, BWD);
}

#[test]
fn conv2d_stride2_pad1_matches_torch() {
    let f = Fixture::load("layers");
    let mut c = Conv2d::new(2, 3, 3, 2, 1, true);
    c.load_state_dict(&state_of(&[
        ("weight", f.values("conv.weight"), vec![3, 2, 3, 3]),
        ("bias", f.values("conv.bias"), vec![3]),
    ]))
    .expect("load");
    let x = f.tensor("conv.x", true);
    let out = c.forward(&x);
    let p = c.parameters();
    check_case(
        &f,
        "conv",
        &out,
        &[
            ("x", &x, None),
            ("weight", &p[0], None),
            ("bias", &p[1], None),
        ],
        CONV_FWD,
        CONV_BWD,
    );
}

#[test]
fn conv_transpose2d_stride2_pad1_matches_torch() {
    let f = Fixture::load("layers");
    let mut c = ConvTranspose2d::new(2, 3, 3, 2, 1, true);
    c.load_state_dict(&state_of(&[
        ("weight", f.values("convt.weight"), vec![2, 3, 3, 3]),
        ("bias", f.values("convt.bias"), vec![3]),
    ]))
    .expect("load");
    let x = f.tensor("convt.x", true);
    let out = c.forward(&x);
    let p = c.parameters();
    check_case(
        &f,
        "convt",
        &out,
        &[
            ("x", &x, None),
            ("weight", &p[0], None),
            ("bias", &p[1], None),
        ],
        CONV_FWD,
        CONV_BWD,
    );
}

#[test]
fn maxpool_matches_torch() {
    let f = Fixture::load("layers");
    let x = f.tensor("maxpool.x", true);
    let out = MaxPool2d::new(2, 2, 0).forward(&x);
    check_case(&f, "maxpool", &out, &[("x", &x, None)], FWD, BWD);
}

#[test]
fn maxpool_all_neg_inf_channel_keeps_gradient_in_its_channel() {
    let f = Fixture::load("layers");
    let x = f.tensor("maxpool_neginf.x", true);
    let out = MaxPool2d::new(2, 2, 0).forward(&x);
    check_case(&f, "maxpool_neginf", &out, &[("x", &x, None)], FWD, BWD);
}

#[test]
fn maxpool_propagates_nan_like_torch() {
    let f = Fixture::load("layers");
    let x = f.tensor("maxpool_nan.x", false);
    let out = MaxPool2d::new(2, 2, 0).forward(&x);
    torch_ref::assert_close(
        "maxpool_nan.out",
        &out.data(),
        &f.values("maxpool_nan.out"),
        FWD,
    );
}

#[test]
fn maxpool_tie_sends_the_whole_gradient_to_the_first_element() {
    let f = Fixture::load("layers");
    let x = f.tensor("maxpool_tie.x", true);
    let out = MaxPool2d::new(2, 2, 0).forward(&x);
    check_case(&f, "maxpool_tie", &out, &[("x", &x, None)], FWD, BWD);
}

#[test]
fn maxpool_with_two_nans_sends_the_gradient_to_the_last() {
    let f = Fixture::load("layers");
    let x = f.tensor("maxpool_nan2.x", true);
    let out = MaxPool2d::new(2, 2, 0).forward(&x);
    check_case(&f, "maxpool_nan2", &out, &[("x", &x, None)], FWD, BWD);
}

#[test]
fn maxpool_all_neg_inf_window_with_padding_keeps_its_first_real_element() {
    // S3: windows that mix padding and real -inf send the gradient to the first real one.
    let f = Fixture::load("layers");
    let x = f.tensor("maxpool_neginf_pad.x", true);
    let out = MaxPool2d::new(2, 2, 1).forward(&x);
    check_case(&f, "maxpool_neginf_pad", &out, &[("x", &x, None)], FWD, BWD);
}

#[test]
fn batchnorm2d_train_and_eval_match_torch() {
    let f = Fixture::load("layers");
    let mut bn = BatchNorm2d::new(2);
    bn.load_state_dict(&state_of(&[
        ("gamma", f.values("bn.gamma"), vec![2]),
        ("beta", f.values("bn.beta"), vec![2]),
    ]))
    .expect("load");
    let x = f.tensor("bn.x", true);
    let out = bn.forward(&x);
    let p = bn.parameters();
    check_case(
        &f,
        "bn",
        &out,
        &[
            ("x", &x, None),
            ("gamma", &p[0], None),
            ("beta", &p[1], None),
        ],
        FWD,
        BWD,
    );
    // The running statistics after one training step, then eval mode with them.
    let state = bn.state_dict();
    torch_ref::assert_close(
        "running_mean",
        &state["running_mean"].to_vec_as::<f32>().expect("f32"),
        &f.values("bn.running_mean"),
        FWD,
    );
    torch_ref::assert_close(
        "running_var",
        &state["running_var"].to_vec_as::<f32>().expect("f32"),
        &f.values("bn.running_var"),
        FWD,
    );
    bn.train(false);
    let xe = f.tensor("bn.x_eval", false);
    torch_ref::assert_close(
        "bn.out_eval",
        &bn.forward(&xe).data(),
        &f.values("bn.out_eval"),
        FWD,
    );
}

/// Compare `a.matmul(b)` for the named fixture case.
fn matmul_case(name: &str) {
    let f = Fixture::load("layers");
    let a = f.tensor(&format!("{name}.a"), true);
    let b = f.tensor(&format!("{name}.b"), true);
    let out = a.matmul(&b);
    check_case(
        &f,
        name,
        &out,
        &[("a", &a, None), ("b", &b, None)],
        1e-4,
        1e-3,
    );
}

#[test]
fn matmul_2d_matches_torch() {
    matmul_case("mm22");
}

#[test]
fn matmul_batched_same_rank_matches_torch() {
    matmul_case("mm33");
}

#[test]
#[ignore = "D6: matmul requires equal operand ranks (documented in MatmulSpec); torch broadcasts [2,3,4] @ [4,5]"]
fn matmul_batched_by_matrix_matches_torch() {
    matmul_case("mm32");
}

#[test]
fn matmul_rank4_with_broadcast_batch_matches_torch() {
    matmul_case("mm44");
}

#[test]
fn permute_3_cycle_matches_torch() {
    let f = Fixture::load("layers");
    let x = f.tensor("permute.x", true);
    let out = x.permute(&[Axis(1), Axis(2), Axis(0)]);
    check_case(&f, "permute", &out, &[("x", &x, None)], FWD, BWD);
}

#[test]
fn pixel_shuffle_matches_torch() {
    let f = Fixture::load("layers");
    let x = f.tensor("pixelshuffle.x", true);
    let out = PixelShuffle::new(2).forward(&x);
    check_case(&f, "pixelshuffle", &out, &[("x", &x, None)], FWD, BWD);
}

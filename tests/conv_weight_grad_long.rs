//! Conv2d's weight gradient over many output positions (#318), per algorithm, against the
//! exact sum.
//!
//! A 1x1 convolution over a `16x1x512x512` input of ones with an upstream gradient of 0.1
//! sums `2^22` copies of `0.1f32` into its one weight. Direct and IGEMM used one sequential
//! f32 chain (4.0% low), while Im2col goes through gemm (1.5e-4), so the algorithm that
//! `ConvAlgo::Auto` picked changed the gradient by 4%. Direct and IGEMM now sum blocked
//! pairwise. CPU only: the GPU kernels are separate and not changed here.

use volta::{Conv2d, ConvAlgo, Module, RawTensor, ones};

/// Output positions: `16 * 512 * 512 = 2^22`.
const SHAPE: [usize; 4] = [16, 1, 512, 512];

/// The weight gradient of a 1x1, bias-free conv over ones, with every upstream gradient 0.1.
fn weight_grad(algo: ConvAlgo) -> f32 {
    let mut conv = Conv2d::new(1, 1, 1, 1, 0, false);
    conv.set_algo(algo);
    let x = ones(&SHAPE);
    let y = conv.forward(&x);
    let upstream = RawTensor::new(vec![0.1; SHAPE.iter().product()], &SHAPE, false);
    y.elem_mul(&upstream)
        .sum()
        .into_loss()
        .expect("scalar loss")
        .backward();
    let [weight] = conv
        .parameters()
        .try_into()
        .expect("a bias-free conv has one parameter");
    let [g]: [f32; 1] = weight
        .grad()
        .expect("weight gradient")
        .try_into()
        .expect("a 1x1x1x1 weight has one gradient element");
    g
}

/// `0.1f32 * 2^22`, exactly.
fn exact() -> f64 {
    f64::from(0.1f32) * f64::from(1u32 << 22)
}

/// Relative error of `got` against the exact sum.
fn rel(got: f32) -> f64 {
    ((f64::from(got) - exact()) / exact()).abs()
}

#[test]
fn direct_and_igemm_weight_gradients_are_pairwise_accurate() {
    for algo in [ConvAlgo::Direct, ConvAlgo::IGEMM] {
        let g = weight_grad(algo);
        // A blocked pairwise sum (128-term runs, then a tree) is within a few times
        // (128 + 22) * 2^-24 at worst; the old chain was 4.0e-2.
        assert!(
            rel(g) < 1e-5,
            "{algo:?}: {g}, exact {}, relative error {:e}",
            exact(),
            rel(g)
        );
    }
}

#[test]
fn algorithms_agree_on_the_weight_gradient() {
    // Im2col's gemm is the least accurate of the three here (about 1.5e-4); before the fix
    // Direct and IGEMM were 4% away from it.
    let im2col = weight_grad(ConvAlgo::Im2col);
    assert!(
        rel(im2col) < 5e-4,
        "Im2col: {im2col}, relative error {:e}",
        rel(im2col)
    );
    for algo in [ConvAlgo::Direct, ConvAlgo::IGEMM] {
        let g = weight_grad(algo);
        let gap = (f64::from(g) - f64::from(im2col)).abs() / exact();
        assert!(gap < 5e-4, "{algo:?} {g} vs Im2col {im2col}: gap {gap:e}");
    }
}

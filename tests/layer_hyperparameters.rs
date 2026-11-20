//! Layer constructors check their hyperparameters (#133): each `try_` form returns an
//! `InvalidParameter` naming the layer and the argument, and the panicking form says the
//! same. The rows are the issue's repro table.

use volta::amp::GradScaler;
use volta::nn::{
    BatchNorm1d, BatchNorm2d, Conv2d, ConvTranspose2d, Dropout, Embedding, FeedForward, LSTMCell,
    LayerNorm, Linear, MaxPool2d, MultiHeadAttention, PixelShuffle, PositionalEncoding,
    PositionalEncodingType, TransformerBlock,
};

/// The `(op, arg)` of a refused construction, or a panic naming `what`.
fn refused<T>(what: &str, r: volta::Result<T>) -> (&'static str, &'static str) {
    let Err(e) = r else {
        panic!("{what} was accepted");
    };
    let d = e
        .detail()
        .unwrap_or_else(|| panic!("{what}: {e:?} has no detail"));
    (d.op, d.arg.unwrap_or_else(|| panic!("{what}: no arg")))
}

#[test]
fn zero_sizes_name_the_layer_and_argument() {
    let rows = [
        (
            refused("Linear(0, 3)", Linear::try_new(0, 3, true)),
            ("Linear", "in_features"),
        ),
        (
            refused("Linear(3, 0)", Linear::try_new(3, 0, true)),
            ("Linear", "out_features"),
        ),
        (
            refused("Conv2d in_ch 0", Conv2d::try_new(0, 1, 3, 1, 0, true)),
            ("Conv2d", "in_ch"),
        ),
        (
            refused("Conv2d kernel 0", Conv2d::try_new(1, 1, 0, 1, 0, true)),
            ("Conv2d", "kernel"),
        ),
        (
            refused("Conv2d stride 0", Conv2d::try_new(1, 1, 3, 0, 0, true)),
            ("Conv2d", "stride"),
        ),
        (
            refused(
                "ConvTranspose2d stride 0",
                ConvTranspose2d::try_new(1, 1, 3, 0, 0, true),
            ),
            ("ConvTranspose2d", "stride"),
        ),
        (
            refused("MaxPool2d kernel 0", MaxPool2d::try_new(0, 1, 0)),
            ("MaxPool2d", "kernel"),
        ),
        (
            refused("MaxPool2d stride 0", MaxPool2d::try_new(2, 0, 0)),
            ("MaxPool2d", "stride"),
        ),
        (
            refused("LSTMCell(0, 4)", LSTMCell::try_new(0, 4, true)),
            ("LSTMCell", "input_size"),
        ),
        (
            refused("BatchNorm1d(0)", BatchNorm1d::try_new(0)),
            ("BatchNorm1d", "num_features"),
        ),
        (
            refused("Embedding(0, 4)", Embedding::try_new(0, 4)),
            ("Embedding", "vocab_size"),
        ),
        (
            refused("PixelShuffle(0)", PixelShuffle::try_new(0)),
            ("PixelShuffle", "upscale_factor"),
        ),
        (
            refused(
                "PositionalEncoding max_len 0",
                PositionalEncoding::try_new(8, 0, PositionalEncodingType::Sinusoidal),
            ),
            ("PositionalEncoding", "max_len"),
        ),
        (
            refused(
                "PositionalEncoding odd sinusoidal",
                PositionalEncoding::try_new(7, 4, PositionalEncodingType::Sinusoidal),
            ),
            ("PositionalEncoding", "embed_dim"),
        ),
        (
            refused("MHA 0 heads", MultiHeadAttention::try_new(8, 0, 0.0)),
            ("MultiHeadAttention", "num_heads"),
        ),
        (
            refused("MHA 8 % 3", MultiHeadAttention::try_new(8, 3, 0.0)),
            ("MultiHeadAttention", "num_heads"),
        ),
        (
            refused("FeedForward expansion 0", FeedForward::try_new(8, 0, 0.0)),
            ("FeedForward", "expansion_factor"),
        ),
        (
            refused(
                "TransformerBlock expansion 0",
                TransformerBlock::try_new(8, 2, 0, 0.0),
            ),
            ("FeedForward", "expansion_factor"),
        ),
    ];
    for (got, want) in rows {
        assert_eq!(got, want);
    }
}

#[test]
fn eps_momentum_and_dropout_are_range_checked() {
    let rows = [
        (
            refused(
                "BatchNorm1d eps -1",
                BatchNorm1d::try_new_with_params(4, -1.0, 0.1),
            ),
            ("BatchNorm1d", "eps"),
        ),
        (
            refused(
                "BatchNorm1d momentum 7",
                BatchNorm1d::try_new_with_params(4, 1e-5, 7.0),
            ),
            ("BatchNorm1d", "momentum"),
        ),
        (
            refused(
                "BatchNorm2d eps NaN",
                BatchNorm2d::try_new_with_params(4, f32::NAN, 0.1),
            ),
            ("BatchNorm2d", "eps"),
        ),
        (
            refused(
                "BatchNorm2d momentum NaN",
                BatchNorm2d::try_new_with_params(4, 1e-5, f32::NAN),
            ),
            ("BatchNorm2d", "momentum"),
        ),
        (
            refused("LayerNorm []", LayerNorm::try_new(vec![])),
            ("LayerNorm", "normalized_shape"),
        ),
        (
            refused("LayerNorm [4, 0]", LayerNorm::try_new(vec![4, 0])),
            ("LayerNorm", "normalized_shape"),
        ),
        (
            refused(
                "LayerNorm eps -1",
                LayerNorm::try_new_with_eps(vec![4], -1.0),
            ),
            ("LayerNorm", "eps"),
        ),
        (
            refused("LayerNorm eps 0", LayerNorm::try_new_with_eps(vec![4], 0.0)),
            ("LayerNorm", "eps"),
        ),
        (
            refused("Dropout 1.5", Dropout::try_new(1.5)),
            ("Dropout", "p"),
        ),
        (
            refused(
                "TransformerBlock dropout 2",
                TransformerBlock::try_new(8, 2, 4, 2.0),
            ),
            ("MultiHeadAttention", "dropout"),
        ),
        (
            refused(
                "GradScaler inf scale",
                GradScaler::try_with_params(f64::INFINITY, 2.0, 0.5, 1),
            ),
            ("GradScaler", "init_scale"),
        ),
        (
            refused(
                "GradScaler inf growth",
                GradScaler::try_with_params(1.0, f64::INFINITY, 0.5, 1),
            ),
            ("GradScaler", "growth_factor"),
        ),
    ];
    for (got, want) in rows {
        assert_eq!(got, want);
    }
}

/// The edges of each range are accepted, as in PyTorch.
#[test]
fn boundary_values_are_accepted() {
    BatchNorm1d::try_new_with_params(1, f32::MIN_POSITIVE, 0.0).expect("momentum 0");
    BatchNorm2d::try_new_with_params(1, 1e-5, 1.0).expect("momentum 1");
    Dropout::try_new(0.0).expect("p 0");
    Dropout::try_new(1.0).expect("p 1");
    MaxPool2d::try_new(1, 1, 0).expect("kernel 1, stride 1");
    LayerNorm::try_new(vec![1]).expect("one feature");
    PositionalEncoding::try_new(7, 4, PositionalEncodingType::Learned).expect("odd learned");
    GradScaler::try_with_params(1.0, 1.5, 0.5, 1).expect("finite settings");
}

/// The panicking form reports the `try_` form's error, naming the layer once.
#[test]
#[should_panic(expected = "Invalid parameter: Linear: in_features must be at least 1, got 0")]
fn new_panics_with_the_try_new_message() {
    let _ = Linear::new(0, 3, true);
}

#[test]
#[should_panic(expected = "Invalid parameter: BatchNorm1d: momentum must be in [0, 1], got 7")]
fn batchnorm_momentum_panic_names_the_value() {
    let _ = BatchNorm1d::new_with_params(4, 1e-5, 7.0);
}

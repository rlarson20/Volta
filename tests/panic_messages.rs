//! Panic messages for caller mistakes name the public op or layer, say what was wrong and
//! show the value (#142). Each row is from the issue's repro table.

use volta::nn::{
    BatchNorm2d, Conv2d, FeedForward, Layer, LayerNorm, Linear, MaxPool2d, MultiHeadAttention,
};
use volta::{Axis, RawTensor, Tensor};

/// The panic message of `f`, or a panic if it returns.
fn message(f: impl FnOnce() + std::panic::UnwindSafe) -> String {
    let payload = std::panic::catch_unwind(f).expect_err("the call should panic");
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
        .expect("a string payload")
}

/// A `[2, 3]` input.
fn a() -> Tensor {
    RawTensor::randn(&[2, 3])
}

#[test]
fn messages_name_the_op_and_show_the_values() {
    let rows: Vec<(String, &str)> = vec![
        (
            message(|| drop(a().sum_dim(Axis(5), false))),
            "sum_dim: Dimension 5 out of bounds for shape [2, 3]",
        ),
        (
            message(|| drop(a().softmax(Axis(5)))),
            "softmax: Dimension 5 out of bounds for shape [2, 3]",
        ),
        (
            message(|| drop(a().log_softmax(Axis(5)))),
            "log_softmax: Dimension 5 out of bounds for shape [2, 3]",
        ),
        (
            message(|| drop(a().shrink(&[(0, 9), (0, 1)]))),
            "Invalid parameter: shrink: range (0, 9) for dimension 0 ends past its size 2 (shape [2, 3])",
        ),
        (
            message(|| drop(a().permute(&[Axis(0), Axis(0)]))),
            "permute: axes [0, 0] are not a permutation of 0..2 (shape [2, 3])",
        ),
        (
            message(|| drop(a().reshape(&[7]))),
            "Invalid parameter: reshape: cannot reshape [2, 3] (6 elements) to [7] (7 elements)",
        ),
        (
            message(|| drop(a().add(&RawTensor::randn(&[4])))),
            "add: Cannot broadcast shapes [2, 3] and [4] at dimension 1",
        ),
        (
            message(|| {
                let v = RawTensor::randn(&[3]);
                drop(volta::cross_entropy_loss(&v, &v));
            }),
            "cross_entropy_loss: logits must be [N, C], got rank 1 ([3])",
        ),
        (
            message(|| drop(Linear::new(4, 2, true).forward(&a()))),
            "Linear(4 -> 2): expected input [.., 4], got [2, 3]",
        ),
        (
            message(|| drop(Conv2d::new(1, 1, 3, 1, 0, true).forward(&a()))),
            "Invalid parameter: Conv2d: input must be 4D, got shape [2, 3]",
        ),
        (
            message(|| drop(MaxPool2d::new(2, 2, 0).forward(&a()))),
            "Invalid parameter: MaxPool2d: input must be 4D, got shape [2, 3]",
        ),
        (
            message(|| drop(BatchNorm2d::new(3).forward(&a()))),
            "BatchNorm2d(3): expected input [B, 3, H, W], got [2, 3]",
        ),
        (
            message(|| drop(LayerNorm::new(vec![4]).forward(&a()))),
            "LayerNorm([4]): expected input [.., 4], got [2, 3]",
        ),
        (
            message(|| {
                let mha = MultiHeadAttention::new(4, 2, 0.0);
                drop(mha.forward(&RawTensor::randn(&[2, 4]), false, None));
            }),
            "MultiHeadAttention: query must be [batch, seq_len, 4], got [2, 4]",
        ),
        (
            message(|| {
                drop(<FeedForward as Layer>::forward(
                    &FeedForward::new(4, 4, 0.0),
                    &a(),
                ));
            }),
            "FeedForward: input must be [batch, seq_len, 4], got [2, 3]",
        ),
    ];
    for (got, want) in rows {
        assert_eq!(got, want);
    }
}

/// None of these says the fault is inside Volta.
#[test]
fn a_rank_mistake_is_not_reported_as_an_index_mapping_bug() {
    let m = message(|| {
        drop(<FeedForward as Layer>::forward(
            &FeedForward::new(4, 4, 0.0),
            &a(),
        ));
    });
    assert!(!m.contains("index-mapping bug"), "{m}");
}

//! Every public type implements `Debug` (#116, enforced by `missing_debug_implementations`
//! in `Cargo.toml`). These pin what the hand-written ones show: shapes and settings, not
//! values.

use volta::io::mapping::StateDictMapper;
use volta::nn::{Linear, ReLU, Sequential};
use volta::{DataLoader, RawTensor};

#[test]
fn a_tensor_shows_its_shape_and_dtype_not_its_values() {
    let t = RawTensor::new(vec![1.5, 2.5], &[2], true);
    let s = format!("{t:?}");
    assert!(s.starts_with("Tensor { shape: [2], dtype: F32"), "{s}");
    assert!(!s.contains("1.5") && !s.contains("RefCell"), "{s}");
}

#[test]
fn a_layer_shows_its_parameters_shapes() {
    let s = format!("{:?}", Linear::new(3, 2, true));
    assert!(
        s.starts_with("Linear { weight: Tensor { shape: [3, 2]"),
        "{s}"
    );
}

#[test]
fn a_sequential_lists_its_layer_names() {
    let m = Sequential::builder()
        .add_named("fc", Box::new(Linear::new(2, 2, true)))
        .add_unnamed(Box::new(ReLU))
        .build();
    assert_eq!(
        format!("{m:?}"),
        r#"Sequential { layers: [Some("fc"), None], parameters: 2 }"#
    );
}

#[test]
fn a_data_loader_shows_sizes_not_samples() {
    let d = DataLoader::new(vec![7.0; 6], vec![9.0; 3], &[2], &[1], 2, false);
    let s = format!("{d:?}");
    assert!(
        s.starts_with("DataLoader { samples: 3, data_shape: [2]"),
        "{s}"
    );
    assert!(!s.contains("7.0"), "{s}");
}

#[test]
fn a_mapper_counts_its_edits() {
    let m = StateDictMapper::new().rename("a", "b").strip_prefix("m.");
    assert_eq!(format!("{m:?}"), "StateDictMapper { transformations: 2 }");
}

/// `unwrap_err` on a `Result<ScalarLoss, _>` needs `ScalarLoss: Debug`.
#[test]
fn a_scalar_loss_result_can_be_unwrapped_as_an_error() {
    let err = RawTensor::new(vec![1.0, 2.0], &[2], true)
        .into_loss()
        .unwrap_err();
    assert!(
        matches!(err, volta::VoltaError::NotScalar { .. }),
        "{err:?}"
    );
}

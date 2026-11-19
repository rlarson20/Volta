//! `Storage` has the same variants with and without the `gpu` feature, and is
//! `#[non_exhaustive]` (#340): this match, written the way a downstream crate must write it,
//! compiles under both feature sets. Before, `Storage::Gpu` existed only with `gpu`, so a
//! match naming it broke CPU-only builds and one without it broke when any crate in the
//! build turned `gpu` on.

use volta::Storage;

/// Where `s` lives, by its variant.
fn kind(s: &Storage) -> &'static str {
    match s {
        Storage::Cpu(_) => "cpu",
        Storage::Gpu(_) => "gpu",
        _ => "other",
    }
}

#[test]
fn storage_matches_the_same_way_in_every_feature_set() {
    let s = Storage::cpu(vec![1.0, 2.0]);
    assert_eq!(kind(&s), "cpu");
    assert!(!s.is_gpu());
}

/// With a GPU, GPU storage matches the `Gpu` arm (lavapipe or a real adapter).
#[cfg(feature = "gpu")]
#[test]
fn gpu_storage_matches_the_gpu_arm() {
    let Some(gpu) = volta::Device::gpu() else {
        return;
    };
    let s = Storage::cpu(vec![1.0, 2.0]).to_device(&gpu);
    assert_eq!(kind(&s), "gpu");
}

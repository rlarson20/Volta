//! `Storage` is an opaque struct (#356): other crates read where it lives through methods,
//! never by matching variants, so the same code compiles with and without the `gpu`
//! feature and the representation can change without a breaking change. That there is no
//! variant to match is checked by a `compile_fail` doctest on `Storage`.

use volta::Device;
use volta::storage::Storage;

/// Where `s` lives, read the way a downstream crate must read it.
fn kind(s: &Storage) -> &'static str {
    if s.is_gpu() { "gpu" } else { "cpu" }
}

#[test]
fn cpu_storage_reads_as_cpu_in_every_feature_set() {
    let s = Storage::cpu(vec![1.0, 2.0]);
    assert_eq!(kind(&s), "cpu");
    assert_eq!(s.len(), 2);
    assert_eq!(s.to_host_f32(), &[1.0, 2.0]);
    assert_eq!(kind(&s.to_device(&Device::CPU)), "cpu");
}

/// With a GPU, GPU storage reads as GPU and comes back unchanged (lavapipe or a real
/// adapter).
#[cfg(feature = "gpu")]
#[test]
fn gpu_storage_reads_as_gpu() {
    let Some(gpu) = Device::gpu() else {
        return;
    };
    let s = Storage::cpu(vec![1.0, 2.0]).to_device(&gpu);
    assert_eq!(kind(&s), "gpu");
    let back = s.to_device(&Device::CPU);
    assert_eq!(kind(&back), "cpu");
    assert_eq!(back.to_host_f32(), &[1.0, 2.0]);
}

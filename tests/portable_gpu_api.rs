//! The eight GPU control functions exist with and without the `gpu` feature (#342), so a
//! portable program needs no `cfg` of its own to call them.
//!
//! The first test compiles and runs under every feature set. The second runs only without
//! `gpu` and pins each stub's answer: what the real function returns when no adapter was
//! found.

use volta::RawTensor;
use volta::gpu::{
    gpu_cleanup, gpu_compact, gpu_pending_count, gpu_pool_stats, gpu_sync, gpu_sync_threshold,
    is_gpu_available,
};

/// Every function is callable through the same paths in every build.
#[test]
fn control_functions_are_callable_in_every_build() {
    let t = RawTensor::new(vec![1.0, 2.0], &[2], false);
    if is_gpu_available() {
        assert!(gpu_sync(), "a sync with nothing pending completes");
    }
    let _ = gpu_pending_count();
    let _ = gpu_sync_threshold();
    let _ = gpu_pool_stats();
    let _ = gpu_cleanup();
    let _ = gpu_compact();
    volta::gpu::invalidate_all_tensor_caches(&[&t]);
    assert_eq!(t.data(), vec![1.0, 2.0], "a CPU tensor is untouched");
}

/// Without `gpu`, each stub gives the no-GPU answer.
#[cfg(not(feature = "gpu"))]
#[test]
fn stubs_give_the_no_gpu_answers() {
    assert!(!is_gpu_available());
    assert!(gpu_sync(), "no GPU: nothing to wait for");
    assert_eq!(gpu_pending_count(), 0);
    assert_eq!(gpu_sync_threshold(), 0);
    assert_eq!(gpu_pool_stats(), None);
    assert!(!gpu_cleanup(), "no GPU to clean up");
    assert!(!gpu_compact(), "no GPU to compact");
    let t = RawTensor::new(vec![3.0], &[1], false);
    volta::gpu::invalidate_all_tensor_caches(&[&t]);
    assert_eq!(t.item(), 3.0);
}

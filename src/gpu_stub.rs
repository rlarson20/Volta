//! The GPU control functions without the `gpu` feature (#342).
//!
//! Mounted as `volta::gpu` when the feature is off, so a portable program calls
//! `volta::is_gpu_available()`, `volta::gpu_sync()` and the rest without a `cfg` of its own
//! (the "gated real function plus a not-gpu stub" rule, `.claude/rules/gpu.md`). Each stub
//! returns what the real function returns when no adapter was found, so a caller sees the
//! same answer from a CPU-only build as from a GPU build on a machine with no GPU. Only the
//! GPU types (`GpuBuffer`) are missing here.

use crate::Tensor;

/// Whether a GPU is available: always `false` without the `gpu` feature.
#[must_use]
pub const fn is_gpu_available() -> bool {
    false
}

/// Wait for pending GPU work: `true` (nothing to wait for) without the `gpu` feature, as
/// the real function returns with no GPU.
#[must_use]
pub const fn gpu_sync() -> bool {
    true
}

/// Pending GPU submissions: always 0 without the `gpu` feature.
#[must_use]
pub const fn gpu_pending_count() -> u32 {
    0
}

/// The automatic-sync threshold: 0 without the `gpu` feature, as with no GPU.
#[must_use]
pub const fn gpu_sync_threshold() -> u32 {
    0
}

/// Clear the GPU buffer pools: `false` (no GPU to clean up) without the `gpu` feature.
#[must_use]
pub const fn gpu_cleanup() -> bool {
    false
}

/// Compact GPU memory: `false` (no GPU) without the `gpu` feature.
#[must_use]
pub const fn gpu_compact() -> bool {
    false
}

/// GPU buffer pool statistics: `None` (no GPU) without the `gpu` feature.
#[must_use]
pub const fn gpu_pool_stats() -> Option<(usize, usize)> {
    None
}

/// Release the host copies of GPU tensors: a no-op without the `gpu` feature, where every
/// tensor is a CPU tensor (as the real function is for CPU tensors).
pub const fn invalidate_all_tensor_caches(_tensors: &[&Tensor]) {}

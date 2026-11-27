//! [`GpuCounters`]: how often the GPU backend talked to the device (#464).
//!
//! Shared by `volta::gpu` and its not-gpu stub (`gpu_stub.rs`), so the type is the same in
//! both builds. The counts live in process-wide atomics, bumped by the GPU backend at the
//! one place each event happens.

/// Counts of GPU backend events since the process started, from [`crate::gpu::gpu_counters`].
///
/// They show where a GPU workload waits on the host: a training step that stays on the
/// device should do no `readbacks` or `uploads` after its data is on the GPU, and `syncs`
/// only where the program asks for them. The counts are process-wide: threads share them,
/// so compare two snapshots ([`Self::since`]) around code that runs alone. Without the
/// `gpu` feature, or without an adapter, every count stays 0.
///
/// `#[non_exhaustive]`: later counters can be added without a breaking change.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuCounters {
    /// Command buffers submitted to the queue (kernel launches, buffer copies and readback
    /// copies; a pass of several kernels is one).
    pub submits: u64,
    /// Waits for the whole queue to drain ([`crate::gpu::gpu_sync`] calls that had work
    /// pending). Back-pressure, which waits only for the oldest submission once
    /// [`crate::gpu::gpu_sync_threshold`] are in flight, is not a drain and is not counted.
    pub syncs: u64,
    /// GPU-to-host copies (each waits for the queue up to that copy).
    pub readbacks: u64,
    /// Host-to-GPU copies of buffer contents (tensor data and index lists; not the small
    /// per-kernel parameter blocks, and not zero fills, which the GPU clears itself).
    pub uploads: u64,
}

impl GpuCounters {
    /// The events between `earlier` and `self` (each count's difference, 0 if `earlier`
    /// is the later snapshot).
    #[must_use]
    pub const fn since(&self, earlier: &Self) -> Self {
        Self {
            submits: self.submits.saturating_sub(earlier.submits),
            syncs: self.syncs.saturating_sub(earlier.syncs),
            readbacks: self.readbacks.saturating_sub(earlier.readbacks),
            uploads: self.uploads.saturating_sub(earlier.uploads),
        }
    }
}

#[cfg(feature = "gpu")]
pub(crate) use counting::{note_readback, note_submit, note_sync, note_upload, snapshot};

/// The atomics behind [`GpuCounters`], bumped only by the GPU backend.
#[cfg(feature = "gpu")]
mod counting {
    use super::GpuCounters;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// [`GpuCounters::submits`].
    static SUBMITS: AtomicU64 = AtomicU64::new(0);
    /// [`GpuCounters::syncs`].
    static SYNCS: AtomicU64 = AtomicU64::new(0);
    /// [`GpuCounters::readbacks`].
    static READBACKS: AtomicU64 = AtomicU64::new(0);
    /// [`GpuCounters::uploads`].
    static UPLOADS: AtomicU64 = AtomicU64::new(0);

    /// Count one queue submission.
    pub(crate) fn note_submit() {
        SUBMITS.fetch_add(1, Ordering::Relaxed);
    }

    /// Count one queue drain.
    pub(crate) fn note_sync() {
        SYNCS.fetch_add(1, Ordering::Relaxed);
    }

    /// Count one GPU-to-host copy.
    pub(crate) fn note_readback() {
        READBACKS.fetch_add(1, Ordering::Relaxed);
    }

    /// Count one host-to-GPU copy.
    pub(crate) fn note_upload() {
        UPLOADS.fetch_add(1, Ordering::Relaxed);
    }

    /// The counts now.
    pub(crate) fn snapshot() -> GpuCounters {
        GpuCounters {
            submits: SUBMITS.load(Ordering::Relaxed),
            syncs: SYNCS.load(Ordering::Relaxed),
            readbacks: READBACKS.load(Ordering::Relaxed),
            uploads: UPLOADS.load(Ordering::Relaxed),
        }
    }
}

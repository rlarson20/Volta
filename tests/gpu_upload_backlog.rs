//! Host-to-device uploads must be counted as pending work, so `gpu_sync` flushes them.
//!
//! `queue.write_buffer` keeps its data in wgpu's staging area until the next submit. When
//! uploads were not counted, `gpu_sync` returned at once, the staging memory piled up
//! (about 1 GB in `just bench-gpu`'s `memory_transfer` group), and the next kernel's sync
//! had to drain the whole backlog and hit the sync timeout.
//!
//! This needs a real adapter. The test returns early without one, so "passed" is only
//! GPU-verified on a machine that has a GPU: run `just test-gpu` there.
//!
//! Everything is one `#[test]`: the GPU context and its pending counter are process-global,
//! and a second test syncing in parallel would reset the counter this one reads.

#![cfg(feature = "gpu")]

mod common;

use volta::RawTensor;
use volta::gpu::{get_process_memory_mb, gpu_pending_count, gpu_sync};

/// Elements per upload: 1 MiB of `f32`.
const ELEMS: usize = 262_144;
/// Uploads in the loop. Unflushed, this is about 500 MiB of staging memory.
const UPLOADS: usize = 500;
/// Allowed growth in process memory over the loop, in MB. Flushed, growth is a few MB.
const MAX_GROWTH_MB: usize = 150;

/// A non-constant vector, so the upload is real data and not a page of zeros.
fn wavy(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * 0.01).sin()).collect()
}

/// An upload is pending work until synced, and a sync loop over uploads does not grow memory.
#[test]
fn uploads_are_flushed_by_sync() {
    if common::skip_if_no_gpu() {
        return;
    }
    let cpu = RawTensor::new(wavy(ELEMS), &[ELEMS], false);

    assert!(gpu_sync(), "initial sync timed out");
    let uploaded = cpu.to_device(common::gpu_device());
    assert!(
        gpu_pending_count() > 0,
        "an upload must count as pending work, or gpu_sync() will skip flushing it"
    );
    assert!(gpu_sync(), "sync after one upload timed out");
    assert_eq!(gpu_pending_count(), 0, "sync must clear the pending count");
    drop(uploaded);

    // Warm the buffer pool so the baseline already includes its steady-state size.
    for _ in 0..8 {
        drop(cpu.to_device(common::gpu_device()));
        assert!(gpu_sync(), "warm-up sync timed out");
    }

    let before = get_process_memory_mb();
    for i in 0..UPLOADS {
        drop(cpu.to_device(common::gpu_device()));
        assert!(gpu_sync(), "sync timed out after upload {i}");
    }
    let growth = get_process_memory_mb().saturating_sub(before);
    assert!(
        growth < MAX_GROWTH_MB,
        "{UPLOADS} synced 1 MiB uploads grew process memory by {growth} MB (limit {MAX_GROWTH_MB} MB)"
    );
}

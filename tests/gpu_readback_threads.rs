//! #437: readbacks from several threads at once. The GPU context is one process-wide
//! device, and a readback that polled with `submission_index: None` (the device's last
//! submission) could name another thread's not-yet-registered submit, tripping wgpu-core's
//! "current submission index should be at least the wait submission index" assert. Each
//! poll now waits on its own submission. The race is timing-dependent: on lavapipe this
//! test did not hit it before the fix in 11 runs, so it guards the fix rather than proving
//! it. Run it repeatedly with `just test-repeat-threads "--test gpu_readback_threads" 30`.

#![cfg(feature = "gpu")]

mod common;

use volta::RawTensor;

/// Threads reading back at once.
const THREADS: usize = 16;
/// Readbacks each thread makes.
const ROUNDS: usize = 200;

/// `ROUNDS` upload, add and read-back round trips; panics on a wrong value.
fn round_trips(seed: usize) {
    let device = common::gpu_device();
    for round in 0..ROUNDS {
        #[expect(clippy::cast_precision_loss, reason = "small test counters")]
        let base = (seed * ROUNDS + round) as f32;
        let a = RawTensor::new(vec![base, base + 1.0, base + 2.0, base + 3.0], &[4], false)
            .to_device(device.clone());
        let got = a.add(&a).data();
        let want = vec![
            2.0 * base,
            2.0 * base + 2.0,
            2.0 * base + 4.0,
            2.0 * base + 6.0,
        ];
        assert_eq!(got, want, "thread {seed}, round {round}");
    }
}

#[test]
fn concurrent_readbacks_from_many_threads() {
    if common::skip_if_no_gpu() {
        return;
    }
    let handles: Vec<_> = (0..THREADS)
        .map(|seed| std::thread::spawn(move || round_trips(seed)))
        .collect();
    for h in handles {
        h.join().expect("a readback thread panicked");
    }
}

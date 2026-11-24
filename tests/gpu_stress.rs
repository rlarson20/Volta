//! GPU stress tests for command queue management.
//!
//! Each test puts load on the command queue (many small ops, buffer-pool churn, a long
//! dependent chain, mixed sizes, repeated readbacks) and then checks what load could
//! break: every `gpu_sync()` completes (it returns `false` on a timeout), the pending
//! count stays bounded by the auto-sync threshold, and the kernels' outputs are still
//! right afterwards, compared with values computed on the host. A recycled pool buffer
//! read before it was written, or ops run out of order, shows up as a value mismatch.
#[cfg(feature = "gpu")]
mod gpu_stress_tests {
    use volta::gpu::{gpu_pending_count, gpu_sync, gpu_sync_threshold};
    use volta::{Device, RawTensor, Tensor, gpu};

    /// Serializes this module's tests: they read and assert the process-wide
    /// pending-submission counter, which any test running in parallel moves.
    static COUNTER_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Hold the counter lock for the rest of the test (poison-tolerant: a failed test
    /// must not cascade into the others).
    fn serial() -> std::sync::MutexGuard<'static, ()> {
        COUNTER_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Waits for the GPU and fails the test if the wait timed out.
    fn synced() {
        assert!(gpu_sync(), "gpu_sync timed out");
    }

    /// Fails the test unless auto-sync has kept the pending count at or below the threshold.
    fn assert_pending_bounded(context: &str) {
        let (pending, threshold) = (gpu_pending_count(), gpu_sync_threshold());
        assert!(
            pending <= threshold,
            "{context}: {pending} pending submissions, over the auto-sync threshold {threshold}"
        );
    }

    /// `size` values with both signs (so `relu` is not the identity), offset by `seed` so
    /// tensors made from different seeds differ.
    fn host_values(size: usize, seed: usize) -> Vec<f32> {
        (0..size)
            .map(|i| ((i + 7 * seed) % 97) as f32 * 0.01 - 0.48)
            .collect()
    }

    /// `host_values(size, seed)` on the GPU.
    ///
    /// The caller has checked `gpu::is_gpu_available()`.
    fn gpu_tensor(size: usize, seed: usize) -> Tensor {
        RawTensor::from_vec(host_values(size, seed), &[size])
            .to_device(Device::gpu().expect("GPU availability checked by the caller"))
    }

    /// Fails the test unless `t` (read back from the GPU) is exactly `relu` of
    /// `host_values(size, seed)`: relu is exact, so there is no tolerance.
    fn assert_relu_of(t: &Tensor, size: usize, seed: usize, context: &str) {
        let got = t.data();
        let expected: Vec<f32> = host_values(size, seed)
            .into_iter()
            .map(|v| v.max(0.0))
            .collect();
        assert_eq!(got.len(), expected.len(), "{context}: length");
        for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
            assert!(g == e, "{context}: element {i} is {g}, expected {e}");
        }
    }

    /// The "many small ops" benchmark pattern: 20 small tensors, 100 rounds of relu with
    /// no explicit sync (like Criterion's warmup).
    #[test]
    fn test_many_small_ops_pattern() {
        let _serial = serial();
        if !gpu::is_gpu_available() {
            return;
        }

        let tensors: Vec<_> = (0..20).map(|s| gpu_tensor(256, s)).collect();
        let mut last = Vec::new();
        for iteration in 0..100 {
            last = tensors.iter().map(Tensor::relu).collect();
            assert_pending_bounded(&format!("iteration {iteration}"));
        }

        synced();
        assert_eq!(gpu_pending_count(), 0, "pending count after sync");
        for (s, r) in last.iter().enumerate() {
            assert_relu_of(r, 256, s, &format!("tensor {s} after 100 rounds"));
        }
    }

    /// Each op is counted: from a clean queue, every relu adds the same number of
    /// submissions (its output allocation and its kernel), until auto-sync empties the
    /// queue; after that the count stays at or below the threshold.
    #[test]
    fn test_submission_tracking_accuracy() {
        let _serial = serial();
        if !gpu::is_gpu_available() {
            return;
        }

        let t = gpu_tensor(1024, 0);
        synced();
        assert_eq!(gpu_pending_count(), 0);

        let threshold = gpu_sync_threshold();
        let mut results = vec![t.relu()];
        let per_op = gpu_pending_count();
        assert!(per_op >= 1, "a relu on the GPU submitted nothing");
        for i in 2..=10 {
            results.push(t.relu());
            let pending = gpu_pending_count();
            if i * per_op < threshold {
                assert_eq!(pending, i * per_op, "pending after {i} ops");
            } else {
                assert!(pending <= threshold, "pending after {i} ops: {pending}");
            }
        }

        synced();
        for (i, r) in results.iter().enumerate() {
            assert_relu_of(r, 1024, 0, &format!("op {i}"));
        }
    }

    /// Buffer-pool churn: five rounds of 100 fresh tensors, used and dropped (their
    /// buffers go back to the pool and are handed out again next round). Each round's
    /// results must match that round's inputs, not a previous round's buffer contents.
    #[test]
    fn test_buffer_pool_churn() {
        let _serial = serial();
        if !gpu::is_gpu_available() {
            return;
        }

        for round in 0..5 {
            let seeds: Vec<usize> = (0..100).map(|i| 100 * round + i).collect();
            let tensors: Vec<_> = seeds.iter().map(|&s| gpu_tensor(1024, s)).collect();
            let results: Vec<_> = tensors.iter().map(Tensor::relu).collect();
            assert_pending_bounded(&format!("round {round}"));

            synced();
            for (r, &s) in results.iter().zip(&seeds) {
                assert_relu_of(r, 1024, s, &format!("round {round}, seed {s}"));
            }
            drop(results);
            drop(tensors);
        }
    }

    /// Auto-sync: `threshold + 5` ops from a clean queue never leave `threshold` pending.
    #[test]
    fn test_auto_sync_threshold() {
        let _serial = serial();
        if !gpu::is_gpu_available() {
            return;
        }

        let threshold = gpu_sync_threshold();
        let t = gpu_tensor(256, 0);
        synced();
        assert_eq!(gpu_pending_count(), 0);

        let mut last = t.relu();
        for i in 2..=threshold + 5 {
            last = t.relu();
            assert!(
                gpu_pending_count() < threshold,
                "auto-sync should keep pending below {threshold}; got {} after {i} ops",
                gpu_pending_count()
            );
        }

        synced();
        assert_relu_of(&last, 256, 0, "last op");
    }

    /// A long dependent chain: each op reads the previous op's output, so the queue holds
    /// 1000 ops in order and every intermediate buffer is live until the next op reads it.
    /// `x + 1` a thousand times is `x + 1000`; the inputs are multiples of 1/100 in
    /// `[-0.48, 0.48]`, so each add rounds by at most half an ulp at 1024 (2^-15).
    #[test]
    fn test_long_operation_chain() {
        let _serial = serial();
        if !gpu::is_gpu_available() {
            return;
        }

        let n = 4096;
        let gpu_dev = Device::gpu().expect("GPU availability checked above");
        let one = RawTensor::from_vec(vec![1.0; n], &[n]).to_device(gpu_dev);
        let mut x = gpu_tensor(n, 0);
        for step in 0..1000 {
            x = x.add(&one);
            assert_pending_bounded(&format!("step {step}"));
        }

        synced();
        let got = x.data();
        let expected: Vec<f32> = host_values(n, 0).into_iter().map(|v| v + 1000.0).collect();
        assert_eq!(got.len(), expected.len());
        // Spacing below 1024 is at most 2^-14: 1000 adds and the expected value's own
        // rounding, each off by at most half of it.
        let tol = 1001.0 * 2f32.powi(-15);
        for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
            assert!(
                (g - e).abs() <= tol,
                "element {i} after 1000 adds is {g}, expected {e} (tolerance {tol})"
            );
        }
    }

    /// Small, medium and large tensors interleaved in one queue.
    #[test]
    fn test_mixed_operation_sizes() {
        let _serial = serial();
        if !gpu::is_gpu_available() {
            return;
        }

        let sizes = [(64, 1), (1024, 2), (16384, 3)];
        let tensors: Vec<_> = sizes.iter().map(|&(n, s)| gpu_tensor(n, s)).collect();
        synced();

        let mut last = Vec::new();
        for i in 0..50 {
            last = tensors.iter().map(Tensor::relu).collect();
            assert_pending_bounded(&format!("iteration {i}"));
        }

        synced();
        for (r, &(n, s)) in last.iter().zip(&sizes) {
            assert_relu_of(r, n, s, &format!("size {n}"));
        }
    }

    /// A readback right after upload returns exactly the uploaded data, and leaves the
    /// queue drained.
    #[test]
    fn test_copy_operations_tracking() {
        let _serial = serial();
        if !gpu::is_gpu_available() {
            return;
        }

        let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let t = RawTensor::from_vec(data.clone(), &[1024])
            .to_device(Device::gpu().expect("GPU availability checked above"));

        synced();
        let result = t.data();
        assert_eq!(result, data, "readback differs from the uploaded data");
        synced();
        assert_eq!(gpu_pending_count(), 0, "pending count after sync");
    }

    /// Repeated readbacks of the same tensors (staging buffers are pooled and reused):
    /// every readback returns each tensor's own data. No pending-count bound here: a
    /// readback's submit is counted but never auto-syncs (its blocking poll waits for the
    /// queue anyway; see `GpuBuffer`'s readback), so the count can pass the threshold.
    #[test]
    fn test_staging_buffer_pooling() {
        let _serial = serial();
        if !gpu::is_gpu_available() {
            return;
        }

        let tensors: Vec<_> = (0..10).map(|s| gpu_tensor(1024, s)).collect();
        for iteration in 0..5 {
            for (s, t) in tensors.iter().enumerate() {
                assert_eq!(
                    t.data(),
                    host_values(1024, s),
                    "tensor {s}, readback {iteration}"
                );
            }
        }
        synced();
    }
}

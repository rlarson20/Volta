//! GPU stress tests for command queue management
//!
//! These tests are designed to reproduce and diagnose GPU timeout issues
//! by stressing the command queue with various workload patterns.
#[cfg(feature = "gpu")]
mod gpu_stress_tests {
    use volta::{Device, RawTensor, gpu, gpu_pending_count, gpu_sync, gpu_sync_threshold};

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

    /// Helper to create a test tensor on GPU
    ///
    /// Note: This function assumes GPU is available. Callers must check
    /// `gpu::is_gpu_available()` before calling this function.
    fn gpu_tensor(size: usize) -> volta::Tensor {
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
        RawTensor::from_vec(data, &[size])
            .to_device(Device::gpu().expect("GPU should be available for these tests"))
    }

    /// Test 1: Reproduce the "many small ops" benchmark pattern
    ///
    /// This should trigger the timeout if the issue still exists.
    #[test]
    fn test_many_small_ops_pattern() {
        let _serial = serial();
        if !gpu::is_gpu_available() {
            println!("Skipping GPU stress test - no GPU available");
            return;
        }

        println!("Sync threshold: {}", gpu_sync_threshold());

        // Create 20 small tensors (matching the benchmark)
        let tensors: Vec<_> = (0..20).map(|_| gpu_tensor(256)).collect();

        println!("After tensor creation, pending: {}", gpu_pending_count());

        // Run multiple iterations without explicit sync (like Criterion warmup)
        for iteration in 0..100 {
            for t in &tensors {
                let _ = t.relu();
            }

            if iteration % 10 == 0 {
                let pending = gpu_pending_count();
                println!("Iteration {iteration}, pending ops: {pending}");
            }
        }

        let pending_before_sync = gpu_pending_count();
        println!("Before final sync, pending: {pending_before_sync}");

        let _ = gpu_sync();

        let pending_after_sync = gpu_pending_count();
        println!("After final sync, pending: {pending_after_sync}");
        assert_eq!(
            pending_after_sync, 0,
            "Pending count should be 0 after sync"
        );
    }

    /// Test 2: Track actual vs reported submissions
    ///
    /// This test measures how many operations we think we're doing
    /// versus what the pending counter shows.
    #[test]
    fn test_submission_tracking_accuracy() {
        let _serial = serial();
        if !gpu::is_gpu_available() {
            println!("Skipping GPU stress test - no GPU available");
            return;
        }

        let t = gpu_tensor(1024);

        // Start clean
        let _ = gpu_sync();
        assert_eq!(gpu_pending_count(), 0);

        // Do 10 operations
        for i in 1..=10 {
            let _ = t.relu();
            let pending = gpu_pending_count();
            println!("After op {i}: pending = {pending}");
        }

        let pending = gpu_pending_count();
        println!("Expected 10 ops (or fewer if auto-synced), got: {pending}");

        let _ = gpu_sync();
    }

    /// Test 3: Rapid buffer allocation and deallocation
    ///
    /// Stresses the buffer pool by creating and dropping many tensors.
    #[test]
    fn test_buffer_pool_churn() {
        let _serial = serial();
        if !gpu::is_gpu_available() {
            println!("Skipping GPU stress test - no GPU available");
            return;
        }

        for round in 0..5 {
            println!("Round {round}");

            // Create 100 tensors (will churn buffer pool)
            let tensors: Vec<_> = (0..100).map(|_| gpu_tensor(1024)).collect();

            println!("After creation, pending: {}", gpu_pending_count());

            // Use them
            for t in &tensors {
                let _ = t.relu();
            }

            println!("After operations, pending: {}", gpu_pending_count());

            // Sync and drop (tensors go back to pool)
            let _ = gpu_sync();
            drop(tensors);
        }
    }

    /// Test 4: Verify auto-sync threshold triggers correctly
    ///
    /// This test checks that `maybe_sync()` actually kicks in when we
    /// exceed the threshold.
    #[test]
    fn test_auto_sync_threshold() {
        let _serial = serial();
        if !gpu::is_gpu_available() {
            println!("Skipping GPU stress test - no GPU available");
            return;
        }

        let threshold = gpu_sync_threshold();
        println!("Sync threshold: {threshold}");

        let t = gpu_tensor(256);

        // Start clean
        let _ = gpu_sync();
        assert_eq!(gpu_pending_count(), 0);

        // Do threshold + 5 operations
        let target_ops = threshold + 5;
        for i in 1..=target_ops {
            let _ = t.relu();
            let pending = gpu_pending_count();

            if pending == 0 && i < target_ops {
                println!("Auto-sync triggered after {i} ops");
                break;
            }
        }

        let final_pending = gpu_pending_count();
        println!("Final pending: {final_pending}");

        // Should have auto-synced, so pending should be less than threshold
        assert!(
            final_pending < threshold,
            "Auto-sync should have triggered. Expected < {threshold}, got {final_pending}"
        );

        let _ = gpu_sync();
    }

    /// Test 5: Stress test - long chain of operations
    ///
    /// Run a very long chain to see if we can cause a timeout
    #[test]
    fn test_long_operation_chain() {
        let _serial = serial();
        if !gpu::is_gpu_available() {
            println!("Skipping GPU stress test - no GPU available");
            return;
        }

        let t = gpu_tensor(4096);

        println!("Starting long chain of 1000 operations");

        for i in 0..1000 {
            let _ = t.relu();

            if i % 100 == 0 {
                let pending = gpu_pending_count();
                println!("Op {i}: pending = {pending}");
            }
        }

        println!("Final pending before sync: {}", gpu_pending_count());
        let _ = gpu_sync();
        println!("After sync: {}", gpu_pending_count());
    }

    /// Test 6: Mixed operation sizes
    ///
    /// Tests with both small and large tensors to see if size matters
    #[test]
    fn test_mixed_operation_sizes() {
        let _serial = serial();
        if !gpu::is_gpu_available() {
            println!("Skipping GPU stress test - no GPU available");
            return;
        }

        let small = gpu_tensor(64);
        let medium = gpu_tensor(1024);
        let large = gpu_tensor(16384);

        let _ = gpu_sync();

        for i in 0..50 {
            let _ = small.relu();
            let _ = medium.relu();
            let _ = large.relu();

            if i % 10 == 0 {
                println!("Iteration {}: pending = {}", i, gpu_pending_count());
            }
        }

        let _ = gpu_sync();
    }

    /// Test 7: Copy operations tracking
    ///
    /// Specifically test whether `copy_region` and `to_vec` track their submissions
    #[test]
    fn test_copy_operations_tracking() {
        let _serial = serial();
        if !gpu::is_gpu_available() {
            println!("Skipping GPU stress test - no GPU available");
            return;
        }

        let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let t = RawTensor::from_vec(data.clone(), &[1024])
            .to_device(Device::gpu().expect("GPU should be available"));

        let _ = gpu_sync();
        let before = gpu_pending_count();
        println!("Before reading back: pending = {before}");

        // Reading data back from GPU does a submit (via to_vec in GpuBuffer)
        let result = t.data();

        let after = gpu_pending_count();
        println!("After reading back: pending = {after}");

        // Verify data is correct
        assert_eq!(result.len(), data.len());

        let _ = gpu_sync();
    }

    /// Test 8: Staging buffer pooling
    ///
    /// Verify staging buffers are reused across multiple GPU→CPU transfers
    #[test]
    fn test_staging_buffer_pooling() {
        let _serial = serial();
        if !gpu::is_gpu_available() {
            println!("Skipping GPU stress test - no GPU available");
            return;
        }

        println!("Testing staging buffer pool with multiple readbacks");

        // Create 10 tensors
        let tensors: Vec<_> = (0..10).map(|_| gpu_tensor(1024)).collect();

        // Read back multiple times - staging buffers should be pooled
        for iteration in 0..5 {
            println!("Readback iteration {iteration}");

            for (i, t) in tensors.iter().enumerate() {
                let data = t.data();
                assert_eq!(data.len(), 1024, "Tensor {i} data length mismatch");

                // Verify data integrity (values should be i * 0.01 for indices)
                let first_val = data[0];
                assert!((0.0..=1.0).contains(&first_val), "Data sanity check failed");
            }

            println!(
                "Iteration {} complete, pending: {}",
                iteration,
                gpu_pending_count()
            );
        }

        let _ = gpu_sync();
        println!("Staging buffer pool test complete");
    }
}

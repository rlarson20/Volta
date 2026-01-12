//! GPU stress tests for command queue management
//!
//! These tests are designed to reproduce and diagnose GPU timeout issues
//! by stressing the command queue with various workload patterns.

#[cfg(feature = "gpu")]
mod gpu_stress_tests {
    use volta::{Device, RawTensor, TensorOps, gpu_pending_count, gpu_sync, gpu_sync_threshold};

    /// Helper to create a test tensor on GPU
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
                println!("Iteration {}, pending ops: {}", iteration, pending);
            }
        }

        let pending_before_sync = gpu_pending_count();
        println!("Before final sync, pending: {}", pending_before_sync);

        gpu_sync();

        let pending_after_sync = gpu_pending_count();
        println!("After final sync, pending: {}", pending_after_sync);
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
        let t = gpu_tensor(1024);

        // Start clean
        gpu_sync();
        assert_eq!(gpu_pending_count(), 0);

        // Do 10 operations
        for i in 1..=10 {
            let _ = t.relu();
            let pending = gpu_pending_count();
            println!("After op {}: pending = {}", i, pending);
        }

        let pending = gpu_pending_count();
        println!(
            "Expected 10 ops (or fewer if auto-synced), got: {}",
            pending
        );

        gpu_sync();
    }

    /// Test 3: Rapid buffer allocation and deallocation
    ///
    /// Stresses the buffer pool by creating and dropping many tensors.
    #[test]
    fn test_buffer_pool_churn() {
        for round in 0..5 {
            println!("Round {}", round);

            // Create 100 tensors (will churn buffer pool)
            let tensors: Vec<_> = (0..100).map(|_| gpu_tensor(1024)).collect();

            println!("After creation, pending: {}", gpu_pending_count());

            // Use them
            for t in &tensors {
                let _ = t.relu();
            }

            println!("After operations, pending: {}", gpu_pending_count());

            // Sync and drop (tensors go back to pool)
            gpu_sync();
            drop(tensors);
        }
    }

    /// Test 4: Verify auto-sync threshold triggers correctly
    ///
    /// This test checks that maybe_sync() actually kicks in when we
    /// exceed the threshold.
    #[test]
    fn test_auto_sync_threshold() {
        let threshold = gpu_sync_threshold();
        println!("Sync threshold: {}", threshold);

        let t = gpu_tensor(256);

        // Start clean
        gpu_sync();
        assert_eq!(gpu_pending_count(), 0);

        // Do threshold + 5 operations
        let target_ops = threshold + 5;
        for i in 1..=target_ops {
            let _ = t.relu();
            let pending = gpu_pending_count();

            if pending == 0 && i < target_ops {
                println!("Auto-sync triggered after {} ops", i);
                break;
            }
        }

        let final_pending = gpu_pending_count();
        println!("Final pending: {}", final_pending);

        // Should have auto-synced, so pending should be less than threshold
        assert!(
            final_pending < threshold,
            "Auto-sync should have triggered. Expected < {}, got {}",
            threshold,
            final_pending
        );

        gpu_sync();
    }

    /// Test 5: Stress test - long chain of operations
    ///
    /// Run a very long chain to see if we can cause a timeout
    #[test]
    fn test_long_operation_chain() {
        let t = gpu_tensor(4096);

        println!("Starting long chain of 1000 operations");

        for i in 0..1000 {
            let _ = t.relu();

            if i % 100 == 0 {
                let pending = gpu_pending_count();
                println!("Op {}: pending = {}", i, pending);
            }
        }

        println!("Final pending before sync: {}", gpu_pending_count());
        gpu_sync();
        println!("After sync: {}", gpu_pending_count());
    }

    /// Test 6: Mixed operation sizes
    ///
    /// Tests with both small and large tensors to see if size matters
    #[test]
    fn test_mixed_operation_sizes() {
        let small = gpu_tensor(64);
        let medium = gpu_tensor(1024);
        let large = gpu_tensor(16384);

        gpu_sync();

        for i in 0..50 {
            let _ = small.relu();
            let _ = medium.relu();
            let _ = large.relu();

            if i % 10 == 0 {
                println!("Iteration {}: pending = {}", i, gpu_pending_count());
            }
        }

        gpu_sync();
    }

    /// Test 7: Copy operations tracking
    ///
    /// Specifically test whether copy_region and to_vec track their submissions
    #[test]
    fn test_copy_operations_tracking() {
        let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let t = RawTensor::from_vec(data.clone(), &[1024]).to_device(Device::gpu().unwrap());

        gpu_sync();
        let before = gpu_pending_count();
        println!("Before reading back: pending = {}", before);

        // Reading data back from GPU does a submit (via to_vec in GpuBuffer)
        let result = t.borrow().data.to_vec();

        let after = gpu_pending_count();
        println!("After reading back: pending = {}", after);

        // Verify data is correct
        assert_eq!(result.len(), data.len());

        gpu_sync();
    }
}

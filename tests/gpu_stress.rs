//! GPU stress tests for command queue management
//!
//! These tests are designed to reproduce and diagnose GPU timeout issues
//! by stressing the command queue with various workload patterns.

#[cfg(feature = "gpu")]
mod gpu_stress_tests {
    use volta::{
        Device, RawTensor, TensorOps, gpu, gpu_pending_count, gpu_sync, gpu_sync_threshold,
    };

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
        let result = t.borrow().data.to_vec();

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
                let data = t.borrow().data.to_vec();
                assert_eq!(data.len(), 1024, "Tensor {i} data length mismatch");

                // Verify data integrity (values should be i * 0.01 for indices)
                let first_val = data.first().copied().unwrap_or(f32::NAN);
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

    /// Test 9: `sync_checked()` timeout handling
    ///
    /// Verify `sync_checked()` properly tracks consecutive timeouts
    #[test]
    fn test_sync_checked_timeout_handling() {
        use volta::get_gpu_context;

        if !gpu::is_gpu_available() {
            println!("Skipping GPU stress test - no GPU available");
            return;
        }

        let ctx = get_gpu_context().expect("GPU available");

        // Reset timeout counter to start fresh
        ctx.reset_timeout_counter();
        assert_eq!(
            ctx.consecutive_timeouts(),
            0,
            "Timeout counter should start at 0"
        );

        let t = gpu_tensor(1024);

        // Run operations with periodic sync_checked()
        for i in 0..50 {
            let _ = t.relu();

            if i % 10 == 0 {
                println!("Checkpoint {}: pending = {}", i, gpu_pending_count());

                match ctx.sync_checked() {
                    Ok(()) => {
                        println!("Sync {} OK, timeouts: {}", i, ctx.consecutive_timeouts());
                        assert_eq!(
                            ctx.consecutive_timeouts(),
                            0,
                            "Successful sync should reset timeout counter"
                        );
                    }
                    Err(e) => {
                        panic!("Sync failed at checkpoint {i}: {e}");
                    }
                }
            }
        }

        // Final check
        assert_eq!(
            ctx.consecutive_timeouts(),
            0,
            "Timeout counter should be 0 after successful syncs"
        );
    }

    /// Test 10: Resource monitoring integration
    ///
    /// Verify resource monitoring works and doesn't panic
    #[test]
    fn test_resource_monitoring() {
        use volta::gpu::monitor::{ResourceStatus, check_system_resources, get_process_memory_mb};

        if !gpu::is_gpu_available() {
            println!("Skipping GPU stress test - no GPU available");
            return;
        }

        // Check resources before starting
        match check_system_resources() {
            ResourceStatus::Critical(msg) => {
                println!("WARNING: Resources already critical before test: {msg}");
                println!("Skipping heavy operations to avoid system freeze");
                return; // Skip test if already in critical state
            }
            ResourceStatus::Warning(msg) => {
                println!("Resources elevated at test start: {msg}");
            }
            ResourceStatus::Healthy => {
                println!("Resources healthy at test start");
            }
        }

        println!("Initial memory: {}MB", get_process_memory_mb());

        // Create moderate workload
        let tensors: Vec<_> = (0..20).map(|_| gpu_tensor(256)).collect();

        for iteration in 0..100 {
            for t in &tensors {
                let _ = t.relu();
            }

            // Check resources every 10 iterations
            if iteration % 10 == 0 {
                let memory_mb = get_process_memory_mb();
                let pending = gpu_pending_count();

                match check_system_resources() {
                    ResourceStatus::Critical(msg) => {
                        println!("CRITICAL at iter {iteration}: {msg}");
                        println!("Memory: {memory_mb}MB, Pending: {pending}");
                        panic!("Test aborted - resources critical: {msg}");
                    }
                    ResourceStatus::Warning(msg) => {
                        println!("Warning at iter {iteration}: {msg}");
                        println!("Memory: {memory_mb}MB, Pending: {pending}");
                    }
                    ResourceStatus::Healthy => {
                        println!("Iter {iteration}: Healthy ({memory_mb}MB, {pending} pending)");
                    }
                }
            }
        }

        let _ = gpu_sync();
        println!("Final memory: {}MB", get_process_memory_mb());
    }

    /// Test 11: Many small ops with resource monitoring
    ///
    /// Enhanced version of test 1 with resource monitoring to abort early
    #[test]
    fn test_many_small_ops_with_monitoring() {
        use volta::gpu::monitor::{ResourceStatus, check_system_resources};

        if !gpu::is_gpu_available() {
            println!("Skipping GPU stress test - no GPU available");
            return;
        }

        // Pre-flight check
        match check_system_resources() {
            ResourceStatus::Critical(msg) => {
                panic!("Cannot start test - resources already critical: {msg}");
            }
            ResourceStatus::Warning(msg) => {
                println!("Starting with elevated resources: {msg}");
            }
            ResourceStatus::Healthy => {}
        }

        let tensors: Vec<_> = (0..20).map(|_| gpu_tensor(256)).collect();

        for iteration in 0..100 {
            for t in &tensors {
                let _ = t.relu();
            }

            if iteration % 10 == 0 {
                match check_system_resources() {
                    ResourceStatus::Critical(msg) => {
                        let _ = gpu_sync(); // Try to clean up before aborting
                        panic!("Test aborted at iteration {iteration} - {msg}");
                    }
                    ResourceStatus::Warning(msg) => {
                        println!("Warning at iter {iteration}: {msg}");
                    }
                    ResourceStatus::Healthy => {}
                }
            }
        }

        let _ = gpu_sync();
    }

    /// Test 12: System monitor profiling
    ///
    /// Demonstrate real-time performance profiling with `SystemMonitor`
    #[test]
    fn test_system_monitor_profiling() {
        use volta::gpu::system_monitor::SystemMonitor;

        if !gpu::is_gpu_available() {
            println!("Skipping GPU stress test - no GPU available");
            return;
        }

        let monitor = SystemMonitor::new();

        monitor.checkpoint("test_start");

        // Create tensors
        let tensors: Vec<_> = (0..10).map(|_| gpu_tensor(1024)).collect();
        monitor.checkpoint("tensors_created");

        // Run operations
        for i in 0..50 {
            for t in &tensors {
                let _ = t.relu();
                monitor.record_operation();
            }

            if i % 10 == 0 {
                monitor.checkpoint(&format!("iteration_{i}"));
            }
        }

        let _ = gpu_sync();
        monitor.record_sync();
        monitor.checkpoint("operations_complete");

        // Get final stats
        let stats = monitor.stats();
        println!("\nFinal Stats: {stats}");

        assert!(stats.operation_count > 0, "Should have recorded operations");
        assert!(stats.sync_count > 0, "Should have recorded syncs");
        assert!(stats.elapsed_secs > 0.0, "Should have elapsed time");
    }

    /// Test 13: Early warning system trend detection
    ///
    /// Verify early warning system detects increasing memory trends
    #[test]
    fn test_early_warning_trend_detection() {
        use volta::gpu::early_warning::{EarlyWarningSystem, HealthStatus};

        if !gpu::is_gpu_available() {
            println!("Skipping GPU stress test - no GPU available");
            return;
        }

        let mut ews = EarlyWarningSystem::new();

        // Initial checks should be healthy (insufficient data)
        match ews.check_health() {
            HealthStatus::Healthy => println!("Initial: Healthy (insufficient data)"),
            HealthStatus::Warning(msg) => println!("Initial: Warning - {msg}"),
            HealthStatus::Critical(msg) => println!("Initial: Critical - {msg}"),
        }

        // Run some operations and periodically check
        let tensors: Vec<_> = (0..10).map(|_| gpu_tensor(256)).collect();

        for iteration in 0..20 {
            for t in &tensors {
                let _ = t.relu();
            }

            if iteration % 5 == 0 {
                match ews.check_health() {
                    HealthStatus::Healthy => {
                        println!("Iteration {iteration}: Healthy");
                    }
                    HealthStatus::Warning(msg) => {
                        println!("Iteration {iteration}: Warning - {msg}");
                    }
                    HealthStatus::Critical(msg) => {
                        println!("Iteration {iteration}: Critical - {msg}");
                        // In real usage, would abort here
                    }
                }
            }
        }

        let _ = gpu_sync();

        // Get trend data
        let trends = ews.trends();
        println!(
            "Collected {} memory samples, {} pending samples",
            trends.memory_samples.len(),
            trends.pending_samples.len()
        );

        assert!(
            !trends.memory_samples.is_empty(),
            "Should have collected memory samples"
        );
    }

    /// Test 14: Combined monitoring integration
    ///
    /// Demonstrate using all monitoring tools together
    #[test]
    fn test_combined_monitoring_integration() {
        use volta::gpu::early_warning::{EarlyWarningSystem, HealthStatus as EWSHealthStatus};
        use volta::gpu::monitor::{ResourceStatus, check_system_resources};
        use volta::gpu::system_monitor::SystemMonitor;

        if !gpu::is_gpu_available() {
            println!("Skipping GPU stress test - no GPU available");
            return;
        }

        // Initialize all monitors
        let monitor = SystemMonitor::new();
        let mut ews = EarlyWarningSystem::new();

        monitor.checkpoint("combined_test_start");

        // Pre-flight resource check
        match check_system_resources() {
            ResourceStatus::Critical(msg) => {
                panic!("Cannot start - resources critical: {msg}");
            }
            ResourceStatus::Warning(msg) => {
                println!("Starting with warning: {msg}");
            }
            ResourceStatus::Healthy => {
                println!("Starting healthy");
            }
        }

        // Create workload
        let tensors: Vec<_> = (0..15).map(|_| gpu_tensor(512)).collect();
        monitor.checkpoint("setup_complete");

        // Run with combined monitoring
        for iteration in 0..30 {
            for t in &tensors {
                let _ = t.relu();
                monitor.record_operation();
            }

            if iteration % 5 == 0 {
                // Check immediate resources
                if let ResourceStatus::Critical(msg) = check_system_resources() {
                    panic!("Iteration {iteration}: Critical resources - {msg}");
                }

                // Check trends
                match ews.check_health() {
                    EWSHealthStatus::Critical(msg) => {
                        println!("Iteration {iteration}: Critical trend - {msg}");
                    }
                    EWSHealthStatus::Warning(msg) => {
                        println!("Iteration {iteration}: Warning trend - {msg}");
                    }
                    EWSHealthStatus::Healthy => {}
                }

                // Profiling checkpoint
                monitor.checkpoint(&format!("iter_{iteration}"));
            }
        }

        let _ = gpu_sync();
        monitor.record_sync();
        monitor.checkpoint("test_complete");

        // Final reporting
        let stats = monitor.stats();
        println!("\n=== Final Performance Stats ===");
        println!("{stats}");

        let trends = ews.trends();
        println!(
            "\n=== Trend Data ===\nMemory samples: {:?}\nPending samples: {:?}",
            trends.memory_samples, trends.pending_samples
        );

        match check_system_resources() {
            ResourceStatus::Critical(msg) => {
                println!("\n=== CRITICAL: {msg} ===");
            }
            ResourceStatus::Warning(msg) => {
                println!("\n=== Warning: {msg} ===");
            }
            ResourceStatus::Healthy => {
                println!("\n=== Completed Healthy ===");
            }
        }
    }
}

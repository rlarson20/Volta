//! Real-time system monitoring for GPU performance profiling
//!
//! Provides detailed performance tracking and profiling capabilities for
//! debugging GPU operations and understanding performance characteristics.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

/// Real-time system monitor for GPU performance profiling
///
/// Tracks performance metrics over time with checkpoint-based profiling.
/// Useful for debugging GPU operations and identifying performance bottlenecks.
///
/// # Example
///
/// ```ignore
/// use volta::gpu::system_monitor::SystemMonitor;
///
/// let monitor = SystemMonitor::new();
///
/// monitor.checkpoint("setup_start");
/// // ... GPU operations ...
/// monitor.checkpoint("setup_complete");
///
/// // ... more operations ...
///
/// let stats = monitor.stats();
/// println!("Ops/sec: {:.2}", stats.ops_per_sec);
/// ```
pub struct SystemMonitor {
    start_time: Instant,
    peak_memory_mb: AtomicUsize,
    operation_count: AtomicU64,
    sync_count: AtomicU64,
    last_checkpoint_ops: AtomicU64,
    last_checkpoint_time: Mutex<Instant>,
}

impl SystemMonitor {
    /// Create a new system monitor
    #[must_use]
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            peak_memory_mb: AtomicUsize::new(0),
            operation_count: AtomicU64::new(0),
            sync_count: AtomicU64::new(0),
            last_checkpoint_ops: AtomicU64::new(0),
            last_checkpoint_time: Mutex::new(Instant::now()),
        }
    }

    /// Record a checkpoint with a descriptive label
    ///
    /// Prints performance metrics including memory usage, operation counts,
    /// operations per second, sync count, and pending GPU operations.
    ///
    /// # Arguments
    /// * `label` - Descriptive name for this checkpoint
    /// # Panics
    /// unwrap `last_time` mutex
    pub fn checkpoint(&self, label: &str) {
        let current_memory_mb = crate::gpu::monitor::get_process_memory_mb();
        let _peak = self
            .peak_memory_mb
            .fetch_max(current_memory_mb, Ordering::Relaxed);
        let ops = self.operation_count.load(Ordering::Relaxed);
        let syncs = self.sync_count.load(Ordering::Relaxed);
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let pending = crate::gpu::gpu_pending_count();

        // Calculate delta since last checkpoint
        let last_ops = self.last_checkpoint_ops.swap(ops, Ordering::Relaxed);
        let ops_delta = ops - last_ops;

        let mut last_time = self.last_checkpoint_time.lock().unwrap();
        let time_delta = last_time.elapsed().as_secs_f64();
        *last_time = Instant::now();
        drop(last_time);

        let ops_per_sec = if time_delta > 0.0 {
            ops_delta as f64 / time_delta
        } else {
            0.0
        };

        println!(
            "[{label}] Mem: {current_memory_mb}MB, Ops: {ops} (+{ops_delta}, {ops_per_sec:.0}/s), Syncs: {syncs}, Pending: {pending}, Time: {elapsed:.2}s"
        );
    }

    /// Record a GPU operation
    ///
    /// Increments the operation counter. Call this after each GPU operation
    /// to track total workload.
    pub fn record_operation(&self) {
        self.operation_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a GPU sync
    ///
    /// Increments the sync counter. Call this after each `gpu_sync()` call
    /// to track synchronization frequency.
    pub fn record_sync(&self) {
        self.sync_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current performance statistics
    ///
    /// Returns a snapshot of performance metrics including elapsed time,
    /// peak memory usage, operation counts, and throughput.
    pub fn stats(&self) -> MonitorStats {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let ops = self.operation_count.load(Ordering::Relaxed);

        MonitorStats {
            elapsed_secs: elapsed,
            peak_memory_mb: self.peak_memory_mb.load(Ordering::Relaxed),
            operation_count: ops,
            sync_count: self.sync_count.load(Ordering::Relaxed),
            ops_per_sec: if elapsed > 0.0 {
                ops as f64 / elapsed
            } else {
                0.0
            },
        }
    }

    /// Reset all counters and timers
    ///
    /// Useful for starting fresh profiling sessions without creating
    /// a new monitor instance.
    /// # Panics
    /// unwrap `last_checkpoint_time` mutex
    pub fn reset(&self) {
        self.peak_memory_mb.store(0, Ordering::Relaxed);
        self.operation_count.store(0, Ordering::Relaxed);
        self.sync_count.store(0, Ordering::Relaxed);
        self.last_checkpoint_ops.store(0, Ordering::Relaxed);
        *self.last_checkpoint_time.lock().unwrap() = Instant::now();
    }
}

impl Default for SystemMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot of system monitor performance statistics
#[derive(Debug, Clone)]
pub struct MonitorStats {
    /// Total elapsed time in seconds
    pub elapsed_secs: f64,
    /// Peak memory usage in megabytes
    pub peak_memory_mb: usize,
    /// Total GPU operations recorded
    pub operation_count: u64,
    /// Total GPU sync operations
    pub sync_count: u64,
    /// Average operations per second
    pub ops_per_sec: f64,
}

impl std::fmt::Display for MonitorStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MonitorStats {{ elapsed: {:.2}s, peak_mem: {}MB, ops: {}, syncs: {}, throughput: {:.0} ops/s }}",
            self.elapsed_secs,
            self.peak_memory_mb,
            self.operation_count,
            self.sync_count,
            self.ops_per_sec
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitor_creation() {
        let monitor = SystemMonitor::new();
        let stats = monitor.stats();
        assert_eq!(stats.operation_count, 0);
        assert_eq!(stats.sync_count, 0);
    }

    #[test]
    fn test_record_operations() {
        let monitor = SystemMonitor::new();

        monitor.record_operation();
        monitor.record_operation();
        monitor.record_operation();

        let stats = monitor.stats();
        assert_eq!(stats.operation_count, 3);
    }

    #[test]
    fn test_record_syncs() {
        let monitor = SystemMonitor::new();

        monitor.record_sync();
        monitor.record_sync();

        let stats = monitor.stats();
        assert_eq!(stats.sync_count, 2);
    }

    #[test]
    fn test_reset() {
        let monitor = SystemMonitor::new();

        monitor.record_operation();
        monitor.record_sync();
        monitor.reset();

        let stats = monitor.stats();
        assert_eq!(stats.operation_count, 0);
        assert_eq!(stats.sync_count, 0);
    }

    #[test]
    fn test_checkpoint_does_not_panic() {
        let monitor = SystemMonitor::new();
        monitor.checkpoint("test");
        // Should complete without panic
    }

    #[test]
    fn test_stats_display() {
        let monitor = SystemMonitor::new();
        monitor.record_operation();
        let stats = monitor.stats();
        let display = format!("{stats}");
        assert!(display.contains("ops: 1"));
    }
}

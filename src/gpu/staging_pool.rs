use std::collections::HashMap;
use std::sync::Mutex;
use wgpu::Buffer;

/// Pool for MAP_READ staging buffers used in GPU→CPU transfers
///
/// **Design**: Exact size matching with global capacity limit. Staging buffers
/// are short-lived read-only buffers used for GPU→CPU data transfers.
///
/// **Why pooling?** `GpuBuffer::to_vec()` creates fresh staging buffers on every
/// call. In benchmark loops with many readbacks, staging buffers accumulate
/// and cause memory pressure. This pool caps concurrent staging buffers at 64,
/// preventing unbounded memory growth.
///
/// **Pool strategy**:
/// - Exact size matching (not power-of-2 bucketing) - simpler, less waste
/// - Global limit of 64 buffers - conservative, sufficient for readback use case
/// - Fail-safe: Returns None if pool full, caller creates new buffer
///
/// # Example
///
/// ```ignore
/// let pool = StagingBufferPool::default();
///
/// // Try to acquire from pool
/// let buffer = pool.acquire(size_bytes)
///     .unwrap_or_else(|| create_new_staging_buffer(size_bytes));
///
/// // ... use buffer for GPU→CPU transfer ...
///
/// // Return to pool (or drop if full)
/// pool.release(buffer, size_bytes);
/// ```
pub struct StagingBufferPool {
    /// Pools organized by exact byte size
    pools: Mutex<HashMap<u64, Vec<Buffer>>>,
    /// Maximum total staging buffers across all sizes
    max_total: usize,
    /// Current number of pooled buffers
    current_count: Mutex<usize>,
}

impl StagingBufferPool {
    /// Create new staging buffer pool with specified capacity
    ///
    /// # Arguments
    /// * `max_total` - Maximum number of concurrent staging buffers: default 64
    #[must_use]
    pub fn new(max_total: usize) -> Self {
        Self {
            pools: Mutex::new(HashMap::new()),
            max_total,
            current_count: Mutex::new(0),
        }
    }

    /// Try to acquire staging buffer of exact size from pool
    ///
    /// Returns `None` if no buffer of this size is available in the pool.
    /// Caller should create a new buffer and use it directly.
    ///
    /// # Arguments
    /// * `size_bytes` - Exact size in bytes of the staging buffer needed
    pub fn acquire(&self, size_bytes: u64) -> Option<Buffer> {
        let mut pools = self.pools.lock().unwrap();
        let mut count = self.current_count.lock().unwrap();

        if let Some(pool) = pools.get_mut(&size_bytes)
            && let Some(buffer) = pool.pop()
        {
            *count = count.saturating_sub(1);

            #[cfg(debug_assertions)]
            if std::env::var("VOLTA_GPU_DEBUG").is_ok() {
                eprintln!(
                    "[StagingPool] Acquired {}B buffer from pool ({} remain)",
                    size_bytes, *count
                );
            }

            return Some(buffer);
        }

        None
    }

    /// Return staging buffer to pool for reuse
    ///
    /// Returns `true` if buffer was added to pool, `false` if pool is full
    /// (buffer will be dropped and freed).
    ///
    /// # Arguments
    /// * `buffer` - The staging buffer to return to the pool
    /// * `size_bytes` - Size of the buffer in bytes (must match buffer's actual size)
    pub fn release(&self, buffer: Buffer, size_bytes: u64) -> bool {
        let mut pools = self.pools.lock().unwrap();
        let mut count = self.current_count.lock().unwrap();

        if *count >= self.max_total {
            #[cfg(debug_assertions)]
            if std::env::var("VOLTA_GPU_DEBUG").is_ok() {
                eprintln!(
                    "[StagingPool] Pool full ({}/{}), dropping {}B buffer",
                    *count, self.max_total, size_bytes
                );
            }

            return false; // Pool full, buffer will be dropped
        }

        pools.entry(size_bytes).or_default().push(buffer);
        *count += 1;

        #[cfg(debug_assertions)]
        if std::env::var("VOLTA_GPU_DEBUG").is_ok() {
            eprintln!(
                "[StagingPool] Returned {}B buffer to pool ({} total)",
                size_bytes, *count
            );
        }

        true
    }

    /// Clear all buffers from the pool
    ///
    /// This drops all pooled buffers, freeing their GPU memory.
    /// Useful for cleanup or resetting state during testing.
    pub fn clear(&self) {
        self.pools.lock().unwrap().clear();
        *self.current_count.lock().unwrap() = 0;

        #[cfg(debug_assertions)]
        if std::env::var("VOLTA_GPU_DEBUG").is_ok() {
            eprintln!("[StagingPool] Cleared all buffers");
        }
    }

    /// Get current number of pooled buffers
    pub fn size(&self) -> usize {
        *self.current_count.lock().unwrap()
    }

    /// Get maximum pool capacity
    pub fn capacity(&self) -> usize {
        self.max_total
    }

    /// Get statistics about pool usage
    ///
    /// Returns a snapshot of the current pool state for diagnostics.
    pub fn stats(&self) -> StagingPoolStats {
        StagingPoolStats {
            total_pooled: *self.current_count.lock().unwrap(),
            max_capacity: self.max_total,
        }
    }
}

/// Statistics about staging buffer pool usage
#[derive(Debug, Clone)]
pub struct StagingPoolStats {
    /// Total number of buffers currently in the pool
    pub total_pooled: usize,
    /// Maximum pool capacity
    pub max_capacity: usize,
}

impl Default for StagingBufferPool {
    fn default() -> Self {
        Self::new(64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_starts_empty() {
        let pool = StagingBufferPool::new(10);
        assert_eq!(pool.size(), 0);
        assert_eq!(pool.capacity(), 10);
    }

    #[test]
    fn test_acquire_from_empty_pool_returns_none() {
        let pool = StagingBufferPool::new(10);
        assert!(pool.acquire(1024).is_none());
    }

    #[test]
    fn test_clear_resets_count() {
        let pool = StagingBufferPool::new(10);
        *pool.current_count.lock().unwrap() = 5; // Simulate some buffers
        pool.clear();
        assert_eq!(pool.size(), 0);
    }

    #[test]
    fn test_capacity_limit() {
        let pool = StagingBufferPool::new(2);
        assert_eq!(pool.capacity(), 2);
    }
}

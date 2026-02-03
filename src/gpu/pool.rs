//! GPU buffer pooling for efficient memory reuse
//!
//! This module provides a buffer pool that reuses GPU buffers across operations,
//! preventing memory exhaustion during repeated GPU operations like benchmarking
//! or training loops.
//!
//! # Design
//!
//! Buffers are organized into size buckets (power-of-2) for efficient matching.
//! When a buffer is requested, the pool first checks for an available buffer
//! of the appropriate size. When a buffer is dropped, it's returned to the pool
//! for future reuse.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

/// Configuration for the buffer pool
#[derive(Debug, Clone)]
pub struct BufferPoolConfig {
    /// Maximum number of buffers per size bucket
    pub max_buffers_per_bucket: usize,
    /// Maximum total buffers across all buckets
    pub max_total_buffers: usize,
    /// Maximum buffer size (in bytes) to pool (larger buffers are not pooled)
    pub max_pooled_size: usize,
}

impl Default for BufferPoolConfig {
    fn default() -> Self {
        Self {
            max_buffers_per_bucket: 8,         // Keep up to 8 buffers of each size
            max_total_buffers: 64,             // Total pool capacity
            max_pooled_size: 64 * 1024 * 1024, // Pool buffers up to 64MB
        }
    }
}

/// A pool of reusable GPU buffers, organized by size buckets
///
/// # Design Rationale
///
/// GPU buffer allocation is expensive (involves driver calls and memory mapping).
/// This pool enables buffer reuse to:
/// 1. Reduce allocation overhead for repeated operations
/// 2. Minimize memory fragmentation
/// 3. Prevent memory exhaustion during benchmarking/training
///
/// Buffers are bucketed by size (rounded up to power-of-2) for efficient matching.
/// A 5000-byte request gets a buffer from the 8192-byte bucket.
///
/// # Thread Safety
///
/// All operations are protected by a Mutex, making the pool safe for concurrent
/// access (though Volta is currently single-threaded, this future-proofs the design).
pub struct BufferPool {
    /// Free buffers organized by size bucket (key = log2(size))
    buckets: Mutex<HashMap<u32, VecDeque<wgpu::Buffer>>>,
    /// Configuration
    config: BufferPoolConfig,
    /// Current total buffer count (for capacity limiting)
    total_count: Mutex<usize>,
}

impl BufferPool {
    /// Create a new buffer pool with the given configuration
    #[must_use]
    pub fn new(config: BufferPoolConfig) -> Self {
        Self {
            buckets: Mutex::new(HashMap::new()),
            config,
            total_count: Mutex::new(0),
        }
    }

    /// Create a buffer pool with default configuration
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(BufferPoolConfig::default())
    }

    /// Round size up to next power of 2 and return the bucket index (log2)
    ///
    /// Examples:
    /// - 1000 bytes → bucket 10 (1024 bytes)
    /// - 5000 bytes → bucket 13 (8192 bytes)
    /// - 1 byte → bucket 0 (1 byte)
    const fn size_to_bucket(size: usize) -> u32 {
        if size == 0 {
            return 0;
        }
        // Find log2(next_power_of_2(size))
        // For size=1000: leading_zeros of 999 = bits needed
        usize::BITS - (size - 1).leading_zeros()
    }

    /// Get the actual byte size for a bucket
    const fn bucket_to_size(bucket: u32) -> usize {
        1usize << bucket
    }

    /// Get the allocation size for a given minimum size
    ///
    /// This rounds up to the next power of 2, ensuring all buffers in a bucket
    /// have the same size and can be safely reused.
    #[must_use]
    pub const fn allocation_size(min_size: usize) -> usize {
        if min_size == 0 {
            return 0;
        }
        let bucket = Self::size_to_bucket(min_size);
        Self::bucket_to_size(bucket)
    }

    /// Try to acquire a buffer from the pool
    ///
    /// Returns `Some(buffer)` if a suitable buffer is available,
    /// `None` if the pool is empty for that size or the size is too large.
    ///
    /// # Arguments
    /// * `min_size` - Minimum required size in bytes
    /// # Panics
    /// Unwrapping bucket mutex
    pub fn acquire(&self, min_size: usize) -> Option<wgpu::Buffer> {
        if min_size > self.config.max_pooled_size {
            return None; // Too large to pool
        }

        let bucket = Self::size_to_bucket(min_size);

        if let Some(queue) = self.buckets.lock().unwrap().get_mut(&bucket)
            && let Some(buffer) = queue.pop_front()
        {
            let mut count = self.total_count.lock().unwrap();
            *count = count.saturating_sub(1);
            drop(count);
            return Some(buffer);
        }

        None
    }

    /// Return a buffer to the pool for reuse
    ///
    /// Returns `true` if the buffer was pooled, `false` if it was rejected
    /// (e.g., pool full or buffer too large). Rejected buffers should be
    /// dropped normally.
    ///
    /// # Arguments
    /// * `buffer` - The GPU buffer to return
    /// * `size` - The buffer size in bytes
    /// # Panics
    /// Unwrapping bucket mutex
    #[expect(
        clippy::significant_drop_tightening,
        reason = "Attempts to fix by merging the constructon with its single usage leads to other errors, will come back to fix this."
    )]
    pub fn release(&self, buffer: wgpu::Buffer, size: usize) -> bool {
        if size > self.config.max_pooled_size {
            return false; // Too large to pool
        }

        let bucket = Self::size_to_bucket(size);
        let mut buckets = self.buckets.lock().unwrap();
        let mut count = self.total_count.lock().unwrap();

        // Check total capacity
        if *count >= self.config.max_total_buffers {
            return false; // Pool full
        }

        let queue = buckets.entry(bucket).or_default();

        // Check per-bucket capacity
        if queue.len() >= self.config.max_buffers_per_bucket {
            return false; // Bucket full
        }

        queue.push_back(buffer);
        *count += 1;
        true
    }

    /// Clear all buffers from the pool
    ///
    /// This immediately frees all GPU memory held by the pool.
    /// Useful for reducing memory pressure or before shutdown.
    /// # Panics
    /// Unwrapping bucket mutex
    pub fn clear(&self) {
        self.buckets.lock().unwrap().clear();
        *self.total_count.lock().unwrap() = 0;
    }

    /// Get statistics about pool usage
    /// # Panics
    /// Unwrapping bucket mutex
    #[allow(dead_code)]
    pub fn stats(&self) -> BufferPoolStats {
        let total = *self.total_count.lock().unwrap();

        let bucket_stats: Vec<(usize, usize)> = self
            .buckets
            .lock()
            .unwrap()
            .iter()
            .map(|(&bucket, queue)| (Self::bucket_to_size(bucket), queue.len()))
            .collect();

        BufferPoolStats {
            total_pooled: total,
            buckets: bucket_stats,
        }
    }
}

/// Statistics about buffer pool usage
#[derive(Debug)]
#[allow(dead_code)]
pub struct BufferPoolStats {
    /// Total number of buffers currently in the pool
    pub total_pooled: usize,
    /// Buffer counts per bucket (`bucket_size`, count)
    pub buckets: Vec<(usize, usize)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_to_bucket() {
        // Power of 2 sizes map exactly
        // size=1 → bucket 0 (1 byte)
        // size=2 → bucket 1 (2 bytes)
        // size=4 → bucket 2 (4 bytes)
        assert_eq!(BufferPool::size_to_bucket(1), 0);
        assert_eq!(BufferPool::size_to_bucket(2), 1);
        assert_eq!(BufferPool::size_to_bucket(4), 2);
        assert_eq!(BufferPool::size_to_bucket(1024), 10);

        // Non-power-of-2 sizes round up
        assert_eq!(BufferPool::size_to_bucket(3), 2); // → bucket 2 (4 bytes)
        assert_eq!(BufferPool::size_to_bucket(1000), 10); // → bucket 10 (1024 bytes)
        assert_eq!(BufferPool::size_to_bucket(5000), 13); // → bucket 13 (8192 bytes)

        // Edge case
        assert_eq!(BufferPool::size_to_bucket(0), 0);
    }

    #[test]
    fn test_bucket_to_size() {
        assert_eq!(BufferPool::bucket_to_size(0), 1);
        assert_eq!(BufferPool::bucket_to_size(1), 2);
        assert_eq!(BufferPool::bucket_to_size(10), 1024);
        assert_eq!(BufferPool::bucket_to_size(20), 1024 * 1024);
    }

    #[test]
    fn test_allocation_size() {
        // Verify allocation_size rounds up to power of 2
        assert_eq!(BufferPool::allocation_size(1), 1);
        assert_eq!(BufferPool::allocation_size(2), 2);
        assert_eq!(BufferPool::allocation_size(3), 4);
        assert_eq!(BufferPool::allocation_size(5), 8);
        assert_eq!(BufferPool::allocation_size(1000), 1024);
        assert_eq!(BufferPool::allocation_size(0), 0);
    }

    #[test]
    fn test_config_limits() {
        let config = BufferPoolConfig {
            max_buffers_per_bucket: 2,
            max_total_buffers: 4,
            max_pooled_size: 1024,
        };
        let pool = BufferPool::new(config);

        // Size above max should not be pooled
        assert!(pool.acquire(2048).is_none());
    }
}

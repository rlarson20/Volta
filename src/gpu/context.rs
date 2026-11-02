//! GPU context management
//!
//! The `GpuContext` holds the wgpu device and queue, which are needed
//! for all GPU operations. Think of it as your "connection" to the GPU.

use wgpu::PipelineCompilationOptions;

use super::pool::{BufferPool, BufferPoolConfig};
use super::staging_pool::StagingBufferPool;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

/// Manages the GPU device, queue, and compiled compute pipelines
/// Default threshold for pending submissions before forcing a sync.
/// This prevents GPU command queue exhaustion during rapid-fire operations.
const DEFAULT_SYNC_THRESHOLD: u32 = 16;

/// Hard timeout for GPU sync operations (M2 Mac optimized)
/// Set to 5s to accommodate unified memory architecture and thermal throttling
const HARD_SYNC_TIMEOUT_SECS: u64 = 5;

/// Maximum consecutive timeouts before aborting (3-strike rule)
const MAX_CONSECUTIVE_TIMEOUTS: u32 = 3;

/// Get the hard sync timeout from environment or default
///
/// Allows runtime configuration for testing:
/// ```bash
/// VOLTA_GPU_SYNC_TIMEOUT=5 cargo bench --bench gpu_comparison
/// ```
fn get_sync_timeout_secs() -> u64 {
    std::env::var("VOLTA_GPU_SYNC_TIMEOUT")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(HARD_SYNC_TIMEOUT_SECS)
}

/// GPU synchronization errors
#[derive(Debug)]
pub enum GpuSyncError {
    /// Sync timed out (potentially recoverable)
    Timeout(String),
    /// GPU appears lost after consecutive timeouts
    GpuLost(String),
}

impl std::fmt::Display for GpuSyncError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Timeout(msg) => write!(f, "GPU sync timeout: {msg}"),
            Self::GpuLost(msg) => write!(f, "GPU lost: {msg}"),
        }
    }
}

impl std::error::Error for GpuSyncError {}

/// Handle to the GPU: device, queue, compiled pipelines, buffer pools, and submission counters.
pub struct GpuContext {
    /// The GPU device - represents the actual hardware
    device: wgpu::Device,
    /// Command queue - where we submit work to the GPU
    queue: wgpu::Queue,
    /// Adapter info for debugging
    adapter_info: wgpu::AdapterInfo,
    /// Pre-compiled compute pipelines for common operations
    pipelines: ComputePipelines,
    /// Buffer pool for reusing GPU memory allocations
    buffer_pool: Arc<BufferPool>,
    /// Staging buffer pool for GPU→CPU transfers
    staging_pool: Arc<StagingBufferPool>,
    /// Counter for pending GPU submissions (for back-pressure throttling)
    pending_submissions: AtomicU32,
    /// Threshold before forcing a sync to prevent command queue exhaustion
    sync_threshold: u32,
    /// Consecutive timeout counter for 3-strike abort rule
    consecutive_timeouts: AtomicU32,
}

/// The backward pipelines of one differentiable binary op, looked up from its
/// `define_ops!` row.
pub(crate) struct BinaryBackwardPipelines<'p> {
    /// Same-shape gradient wrt the first input.
    pub a: &'p wgpu::ComputePipeline,
    /// Same-shape gradient wrt the second input.
    pub b: &'p wgpu::ComputePipeline,
    /// Broadcast backward, scatter pass.
    pub pass1: &'p wgpu::ComputePipeline,
    /// Broadcast backward, reduce pass.
    pub pass2: &'p wgpu::ComputePipeline,
}

/// Collection of pre-compiled compute pipelines
///
/// Compiling shaders is expensive, so we do it once at initialization
/// and reuse the pipelines for all operations.
pub struct ComputePipelines {
    // Element-wise operations
    /// Elementwise `a + b`.
    pub add: wgpu::ComputePipeline,
    /// Elementwise `a - b`.
    pub sub: wgpu::ComputePipeline,
    /// Elementwise `a * b`.
    pub mul: wgpu::ComputePipeline,
    /// Elementwise `a / b`.
    pub div: wgpu::ComputePipeline,
    /// Elementwise `max(a, b)`.
    pub max: wgpu::ComputePipeline,
    /// Elementwise `a mod b`.
    pub mod_op: wgpu::ComputePipeline,
    /// Elementwise the comparison `a < b`.
    pub cmplt: wgpu::ComputePipeline,

    // Unary operations
    /// Elementwise `-x`.
    pub neg: wgpu::ComputePipeline,
    /// Elementwise `exp(x)`.
    pub exp: wgpu::ComputePipeline,
    /// Elementwise `ln(x)`.
    pub log: wgpu::ComputePipeline,
    /// Elementwise `max(x, 0)`.
    pub relu: wgpu::ComputePipeline,
    /// Elementwise `sigmoid(x)`.
    pub sigmoid: wgpu::ComputePipeline,
    /// Elementwise `tanh(x)`.
    pub tanh: wgpu::ComputePipeline,
    /// Elementwise `sqrt(x)`.
    pub sqrt: wgpu::ComputePipeline,
    /// Elementwise `1 / x`.
    pub recip: wgpu::ComputePipeline,
    /// Elementwise `2^x`.
    pub exp2: wgpu::ComputePipeline,
    /// Elementwise `log2(x)`.
    pub log2: wgpu::ComputePipeline,
    /// Elementwise `sin(x)`.
    pub sin: wgpu::ComputePipeline,
    /// Elementwise `cos(x)`.
    pub cos: wgpu::ComputePipeline,
    /// Elementwise `erf(x)`.
    pub erf: wgpu::ComputePipeline,

    // Unary backward operations
    /// Backward of `neg`: the input gradient.
    pub neg_backward: wgpu::ComputePipeline,
    /// Backward of `exp`: the input gradient.
    pub exp_backward: wgpu::ComputePipeline,
    /// Backward of `log`: the input gradient.
    pub log_backward: wgpu::ComputePipeline,
    /// Backward of `relu`: the input gradient.
    pub relu_backward: wgpu::ComputePipeline,
    /// Backward of `sigmoid`: the input gradient.
    pub sigmoid_backward: wgpu::ComputePipeline,
    /// Backward of `tanh`: the input gradient.
    pub tanh_backward: wgpu::ComputePipeline,
    /// Backward of `sqrt`: the input gradient.
    pub sqrt_backward: wgpu::ComputePipeline,
    /// Backward of `recip`: the input gradient.
    pub recip_backward: wgpu::ComputePipeline,
    /// Backward of `exp2`: the input gradient.
    pub exp2_backward: wgpu::ComputePipeline,
    /// Backward of `log2`: the input gradient.
    pub log2_backward: wgpu::ComputePipeline,
    /// Backward of `sin`: the input gradient.
    pub sin_backward: wgpu::ComputePipeline,
    /// Backward of `cos`: the input gradient.
    pub cos_backward: wgpu::ComputePipeline,
    /// Backward of `erf`: the input gradient.
    pub erf_backward: wgpu::ComputePipeline,

    // Binary backward operations (same-shape only, legacy)
    /// Same-shape backward of `add` for the first operand.
    pub add_backward_a: wgpu::ComputePipeline,
    /// Same-shape backward of `add` for the second operand.
    pub add_backward_b: wgpu::ComputePipeline,
    /// Same-shape backward of `sub` for the first operand.
    pub sub_backward_a: wgpu::ComputePipeline,
    /// Same-shape backward of `sub` for the second operand.
    pub sub_backward_b: wgpu::ComputePipeline,
    /// Same-shape backward of `mul` for the first operand.
    pub mul_backward_a: wgpu::ComputePipeline,
    /// Same-shape backward of `mul` for the second operand.
    pub mul_backward_b: wgpu::ComputePipeline,
    /// Same-shape backward of `div` for the first operand.
    pub div_backward_a: wgpu::ComputePipeline,
    /// Same-shape backward of `div` for the second operand.
    pub div_backward_b: wgpu::ComputePipeline,
    /// Same-shape backward of `max` for the first operand.
    pub max_backward_a: wgpu::ComputePipeline,
    /// Same-shape backward of `max` for the second operand.
    pub max_backward_b: wgpu::ComputePipeline,

    // Binary backward operations with RACE-FREE broadcasting (two-pass)
    /// Broadcast backward of `add`, scatter pass: records each output element's contribution in a temporary buffer.
    pub add_broadcast_pass1: wgpu::ComputePipeline,
    /// Broadcast backward of `add`, reduce pass: accumulates the temporary buffer into the operand gradients without write races.
    pub add_broadcast_pass2: wgpu::ComputePipeline,
    /// Broadcast backward of `sub`, scatter pass: records each output element's contribution in a temporary buffer.
    pub sub_broadcast_pass1: wgpu::ComputePipeline,
    /// Broadcast backward of `sub`, reduce pass: accumulates the temporary buffer into the operand gradients without write races.
    pub sub_broadcast_pass2: wgpu::ComputePipeline,
    /// Broadcast backward of `mul`, scatter pass: records each output element's contribution in a temporary buffer.
    pub mul_broadcast_pass1: wgpu::ComputePipeline,
    /// Broadcast backward of `mul`, reduce pass: accumulates the temporary buffer into the operand gradients without write races.
    pub mul_broadcast_pass2: wgpu::ComputePipeline,
    /// Broadcast backward of `div`, scatter pass: records each output element's contribution in a temporary buffer.
    pub div_broadcast_pass1: wgpu::ComputePipeline,
    /// Broadcast backward of `div`, reduce pass: accumulates the temporary buffer into the operand gradients without write races.
    pub div_broadcast_pass2: wgpu::ComputePipeline,
    /// Broadcast backward of `max`, scatter pass: records each output element's contribution in a temporary buffer.
    pub max_broadcast_pass1: wgpu::ComputePipeline,
    /// Broadcast backward of `max`, reduce pass: accumulates the temporary buffer into the operand gradients without write races.
    pub max_broadcast_pass2: wgpu::ComputePipeline,

    // Reductions
    /// Reduces all elements to their sum.
    pub sum_reduce: wgpu::ComputePipeline,
    /// Reduces all elements to their maximum.
    pub max_reduce: wgpu::ComputePipeline,
    /// Reduces all elements to their mean.
    pub mean_reduce: wgpu::ComputePipeline,

    // Movement operations
    /// Forward `permute` movement op.
    pub permute: wgpu::ComputePipeline,
    /// Forward `expand` movement op.
    pub expand: wgpu::ComputePipeline,
    /// Forward `pad` movement op.
    pub pad: wgpu::ComputePipeline,
    /// Forward `shrink` movement op.
    pub shrink: wgpu::ComputePipeline,
    /// Forward `stride` movement op.
    pub stride: wgpu::ComputePipeline,

    // Movement backward operations
    /// Backward of the `permute` movement op.
    pub permute_backward: wgpu::ComputePipeline,
    /// Backward of the `expand` movement op.
    pub expand_backward: wgpu::ComputePipeline,
    /// Backward of the `pad` movement op.
    pub pad_backward: wgpu::ComputePipeline,
    /// Backward of the `shrink` movement op.
    pub shrink_backward: wgpu::ComputePipeline,
    /// Backward of the `stride` movement op.
    pub stride_backward: wgpu::ComputePipeline,

    // Matrix multiplication (this is the big one for ML!)
    /// Batched matrix product.
    pub matmul: wgpu::ComputePipeline,

    // Matrix multiplication backward
    /// Matmul gradient for the left operand.
    pub matmul_backward_a: wgpu::ComputePipeline,
    /// Matmul gradient for the right operand.
    pub matmul_backward_b: wgpu::ComputePipeline,

    // Reduction backward
    /// Backward of the full sum reduction.
    pub sum_backward: wgpu::ComputePipeline,
    /// Backward of the full mean reduction.
    pub mean_backward: wgpu::ComputePipeline,
    /// Backward of the full max reduction: the gradient goes to the maximum element.
    pub max_backward: wgpu::ComputePipeline,

    // Optimizer step
    /// Parameter update shared by SGD, SGD with momentum, and Adam.
    pub optimizer_step: wgpu::ComputePipeline,

    // Muon: momentum, Frobenius norm, and the elementwise pieces of Newton-Schulz
    /// Muon momentum buffer update (optionally Nesterov).
    pub muon_momentum: wgpu::ComputePipeline,
    /// Sum of squares of a buffer, for the Frobenius norm.
    pub muon_sum_squares: wgpu::ComputePipeline,
    /// Divides a buffer by the norm from `muon_sum_squares` (plus an epsilon).
    pub muon_normalize: wgpu::ComputePipeline,
    /// Computes `alpha * x + beta * y`.
    pub muon_axpby: wgpu::ComputePipeline,

    // Image-to-column transformation for convolution
    /// Unfolds input windows into columns.
    pub im2col: wgpu::ComputePipeline,
    /// Folds columns back into an image, summing overlaps.
    pub col2im: wgpu::ComputePipeline,

    // Direct convolution (memory efficient)
    /// Direct sliding-window convolution.
    pub direct_conv: wgpu::ComputePipeline,

    // Implicit GEMM convolution (balanced performance/memory)
    /// Implicit-GEMM convolution.
    pub igemm: wgpu::ComputePipeline,

    // Convolution backward operations
    /// Input gradient of direct convolution.
    pub conv_backward_input: wgpu::ComputePipeline,
    /// Weight gradient of direct convolution.
    pub conv_backward_weight: wgpu::ComputePipeline,

    // iGEMM backward operations
    /// Input gradient of implicit-GEMM convolution.
    pub igemm_backward_input: wgpu::ComputePipeline,
    /// Weight gradient of implicit-GEMM convolution.
    pub igemm_backward_weight: wgpu::ComputePipeline,
}

impl GpuContext {
    /// Initialize the GPU context
    ///
    /// This is an expensive operation that:
    /// 1. Finds a suitable GPU adapter
    /// 2. Creates a device and queue
    /// 3. Compiles all our compute shaders
    /// # Errors
    /// A message naming the step that failed: no suitable adapter, or the device request was refused.
    pub fn new() -> Result<Self, String> {
        // wgpu is async, but we want a sync API for simplicity
        // pollster::block_on runs async code synchronously
        pollster::block_on(Self::new_async())
    }

    /// Selects an adapter, opens a device, and compiles every pipeline; the error says which step failed.
    async fn new_async() -> Result<Self, String> {
        // Step 1: Create a wgpu instance
        // This is the entry point to wgpu
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(), // Try all available backends (Vulkan, Metal, DX12, etc.)
            ..Default::default()
        });

        // Step 2: Request an adapter (represents a physical GPU)
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None, // We don't need a surface for compute
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| format!("No suitable GPU adapter found: {e}"))?;

        let adapter_info = adapter.get_info();

        // Step 3: Request a device (logical connection to the GPU)

        let device_descriptor = wgpu::DeviceDescriptor {
            label: Some("Volta GPU Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        };

        let (device, queue) = adapter
            .request_device(&device_descriptor)
            .await
            .map_err(|e| format!("Failed to create device: {e}"))?;
        // Step 4: Compile all our compute shaders
        let pipelines = Self::create_pipelines(&device);

        // Step 5: Create the buffer pool for memory reuse
        let buffer_pool = Arc::new(BufferPool::new(BufferPoolConfig::default()));

        // Step 6: Create the staging buffer pool for GPU→CPU transfers
        let staging_pool = Arc::new(StagingBufferPool::default());

        Ok(Self {
            device,
            queue,
            adapter_info,
            pipelines,
            buffer_pool,
            staging_pool,
            pending_submissions: AtomicU32::new(0),
            sync_threshold: DEFAULT_SYNC_THRESHOLD,
            consecutive_timeouts: AtomicU32::new(0),
        })
    }

    /// Get the GPU device name for display
    pub fn device_name(&self) -> &str {
        &self.adapter_info.name
    }

    /// Get a reference to the wgpu device
    pub const fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get a reference to the command queue
    pub const fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Get the compiled pipelines
    pub const fn pipelines(&self) -> &ComputePipelines {
        &self.pipelines
    }

    /// Get a reference to the buffer pool
    pub fn buffer_pool(&self) -> &BufferPool {
        &self.buffer_pool
    }

    /// Get an Arc reference to the buffer pool (for `GpuBuffer` to hold)
    pub fn buffer_pool_arc(&self) -> Arc<BufferPool> {
        Arc::clone(&self.buffer_pool)
    }

    /// Get a reference to the staging buffer pool
    pub fn staging_pool(&self) -> &StagingBufferPool {
        &self.staging_pool
    }

    /// Get an Arc reference to the staging buffer pool
    pub fn staging_pool_arc(&self) -> Arc<StagingBufferPool> {
        Arc::clone(&self.staging_pool)
    }

    /// Increment the pending submission counter.
    /// Call this after each `queue.submit()`.
    pub fn increment_pending(&self) {
        let new_count = self.pending_submissions.fetch_add(1, Ordering::Relaxed) + 1;
        log::trace!("[GPU] pending count: {} -> {new_count}", new_count - 1);
    }

    /// Force GPU synchronization - wait for all pending commands to complete.
    ///
    /// This polls the device until all submitted work is finished, then resets
    /// the pending counter. Use this when you need a clean sync point, such as
    /// between benchmark iterations or before reading results.
    ///
    /// Submits an empty command buffer first so that queued `write_buffer` uploads are flushed
    /// (and their staging memory can be freed) rather than left waiting for the next kernel.
    ///
    /// Returns true if sync completed successfully, false if it timed out.
    /// A timeout doesn't necessarily mean the device is lost - the GPU may
    /// just be under heavy load. The pending counter is reset regardless.
    pub fn sync(&self) -> bool {
        let pending = self.pending_submissions.load(Ordering::Relaxed);

        // Only sync if there are pending submissions
        if pending == 0 {
            return true;
        }

        log::debug!("[GPU] syncing {pending} pending submissions...");

        // `write_buffer` data sits in wgpu's staging area until the next submit, so an upload
        // alone is never in flight. An empty submit flushes it and gives `poll` something to wait on.
        self.queue.submit(std::iter::empty::<wgpu::CommandBuffer>());

        // Poll the device until all work completes
        let result = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(get_sync_timeout_secs())),
        });

        // Reset the counter regardless of success/failure
        // (we may have completed some work even on timeout)
        self.pending_submissions.store(0, Ordering::Relaxed);

        match result {
            Ok(_) => {
                log::debug!("[GPU] sync complete, pending reset to 0");
                true
            }
            Err(e) => {
                log::warn!(
                    "[GPU] sync timeout: {e:?}. This may indicate GPU overload; consider reducing batch size, adding more frequent sync points, or using smaller tensors."
                );
                false
            }
        }
    }

    /// Force sync with retry on timeout
    ///
    /// Attempts to sync up to `max_retries` times, returning true only
    /// if sync eventually succeeds.
    pub fn sync_with_retry(&self, max_retries: u32) -> bool {
        for attempt in 0..max_retries {
            if self.sync() {
                return true;
            }
            log::warn!(
                "[GPU] sync retry {}/{max_retries} - GPU may be overwhelmed",
                attempt + 1
            );
            // Small delay between retries
            std::thread::sleep(Duration::from_millis(100));
        }
        log::warn!("[GPU] all sync retries failed - GPU may be lost");
        false
    }

    /// Conditionally sync if pending submissions exceed the threshold.
    ///
    /// This provides automatic back-pressure to prevent GPU command queue
    /// exhaustion during rapid-fire operations. Call after each submit to
    /// ensure the queue doesn't grow unbounded.
    pub fn maybe_sync(&self) {
        let pending = self.pending_submissions.load(Ordering::Relaxed);
        if pending >= self.sync_threshold {
            log::trace!(
                "[GPU] auto-sync triggered: {pending} >= {}",
                self.sync_threshold
            );
            self.sync();
        }
    }

    /// Get the current pending submission count (diagnostic)
    ///
    /// This is useful for debugging GPU command queue issues.
    /// Returns the number of unsynced GPU submissions.
    pub fn pending_count(&self) -> u32 {
        self.pending_submissions.load(Ordering::Relaxed)
    }

    /// Get the sync threshold (diagnostic)
    pub const fn sync_threshold(&self) -> u32 {
        self.sync_threshold
    }

    /// Force sync with hard timeout and consecutive failure tracking
    ///
    /// This method implements a 3-strike rule for GPU timeouts:
    /// - Returns Ok(()) on successful sync (resets timeout counter)
    /// - Returns Err(Timeout) on timeout (increments counter)
    /// - Returns Err(GpuLost) if 3+ consecutive timeouts occur
    ///
    /// Use this instead of `sync()` in critical paths where you need
    /// error handling for GPU failures.
    /// # Errors
    /// Whatever errors come from async
    pub fn sync_checked(&self) -> Result<(), GpuSyncError> {
        let timeout_count = self.consecutive_timeouts.load(Ordering::Relaxed);

        // 3-strike rule: abort after consecutive failures
        if timeout_count >= MAX_CONSECUTIVE_TIMEOUTS {
            return Err(GpuSyncError::GpuLost(format!(
                "{timeout_count} consecutive timeouts - GPU unresponsive"
            )));
        }

        let pending = self.pending_submissions.load(Ordering::Relaxed);
        if pending == 0 {
            return Ok(());
        }

        log::debug!("[GPU] sync_checked: {pending} pending submissions...");

        // Flush queued `write_buffer` uploads; see `sync`.
        self.queue.submit(std::iter::empty::<wgpu::CommandBuffer>());

        // Use the hard sync timeout
        let result = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(get_sync_timeout_secs())),
        });

        self.pending_submissions.store(0, Ordering::Relaxed);

        match result {
            Ok(_) => {
                // Success - reset timeout counter
                self.consecutive_timeouts.store(0, Ordering::Relaxed);

                log::debug!("[GPU] sync_checked: success, timeout counter reset");

                Ok(())
            }
            Err(e) => {
                // Timeout - increment counter
                let new_count = self.consecutive_timeouts.fetch_add(1, Ordering::Relaxed) + 1;

                log::warn!("[GPU] sync timeout #{new_count}: {e:?}");

                if new_count >= MAX_CONSECUTIVE_TIMEOUTS {
                    Err(GpuSyncError::GpuLost(format!(
                        "{new_count} consecutive timeouts - aborting"
                    )))
                } else {
                    Err(GpuSyncError::Timeout(format!(
                        "Timeout #{new_count} of {MAX_CONSECUTIVE_TIMEOUTS}"
                    )))
                }
            }
        }
    }

    /// Get consecutive timeout count (diagnostic)
    ///
    /// Returns the number of consecutive sync timeouts. This counter
    /// resets to 0 on successful sync or manual reset.
    pub fn consecutive_timeouts(&self) -> u32 {
        self.consecutive_timeouts.load(Ordering::Relaxed)
    }

    /// Reset the consecutive timeout counter (manual recovery)
    ///
    /// Use this to manually clear the timeout counter after diagnosing
    /// and resolving GPU issues.
    pub fn reset_timeout_counter(&self) {
        self.consecutive_timeouts.store(0, Ordering::Relaxed);

        log::debug!("[GPU] timeout counter manually reset");
    }

    /// Create all compute pipelines by compiling shaders
    fn create_pipelines(device: &wgpu::Device) -> ComputePipelines {
        // Load and compile shader modules
        let elementwise_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Elementwise Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/elementwise.wgsl").into()),
        });

        let unary_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Unary Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/unary.wgsl").into()),
        });

        let reduce_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Reduce Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/reduce.wgsl").into()),
        });

        let matmul_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MatMul Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/matmul.wgsl").into()),
        });

        let movement_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Movement Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/movement.wgsl").into()),
        });

        let movement_backward_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Movement Backward Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/movement_backward.wgsl").into()),
        });

        let unary_backward_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Unary Backward Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/unary_backward.wgsl").into()),
        });

        let binary_backward_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Binary Backward Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/binary_backward.wgsl").into()),
        });

        let binary_backward_safe_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Binary Backward Safe Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/binary_backward_safe.wgsl").into(),
                ),
            });

        let matmul_backward_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MatMul Backward Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/matmul_backward.wgsl").into()),
        });

        let reduce_backward_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Reduce Backward Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/reduce_backward.wgsl").into()),
        });

        let optimizer_step_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Optimizer Step Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/optimizer_step.wgsl").into()),
        });

        let muon_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Muon Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/muon.wgsl").into()),
        });

        let im2col_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Im2col Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/im2col.wgsl").into()),
        });

        let col2im_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Col2im Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/col2im.wgsl").into()),
        });

        let direct_conv_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Direct Conv Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/direct_conv.wgsl").into()),
        });

        let igemm_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("iGEMM Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/igemm.wgsl").into()),
        });

        let igemm_backward_input_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("iGEMM Backward Input Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/igemm_backward_input.wgsl").into(),
                ),
            });

        let igemm_backward_weight_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("iGEMM Backward Weight Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/igemm_backward_weight.wgsl").into(),
                ),
            });

        let conv_backward_input_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Conv Backward Input Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/conv_backward_input.wgsl").into(),
                ),
            });

        let conv_backward_weight_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Conv Backward Weight Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/conv_backward_weight.wgsl").into(),
                ),
            });

        // Helper to create a compute pipeline
        let create_pipeline = |shader: &wgpu::ShaderModule, entry_point: &str, label: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None, // Auto-generate layout from shader
                module: shader,
                entry_point: Some(entry_point),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            })
        };

        ComputePipelines {
            // Element-wise binary ops
            add: create_pipeline(&elementwise_shader, "add", "Add Pipeline"),
            sub: create_pipeline(&elementwise_shader, "sub", "Sub Pipeline"),
            mul: create_pipeline(&elementwise_shader, "mul", "Mul Pipeline"),
            div: create_pipeline(&elementwise_shader, "div", "Div Pipeline"),
            max: create_pipeline(&elementwise_shader, "max_elem", "Max Pipeline"),
            mod_op: create_pipeline(&elementwise_shader, "mod_op", "Mod Pipeline"),
            cmplt: create_pipeline(&elementwise_shader, "cmplt", "Cmplt Pipeline"),

            // Unary ops
            neg: create_pipeline(&unary_shader, "neg", "Neg Pipeline"),
            exp: create_pipeline(&unary_shader, "exp_op", "Exp Pipeline"),
            log: create_pipeline(&unary_shader, "log_op", "Log Pipeline"),
            relu: create_pipeline(&unary_shader, "relu", "ReLU Pipeline"),
            sigmoid: create_pipeline(&unary_shader, "sigmoid", "Sigmoid Pipeline"),
            tanh: create_pipeline(&unary_shader, "tanh_op", "Tanh Pipeline"),
            sqrt: create_pipeline(&unary_shader, "sqrt_op", "Sqrt Pipeline"),
            recip: create_pipeline(&unary_shader, "recip", "Recip Pipeline"),
            exp2: create_pipeline(&unary_shader, "exp2_op", "Exp2 Pipeline"),
            log2: create_pipeline(&unary_shader, "log2_op", "Log2 Pipeline"),
            sin: create_pipeline(&unary_shader, "sin_op", "Sin Pipeline"),
            erf: create_pipeline(&unary_shader, "erf_op", "Erf Pipeline"),
            cos: create_pipeline(&unary_shader, "cos_op", "Cos Pipeline"),

            // Unary backward ops
            neg_backward: create_pipeline(
                &unary_backward_shader,
                "neg_backward",
                "Neg Backward Pipeline",
            ),
            exp_backward: create_pipeline(
                &unary_backward_shader,
                "exp_backward",
                "Exp Backward Pipeline",
            ),
            log_backward: create_pipeline(
                &unary_backward_shader,
                "log_backward",
                "Log Backward Pipeline",
            ),
            relu_backward: create_pipeline(
                &unary_backward_shader,
                "relu_backward",
                "ReLU Backward Pipeline",
            ),
            sigmoid_backward: create_pipeline(
                &unary_backward_shader,
                "sigmoid_backward",
                "Sigmoid Backward Pipeline",
            ),
            tanh_backward: create_pipeline(
                &unary_backward_shader,
                "tanh_backward",
                "Tanh Backward Pipeline",
            ),
            sqrt_backward: create_pipeline(
                &unary_backward_shader,
                "sqrt_backward",
                "Sqrt Backward Pipeline",
            ),
            recip_backward: create_pipeline(
                &unary_backward_shader,
                "recip_backward",
                "Recip Backward Pipeline",
            ),
            exp2_backward: create_pipeline(
                &unary_backward_shader,
                "exp2_backward",
                "Exp2 Backward Pipeline",
            ),
            log2_backward: create_pipeline(
                &unary_backward_shader,
                "log2_backward",
                "Log2 Backward Pipeline",
            ),
            sin_backward: create_pipeline(
                &unary_backward_shader,
                "sin_backward",
                "Sin Backward Pipeline",
            ),
            cos_backward: create_pipeline(
                &unary_backward_shader,
                "cos_backward",
                "Cos Backward Pipeline",
            ),
            erf_backward: create_pipeline(
                &unary_backward_shader,
                "erf_backward",
                "Erf Backward Pipeline",
            ),

            // Binary backward ops
            add_backward_a: create_pipeline(
                &binary_backward_shader,
                "add_backward_a",
                "Add Backward A Pipeline",
            ),
            add_backward_b: create_pipeline(
                &binary_backward_shader,
                "add_backward_b",
                "Add Backward B Pipeline",
            ),
            sub_backward_a: create_pipeline(
                &binary_backward_shader,
                "sub_backward_a",
                "Sub Backward A Pipeline",
            ),
            sub_backward_b: create_pipeline(
                &binary_backward_shader,
                "sub_backward_b",
                "Sub Backward B Pipeline",
            ),
            mul_backward_a: create_pipeline(
                &binary_backward_shader,
                "mul_backward_a",
                "Mul Backward A Pipeline",
            ),
            mul_backward_b: create_pipeline(
                &binary_backward_shader,
                "mul_backward_b",
                "Mul Backward B Pipeline",
            ),
            div_backward_a: create_pipeline(
                &binary_backward_shader,
                "div_backward_a",
                "Div Backward A Pipeline",
            ),
            div_backward_b: create_pipeline(
                &binary_backward_shader,
                "div_backward_b",
                "Div Backward B Pipeline",
            ),
            max_backward_a: create_pipeline(
                &binary_backward_shader,
                "max_backward_a",
                "Max Backward A Pipeline",
            ),
            max_backward_b: create_pipeline(
                &binary_backward_shader,
                "max_backward_b",
                "Max Backward B Pipeline",
            ),

            // Binary backward with RACE-FREE broadcasting (two-pass)
            add_broadcast_pass1: create_pipeline(
                &binary_backward_safe_shader,
                "add_broadcast_pass1",
                "Add Broadcast Pass1 Pipeline",
            ),
            add_broadcast_pass2: create_pipeline(
                &binary_backward_safe_shader,
                "add_broadcast_pass2",
                "Add Broadcast Pass2 Pipeline",
            ),
            sub_broadcast_pass1: create_pipeline(
                &binary_backward_safe_shader,
                "sub_broadcast_pass1",
                "Sub Broadcast Pass1 Pipeline",
            ),
            sub_broadcast_pass2: create_pipeline(
                &binary_backward_safe_shader,
                "sub_broadcast_pass2",
                "Sub Broadcast Pass2 Pipeline",
            ),
            mul_broadcast_pass1: create_pipeline(
                &binary_backward_safe_shader,
                "mul_broadcast_pass1",
                "Mul Broadcast Pass1 Pipeline",
            ),
            mul_broadcast_pass2: create_pipeline(
                &binary_backward_safe_shader,
                "mul_broadcast_pass2",
                "Mul Broadcast Pass2 Pipeline",
            ),
            div_broadcast_pass1: create_pipeline(
                &binary_backward_safe_shader,
                "div_broadcast_pass1",
                "Div Broadcast Pass1 Pipeline",
            ),
            div_broadcast_pass2: create_pipeline(
                &binary_backward_safe_shader,
                "div_broadcast_pass2",
                "Div Broadcast Pass2 Pipeline",
            ),
            max_broadcast_pass1: create_pipeline(
                &binary_backward_safe_shader,
                "max_broadcast_pass1",
                "Max Broadcast Pass1 Pipeline",
            ),
            max_broadcast_pass2: create_pipeline(
                &binary_backward_safe_shader,
                "max_broadcast_pass2",
                "Max Broadcast Pass2 Pipeline",
            ),

            // Reductions
            sum_reduce: create_pipeline(&reduce_shader, "sum_reduce", "Sum Reduce Pipeline"),
            max_reduce: create_pipeline(&reduce_shader, "max_reduce", "Max Reduce Pipeline"),
            mean_reduce: create_pipeline(&reduce_shader, "mean_reduce", "Mean Reduce Pipeline"),

            // Movement operations
            permute: create_pipeline(&movement_shader, "permute", "Permute Pipeline"),
            expand: create_pipeline(&movement_shader, "expand", "Expand Pipeline"),
            pad: create_pipeline(&movement_shader, "pad", "Pad Pipeline"),
            shrink: create_pipeline(&movement_shader, "shrink", "Shrink Pipeline"),
            stride: create_pipeline(&movement_shader, "stride", "Stride Pipeline"),

            // Movement backward operations
            permute_backward: create_pipeline(
                &movement_backward_shader,
                "permute_backward",
                "Permute Backward Pipeline",
            ),
            expand_backward: create_pipeline(
                &movement_backward_shader,
                "expand_backward",
                "Expand Backward Pipeline",
            ),
            pad_backward: create_pipeline(
                &movement_backward_shader,
                "pad_backward",
                "Pad Backward Pipeline",
            ),
            shrink_backward: create_pipeline(
                &movement_backward_shader,
                "shrink_backward",
                "Shrink Backward Pipeline",
            ),
            stride_backward: create_pipeline(
                &movement_backward_shader,
                "stride_backward",
                "Stride Backward Pipeline",
            ),

            // Matrix multiplication
            matmul: create_pipeline(&matmul_shader, "matmul", "MatMul Pipeline"),

            // Matrix multiplication backward
            matmul_backward_a: create_pipeline(
                &matmul_backward_shader,
                "matmul_backward_a",
                "MatMul Backward A Pipeline",
            ),
            matmul_backward_b: create_pipeline(
                &matmul_backward_shader,
                "matmul_backward_b",
                "MatMul Backward B Pipeline",
            ),

            // Reduction backward
            sum_backward: create_pipeline(&reduce_backward_shader, "main", "Sum Backward Pipeline"),
            mean_backward: create_pipeline(
                &reduce_backward_shader,
                "main",
                "Mean Backward Pipeline",
            ),
            max_backward: create_pipeline(&reduce_backward_shader, "main", "Max Backward Pipeline"),

            // Optimizer step
            optimizer_step: create_pipeline(
                &optimizer_step_shader,
                "main",
                "Optimizer Step Pipeline",
            ),

            // Muon
            muon_momentum: create_pipeline(&muon_shader, "momentum", "Muon Momentum Pipeline"),
            muon_sum_squares: create_pipeline(
                &muon_shader,
                "sum_squares",
                "Muon Sum Squares Pipeline",
            ),
            muon_normalize: create_pipeline(&muon_shader, "normalize", "Muon Normalize Pipeline"),
            muon_axpby: create_pipeline(&muon_shader, "axpby", "Muon Axpby Pipeline"),

            // Image-to-column transformation
            im2col: create_pipeline(&im2col_shader, "im2col_main", "Im2col Pipeline"),
            col2im: create_pipeline(&col2im_shader, "col2im_main", "Col2im Pipeline"),

            // Direct convolution
            direct_conv: create_pipeline(
                &direct_conv_shader,
                "direct_conv_main",
                "Direct Conv Pipeline",
            ),

            // Implicit GEMM convolution
            igemm: create_pipeline(&igemm_shader, "igemm_main", "iGEMM Pipeline"),

            // Convolution backward
            conv_backward_input: create_pipeline(
                &conv_backward_input_shader,
                "conv_backward_input_main",
                "Conv Backward Input Pipeline",
            ),
            conv_backward_weight: create_pipeline(
                &conv_backward_weight_shader,
                "conv_backward_weight_main",
                "Conv Backward Weight Pipeline",
            ),

            // iGEMM backward
            igemm_backward_input: create_pipeline(
                &igemm_backward_input_shader,
                "igemm_backward_input_main",
                "iGEMM Backward Input Pipeline",
            ),
            igemm_backward_weight: create_pipeline(
                &igemm_backward_weight_shader,
                "igemm_backward_weight_main",
                "iGEMM Backward Weight Pipeline",
            ),
        }
    }
}

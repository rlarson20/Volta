# Isolated Async Implementation Plan for Volta

> **Status**: Future Implementation Plan
> **Priority**: Low
> **Estimate**: 2-3 weeks
> **Dependencies**: None

---

## Overview

This document outlines a conservative implementation plan for adding **isolated async features** to Volta while preserving the synchronous nature of core tensor operations. This plan follows the strategic analysis recommendation to "Keep core sync, add async at boundaries."

**Core Principle**: The tensor API remains fully synchronous. Async is only added for I/O-bound operations (data loading) and optional GPU APIs.

---

## Phase 1: Async Data Loading (1-2 weeks)

### Goal

Add an `AsyncDataLoader` type separate from the existing synchronous `DataLoader`, allowing users to opt-in to async data loading without affecting the core API.

### Design

```rust
// New file: src/data/async_loader.rs
use std::sync::Arc;
use tokio::task::JoinHandle;

pub struct AsyncDataLoader {
    inner: DataLoader,
    prefetch_size: usize,
    buffer: VecDeque<(Tensor, Tensor)>,
    prefetch_task: Option<JoinHandle<()>>,
    shutdown_tx: Option<mpsc::Sender<()>>,
}

impl AsyncDataLoader {
    /// Create a new async data loader with background prefetching
    pub async fn new(
        data: Tensor,
        targets: Tensor,
        batch_size: usize,
        shuffle: bool,
        prefetch_size: usize,
    ) -> Result<Self, VoltaError> {
        let inner = DataLoader::new(data, targets, batch_size, shuffle)?;
        let (shutdown_tx, mut shutdown_rx) = mpsc::channel(1);

        // Start background prefetch task
        let buffer = Arc::new(Mutex::new(VecDeque::new()));
        let buffer_clone = buffer.clone();

        let task = tokio::spawn(async move {
            let mut inner = inner;
            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => break,
                    Some(batch) = async {
                        // Simulate blocking operation
                        tokio::task::yield_now().await;
                        inner.next()
                    } => {
                        if let Some(batch) = batch {
                            buffer_clone.lock().await.push_back(batch);
                        }
                    }
                }
            }
        });

        Ok(Self {
            inner,
            prefetch_size,
            buffer: VecDeque::new(),
            prefetch_task: Some(task),
            shutdown_tx: Some(shutdown_tx),
        })
    }

    /// Get the next batch, returning immediately if buffer has data
    pub async fn next_batch(&mut self) -> Option<(Tensor, Tensor)> {
        // Check buffer first
        if let Some(batch) = self.buffer.pop_front() {
            return Some(batch);
        }

        // If buffer empty, wait for prefetch
        // Implementation details...
        None
    }

    /// Get the next batch synchronously (blocks if needed)
    pub fn next_batch_blocking(&mut self) -> Option<(Tensor, Tensor)> {
        self.inner.next()
    }
}

impl Drop for AsyncDataLoader {
    fn drop(&mut self) {
        // Signal shutdown to background task
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.blocking_send(());
        }
    }
}
```

### Implementation Steps

1. **Create new module structure**
   - `src/data/mod.rs` - Data loading module
   - `src/data/loader.rs` - Move existing `DataLoader` here
   - `src/data/async_loader.rs` - New `AsyncDataLoader`

2. **Add async dependencies**
   - Add `tokio` as optional dependency
   - Add `async-trait` if needed
   - Update `Cargo.toml` with feature flag

3. **Implement AsyncDataLoader**
   - Background prefetch task
   - Thread-safe buffer (using channels or `async-lock`)
   - Graceful shutdown handling

4. **Add comprehensive tests**
   - Test basic async iteration
   - Test prefetch buffer behavior
   - Test shutdown/cleanup
   - Test concurrent access patterns

5. **Documentation**
   - Examples showing both sync and async usage
   - Performance benchmarks comparing sync vs async
   - Migration guide for existing users

### Files to Modify

- `Cargo.toml` - Add tokio dependency (optional, feature-gated)
- `src/lib.rs` - Export new `data` module
- `src/tensor.rs` - Move `DataLoader` to `src/data/loader.rs` (lines 1205-1336)
- `src/data/mod.rs` - New file
- `src/data/loader.rs` - New file (move existing code)
- `src/data/async_loader.rs` - New file (async implementation)
- `examples/async_data_loading.rs` - New example

### Feature Flag

```toml
[features]
default = []
async = ["tokio", "async-lock"]
```

---

## Phase 2: Optional Async GPU API (1 week)

### Goal

Expose optional async methods for GPU operations while keeping the synchronous API intact. This is a power-user feature for those who want to integrate Volta into existing async applications.

### Design

```rust
// Extend src/gpu/tensor_ops.rs with async variants

#[cfg(feature = "gpu")]
impl Tensor {
    /// Existing sync API (unchanged)
    pub fn to_device(&self, device: Device) -> Result<Tensor, String> {
        // Current implementation
    }

    /// New async API (opt-in)
    #[cfg(feature = "async")]
    pub async fn to_device_async(&self, device: Device) -> Result<Tensor, String> {
        // Async implementation
        let device_clone = device.clone();
        let tensor_clone = self.clone();

        tokio::task::spawn_blocking(move || {
            // Call sync implementation in thread pool
            tensor_clone.to_device(device_clone)
        })
        .await
        .map_err(|e| format!("Task failed: {}", e))?
    }

    /// Async GPU sync (explicit synchronization point)
    #[cfg(all(feature = "gpu", feature = "async"))]
    pub async fn gpu_sync_async(&self) -> Result<(), String> {
        let tensor_clone = self.clone();
        tokio::task::spawn_blocking(move || {
            tensor_clone.gpu_sync()
        })
        .await
        .map_err(|e| format!("Task failed: {}", e))?
    }
}
```

### Implementation Steps

1. **Add async GPU methods**
   - `to_device_async()`
   - `gpu_sync_async()`
   - `gpu_cleanup_async()`

2. **Use `spawn_blocking` appropriately**
   - GPU operations are already async internally (WGPU)
   - Wrap sync API calls in `spawn_blocking`
   - Avoid blocking async runtime

3. **Add integration tests**
   - Test async GPU operations
   - Test concurrent GPU operations
   - Test error handling

4. **Documentation**
   - When to use async vs sync GPU API
   - Performance considerations
   - Integration examples

### Files to Modify

- `src/gpu/mod.rs` - Add async feature gate
- `src/gpu/tensor_ops.rs` - Add async variants of GPU methods
- `examples/async_gpu.rs` - New example
- `dev-docs/GPU.md` - Update GPU documentation

---

## Phase 3: Model Serving Utilities (Optional, Future)

> **Note**: This phase is speculative and represents a potential future enhancement. Not recommended for initial implementation.

### Goal

Provide async utilities for model serving (HTTP/WebSocket servers) while keeping tensor operations synchronous.

### Design

```rust
// Hypothetical: src/serving/mod.rs

use tokio::net::TcpListener;
use hyper::{Body, Request, Response, Server};

/// Simple HTTP inference server
pub struct InferenceServer<M: Module> {
    model: Arc<M>,
    addr: SocketAddr,
}

impl<M: Module + Send + Sync + 'static> InferenceServer<M> {
    pub fn new(model: M, addr: SocketAddr) -> Self {
        Self {
            model: Arc::new(model),
            addr,
        }
    }

    pub async fn serve(self) -> Result<(), String> {
        let model = self.model.clone();

        let make_svc = hyper::service::make_service_fn(move |_| {
            let model = model.clone();
            async move {
                Ok::<_, hyper::Error>(hyper::service::service_fn(move |req| {
                    Self::handle_inference(req, model.clone())
                }))
            }
        });

        let server = Server::bind(&self.addr).serve(make_svc);

        server.await.map_err(|e| e.to_string())
    }

    async fn handle_inference(
        req: Request<Body>,
        model: Arc<M>,
    ) -> Result<Response<Body>, hyper::Error> {
        // Parse request (async)
        // Run inference (sync!)
        // Serialize response (async)
        todo!()
    }
}
```

### Implementation Steps

1. **Add serving dependencies**
   - `hyper` or `axum` for HTTP
   - `tokio` for async runtime
   - `serde` for JSON serialization

2. **Implement basic inference server**
   - JSON request/response
   - Batch inference support
   - Error handling

3. **Add examples**
   - REST API inference
   - WebSocket streaming
   - gRPC (optional)

4. **Documentation**
   - Deployment guide
   - Performance tuning
   - Security considerations

### Files to Create

- `src/serving/mod.rs` - Serving module
- `src/serving/http.rs` - HTTP server
- `src/serving/ws.rs` - WebSocket support (optional)
- `examples/serving.rs` - Serving example

---

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_async_data_loader_basic() {
        // Test async iteration
    }

    #[tokio::test]
    async fn test_async_data_loader_prefetch() {
        // Test buffer behavior
    }

    #[tokio::test]
    async fn test_async_gpu_transfer() {
        // Test async GPU operations
        #[cfg(feature = "gpu")]
        {
            let tensor = Tensor::zeros(&[10, 10]);
            let gpu_tensor = tensor.to_device_async(Device::Gpu).await?;
            // Assertions...
        }
    }
}
```

### Integration Tests

```rust
// tests/async_integration.rs

#[tokio::test]
async fn test_async_training_loop() {
    // Full training loop using AsyncDataLoader
    let loader = AsyncDataLoader::new(/* ... */).await?;

    for epoch in 0..10 {
        while let Some((batch_x, batch_y)) = loader.next_batch().await {
            // Forward (sync)
            let output = model.forward(&batch_x);
            // Backward (sync)
            output.backward();
            // Step (sync)
            optimizer.step(&model.parameters);
        }
    }
}
```

### Benchmarks

```rust
// benches/async_data_loading.rs

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_sync_vs_async(c: &mut Criterion) {
    c.bench_function("sync_data_loader", |b| {
        b.iter(|| {
            // Sync loader
        });
    });

    c.bench_function("async_data_loader", |b| {
        b.iter(|| {
            // Async loader (in runtime)
        });
    });
}

criterion_group!(benches, benchmark_sync_vs_async);
criterion_main!(benches);
```

---

## Dependencies

### Required Additions

```toml
[dependencies]
# Core async runtime (optional)
tokio = { version = "1.40", optional = true, features = ["rt-multi-thread", "sync", "macros"] }

# Async primitives
async-lock = { version = "3.4", optional = true }

# For serving (future)
hyper = { version = "1.0", optional = true, features = ["full"] }
axum = { version = "0.7", optional = true }
```

### Feature Flags

```toml
[features]
default = []

# Enable async data loading
async = ["tokio", "async-lock"]

# Enable async GPU API
async-gpu = ["async", "gpu"]

# Enable model serving utilities
serving = ["async", "hyper", "serde", "serde_json"]
```

---

## Migration Guide for Users

### Before (Sync Only)

```rust
use volta::prelude::*;

let loader = DataLoader::new(data, targets, 32, true);

for (batch_x, batch_y) in loader {
    let output = model.forward(&batch_x);
    // ...
}
```

### After (Async Available)

```rust
use volta::prelude::*;

// Sync API still works (no changes required)
let loader = DataLoader::new(data, targets, 32, true);
for (batch_x, batch_y) in loader {
    let output = model.forward(&batch_x);
    // ...
}

// OR use async API (opt-in)
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut loader = AsyncDataLoader::new(data, targets, 32, true, 2).await?;

    while let Some((batch_x, batch_y)) = loader.next_batch().await {
        // Tensor operations remain synchronous!
        let output = model.forward(&batch_x);
        // ...
    }

    Ok(())
}
```

---

## Timeline

### Week 1: Async Data Loading
- Day 1-2: Module structure, move existing DataLoader
- Day 3-4: Implement AsyncDataLoader core logic
- Day 5: Tests and documentation

### Week 2: Async GPU API
- Day 1-2: Implement async GPU methods
- Day 3: Integration tests
- Day 4-5: Documentation and examples

### Week 3: Polish and Review
- Day 1-2: Code review and refinement
- Day 3: Performance benchmarks
- Day 4: Final documentation updates
- Day 5: Release preparation

---

## Success Criteria

1. ✅ Core tensor API remains 100% synchronous
2. ✅ No breaking changes to existing code
3. ✅ Async features are opt-in via feature flags
4. ✅ All async code has comprehensive tests
5. ✅ Documentation clearly explains when to use async vs sync
6. ✅ Performance benchmarks show async data loading benefits

---

## Risks and Mitigations

### Risk 1: Runtime Coupling
**Concern**: Users must adopt Tokio for async features.
**Mitigation**: Document clearly that async is opt-in. Sync API remains fully functional.

### Risk 2: Increased Complexity
**Concern**: Adding async increases maintenance burden.
**Mitigation**: Isolate async code to separate modules. Keep sync API as primary.

### Risk 3: Performance Regression
**Concern**: Async overhead might hurt single-threaded performance.
**Mitigation**: Extensive benchmarking. Async is opt-in, so users can avoid overhead.

### Risk 4: Confusing API
**Concern**: Two ways to do the same thing (sync vs async).
**Mitigation**: Clear documentation. Sync is default/recommended. Async is for specific use cases.

---

## Open Questions

1. **Should we support multiple async runtimes?**
   - Recommendation: No. Tokio is de facto standard. Use `spawn_blocking` to avoid coupling.

2. **Should AsyncDataLoader use channels or atomic buffer?**
   - Recommendation: Channels for simplicity. Can optimize later if needed.

3. **Should we provide a `Stream` interface?**
   - Recommendation: Yes, but as a convenience wrapper around `next_batch_async()`.

4. **What error handling strategy?**
   - Recommendation: Convert async errors to `VoltaError`. Keep consistency with sync API.

---

## Further Reading

- [Tokio Best Practices](https://tokio.rs/tokio/tutorial)
- [Async in Rust](https://rust-lang.github.io/async-book/)
- [Hyper Examples](https://github.com/hyperium/hyper/tree/master/examples)

---

**Document Version**: 1.0
**Last Updated**: 2026-02-23
**Author**: Claude Code Implementation Planning

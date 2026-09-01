# Gradient Clipping Design

**Date:** 2026-05-14
**Status:** Approved
**Priority:** High (NEXT_STEPS #2)

## Context

Volta currently lacks gradient clipping functionality, which is essential for training stable deep learning models—especially RNNs/LSTMs which are prone to exploding gradients. The existing examples (`char_language_model.rs`, `mnist_rnn.rs`) train LSTMs without any gradient clipping, leaving them vulnerable to gradient explosion.

This design implements PyTorch-compatible gradient clipping to fix this gap and enable stable training at higher learning rates.

## Requirements

1. **PyTorch-compatible API** - Match `torch.nn.utils.clip_grad_norm_` and `clip_grad_value_`
2. **Support multiple norm types** - L1, L2, L∞ norms
3. **Device-agnostic** - Work with both CPU and GPU tensors
4. **Non-invasive** - Standalone functions, no required optimizer changes
5. **Return clipping statistics** - Enable logging for monitoring gradient health

## Architecture

Gradient clipping is implemented as standalone functions in a new module `src/nn/optim/clip.rs`. The design prioritizes:

- **Separation of concerns** - Clipping logic independent of optimizers
- **PyTorch compatibility** - Exact API match for easy migration
- **Testability** - Pure functions easy to unit test
- **Future-proof** - Works with any optimizer (existing, custom, future)

### File Structure

```
src/nn/optim/
├── mod.rs          # Re-exports clip module
├── adam.rs         # Existing (unchanged)
├── sgd.rs          # Existing (unchanged)
└── clip.rs         # NEW: gradient clipping functions
```

## Components

### Core API

```rust
/// Clips gradient norm of all parameters in-place.
/// Returns the total norm before clipping (useful for logging).
pub fn clip_grad_norm(params: &[Tensor], max_norm: f32, norm_type: f32) -> f32;

/// Clips gradient values element-wise to [-clip_value, clip_value].
pub fn clip_grad_value(params: &[Tensor], clip_value: f32);
```

### Internal Helpers

```rust
fn compute_norm(storage: &Storage, norm_type: f32) -> f32;
fn compute_norm_cpu(data: &[f32], norm_type: f32) -> f32;
fn scale_gradients(grad: &mut Storage, scale: f32);
fn clamp_gradients(grad: &mut Storage, clip_value: f32);
```

### Optional Optimizer Integration

```rust
impl Adam {
    pub fn step_with_clip(&mut self, max_norm: f32, norm_type: f32) -> f32 {
        let total_norm = clip_grad_norm(&self.params, max_norm, norm_type);
        self.step();
        total_norm
    }
}
```

## Data Flow

### Training Loop Integration

```rust
for epoch in 1..=epochs {
    for batch in dataloader {
        let loss = model.forward(&input);
        optimizer.zero_grad();
        loss.backward();

        // Clip gradients before optimizer step
        let total_norm = clip_grad_norm(&params, 1.0, 2.0);
        if total_norm > 1.0 {
            println!("Clipped gradient norm from {} to 1.0", total_norm);
        }

        optimizer.step();
    }
}
```

### Internal Flow for `clip_grad_norm`

1. **Validate input** - Check `max_norm > 0`, `norm_type > 0`
2. **Filter parameters** - Collect only params with gradients
3. **Compute total norm** - Sum across all parameter gradients
4. **Clipping decision** - If `total_norm > max_norm`, compute scaling factor
5. **In-place scaling** - Modify each gradient's `Storage` directly
6. **Return** - Total norm before clipping (for logging)

## Error Handling

### Input Validation

- `max_norm <= 0.0` → panic
- `norm_type <= 0.0` → panic
- Empty parameter list or no gradients → return 0.0

### Gradient Access Safety

Use safe `RefCell` borrowing pattern:

```rust
for param in params {
    let mut tensor = param.borrow_mut();
    if let Some(ref mut grad) = tensor.grad {
        // Work with grad directly
        // No other borrows active at this point
    }
    // borrow_mut() released here
}
```

### Edge Cases

- **Empty params** → return 0.0
- **No gradients** → skip, return 0.0
- **Zero gradients** → return 0.0, no scaling
- **NaN/Inf** → propagate naturally (don't mask errors)
- **Mixed devices** → handle each according to device

## Testing Strategy

### Unit Tests (`src/nn/optim/clip.rs`)

1. **Basic clipping** - Gradients with norm 10 clipped to max_norm=1.0 → final norm exactly 1.0
2. **No clipping** - Gradients with norm 0.5, max_norm=1.0 → unchanged
3. **Norm types** - L1, L2, L∞ all produce correct values
4. **Value clipping** - Element-wise clamping to range
5. **Edge cases** - Empty params, no gradients, zero gradients, single param
6. **Validation** - Panic on invalid max_norm/norm_type

### Integration Tests (Examples)

1. **`char_language_model.rs`** - Add clipping, verify no NaN/Inf at higher LR
2. **`mnist_rnn.rs`** - Add clipping, verify stable training

### Verification Tests

1. **Gradient correctness** - Numerical gradient check still passes
2. **Optimizer convergence** - Training still converges with clipping
3. **Return value** - Total norm matches manual computation

## Implementation Notes

### Norm Computation

- **L1 norm**: `sum(|g|)` across all gradient elements
- **L2 norm**: `sqrt(sum(g²))` across all gradient elements
- **L∞ norm**: `max(|g|)` across all gradient elements

### Storage Operations

Gradients are accessed through the `RefCell<RawTensor>` pattern:

```rust
pub struct RawTensor {
    pub data: Storage,
    pub grad: Option<Storage>,  // Target for clipping
    // ...
}
pub type Tensor = Rc<RefCell<RawTensor>>;
```

Clipping modifies `grad: Storage` in-place, which is safe because:
- Backward pass is complete (autograd graph consumed)
- No other references to grad exist during clipping
- Optimizer step reads grad after clipping

## Acceptance Criteria

From NEXT_STEPS #2:

- ✅ Unit tests: gradients with norm 10 clipped to max_norm=1.0 end up with norm exactly 1.0
- ✅ Unit tests: gradients already under threshold are untouched
- ✅ Numerical gradient check still passes on clipped path
- ✅ L1, L2, L∞ norm types all exercised
- ✅ Updated examples train without NaN/Inf at higher LRs

## Risks & Mitigations

### Risk: RefCell Borrow Conflicts

**Mitigation**: Clipping happens after `backward()` completes and before `optimizer.step()` begins. No other borrows should be active. Verified by ensuring examples train correctly.

### Risk: GPU Implementation Complexity

**Mitigation**: Start with CPU-only implementation. GPU support can be added via Storage abstraction if time permits in the session.

## Future Enhancements

- GPU acceleration for norm computation
- Per-parameter clipping (different max_norm per layer)
- Adaptive clipping strategies (e.g., dynamic max_norm)
- Clipping statistics tracking (history of clip operations)

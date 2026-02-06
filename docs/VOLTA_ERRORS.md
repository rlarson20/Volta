# Volta Error Handling & Type System Refactoring Report

This report outlines the necessary changes to move Volta from a panic-heavy, runtime-checked library to a robust, type-safe ML framework. The goal is to eliminate `unwrap()` and `panic!()` in favor of `Result` types and to leverage the Rust type system for compile-time safety where appropriate.

## 1. Areas Needing Error Handling Changes

The current codebase relies heavily on panics for control flow and validation. The following areas are the highest priority for refactoring:

### A. Tensor Construction & Factory Methods
- **File**: `src/tensor.rs`
- **Issues**:
    - `RawTensor::new` panics if data length doesn't match shape.
    - `he_initialization` panics on empty shapes.
    - `randn` unwraps `Normal::new`.
- **Refactoring**:
    - Change factory methods to return `Result<Tensor, VoltaError>`.
    - Validate inputs (shapes, data lengths) and return variants like `VoltaError::ShapeMismatch`.

### B. Storage & Data Access
- **File**: `src/storage.rs`
- **Issues**:
    - `from_bytes` panics if length isn't divisible by dtype size.
    - `as_f32_slice` panics/asserts if dtype is wrong.
    - `index` traits (`Index`, `IndexMut`) panic on out-of-bounds access.
- **Refactoring**:
    - `Index` traits cannot return `Result`. We should provide safe alternatives e.g., `get_typed<T>() -> Result<&[T], VoltaError>`.
    - Internal conversions should propagate errors instead of asserting.

### C. Operations & Broadcasting
- **File**: `src/ops/binary.rs`, `src/tensor.rs` (reduction ops)
- **Issues**:
    - `broadcast_shape` panics on incompatible shapes.
    - `sum_dim`, `max_dim` panic if dim is out of bounds.
    - `binary_op` asserts/panics on broadcast failure.
- **Refactoring**:
    - Operations should return `Result<Tensor, VoltaError>`.
    - Broadcasting logic should return `Result<Vec<usize>, VoltaError>`.

### D. Device & DType Handling
- **File**: `src/device.rs`, `src/storage.rs`
- **Issues**:
    - GPU fallback logic prints warnings but might mask actual failures or panic later.
    - Casting/Converting types relies on `unwrap()`.
- **Refactoring**:
    - explicit `to_device` fallibility.

## 2. Errors to be Handled

We should introduce a unified `VoltaError` enum (likely in `src/error.rs`) to capture these failure modes:

```rust
#[derive(Debug, thiserror::Error)]
pub enum VoltaError {
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },

    #[error("Broadcast error: shapes {0:?} and {1:?} are incompatible")]
    BroadcastError(Vec<usize>, Vec<usize>),

    #[error("Dimension {dim} out of bounds for shape {shape:?}")]
    DimOutOfBounds { dim: usize, shape: Vec<usize> },

    #[error("DType mismatch: expected {expected:?}, got {got:?}")]
    DTypeMismatch { expected: DType, got: DType },

    #[error("Device mismatch: expected {expected:?}, got {got:?}")]
    DeviceMismatch { expected: String, got: String },

    #[error("Invalid initialization: {0}")]
    InitError(String),

    #[error("Out of memory: {0}")]
    OutOfMemory(String),
}
```

## 3. Type System Refactoring (Runtime -> Compile Time)

Currently, `Tensor` is a dynamic wrapper around `RawTensor`:
```rust
pub type Tensor = Rc<RefCell<RawTensor>>;
```
This means DType, Device, and Shape are all checked at runtime. To move checks to compile time, we can introduce generics.

### Phase 1: Typed Tensors (DType & Device)
We can split the monolithic `Tensor` into a generic struct.

```rust
// Proposed Signature
pub struct TypedTensor<D: DTypeTrait, B: Backend> {
    inner: Rc<RefCell<RawTensor>>, // Keep mostly same internals for now
    _marker: PhantomData<(D, B)>,
}

trait DTypeTrait { fn dtype() -> DType; }
struct F32; impl DTypeTrait for F32 { ... }

trait Backend { fn device() -> Device; }
struct Cpu;
struct Cuda;
```

**Benefits**:
- `tensor.to_dtype::<F32>()` returns `TypedTensor<F32, ...>`.
- Binary ops can enforce same-dtype inputs at compile time:
  `fn add(a: TypedTensor<F32, ...>, b: TypedTensor<F32, ...>)`
- Eliminates "DType mismatch" panics at runtime for typed paths.

### Phase 2: Const Generics for Rank (Shapes)
Rust's `const generics` are becoming mature enough for this. We can track the *rank* (number of dimensions) at compile time, if not the exact shape.

```rust
pub struct Tensor<D, B, const RANK: usize> {
    ...
}
```

**Benefits**:
- `matmul` can accept `Tensor<..., 2>` explicitly.
- `sum_dim` checks `dim < RANK` at compile time (mostly).
- `broadcast` logic can be specialized for specific ranks.

**Challenges**:
- Dynamic graphs often invoke reshaping where rank changes at runtime (though usually predictable).
- Writing generic kernels for `const RANK` is complex but doable (see `dfdx`).

### Recommendation
Start with **Phase 1 (DType/Device generics)**.
1.  Keep the dynamic `Tensor` for ease of use in rapid prototyping / scripting.
2.  Introduce `TypedTensor<D, B>` as a zero-cost wrapper around it.
3.  Refactor internal kernels (`ops/*.rs`) to generic traits to support this safely.
4.  This aligns with "avoiding overuse of unwrap" by making invalid states unrepresentable.

## Summary of Action Plan
1.  **Immediate**: Create `src/error.rs` and `VoltaError`.
2.  **Short-term**: Refactor `Tensor::new`, `storage`, and `ops` to return `Result`. Replace `unwrap()` with `?`.
3.  **Medium-term**: Introduce `TypedTensor<D>` phantom wrapper to enforce dtype safety without major internal rewrites.

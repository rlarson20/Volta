# Compile-Time Safety Opportunities

Generated: 2026-01-29

---

## [High Priority] Pattern: Typestate for Storage

**Current Implementation:**
`src/storage.rs` uses a single `enum Storage` with variants for `Cpu`, `Gpu`, and different dtypes (`F32`, `F16`, etc.). Methods like `as_f32_slice` panic if the variant doesn't match.

```rust
// src/storage.rs
pub enum Storage {
    Cpu(Vec<f32>),
    CpuF16(Vec<f16>),
    Gpu(GpuBuffer),
    // ...
}

impl Storage {
    pub fn as_f32_slice(&self) -> &[f32] {
        match self {
            Storage::Cpu(data) => data,
            _ => panic!("Wrong storage datatype"),
        }
    }
}
```

**Problem:**
Every access to tensor data carries reduced confidence and potential runtime panics. The compiler cannot verify that a `matmul` operation is receiving `F16` storage on `GPU` vs `F32` on `CPU`, leading to extensive `match` statements and runtime checks deep in the kernel dispatch logic.

**Type-Safe Alternative:**
Use the **Typestate Pattern** to encode the storage location and data type in the type system.

```rust
pub struct Storage<D: Device, T: DType> {
    data: D::Buffer<T>,
    _marker: PhantomData<T>,
}

trait Device {
    type Buffer<T>;
}

struct Cpu;
struct Gpu;

// Now `Storage<Cpu, f32>` is a distinct type from `Storage<Gpu, f16>`
// Functions can accept `Storage<Gpu, T>` and be guaranteed to not need runtime checks.
```

**Migration Path:**

1. Create `Storage<D, T>` struct.
2. Refactor `Tensor` to `Tensor<D, T>`.
3. Update operations to Generic boundaries (e.g., `impl Tensor<Cpu, f32>`).
4. Provide a `DynamicTensor` (Box<dyn>) or `enum AnyTensor` only at the very top level (API boundary) if runtime dynamism is strictly required.

**Benefits:**

- **Zero cost abstractions**: No matching on variants for every op.
- **Impossible to misuse**: Can't accidentally run CPU ops on GPU tensors.
- **Performance**: Static dispatch for kernels.

**Effort**: Large (Fundamental architectural change)

---

## [High Priority] Pattern: Const Generics for Tensor Shapes

**Current Implementation:**
`src/tensor.rs` and `src/ops/matmul.rs` check shapes at runtime.

```rust
// src/ops/matmul.rs
assert_eq!(n, n2, "Matmul dimension mismatch: ({m},{n}) @ ({n2},{p})");
```

**Problem:**
Dimension mismatches are the most common source of ML bugs, typically discovered late in training or production. Current code checks these at the last possible moment (inside the op).

**Type-Safe Alternative:**
Use **Const Generics** to track dimensions.

```rust
pub struct Tensor<T, S: Shape> {
    data: Storage<T>,
    shape: S,
}

pub trait Shape {
    const RANK: usize;
    fn dims() -> [usize; Self::RANK];
}

pub struct Rank2<const M: usize, const N: usize>;

impl<T, const M: usize, const N: usize, const K: usize> Mul<Tensor<T, Rank2<N, K>>>
    for Tensor<T, Rank2<M, N>>
{
    type Output = Tensor<T, Rank2<M, K>>;

    fn mul(self, rhs: Tensor<T, Rank2<N, K>>) -> Self::Output {
        // No runtime check needed! M, N, K are known match.
        // ...
    }
}
```

**Migration Path:**

1. Introduce `Shape` trait and `Rank` structs.
2. Add `const N: usize` variants for fixed-size tensors.
3. Keep `DynamicTensor` for when shapes are truly unknown (loading from disk).

**Benefits:**

- **Compile-time shape checking**: `matmul(a, b)` fails to compile if dimensions don't align.
- **Optimization**: Compiler knows loop bounds.

**Effort**: Medium/Large

---

## [Medium Priority] Pattern: Autograd State Enforcement

**Current Implementation:**
`backward()` panics if called on a tensor that doesn't track gradients, or if run twice without clearing.

```rust
// src/autograd.rs
assert!(
    tensor.requires_grad,
    "Called backward on a tensor that doesn't require grad"
);
```

**Problem:**
Logic errors in training loops (forgetting `requires_grad=True` on weights) manifest as runtime panics or silent failures.

**Type-Safe Alternative:**
Encode gradient tracking state in the type.

```rust
struct WithGrad;
struct NoGrad;

struct Tensor<T, S, G = NoGrad> { ... }

impl<T, S> Tensor<T, S, WithGrad> {
    pub fn backward(&self) { ... }
}

// Logic ensures you can only call backward on appropriate tensors
```

**Effort**: Medium

---

## [Medium Priority] Pattern: Enforcing Graph Topology in Ops

**Current Implementation:**
`ops/ternary.rs` assumes specific parent structure in `GradFn::backward`.

```rust
// src/ops/ternary.rs
let x_ref = parents.first().cloned().unwrap();
let y_ref = parents.get(1).cloned().unwrap();
```

**Problem:**
`GradFn` takes a generic `Vec<Tensor>` of parents. Compile-time structure is lost. If an op implementation drifts from its `parents` list construction, `backward` panics.

**Type-Safe Alternative:**
Associate the gradient function with the specific input topology.

```rust
trait DifferentiableOp {
    type Inputs; // Tuple of inputs e.g. (Tensor, Tensor)
    fn backward(ctx: Context, grad: Tensor) -> Self::Inputs;
}
```

**Effort**: Medium

---

## [Low Priority] Pattern: Validated Builders

**Current Implementation:**
`RawTensor::new` checks that data length matches shape size.

```rust
// src/tensor.rs
// # Panics
// Panics if `data.len()` != `shape.product()`
```

**Problem:**
Constructing invalid tensors is possible (panic), requiring runtime validation.

**Type-Safe Alternative:**
Use a Builder pattern that forces consistency or returns `Result`.

```rust
impl Tensor {
    pub fn try_new(data: Vec<T>, shape: &[usize]) -> Result<Self, ShapeError> { ... }
}
```

Or with Const Generics (see above), `from_array` is always safe.

**Effort**: Small

---

## Summary Recommendation

The highest value move is exploring **Typestate for Storage**. This touches the core of the engine and eliminates the most pervasive runtime checking (`match storage { ... }` in every op).

Start by investigating a `Storage<D, T>` refactor.

// Matrix multiplication backward pass
// Computes gradients for matmul: C = A @ B
//
// Two entry points:
// 1. matmul_backward_a: dA = grad @ B^T  (grad is [M,N], B is [K,N], dA is [M,K])
// 2. matmul_backward_b: dB = A^T @ grad  (A is [M,K], grad is [M,N], dB is [K,N])
//
// Uses tiled approach with shared memory, same as forward matmul

struct BackwardParams {
    m: u32,  // For backward_a: rows of grad / rows of dA
             // For backward_b: rows of A / cols of grad
    k: u32,  // For backward_a: cols of grad / rows of B
             // For backward_b: cols of A / rows of grad
    n: u32,  // For backward_a: cols of B / cols of grad
             // For backward_b: cols of grad / cols of B
    _padding: u32,
}

// ============ matmul_backward_a: dA = grad @ B^T =============
// grad: [M, N], B: [K, N], dA: [M, K]
// We compute grad @ B^T which is [M,N] @ [N,K] = [M,K]
// To avoid explicit transpose, we read B column-major during tile load

@group(0) @binding(0) var<storage, read> grad_a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> dA: array<f32>;
@group(0) @binding(3) var<uniform> params_a: BackwardParams;

const TILE_SIZE: u32 = 16u;
var<workgroup> tile_grad: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn matmul_backward_a(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let row = global_id.y;  // Row in dA (also row in grad)
    let col = global_id.x;  // Col in dA (also row in B, which is col in B^T)
    let local_row = local_id.y;
    let local_col = local_id.x;

    var sum: f32 = 0.0;

    // Number of tiles along the N dimension (shared dimension of grad @ B^T)
    let num_tiles = (params_a.n + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // Load tile of grad [M, N] at position [row, t*TILE_SIZE + local_col]
        let grad_row = row;
        let grad_col = t * TILE_SIZE + local_col;
        if (grad_row < params_a.m && grad_col < params_a.n) {
            tile_grad[local_row * TILE_SIZE + local_col] = grad_a[grad_row * params_a.n + grad_col];
        } else {
            tile_grad[local_row * TILE_SIZE + local_col] = 0.0;
        }

        // Load tile of B^T (which is B transposed)
        // B^T[row, col] = B[col, row], so we read B transposed
        // B is [K, N], B^T is [N, K]
        // We need B^T at position [t*TILE_SIZE + local_row, col]
        // Which is B at position [col, t*TILE_SIZE + local_row]
        let bt_row = t * TILE_SIZE + local_row;
        let bt_col = col;
        if (bt_row < params_a.n && bt_col < params_a.k) {
            // Read B[bt_col, bt_row] to get B^T[bt_row, bt_col]
            tile_b[local_row * TILE_SIZE + local_col] = b[bt_col * params_a.n + bt_row];
        } else {
            tile_b[local_row * TILE_SIZE + local_col] = 0.0;
        }

        workgroupBarrier();

        // Compute partial dot product
        for (var i: u32 = 0u; i < TILE_SIZE; i = i + 1u) {
            sum = sum + tile_grad[local_row * TILE_SIZE + i] * tile_b[i * TILE_SIZE + local_col];
        }

        workgroupBarrier();
    }

    // Write result
    if (row < params_a.m && col < params_a.k) {
        dA[row * params_a.k + col] = sum;
    }
}

// ============ matmul_backward_b: dB = A^T @ grad =============
// A: [M, K], grad: [M, N], dB: [K, N]
// We compute A^T @ grad which is [K,M] @ [M,N] = [K,N]

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> grad_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> dB: array<f32>;
@group(0) @binding(3) var<uniform> params_b: BackwardParams;

@compute @workgroup_size(16, 16)
fn matmul_backward_b(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let row = global_id.y;  // Row in dB (also col in A, which is row in A^T)
    let col = global_id.x;  // Col in dB (also col in grad)
    let local_row = local_id.y;
    let local_col = local_id.x;

    var sum: f32 = 0.0;

    // Number of tiles along the M dimension (shared dimension of A^T @ grad)
    let num_tiles = (params_b.m + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // Load tile of A^T (which is A transposed)
        // A is [M, K], A^T is [K, M]
        // We need A^T at position [row, t*TILE_SIZE + local_col]
        // Which is A at position [t*TILE_SIZE + local_col, row]
        let at_row = row;
        let at_col = t * TILE_SIZE + local_col;
        if (at_row < params_b.k && at_col < params_b.m) {
            // Read A[at_col, at_row] to get A^T[at_row, at_col]
            tile_a[local_row * TILE_SIZE + local_col] = a[at_col * params_b.k + at_row];
        } else {
            tile_a[local_row * TILE_SIZE + local_col] = 0.0;
        }

        // Load tile of grad [M, N] at position [t*TILE_SIZE + local_row, col]
        let grad_row = t * TILE_SIZE + local_row;
        let grad_col = col;
        if (grad_row < params_b.m && grad_col < params_b.n) {
            tile_grad[local_row * TILE_SIZE + local_col] = grad_b[grad_row * params_b.n + grad_col];
        } else {
            tile_grad[local_row * TILE_SIZE + local_col] = 0.0;
        }

        workgroupBarrier();

        // Compute partial dot product
        for (var i: u32 = 0u; i < TILE_SIZE; i = i + 1u) {
            sum = sum + tile_a[local_row * TILE_SIZE + i] * tile_grad[i * TILE_SIZE + local_col];
        }

        workgroupBarrier();
    }

    // Write result
    if (row < params_b.k && col < params_b.n) {
        dB[row * params_b.n + col] = sum;
    }
}

var<workgroup> tile_a: array<f32, 256>;

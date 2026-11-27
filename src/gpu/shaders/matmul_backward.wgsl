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
             // For backward_b: rows of A / rows of grad (the shared dimension)
    k: u32,  // For backward_a: cols of dA / rows of B
             // For backward_b: cols of A / rows of dB
    n: u32,  // For backward_a: cols of grad / cols of B (the shared dimension)
             // For backward_b: cols of grad / cols of dB
    batch: u32, // Independent products, stacked as in `matmul.wgsl`
}

// ============ matmul_backward_a: dA = grad @ B^T =============
// grad: [M, N], B: [K, N], dA: [M, K]
// We compute grad @ B^T which is [M,N] @ [N,K] = [M,K]
// To avoid explicit transpose, we read B column-major during tile load

@group(0) @binding(0) var<storage, read> grad_a: array<ELEM>;
@group(0) @binding(1) var<storage, read> b: array<ELEM>;
@group(0) @binding(2) var<storage, read_write> dA: array<ELEM>;
@group(0) @binding(3) var<uniform> params_a: BackwardParams;

const TILE_SIZE: u32 = 16u;
var<workgroup> tile_grad: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn matmul_backward_a(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let global_id = tile_global_id(wg_id, num_wg, local_id, (params_a.k + 15u) / 16u);
    // Stacked products, as in `matmul.wgsl`.
    let tiles_y = (params_a.m + 15u) / 16u;
    let batch_raw = (global_id.y / 16u) / tiles_y;
    let in_batch = batch_raw < params_a.batch;
    let batch = select(0u, batch_raw, in_batch);
    let g_off = batch * params_a.m * params_a.n;
    let b_off = batch * params_a.k * params_a.n;
    let d_off = batch * params_a.m * params_a.k;
    // Row in dA (also row in grad)
    let row = select(params_a.m, ((global_id.y / 16u) % tiles_y) * 16u + local_id.y, in_batch);
    let col = global_id.x;  // Col in dA (also row in B, which is col in B^T)
    let local_row = local_id.y;
    let local_col = local_id.x;

    // One partial per tile, summed pairwise across tiles (#319): a single running sum over
    // all of k loses about k * 2^-24 relative, 4% at k = 2^22.
    var acc: PairwiseSum;

    // Number of tiles along the N dimension (shared dimension of grad @ B^T)
    let num_tiles = (params_a.n + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // Load tile of grad [M, N] at position [row, t*TILE_SIZE + local_col]
        let grad_row = row;
        let grad_col = t * TILE_SIZE + local_col;
        if (grad_row < params_a.m && grad_col < params_a.n) {
            tile_grad[local_row * TILE_SIZE + local_col] = ld_grad_a(g_off + grad_row * params_a.n + grad_col);
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
            tile_b[local_row * TILE_SIZE + local_col] = ld_b(b_off + bt_col * params_a.n + bt_row);
        } else {
            tile_b[local_row * TILE_SIZE + local_col] = 0.0;
        }

        workgroupBarrier();

        // Compute partial dot product
        var partial: f32 = 0.0;
        for (var i: u32 = 0u; i < TILE_SIZE; i = i + 1u) {
            partial = partial + tile_grad[local_row * TILE_SIZE + i] * tile_b[i * TILE_SIZE + local_col];
        }
        pairwise_push(&acc, partial);

        workgroupBarrier();
    }

    // Write result
    if (row < params_a.m && col < params_a.k) {
        st_dA(d_off + row * params_a.k + col, pairwise_total(&acc));
    }
}

// ============ matmul_backward_b: dB = A^T @ grad =============
// A: [M, K], grad: [M, N], dB: [K, N]
// We compute A^T @ grad which is [K,M] @ [M,N] = [K,N]

@group(0) @binding(0) var<storage, read> a: array<ELEM>;
@group(0) @binding(1) var<storage, read> grad_b: array<ELEM>;
@group(0) @binding(2) var<storage, read_write> dB: array<ELEM>;
@group(0) @binding(3) var<uniform> params_b: BackwardParams;

@compute @workgroup_size(16, 16)
fn matmul_backward_b(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let global_id = tile_global_id(wg_id, num_wg, local_id, (params_b.n + 15u) / 16u);
    // Stacked products, as in `matmul.wgsl`.
    let tiles_y = (params_b.k + 15u) / 16u;
    let batch_raw = (global_id.y / 16u) / tiles_y;
    let in_batch = batch_raw < params_b.batch;
    let batch = select(0u, batch_raw, in_batch);
    let a_off = batch * params_b.m * params_b.k;
    let g_off = batch * params_b.m * params_b.n;
    let d_off = batch * params_b.k * params_b.n;
    // Row in dB (also col in A, which is row in A^T)
    let row = select(params_b.k, ((global_id.y / 16u) % tiles_y) * 16u + local_id.y, in_batch);
    let col = global_id.x;  // Col in dB (also col in grad)
    let local_row = local_id.y;
    let local_col = local_id.x;

    // One partial per tile, summed pairwise across tiles (#319): a single running sum over
    // all of k loses about k * 2^-24 relative, 4% at k = 2^22.
    var acc: PairwiseSum;

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
            tile_a[local_row * TILE_SIZE + local_col] = ld_a(a_off + at_col * params_b.k + at_row);
        } else {
            tile_a[local_row * TILE_SIZE + local_col] = 0.0;
        }

        // Load tile of grad [M, N] at position [t*TILE_SIZE + local_row, col]
        let grad_row = t * TILE_SIZE + local_row;
        let grad_col = col;
        if (grad_row < params_b.m && grad_col < params_b.n) {
            tile_grad[local_row * TILE_SIZE + local_col] = ld_grad_b(g_off + grad_row * params_b.n + grad_col);
        } else {
            tile_grad[local_row * TILE_SIZE + local_col] = 0.0;
        }

        workgroupBarrier();

        // Compute partial dot product
        var partial: f32 = 0.0;
        for (var i: u32 = 0u; i < TILE_SIZE; i = i + 1u) {
            partial = partial + tile_a[local_row * TILE_SIZE + i] * tile_grad[i * TILE_SIZE + local_col];
        }
        pairwise_push(&acc, partial);

        workgroupBarrier();
    }

    // Write result
    if (row < params_b.k && col < params_b.n) {
        st_dB(d_off + row * params_b.n + col, pairwise_total(&acc));
    }
}

var<workgroup> tile_a: array<f32, 256>;

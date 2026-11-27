// Matrix multiplication: C = A @ B
// Uses tiled approach with shared memory for better performance
//
// This is a classic GPU optimization technique:
// 1. Load tiles of A and B into fast shared memory
// 2. Compute partial results within the tile
// 3. Accumulate across tiles

@group(0) @binding(0) var<storage, read> a: array<ELEM>;
@group(0) @binding(1) var<storage, read> b: array<ELEM>;
@group(0) @binding(2) var<storage, read_write> c: array<ELEM>;

struct Params {
    m: u32,  // Rows of A
    k: u32,  // Cols of A / Rows of B
    n: u32,  // Cols of B
    batch: u32, // Independent products, stacked: A is [batch, M, K], B [batch, K, N], C [batch, M, N]
}

@group(0) @binding(3) var<uniform> params: Params;

// Tile size - 16x16 is a common choice
const TILE_SIZE: u32 = 16u;

// Shared memory for tiles
var<workgroup> tile_a: array<f32, 256>;  // 16 * 16
var<workgroup> tile_b: array<f32, 256>;  // 16 * 16

@compute @workgroup_size(16, 16)
fn matmul(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let global_id = tile_global_id(wg_id, num_wg, local_id, (params.n + 15u) / 16u);
    // Tile rows are stacked per product: tile row r is row tile r % tiles_y of product
    // r / tiles_y. A workgroup past the last product (a 2-D dispatch's tail) reads product 0
    // and stores nothing (its row is out of range).
    let tiles_y = (params.m + 15u) / 16u;
    let batch_raw = (global_id.y / 16u) / tiles_y;
    let in_batch = batch_raw < params.batch;
    let batch = select(0u, batch_raw, in_batch);
    let row = select(params.m, ((global_id.y / 16u) % tiles_y) * 16u + local_id.y, in_batch);
    let col = global_id.x;
    let a_off = batch * params.m * params.k;
    let b_off = batch * params.k * params.n;
    let c_off = batch * params.m * params.n;
    let local_row = local_id.y;
    let local_col = local_id.x;

    // One partial per tile, summed pairwise across tiles (#319): a single running sum over
    // all of k loses about k * 2^-24 relative, 4% at k = 2^22.
    var acc: PairwiseSum;

    // Number of tiles we need to process
    let num_tiles = (params.k + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // Load tile of A into shared memory
        let a_row = row;
        let a_col = t * TILE_SIZE + local_col;
        if (a_row < params.m && a_col < params.k) {
            tile_a[local_row * TILE_SIZE + local_col] = ld_a(a_off + a_row * params.k + a_col);
        } else {
            tile_a[local_row * TILE_SIZE + local_col] = 0.0;
        }

        // Load tile of B into shared memory
        let b_row = t * TILE_SIZE + local_row;
        let b_col = col;
        if (b_row < params.k && b_col < params.n) {
            tile_b[local_row * TILE_SIZE + local_col] = ld_b(b_off + b_row * params.n + b_col);
        } else {
            tile_b[local_row * TILE_SIZE + local_col] = 0.0;
        }

        // Synchronize to make sure the tile is loaded
        workgroupBarrier();

        // Compute partial dot product for this tile
        var partial: f32 = 0.0;
        for (var i: u32 = 0u; i < TILE_SIZE; i = i + 1u) {
            partial = partial + tile_a[local_row * TILE_SIZE + i] * tile_b[i * TILE_SIZE + local_col];
        }
        pairwise_push(&acc, partial);

        // Synchronize before loading next tile
        workgroupBarrier();
    }

    // Write result
    if (row < params.m && col < params.n) {
        st_c(c_off + row * params.n + col, pairwise_total(&acc));
    }
}

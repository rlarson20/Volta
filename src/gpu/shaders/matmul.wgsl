// Matrix multiplication: C = A @ B
// Uses tiled approach with shared memory for better performance
//
// This is a classic GPU optimization technique:
// 1. Load tiles of A and B into fast shared memory
// 2. Compute partial results within the tile
// 3. Accumulate across tiles

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

struct Params {
    m: u32,  // Rows of A
    k: u32,  // Cols of A / Rows of B
    n: u32,  // Cols of B
    _padding: u32,
}

@group(0) @binding(3) var<uniform> params: Params;

// Tile size - 16x16 is a common choice
const TILE_SIZE: u32 = 16u;

// Shared memory for tiles
var<workgroup> tile_a: array<f32, 256>;  // 16 * 16
var<workgroup> tile_b: array<f32, 256>;  // 16 * 16

@compute @workgroup_size(16, 16)
fn matmul(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let row = global_id.y;
    let col = global_id.x;
    let local_row = local_id.y;
    let local_col = local_id.x;

    var sum: f32 = 0.0;

    // Number of tiles we need to process
    let num_tiles = (params.k + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // Load tile of A into shared memory
        let a_row = row;
        let a_col = t * TILE_SIZE + local_col;
        if (a_row < params.m && a_col < params.k) {
            tile_a[local_row * TILE_SIZE + local_col] = a[a_row * params.k + a_col];
        } else {
            tile_a[local_row * TILE_SIZE + local_col] = 0.0;
        }

        // Load tile of B into shared memory
        let b_row = t * TILE_SIZE + local_row;
        let b_col = col;
        if (b_row < params.k && b_col < params.n) {
            tile_b[local_row * TILE_SIZE + local_col] = b[b_row * params.n + b_col];
        } else {
            tile_b[local_row * TILE_SIZE + local_col] = 0.0;
        }

        // Synchronize to make sure the tile is loaded
        workgroupBarrier();

        // Compute partial dot product for this tile
        for (var i: u32 = 0u; i < TILE_SIZE; i = i + 1u) {
            sum = sum + tile_a[local_row * TILE_SIZE + i] * tile_b[i * TILE_SIZE + local_col];
        }

        // Synchronize before loading next tile
        workgroupBarrier();
    }

    // Write result
    if (row < params.m && col < params.n) {
        c[row * params.n + col] = sum;
    }
}

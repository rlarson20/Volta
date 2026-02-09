// Implicit GEMM (iGEMM) Convolution GPU Shader
// Tiled computation without materializing im2col matrix
// Each workgroup computes a tile of output channels and spatial positions

struct IGEMMParams {
    batch_size: u32,
    in_channels: u32,
    out_channels: u32,
    height: u32,
    width: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
    h_out: u32,
    w_out: u32,
    // Tile configuration
    oc_tile: u32,  // Output channels per tile
    ic_tile: u32,  // Input channels per tile
    s_tile: u32,   // Spatial positions per tile
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: IGEMMParams;

// Workgroup-local storage for accumulating partial results
// Each thread accumulates its own result, no shared memory needed
// for this simple tiled approach

@compute @workgroup_size(16, 16, 1)
fn igemm_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_size = params.batch_size;
    let in_channels = params.in_channels;
    let out_channels = params.out_channels;
    let height = params.height;
    let width = params.width;
    let kernel_h = params.kernel_h;
    let kernel_w = params.kernel_w;
    let stride_h = params.stride_h;
    let stride_w = params.stride_w;
    let pad_h = params.pad_h;
    let pad_w = params.pad_w;
    let h_out = params.h_out;
    let w_out = params.w_out;
    let oc_tile = params.oc_tile;
    let ic_tile = params.ic_tile;
    let s_tile = params.s_tile;

    // Decode global_id to (out_channel, spatial_position)
    // global_id.x = output channel (tiled)
    // global_id.y = spatial position in batch (tiled)
    // global_id.z = tile ID (oc_tile * s_tile workgroups per tile)

    let oc = global_id.x;
    let spatial_idx = global_id.y;
    let tile_id = global_id.z;

    // Calculate which tile we're in
    let tiles_per_oc = (out_channels + oc_tile - 1u) / oc_tile;
    let tiles_per_spatial = ((batch_size * h_out * w_out) + s_tile - 1u) / s_tile;

    let oc_tile_idx = tile_id / tiles_per_spatial;
    let spatial_tile_idx = tile_id % tiles_per_spatial;

    // Calculate actual output channel and spatial position
    let actual_oc = oc_tile_idx * oc_tile + oc;
    if (actual_oc >= out_channels) {
        return;
    }

    let actual_spatial = spatial_tile_idx * s_tile + spatial_idx;
    let total_spatial = batch_size * h_out * w_out;
    if (actual_spatial >= total_spatial) {
        return;
    }

    // Decode spatial index to (batch, h_out, w_out)
    let b = actual_spatial / (h_out * w_out);
    let remaining = actual_spatial % (h_out * w_out);
    let oh = remaining / w_out;
    let ow = remaining % w_out;

    // Compute output[b, actual_oc, oh, ow] using tiled accumulation
    var sum: f32 = 0.0;

    // Iterate over input channel tiles
    var ic_start: u32 = 0u;
    while (ic_start < in_channels) {
        let ic_end = min(ic_start + ic_tile, in_channels);

        // Accumulate over this input channel tile
        for (var ic: u32 = ic_start; ic < ic_end; ic++) {
            // For each kernel position
            for (var kh: u32 = 0u; kh < kernel_h; kh++) {
                for (var kw: u32 = 0u; kw < kernel_w; kw++) {
                    // Input position with padding
                    let h_in = oh * stride_h + kh - pad_h;
                    let w_in = ow * stride_w + kw - pad_w;

                    // Check if within valid bounds (not in padding area)
                    if (h_in >= 0u && h_in < height && w_in >= 0u && w_in < width) {
                        // Input index: NCHW format
                        // input[b, ic, h_in, w_in]
                        let input_idx = b * in_channels * height * width +
                                       ic * height * width +
                                       h_in * width +
                                       w_in;

                        // Weight index: out_channels, in_channels, kernel_h, kernel_w
                        // weight[actual_oc, ic, kh, kw]
                        let weight_idx = actual_oc * in_channels * kernel_h * kernel_w +
                                        ic * kernel_h * kernel_w +
                                        kh * kernel_w +
                                        kw;

                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }

        ic_start = ic_end;
    }

    // Output index: NCHW format
    // output[b, actual_oc, oh, ow]
    let output_idx = b * out_channels * h_out * w_out +
                    actual_oc * h_out * w_out +
                    oh * w_out +
                    ow;

    output[output_idx] = sum;
}

// col2im: Column-to-image transformation for convolution gradient computation
// Inverse of im2col: transforms 2D matrix (B*H_out*W_out, C*K_h*K_w) back to 4D tensor (B, C, H, W)
// Each input position accumulates gradients from all output positions where it contributes

@group(0) @binding(0) var<storage, read> col: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Col2imParams {
    batch_size: u32,
    channels: u32,
    height: u32,
    width: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_h: u32,
    stride_w: u32,
    h_out: u32,
    w_out: u32,
    _padding: vec2<u32>,
}

@group(0) @binding(2) var<uniform> params: Col2imParams;

// Convert 4D coordinates to linear index (NCHW format)
fn coords_to_idx_4d(n: u32, c: u32, h: u32, w: u32) -> u32 {
    let c_stride = params.height * params.width;
    let h_stride = params.width;
    return n * params.channels * c_stride + c * c_stride + h * h_stride + w;
}

// Main col2im transformation
@compute @workgroup_size(16, 16, 1)
fn col2im_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Each thread handles one input position (b, c, h, w)
    let ch = global_id.x;
    let flat_spatial = global_id.y;

    // Check bounds
    if (ch >= params.channels) {
        return;
    }

    let spatial_size = params.batch_size * params.height * params.width;
    if (flat_spatial >= spatial_size) {
        return;
    }

    // Decode spatial position: (batch, height, width)
    let batch = flat_spatial / (params.height * params.width);
    let remainder = flat_spatial % (params.height * params.width);
    let h = remainder / params.width;
    let w = remainder % params.width;

    // Accumulate gradient from all output positions where this input contributes
    var acc: f32 = 0.0;
    let cols = params.channels * params.kernel_h * params.kernel_w;

    // Iterate over all output positions
    for (var oh: u32 = 0u; oh < params.h_out; oh = oh + 1u) {
        for (var ow: u32 = 0u; ow < params.w_out; ow = ow + 1u) {
            // Calculate starting position in input (top-left of receptive field)
            let h_start = oh * params.stride_h;
            let w_start = ow * params.stride_w;

            // Check if this input position is in the receptive field
            if (h >= h_start && h < h_start + params.kernel_h &&
                w >= w_start && w < w_start + params.kernel_w) {

                // Calculate position in kernel
                let kh = h - h_start;
                let kw = w - w_start;

                // Calculate row index in col matrix
                let row_idx = (batch * params.h_out + oh) * params.w_out + ow;

                // Calculate column index in col matrix
                let col_idx = ch * (params.kernel_h * params.kernel_w) + kh * params.kernel_w + kw;

                // Read from col matrix and accumulate
                let col_data_idx = row_idx * cols + col_idx;
                acc += col[col_data_idx];
            }
        }
    }

    // Write accumulated gradient to output
    let output_idx = coords_to_idx_4d(batch, ch, h, w);
    output[output_idx] = acc;
}

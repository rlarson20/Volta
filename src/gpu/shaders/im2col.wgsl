// im2col: Image-to-column transformation for GPU convolution
// Transforms 4D input (B, C, H, W) into 2D matrix (B*H_out*W_out, C*K_h*K_w)
// Each row represents a flattened receptive field for one output position

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Im2colParams {
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

@group(0) @binding(2) var<uniform> params: Im2colParams;

// Convert 4D coordinates to linear index (NCHW format)
fn coords_to_idx_4d(n: u32, c: u32, h: u32, w: u32) -> u32 {
    let c_stride = params.height * params.width;
    let h_stride = params.width;
    return n * params.channels * c_stride + c * c_stride + h * h_stride + w;
}

// Main im2col transformation
@compute @workgroup_size(256, 1, 1)
fn im2col_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Each thread handles one output position (one row of the output matrix)
    let output_row = global_id.x;
    let total_outputs = params.batch_size * params.h_out * params.w_out;

    if (output_row >= total_outputs) {
        return;
    }

    // Decode output position: (batch, h_out_pos, w_out_pos)
    let hw_out = params.h_out * params.w_out;
    let b = output_row / hw_out;
    let remaining = output_row % hw_out;
    let h_out_pos = remaining / params.w_out;
    let w_out_pos = remaining % params.w_out;

    // Calculate starting position in input (top-left of receptive field)
    let h_start = h_out_pos * params.stride_h;
    let w_start = w_out_pos * params.stride_w;

    // Fill the row: flatten receptive field [C, K_h, K_w] into columns
    let cols = params.channels * params.kernel_h * params.kernel_w;

    for (var col_idx: u32 = 0u; col_idx < cols; col_idx = col_idx + 1u) {
        // Decode column_idx to (c, kh, kw)
        let khkw = params.kernel_h * params.kernel_w;
        let c = col_idx / khkw;
        let rem = col_idx % khkw;
        let kh = rem / params.kernel_w;
        let kw = rem % params.kernel_w;

        // Calculate position in input tensor
        let h_pos = h_start + kh;
        let w_pos = w_start + kw;

        // Read input value
        let in_idx = coords_to_idx_4d(b, c, h_pos, w_pos);
        let value = input[in_idx];

        // Write to output matrix
        let out_idx = output_row * cols + col_idx;
        output[out_idx] = value;
    }
}

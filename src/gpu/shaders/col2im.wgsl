// col2im: Column-to-image transformation for convolution gradient computation
// Inverse of im2col: transforms 2D matrix (B*H_out*W_out, C*K_h*K_w) back to 4D tensor (B, C, H, W)
// Each input position accumulates gradients from all output positions where it contributes

@group(0) @binding(0) var<storage, read> col: array<ELEM>;
@group(0) @binding(1) var<storage, read_write> output: array<ELEM>;

@group(0) @binding(2) var<uniform> params: WindowParams;

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

    // Only the output positions whose window covers (h, w) contribute (#309): along each
    // axis, `o * stride <= h < o * stride + kernel`, so `o` runs from
    // `ceil((h + 1 - kernel) / stride)` (0 if `h < kernel`) to `h / stride`, capped at the
    // last output position. The order is the same as a scan over every position, so the
    // sum is bit-identical; it just skips the positions that never matched.
    // The guards keep `h + 1 - kernel` from wrapping below zero.
    var oh_lo: u32 = 0u;
    if (h + 1u > params.kernel_h) {
        oh_lo = (h + 1u - params.kernel_h + params.stride_h - 1u) / params.stride_h;
    }
    var ow_lo: u32 = 0u;
    if (w + 1u > params.kernel_w) {
        ow_lo = (w + 1u - params.kernel_w + params.stride_w - 1u) / params.stride_w;
    }
    let oh_end = min(h / params.stride_h + 1u, params.h_out);
    let ow_end = min(w / params.stride_w + 1u, params.w_out);

    for (var oh: u32 = oh_lo; oh < oh_end; oh = oh + 1u) {
        for (var ow: u32 = ow_lo; ow < ow_end; ow = ow + 1u) {
            // Position in the kernel: in range by the loop bounds.
            let kh = h - oh * params.stride_h;
            let kw = w - ow * params.stride_w;

            // Calculate row index in col matrix
            let row_idx = (batch * params.h_out + oh) * params.w_out + ow;

            // Calculate column index in col matrix
            let col_idx = ch * (params.kernel_h * params.kernel_w) + kh * params.kernel_w + kw;

            // Read from col matrix and accumulate
            let col_data_idx = row_idx * cols + col_idx;
            acc += ld_col(col_data_idx);
        }
    }

    // Write accumulated gradient to output
    let output_idx = coords_to_idx_4d(batch, ch, h, w);
    st_output(output_idx, acc);
}

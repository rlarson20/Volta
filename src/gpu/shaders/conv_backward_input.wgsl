// GPU-accelerated convolution backward: gradient with respect to input
// Computes: ∂L/∂X[b,ic,ih,iw] = Σ ∂L/∂Y[b,oc,oh,ow] * W[oc,ic,kh,kw]

struct ConvBackwardInputParams {
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
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> out_grad: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_input: array<f32>;
@group(0) @binding(3) var<uniform> params: ConvBackwardInputParams;

@compute @workgroup_size(16, 16, 1)
fn conv_backward_input_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    // Decode thread position
    let in_ch = global_id.x;
    let flat_spatial = global_id.y;

    // Check bounds
    if (in_ch >= params.in_channels) {
        return;
    }

    let spatial_size = params.batch_size * params.height * params.width;
    if (flat_spatial >= spatial_size) {
        return;
    }

    // Decode spatial position: [batch, height, width]
    let batch = flat_spatial / (params.height * params.width);
    let remainder = flat_spatial % (params.height * params.width);
    let ih = remainder / params.width;
    let iw = remainder % params.width;

    // Accumulate gradient from all output positions and kernel positions
    var sum: f32 = 0.0;

    // Iterate over output channels
    for (var oc: u32 = 0u; oc < params.out_channels; oc++) {
        // Iterate over kernel positions
        for (var kh: u32 = 0u; kh < params.kernel_h; kh++) {
            for (var kw: u32 = 0u; kw < params.kernel_w; kw++) {
                // Find which output position this kernel position contributes to
                // In forward: out[oh,ow] = in[ih-ph+kh, iw-pw+kw] * weight[kh,kw]
                // So: ih = oh * stride_h - pad_h + kh
                // Therefore: oh = (ih + pad_h - kh) / stride_h
                let oh_candidate = (ih + params.pad_h - kh);
                let ow_candidate = (iw + params.pad_w - kw);

                // Check if this is a valid output position
                if (oh_candidate % params.stride_h == 0u && ow_candidate % params.stride_w == 0u) {
                    let oh = oh_candidate / params.stride_h;
                    let ow = ow_candidate / params.stride_w;

                    if (oh < params.h_out && ow < params.w_out) {
                        // Get output gradient at this position
                        let out_grad_idx = ((batch * params.out_channels + oc) * params.h_out + oh) * params.w_out + ow;
                        let grad_val = out_grad[out_grad_idx];

                        // Get weight at this kernel position
                        let weight_idx = ((oc * params.in_channels + in_ch) * params.kernel_h + kh) * params.kernel_w + kw;
                        let weight_val = weight[weight_idx];

                        sum += grad_val * weight_val;
                    }
                }
            }
        }
    }

    // Write result
    let input_idx = ((batch * params.in_channels + in_ch) * params.height + ih) * params.width + iw;
    grad_input[input_idx] = sum;
}

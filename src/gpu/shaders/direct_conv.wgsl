// Direct Convolution GPU Shader
// Memory-efficient convolution without intermediate im2col matrix
// Each thread computes ONE output pixel

struct DirectConvParams {
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
    _padding: u32,  // 16-byte alignment
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: DirectConvParams;

@compute @workgroup_size(16, 16, 1)
fn direct_conv_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
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

    // Decode global_id to (batch, out_ch, h_out, w_out)
    // global_id.x = out_channel
    // global_id.y = batch * h_out * w_out (flattened)
    // global_id.z = unused (workgroup z dimension)

    let oc = global_id.x;
    if (oc >= out_channels) {
        return;
    }

    let flattened_idx = global_id.y;
    if (flattened_idx >= batch_size * h_out * w_out) {
        return;
    }

    // Decode flattened_idx to (batch, h_out, w_out)
    let b = flattened_idx / (h_out * w_out);
    let remaining = flattened_idx % (h_out * w_out);
    let oh = remaining / w_out;
    let ow = remaining % w_out;

    // Compute output pixel (b, oc, oh, ow)
    var sum: f32 = 0.0;

    // Convolution: sum over input channels and kernel
    for (var ic: u32 = 0u; ic < in_channels; ic++) {
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
                    // weight[oc, ic, kh, kw]
                    let weight_idx = oc * in_channels * kernel_h * kernel_w +
                                    ic * kernel_h * kernel_w +
                                    kh * kernel_w +
                                    kw;

                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Output index: NCHW format
    // output[b, oc, oh, ow]
    let output_idx = b * out_channels * h_out * w_out +
                    oc * h_out * w_out +
                    oh * w_out +
                    ow;

    output[output_idx] = sum;
}

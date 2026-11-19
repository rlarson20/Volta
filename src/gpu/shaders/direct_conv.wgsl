// Direct Convolution GPU Shader
// Memory-efficient convolution without intermediate im2col matrix
// Each thread computes ONE output pixel

@group(0) @binding(0) var<storage, read> input: array<ELEM>;
@group(0) @binding(1) var<storage, read> weight: array<ELEM>;
@group(0) @binding(2) var<storage, read_write> output: array<ELEM>;
@group(0) @binding(3) var<uniform> params: ConvParams;

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
                // Position in the padded input. The padding test compares before subtracting
                // `pad`, so no coordinate relies on u32 wraparound (#314).
                let h_pad = oh * stride_h + kh;
                let w_pad = ow * stride_w + kw;

                // Skip the padding rows and columns (they contribute zero)
                if (h_pad >= pad_h && h_pad < pad_h + height && w_pad >= pad_w && w_pad < pad_w + width) {
                    let h_in = h_pad - pad_h;
                    let w_in = w_pad - pad_w;
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

                    sum += ld_input(input_idx) * ld_weight(weight_idx);
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

    st_output(output_idx, sum);
}

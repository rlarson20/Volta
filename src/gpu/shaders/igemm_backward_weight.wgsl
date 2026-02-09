// GPU-accelerated iGEMM convolution backward: gradient with respect to weights
// Computes: ∂L/∂W[oc,ic,kh,kw] = Σ X[b,ic,oh*sh+kh-ph,ow*sw+kw-pw] * ∂L/∂Y[b,oc,oh,ow]
// Uses the same algorithm as regular conv backward (gradient computation is independent of forward strategy)

struct IGEMMBackwardWeightParams {
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
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_weight: array<f32>;
@group(0) @binding(3) var<uniform> params: IGEMMBackwardWeightParams;

@compute @workgroup_size(16, 16, 1)
fn igemm_backward_weight_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    // Decode thread position
    let out_ch = global_id.x;
    let flat_kernel = global_id.y;

    // Check bounds
    if (out_ch >= params.out_channels) {
        return;
    }

    let kernel_size = params.in_channels * params.kernel_h * params.kernel_w;
    if (flat_kernel >= kernel_size) {
        return;
    }

    // Decode kernel position: [in_ch, kernel_h, kernel_w]
    let in_ch = flat_kernel / (params.kernel_h * params.kernel_w);
    let remainder = flat_kernel % (params.kernel_h * params.kernel_w);
    let kh = remainder / params.kernel_w;
    let kw = remainder % params.kernel_w;

    // Accumulate gradient from all batch and output positions
    var sum: f32 = 0.0;

    for (var batch: u32 = 0u; batch < params.batch_size; batch++) {
        // Iterate over output positions
        for (var oh: u32 = 0u; oh < params.h_out; oh++) {
            for (var ow: u32 = 0u; ow < params.w_out; ow++) {
                // Find input position that corresponds to this kernel position
                let ih = oh * params.stride_h + kh;
                let iw = ow * params.stride_w + kw;

                // Check if this is a valid input position (considering padding)
                // Input was padded, so we need to check if the unpadded input position is valid
                let pad_h_i32 = i32(params.pad_h);
                let pad_w_i32 = i32(params.pad_w);
                let ih_unpadded = i32(ih) - pad_h_i32;
                let iw_unpadded = i32(iw) - pad_w_i32;
                let height_i32 = i32(params.height);
                let width_i32 = i32(params.width);

                if (ih_unpadded >= 0 && ih_unpadded < height_i32 &&
                    iw_unpadded >= 0 && iw_unpadded < width_i32) {
                    // Get input value at this position
                    let input_idx = ((batch * params.in_channels + in_ch) * params.height + u32(ih_unpadded)) * params.width + u32(iw_unpadded);
                    let input_val = input[input_idx];

                    // Get output gradient at this position
                    let out_grad_idx = ((batch * params.out_channels + out_ch) * params.h_out + oh) * params.w_out + ow;
                    let grad_val = out_grad[out_grad_idx];

                    sum += input_val * grad_val;
                }
            }
        }
    }

    // Write result
    let weight_idx = ((out_ch * params.in_channels + in_ch) * params.kernel_h + kh) * params.kernel_w + kw;
    grad_weight[weight_idx] = sum;
}

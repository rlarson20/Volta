// `ConvTranspose2d`'s column steps on the GPU. The forward is `col = X_r @ W_r` (a matmul,
// `[B*H_in*W_in, C_out*K_h*K_w]`), then `col2im_t` folds the columns into the output image
// and adds the bias; backward unfolds the output gradient with `im2col_t` and runs two more
// matmuls. Both are gathers (one thread per element written), so there are no atomics.
//
// `col2im_t` visits an output pixel's taps in ascending (ih, iw), the CPU loop's order.
//
// Bindings: 0 = input (ELEM), 1 = out (ELEM), 2 = bias (ELEM, col2im_t only), 3 = params.

struct Params {
    batch: u32,
    out_ch: u32,
    in_h: u32,
    in_w: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
    out_h: u32,
    out_w: u32,
    has_bias: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<ELEM>;
@group(0) @binding(1) var<storage, read_write> out: array<ELEM>;
@group(0) @binding(2) var<storage, read> bias: array<ELEM>;
@group(0) @binding(3) var<uniform> params: Params;

// One thread per output pixel (b, c, oh, ow): the sum of every column entry that lands on
// it, plus the channel's bias.
@compute @workgroup_size(256)
fn col2im_t(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let o = flat_global_id(gid, num_wg).x;
    let plane = params.out_h * params.out_w;
    if (o >= params.batch * params.out_ch * plane) {
        return;
    }
    let b = o / (params.out_ch * plane);
    let c = (o / plane) % params.out_ch;
    let h_pos = (o % plane) / params.out_w + params.pad_h;
    let w_pos = o % params.out_w + params.pad_w;
    let cols = params.out_ch * params.kernel_h * params.kernel_w;
    var acc = 0.0;
    // h_pos = ih * stride_h + kh: kh from high to low is ih ascending.
    for (var kh_r = 0u; kh_r < params.kernel_h; kh_r = kh_r + 1u) {
        let kh = params.kernel_h - 1u - kh_r;
        if (kh > h_pos || (h_pos - kh) % params.stride_h != 0u) {
            continue;
        }
        let ih = (h_pos - kh) / params.stride_h;
        if (ih >= params.in_h) {
            continue;
        }
        for (var kw_r = 0u; kw_r < params.kernel_w; kw_r = kw_r + 1u) {
            let kw = params.kernel_w - 1u - kw_r;
            if (kw > w_pos || (w_pos - kw) % params.stride_w != 0u) {
                continue;
            }
            let iw = (w_pos - kw) / params.stride_w;
            if (iw >= params.in_w) {
                continue;
            }
            let row = (b * params.in_h + ih) * params.in_w + iw;
            acc = acc + ld_input(row * cols + (c * params.kernel_h + kh) * params.kernel_w + kw);
        }
    }
    if (params.has_bias != 0u) {
        acc = acc + ld_bias(c);
    }
    st_out(o, acc);
}

// One thread per column entry (row, c, kh, kw): the output-gradient pixel that tap wrote,
// or zero for a tap that landed in the padding.
@compute @workgroup_size(256)
fn im2col_t(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let o = flat_global_id(gid, num_wg).x;
    let taps = params.kernel_h * params.kernel_w;
    let cols = params.out_ch * taps;
    let rows = params.batch * params.in_h * params.in_w;
    if (o >= rows * cols) {
        return;
    }
    let row = o / cols;
    let c = (o % cols) / taps;
    let kh = (o % taps) / params.kernel_w;
    let kw = o % params.kernel_w;
    let b = row / (params.in_h * params.in_w);
    let ih = (row / params.in_w) % params.in_h;
    let iw = row % params.in_w;
    let h_pos = ih * params.stride_h + kh;
    let w_pos = iw * params.stride_w + kw;
    var v = 0.0;
    if (h_pos >= params.pad_h && h_pos < params.out_h + params.pad_h
        && w_pos >= params.pad_w && w_pos < params.out_w + params.pad_w) {
        let oh = h_pos - params.pad_h;
        let ow = w_pos - params.pad_w;
        v = ld_input(((b * params.out_ch + c) * params.out_h + oh) * params.out_w + ow);
    }
    st_out(o, v);
}

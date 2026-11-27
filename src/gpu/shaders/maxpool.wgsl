// `MaxPool2d` forward and backward over an already padded input `[planes, padded_h,
// padded_w]` (planes = batch * channels). A window skips padding coordinates (padding is
// -inf and never wins), starts from its first real element, and follows `max_pool2d`'s
// scan (`max_rule::pool_scan_replaces`): a larger value replaces the best, and so does
// every NaN, so the last NaN wins; a tie keeps the earlier element.
//
// Backward is a gather, not a scatter: one thread per padded-input element walks the
// windows that contain it in ascending output order, recomputes each window's winner, and
// adds that window's gradient when the winner is this element. That is the CPU's order of
// `+=` per element, with no atomics.
//
// Bindings: 0 = input (ELEM), 1 = out (ELEM), 2 = grad (ELEM, backward only), 3 = params.

struct Params {
    planes: u32,
    padded_h: u32,
    padded_w: u32,
    pad_h: u32,
    pad_w: u32,
    in_h: u32,
    in_w: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_h: u32,
    stride_w: u32,
    out_h: u32,
    out_w: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<ELEM>;
@group(0) @binding(1) var<storage, read_write> out: array<ELEM>;
@group(0) @binding(2) var<storage, read> grad: array<ELEM>;
@group(0) @binding(3) var<uniform> params: Params;

// `max_rule::pool_scan_replaces`.
fn scan_replaces(cand: f32, best: f32) -> bool {
    return is_nan(cand) || (!is_nan(best) && cand > best);
}

// The padded-input index of window (plane, oh, ow)'s winner.
fn winner(plane: u32, oh: u32, ow: u32) -> u32 {
    let base = plane * params.padded_h * params.padded_w;
    var best = 0.0;
    var best_idx = 0xffffffffu;
    for (var kh = 0u; kh < params.kernel_h; kh = kh + 1u) {
        let h = oh * params.stride_h + kh;
        if (h < params.pad_h || h >= params.pad_h + params.in_h) {
            continue;
        }
        for (var kw = 0u; kw < params.kernel_w; kw = kw + 1u) {
            let w = ow * params.stride_w + kw;
            if (w < params.pad_w || w >= params.pad_w + params.in_w) {
                continue;
            }
            let idx = base + h * params.padded_w + w;
            let v = ld_input(idx);
            if (best_idx == 0xffffffffu || scan_replaces(v, best)) {
                best = v;
                best_idx = idx;
            }
        }
    }
    return best_idx;
}

@compute @workgroup_size(256)
fn maxpool_forward(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let o = flat_global_id(gid, num_wg).x;
    let per_plane = params.out_h * params.out_w;
    if (o >= params.planes * per_plane) {
        return;
    }
    let plane = o / per_plane;
    let rem = o % per_plane;
    st_out(o, ld_input(winner(plane, rem / params.out_w, rem % params.out_w)));
}

// The first output index along one axis whose window (start `o * stride`, length `kernel`)
// contains coordinate `x`.
fn first_window(x: u32, kernel: u32, stride: u32) -> u32 {
    if (x < kernel) {
        return 0u;
    }
    return (x - kernel) / stride + 1u;
}

@compute @workgroup_size(256)
fn maxpool_backward(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let i = flat_global_id(gid, num_wg).x;
    let per_plane = params.padded_h * params.padded_w;
    if (i >= params.planes * per_plane) {
        return;
    }
    let plane = i / per_plane;
    let h = (i % per_plane) / params.padded_w;
    let w = i % params.padded_w;
    var acc = 0.0;
    let oh_end = min(h / params.stride_h + 1u, params.out_h);
    let ow_end = min(w / params.stride_w + 1u, params.out_w);
    for (var oh = first_window(h, params.kernel_h, params.stride_h); oh < oh_end; oh = oh + 1u) {
        for (var ow = first_window(w, params.kernel_w, params.stride_w); ow < ow_end; ow = ow + 1u) {
            if (winner(plane, oh, ow) == i) {
                acc = acc + ld_grad((plane * params.out_h + oh) * params.out_w + ow);
            }
        }
    }
    st_out(i, acc);
}

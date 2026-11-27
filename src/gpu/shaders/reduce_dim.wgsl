// Reductions along one axis (`sum_dim`, `mean_dim`, `max_dim`) and their backward passes.
//
// The input is viewed as [outer, dim, inner] and reduced over `dim`, so output `g` is
// (o, j) = (g / inner, g % inner) and its k-th input is `o * dim * inner + k * inner + j`.
// Each reduction has two launch shapes, picked by the host:
//
// - `*_cols`: one thread per output, which scans its `dim` inputs. For short axes.
// - `*_rows`: one 256-thread workgroup per output. Thread t folds k = t, t + 256, ..., and
//   the threads' results are combined by a tree in shared memory. For long axes.
//
// Sums accumulate in f32 through `PairwiseSum` (and the tree), so the rounding error grows
// with log2(dim), as the CPU's pairwise sum does (S6). `max_dim` follows `torch.max(x, dim)`
// (`max_rule::dim_scan_replaces`): a larger value wins, the first NaN wins, and a tie keeps
// the earlier index. Its backward recomputes that index from the saved input and writes the
// whole row: the gradient at the winner, zero elsewhere.
//
// Bindings (each entry binds only those it uses; the pipelines use an auto layout):
//   0 = input (ELEM), 1 = out (ELEM), 2 = grad (ELEM, backward only), 3 = params.

struct Params {
    outer: u32,
    dim: u32,
    inner: u32,
    divisor: f32, // each sum is divided by this (1.0, or the axis length for a mean)
}

@group(0) @binding(0) var<storage, read> input: array<ELEM>;
@group(0) @binding(1) var<storage, read_write> out: array<ELEM>;
@group(0) @binding(2) var<storage, read> grad: array<ELEM>;
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;

var<workgroup> sh_val: array<f32, 256>;
var<workgroup> sh_idx: array<u32, 256>;

// The flat index of this workgroup in a (possibly 2-D) dispatch (`kernels::dispatch_flat`).
fn group_id(wid: vec3<u32>, num_wg: vec3<u32>) -> u32 {
    return wid.x + wid.y * num_wg.x;
}

// The index of output `g`'s first input; its k-th input is at `row_base(g) + k * inner`.
fn row_base(g: u32) -> u32 {
    return (g / params.inner) * params.dim * params.inner + g % params.inner;
}

// `max_rule::dim_scan_replaces`: whether `cand` replaces the running best of a scan in
// ascending index order.
fn scan_replaces(cand: f32, best: f32) -> bool {
    return (!is_nan(cand) && !is_nan(best) && cand > best) || (is_nan(cand) && !is_nan(best));
}

// Whether (vb, ib) beats (va, ia) as `torch.max(x, dim)`'s winner, whatever order the two
// were found in: the first NaN, else the largest value, else (a tie) the smaller index.
fn beats(vb: f32, ib: u32, va: f32, ia: u32) -> bool {
    let na = is_nan(va);
    let nb = is_nan(vb);
    if (na || nb) {
        return nb && (!na || ib < ia);
    }
    return vb > va || (vb == va && ib < ia);
}

// The index along the axis of output `g`'s maximum, scanned by one thread.
fn argmax_scan(base: u32) -> u32 {
    var best = ld_input(base);
    var best_k = 0u;
    for (var k = 1u; k < params.dim; k = k + 1u) {
        let v = ld_input(base + k * params.inner);
        if (scan_replaces(v, best)) {
            best = v;
            best_k = k;
        }
    }
    return best_k;
}

// The index along the axis of output `g`'s maximum, found by the whole workgroup: every
// thread gets it. `tid` must be the local index, and every thread of the group must call it.
fn argmax_tree(tid: u32, base: u32) -> u32 {
    // Thread `tid` scans k = tid, tid + WG, ... in ascending order; a thread with no input
    // holds a NaN-free sentinel that every real element beats (index past the axis).
    var best = 0.0;
    var best_k = params.dim;
    for (var k = tid; k < params.dim; k = k + WG) {
        let v = ld_input(base + k * params.inner);
        if (best_k == params.dim || scan_replaces(v, best)) {
            best = v;
            best_k = k;
        }
    }
    sh_val[tid] = best;
    sh_idx[tid] = best_k;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            let ia = sh_idx[tid];
            let ib = sh_idx[tid + s];
            // An empty slot (index `dim`) never wins and always loses.
            if (ib != params.dim && (ia == params.dim || beats(sh_val[tid + s], ib, sh_val[tid], ia))) {
                sh_val[tid] = sh_val[tid + s];
                sh_idx[tid] = ib;
            }
        }
        workgroupBarrier();
    }
    let winner = sh_idx[0];
    workgroupBarrier();
    return winner;
}

@compute @workgroup_size(256)
fn sum_dim_cols(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let g = flat_global_id(gid, num_wg).x;
    if (g >= params.outer * params.inner) {
        return;
    }
    let base = row_base(g);
    var acc: PairwiseSum;
    for (var k = 0u; k < params.dim; k = k + 1u) {
        pairwise_push(&acc, ld_input(base + k * params.inner));
    }
    st_out(g, pairwise_total(&acc) / params.divisor);
}

@compute @workgroup_size(256)
fn sum_dim_rows(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let g = group_id(wid, num_wg);
    let tid = lid.x;
    // `g` is the same for the whole workgroup, so this exit keeps the barriers uniform.
    if (g >= params.outer * params.inner) {
        return;
    }
    let base = row_base(g);
    var acc: PairwiseSum;
    for (var k = tid; k < params.dim; k = k + WG) {
        pairwise_push(&acc, ld_input(base + k * params.inner));
    }
    sh_val[tid] = pairwise_total(&acc);
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            sh_val[tid] = sh_val[tid] + sh_val[tid + s];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        st_out(g, sh_val[0] / params.divisor);
    }
}

@compute @workgroup_size(256)
fn max_dim_cols(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let g = flat_global_id(gid, num_wg).x;
    if (g >= params.outer * params.inner) {
        return;
    }
    let base = row_base(g);
    st_out(g, ld_input(base + argmax_scan(base) * params.inner));
}

@compute @workgroup_size(256)
fn max_dim_rows(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let g = group_id(wid, num_wg);
    if (g >= params.outer * params.inner) {
        return;
    }
    let base = row_base(g);
    let k = argmax_tree(lid.x, base);
    if (lid.x == 0u) {
        st_out(g, ld_input(base + k * params.inner));
    }
}

// Backward of `sum_dim`/`mean_dim`: one thread per input element, `grad[g] / divisor`.
@compute @workgroup_size(256)
fn sum_dim_backward(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let i = flat_global_id(gid, num_wg).x;
    if (i >= params.outer * params.dim * params.inner) {
        return;
    }
    let g = (i / (params.dim * params.inner)) * params.inner + i % params.inner;
    st_out(i, ld_grad(g) / params.divisor);
}

@compute @workgroup_size(256)
fn max_dim_backward_cols(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let g = flat_global_id(gid, num_wg).x;
    if (g >= params.outer * params.inner) {
        return;
    }
    let base = row_base(g);
    let winner = argmax_scan(base);
    let gv = ld_grad(g);
    for (var k = 0u; k < params.dim; k = k + 1u) {
        st_out(base + k * params.inner, select(0.0, gv, k == winner));
    }
}

@compute @workgroup_size(256)
fn max_dim_backward_rows(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let g = group_id(wid, num_wg);
    if (g >= params.outer * params.inner) {
        return;
    }
    let base = row_base(g);
    let winner = argmax_tree(lid.x, base);
    let gv = ld_grad(g);
    for (var k = lid.x; k < params.dim; k = k + WG) {
        st_out(base + k * params.inner, select(0.0, gv, k == winner));
    }
}

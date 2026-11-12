// Parallel full-tensor reductions (sum, mean, sum of squares, max with index).
//
// One pass turns `params.n` values into `ceil(n / 1024)` partial results: a workgroup of
// 256 threads owns 1024 consecutive inputs, each thread folds 4 of them (stride 256, so
// loads coalesce), and the threads' results are combined by a tree in shared memory. The
// host runs the `*_first` entry on the input, then the `*_next` entry on the partials,
// until one value is left. Every partial is f32, whatever the input's element type, and
// the adds form a tree of depth ~log(n) rather than a chain of n, which is what makes a sum
// of 2^25 ones exact (S6). The reduction is never read back: the last pass leaves the
// scalar in a one-element buffer.
//
// Bindings (each entry binds only the ones it uses; the pipelines use an auto layout):
//   *_first: 0 = input (ELEM),  2 = out (f32),  4 = out_index (u32, max only), 5 = params
//   *_next:  1 = partials (f32), 2 = out (f32), 3 = in_index, 4 = out_index (max), 5 = params
//
// Every entry reads its element count from `params.n`, never from `arrayLength`: pooled
// buffers are allocated in power-of-2 buckets, so a binding can be longer than its tensor.

struct Params {
    n: u32,        // values this pass reads
    divisor: f32,  // each output is divided by this (1.0, except the last pass of a mean)
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<ELEM>;
@group(0) @binding(1) var<storage, read> partials: array<ELEM2>;
@group(0) @binding(2) var<storage, read_write> out: array<ELEM2>;
@group(0) @binding(3) var<storage, read> in_index: array<u32>;
@group(0) @binding(4) var<storage, read_write> out_index: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;

const WG: u32 = 256u;
const PER_THREAD: u32 = 4u;
const PER_GROUP: u32 = 1024u; // WG * PER_THREAD; the host depends on it

var<workgroup> sh_val: array<f32, 256>;
var<workgroup> sh_idx: array<u32, 256>;

// The flat index of this workgroup in a (possibly 2-D) dispatch (`kernels::dispatch_flat`).
fn group_id(wid: vec3<u32>, num_wg: vec3<u32>) -> u32 {
    return wid.x + wid.y * num_wg.x;
}

// The sum of `v` over the workgroup, in a fixed tree order.
fn tree_sum(tid: u32, v: f32) -> f32 {
    sh_val[tid] = v;
    workgroupBarrier();
    for (var s: u32 = WG / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            sh_val[tid] = sh_val[tid] + sh_val[tid + s];
        }
        workgroupBarrier();
    }
    return sh_val[0];
}

@compute @workgroup_size(256)
fn sum_first(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let g = group_id(wid, num_wg);
    let base = g * PER_GROUP;
    var acc: f32 = 0.0;
    for (var k: u32 = 0u; k < PER_THREAD; k = k + 1u) {
        let i = base + lid.x + k * WG;
        if (i < params.n) {
            acc = acc + ld_input(i);
        }
    }
    let total = tree_sum(lid.x, acc);
    if (lid.x == 0u && base < params.n) {
        st_out(g, total / params.divisor);
    }
}

@compute @workgroup_size(256)
fn sum_next(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let g = group_id(wid, num_wg);
    let base = g * PER_GROUP;
    var acc: f32 = 0.0;
    for (var k: u32 = 0u; k < PER_THREAD; k = k + 1u) {
        let i = base + lid.x + k * WG;
        if (i < params.n) {
            acc = acc + ld_partials(i);
        }
    }
    let total = tree_sum(lid.x, acc);
    if (lid.x == 0u && base < params.n) {
        st_out(g, total / params.divisor);
    }
}

// The first pass of a sum of squares (the Muon Frobenius norm); `sum_next` finishes it.
@compute @workgroup_size(256)
fn sumsq_first(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let g = group_id(wid, num_wg);
    let base = g * PER_GROUP;
    var acc: f32 = 0.0;
    for (var k: u32 = 0u; k < PER_THREAD; k = k + 1u) {
        let i = base + lid.x + k * WG;
        if (i < params.n) {
            let v = ld_input(i);
            acc = acc + v * v;
        }
    }
    let total = tree_sum(lid.x, acc);
    if (lid.x == 0u && base < params.n) {
        st_out(g, total / params.divisor);
    }
}

// Whether candidate (v, i) replaces the running best (bv, bi). This is the one place the
// tie rule lives: the larger value wins and an exact tie goes to the higher index, which
// is the CPU's `Iterator::max_by` (the last maximum). It is a total order on (value,
// index), so the result does not depend on how the work is grouped. NaN never wins.
fn better(v: f32, i: u32, bv: f32, bi: u32) -> bool {
    return v > bv || (v == bv && i > bi);
}

// Combine the workgroup's (value, index) pairs with `better`, leaving the winner in slot 0.
fn tree_max(tid: u32) {
    workgroupBarrier();
    for (var s: u32 = WG / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            let ov = sh_val[tid + s];
            let oi = sh_idx[tid + s];
            if (better(ov, oi, sh_val[tid], sh_idx[tid])) {
                sh_val[tid] = ov;
                sh_idx[tid] = oi;
            }
        }
        workgroupBarrier();
    }
}

// Max of the input. Every thread starts from the group's first element (never from a
// sentinel such as -1e9), so a thread with no elements of its own repeats an element that
// is in the group anyway, which `better` ignores.
@compute @workgroup_size(256)
fn max_first(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let g = group_id(wid, num_wg);
    let base = g * PER_GROUP;
    var best_v: f32 = 0.0;
    var best_i: u32 = 0u;
    if (base < params.n) {
        best_v = ld_input(base);
        best_i = base;
    }
    for (var k: u32 = 0u; k < PER_THREAD; k = k + 1u) {
        let i = base + lid.x + k * WG;
        if (i < params.n) {
            let v = ld_input(i);
            if (better(v, i, best_v, best_i)) {
                best_v = v;
                best_i = i;
            }
        }
    }
    sh_val[lid.x] = best_v;
    sh_idx[lid.x] = best_i;
    tree_max(lid.x);
    if (lid.x == 0u && base < params.n) {
        st_out(g, sh_val[0]);
        out_index[g] = sh_idx[0];
    }
}

// Max of the partial (value, index) pairs from an earlier pass; `in_index` carries each
// partial's index into the original input.
@compute @workgroup_size(256)
fn max_next(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let g = group_id(wid, num_wg);
    let base = g * PER_GROUP;
    var best_v: f32 = 0.0;
    var best_i: u32 = 0u;
    if (base < params.n) {
        best_v = ld_partials(base);
        best_i = in_index[base];
    }
    for (var k: u32 = 0u; k < PER_THREAD; k = k + 1u) {
        let j = base + lid.x + k * WG;
        if (j < params.n) {
            let v = ld_partials(j);
            let i = in_index[j];
            if (better(v, i, best_v, best_i)) {
                best_v = v;
                best_i = i;
            }
        }
    }
    sh_val[lid.x] = best_v;
    sh_idx[lid.x] = best_i;
    tree_max(lid.x);
    if (lid.x == 0u && base < params.n) {
        st_out(g, sh_val[0]);
        out_index[g] = sh_idx[0];
    }
}

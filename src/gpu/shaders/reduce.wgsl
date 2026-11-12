// Parallel full-tensor reductions (sum, mean, sum of squares, max with its tie count).
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
//   *_first: 0 = input (ELEM),  2 = out (f32),  4 = out_count (u32, max only), 5 = params
//   *_next:  1 = partials (f32), 2 = out (f32), 3 = in_count, 4 = out_count (max), 5 = params
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
@group(0) @binding(3) var<storage, read> in_count: array<u32>;
@group(0) @binding(4) var<storage, read_write> out_count: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;

const WG: u32 = 256u;
const PER_THREAD: u32 = 4u;
const PER_GROUP: u32 = 1024u; // WG * PER_THREAD; the host depends on it

var<workgroup> sh_val: array<f32, 256>;
var<workgroup> sh_cnt: array<u32, 256>;

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

// A running maximum and how many elements equal it: `torch.max()` splits its gradient over
// them (S4). `c == 0` is the empty identity, so a thread with no elements of its own
// contributes nothing (never a sentinel such as -1e9).
struct Best {
    v: f32,
    c: u32,
}

// Merge two (maximum, count) pairs. A NaN beats every number and NaNs count together, equal
// maxima add their counts, and the larger value wins otherwise. It is commutative and
// associative, so the result does not depend on how the work is grouped. NaN is tested
// from its bits (`is_nan`) before any float comparison.
fn combine(a: Best, b: Best) -> Best {
    if (a.c == 0u) { return b; }
    if (b.c == 0u) { return a; }
    let na = is_nan(a.v);
    let nb = is_nan(b.v);
    if (na && nb) { return Best(a.v, a.c + b.c); }
    if (na) { return a; }
    if (nb) { return b; }
    if (a.v > b.v) { return a; }
    if (b.v > a.v) { return b; }
    return Best(a.v, a.c + b.c);
}

// Combine the workgroup's (value, count) pairs, leaving the result in slot 0.
fn tree_max(tid: u32) {
    workgroupBarrier();
    for (var s: u32 = WG / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            let m = combine(Best(sh_val[tid], sh_cnt[tid]), Best(sh_val[tid + s], sh_cnt[tid + s]));
            sh_val[tid] = m.v;
            sh_cnt[tid] = m.c;
        }
        workgroupBarrier();
    }
}

// Max of the input and the number of elements equal to it.
@compute @workgroup_size(256)
fn max_first(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let g = group_id(wid, num_wg);
    let base = g * PER_GROUP;
    var best = Best(0.0, 0u);
    for (var k: u32 = 0u; k < PER_THREAD; k = k + 1u) {
        let i = base + lid.x + k * WG;
        if (i < params.n) {
            best = combine(best, Best(ld_input(i), 1u));
        }
    }
    sh_val[lid.x] = best.v;
    sh_cnt[lid.x] = best.c;
    tree_max(lid.x);
    if (lid.x == 0u && base < params.n) {
        st_out(g, sh_val[0]);
        out_count[g] = sh_cnt[0];
    }
}

// Max of the partial (value, count) pairs from an earlier pass.
@compute @workgroup_size(256)
fn max_next(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let g = group_id(wid, num_wg);
    let base = g * PER_GROUP;
    var best = Best(0.0, 0u);
    for (var k: u32 = 0u; k < PER_THREAD; k = k + 1u) {
        let j = base + lid.x + k * WG;
        if (j < params.n) {
            best = combine(best, Best(ld_partials(j), in_count[j]));
        }
    }
    sh_val[lid.x] = best.v;
    sh_cnt[lid.x] = best.c;
    tree_max(lid.x);
    if (lid.x == 0u && base < params.n) {
        st_out(g, sh_val[0]);
        out_count[g] = sh_cnt[0];
    }
}

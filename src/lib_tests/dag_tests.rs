//! Random-DAG gradient checks (ADDENDUM §10.2, #532): the safety net for the Phase 4 graph
//! rewrites.
//!
//! Each op's backward already has its own finite-difference test, one graph and one hand-picked
//! loss at a time. Nothing there composes ops the way a model does, so a backward that is only
//! wrong under composition (a broadcast meeting a reduction, a permute feeding a matmul, a node
//! used twice) can pass every one of them. This suite generates random DAGs over the core ops,
//! backpropagates a randomly weighted sum, and checks the gradient of **every leaf** against a
//! central difference, in f64 at the tolerance `f64_tests` uses.
//!
//! # How a DAG is generated
//!
//! proptest generates a vector of `u32` choices. [`Dag::build`] reads it like a stream of dice
//! rolls and tracks every value's shape as it goes, so each op it emits is well formed
//! (rank-mismatched broadcasts and unequal-rank matmuls included). A shorter or smaller vector
//! reads as the first option at every roll, so proptest shrinks a failure toward a short
//! program of simple ops. [`Dag`]'s `Display` prints that program as a test body to paste into a
//! regression test.
//!
//! Operands are picked from every earlier value, with a bias toward the latest, so DAGs have
//! diamonds (a value consumed twice) and leaves that feed several ops: the paths through
//! `run_backward`'s `pending`/`visited` bookkeeping that the arena graph replaces.
//!
//! # Inputs where finite differences mislead
//!
//! A kink (`relu` near 0), a pole (`div` by a value near 0) or large values make the central
//! difference disagree with the analytic gradient with no bug in either. The generator keeps
//! domain-restricted operands positive by construction, and [`Dag::reject_reason`] discards a
//! case after a forward pass whenever such an input lands inside a margin.

use super::*;
use crate::gradcheck_f64::{TOL, gradcheck_all, leaf, leaf_from, values, weighted};
use crate::index::{fetch, slot, span};
use proptest::prelude::*;
use std::fmt;

/// The highest rank a generated value has.
const MAX_RANK: usize = 3;
/// The most ops one DAG has.
const MAX_OPS: usize = 12;
/// The most leaf elements one DAG has. A central difference costs two forward passes per leaf
/// element, so this bounds a case's cost.
const MAX_LEAF_ELEMS: usize = 150;
/// The largest entropy vector proptest generates.
const MAX_ENTROPY: usize = 240;
/// Cases per run unless `PROPTEST_CASES` says otherwise.
const DEFAULT_CASES: u32 = 256;
/// A `relu` input closer to 0 than this makes the central difference straddle the kink.
const RELU_MARGIN: f64 = 1e-3;
/// A `div` denominator smaller than this in magnitude makes the quotient's slope blow up.
const DENOM_MIN: f64 = 0.1;
/// Values larger than this in magnitude cost the central difference its precision: its
/// rounding noise is about `1e-16 * |loss| / EPS`, so the error grows with the values. Measured
/// over 1500 DAGs (see the PR): with every value and the loss under 50 the worst leaf error was
/// 7e-9, 14 times under `TOL`; the cases above 50 reached 1.1e-7.
const MAX_ABS: f64 = 50.0;
/// The same bound on the loss itself, which sums many values.
const MAX_LOSS: f64 = 50.0;
/// Seed of the loss weights.
const LOSS_SEED: u64 = 9001;
/// The sizes a generated dimension takes: 1 for broadcasting, the rest not powers of two.
const DIM_MENU: [usize; 5] = [1, 2, 3, 4, 5];

/// A shape of rank `1..=MAX_RANK`, small and `Copy` so the builder can index it with `fetch`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Dims {
    /// How many entries of `d` are real.
    rank: usize,
    /// The dimensions, padded with 1 past `rank`.
    d: [usize; MAX_RANK],
}

impl Dims {
    /// A shape from its dimensions.
    ///
    /// # Panics
    ///
    /// If the rank is 0 or above `MAX_RANK`.
    fn new(dims: &[usize]) -> Self {
        assert!(
            (1..=MAX_RANK).contains(&dims.len()),
            "rank {} is outside 1..={MAX_RANK}",
            dims.len()
        );
        let mut d = [1; MAX_RANK];
        for (i, &x) in dims.iter().enumerate() {
            *slot(&mut d, i) = x;
        }
        Self {
            rank: dims.len(),
            d,
        }
    }

    /// The real dimensions.
    fn dims(&self) -> &[usize] {
        span(&self.d, 0..self.rank)
    }

    /// The element count.
    fn numel(&self) -> usize {
        self.dims().iter().product()
    }

    /// The innermost dimension.
    fn last(&self) -> usize {
        fetch(&self.d, self.rank - 1)
    }
}

/// The broadcast of two dimension lists (right-aligned, each pair equal or one of them 1), or
/// `None` if they do not broadcast. Empty lists broadcast to an empty list.
fn broadcast_slices(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut out = Vec::with_capacity(rank);
    for i in 0..rank {
        let da = if i + a.len() >= rank {
            fetch(a, i + a.len() - rank)
        } else {
            1
        };
        let db = if i + b.len() >= rank {
            fetch(b, i + b.len() - rank)
        } else {
            1
        };
        match (da, db) {
            (x, y) if x == y => out.push(x),
            (1, y) => out.push(y),
            (x, 1) => out.push(x),
            _ => return None,
        }
    }
    Some(out)
}

/// The broadcast of two shapes, or `None` if they do not broadcast.
fn broadcast(a: Dims, b: Dims) -> Option<Dims> {
    broadcast_slices(a.dims(), b.dims()).map(|d| Dims::new(&d))
}

/// The shape of `a.matmul(b)` under Volta's rules (`RawTensor::matmul`, `MatmulSpec`), or
/// `None` if Volta rejects the pair or this suite does not generate it. Supported: matrices of
/// equal rank (batch dimensions broadcast), `[m, k] @ [k]` and `[k] @ [k, n]`. Rejected by
/// design: matrices of unequal rank (that is what makes B4 unreachable). Not generated: the
/// rank-1 dot product, whose result is a scalar.
fn matmul_dims(a: Dims, b: Dims) -> Option<Dims> {
    let out = match (a.rank, b.rank) {
        (2, 1) => {
            if a.last() != b.last() {
                return None;
            }
            vec![fetch(&a.d, 0)]
        }
        (1, 2) => {
            if a.last() != fetch(&b.d, 0) {
                return None;
            }
            vec![b.last()]
        }
        (ra, rb) if ra == rb && ra >= 2 => {
            if a.last() != fetch(&b.d, rb - 2) {
                return None;
            }
            let mut out = broadcast_slices(span(a.dims(), 0..ra - 2), span(b.dims(), 0..rb - 2))?;
            out.push(fetch(&a.d, ra - 2));
            out.push(b.last());
            out
        }
        _ => return None,
    };
    Some(Dims::new(&out))
}

/// One op of a generated DAG. Operands are indices of earlier steps.
#[derive(Clone, Copy, Debug)]
enum Op {
    /// `a + b`, broadcasting.
    Add(usize, usize),
    /// `a - b`, broadcasting.
    Sub(usize, usize),
    /// `a * b`, broadcasting.
    Mul(usize, usize),
    /// `a / b`, broadcasting; `b` is positive by construction.
    Div(usize, usize),
    /// `a @ b`, with batch broadcasting and rank-1 sides.
    Matmul(usize, usize),
    /// `relu(x)`.
    Relu(usize),
    /// `exp(x)`.
    Exp(usize),
    /// `tanh(x)`.
    Tanh(usize),
    /// `sum_dim(x, axis, keepdim)`.
    SumDim {
        /// The reduced value.
        x: usize,
        /// The reduced axis.
        axis: usize,
        /// Whether the axis stays as a size-1 dimension.
        keepdim: bool,
    },
    /// `permute(x, axes)`.
    Permute {
        /// The permuted value.
        x: usize,
        /// The first `rank` entries are the permutation.
        axes: [usize; MAX_RANK],
        /// The rank of `x`.
        rank: usize,
    },
    /// `reshape(x, to)`.
    Reshape {
        /// The reshaped value.
        x: usize,
        /// The new shape, with `x`'s element count.
        to: Dims,
    },
}

impl Op {
    /// The steps this op reads.
    fn operands(self) -> Vec<usize> {
        match self {
            Self::Add(a, b)
            | Self::Sub(a, b)
            | Self::Mul(a, b)
            | Self::Div(a, b)
            | Self::Matmul(a, b) => vec![a, b],
            Self::Relu(x)
            | Self::Exp(x)
            | Self::Tanh(x)
            | Self::SumDim { x, .. }
            | Self::Permute { x, .. }
            | Self::Reshape { x, .. } => vec![x],
        }
    }

    /// The op applied to earlier values.
    fn apply(self, vals: &[Tensor]) -> Tensor {
        let t = |id: usize| at(vals, id);
        match self {
            Self::Add(a, b) => t(a).add(&t(b)),
            Self::Sub(a, b) => t(a).sub(&t(b)),
            Self::Mul(a, b) => t(a).elem_mul(&t(b)),
            Self::Div(a, b) => t(a).div(&t(b)),
            Self::Matmul(a, b) => t(a).matmul(&t(b)),
            Self::Relu(x) => t(x).relu(),
            Self::Exp(x) => t(x).exp(),
            Self::Tanh(x) => t(x).tanh(),
            Self::SumDim { x, axis, keepdim } => t(x).sum_dim(Axis(axis), keepdim),
            Self::Permute { x, axes, rank } => {
                let axes: Vec<Axis> = span(&axes, 0..rank).iter().map(|&a| Axis(a)).collect();
                t(x).permute(&axes)
            }
            Self::Reshape { x, to } => t(x).reshape(to.dims()),
        }
    }
}

/// One step of a generated DAG: a leaf, or an op over earlier steps.
#[derive(Clone, Copy, Debug)]
enum Step {
    /// A leaf of normal values, or of `0.5 + |v|` when `positive`.
    Leaf {
        /// The leaf's shape.
        dims: Dims,
        /// Seed of the leaf's values.
        seed: u64,
        /// Whether the values are bounded away from 0 and positive.
        positive: bool,
    },
    /// An op over earlier steps.
    Op(Op),
}

/// What the builder knows about a value: its shape, and whether it is positive by construction.
#[derive(Clone, Copy, Debug)]
struct Val {
    /// The value's shape.
    dims: Dims,
    /// Whether every element is positive, by construction (a positive leaf, `exp`, and sums or
    /// products of positive values).
    positive: bool,
}

/// A generated DAG: steps in evaluation order, each with its known shape. The output is the
/// last step.
#[derive(Clone, Debug)]
struct Dag {
    /// The steps, leaves and ops interleaved.
    steps: Vec<Step>,
    /// One entry per step.
    vals: Vec<Val>,
}

/// A clone of element `i`, panicking if it is out of bounds (`fetch` for values that are not
/// `Copy`).
fn at<T: Clone>(s: &[T], i: usize) -> T {
    span(s, i..i + 1).first().cloned().expect("a span of one")
}

/// A stream of dice rolls read from proptest's entropy vector.
struct Choices<'a> {
    /// The entropy.
    v: &'a [u32],
    /// How many rolls have been read.
    pos: usize,
}

impl Choices<'_> {
    /// The next raw value; an exhausted stream reads as 0, so a shrunk (shorter) vector takes
    /// the first option at every roll it no longer covers.
    fn raw(&mut self) -> u32 {
        let raw = if self.pos < self.v.len() {
            fetch(self.v, self.pos)
        } else {
            0 // index-ok: exhausted entropy reads as 0 by design (shrinking)
        };
        self.pos += 1;
        raw
    }

    /// The next roll, in `0..n`.
    fn next(&mut self, n: usize) -> usize {
        usize::try_from(self.raw()).expect("u32 fits usize") % n
    }

    /// The next roll as a seed.
    fn seed(&mut self) -> u64 {
        u64::from(self.raw())
    }
}

/// The op kinds the builder rolls between.
const KINDS: usize = 11;

/// Builds a [`Dag`] from the entropy stream.
struct Builder<'a> {
    /// The dice rolls.
    ch: Choices<'a>,
    /// The steps so far.
    steps: Vec<Step>,
    /// The known value of each step.
    vals: Vec<Val>,
    /// The leaf elements spent so far.
    leaf_elems: usize,
}

impl Builder<'_> {
    /// A random shape.
    fn random_dims(&mut self) -> Dims {
        let rank = 1 + self.ch.next(MAX_RANK);
        let dims: Vec<usize> = (0..rank)
            .map(|_| fetch(&DIM_MENU, self.ch.next(DIM_MENU.len())))
            .collect();
        Dims::new(&dims)
    }

    /// Adds a leaf, or `None` if the leaf-element budget is spent.
    fn push_leaf(&mut self, dims: Dims, positive: bool) -> Option<usize> {
        if self.leaf_elems + dims.numel() > MAX_LEAF_ELEMS {
            return None;
        }
        self.leaf_elems += dims.numel();
        let seed = self.ch.seed();
        self.steps.push(Step::Leaf {
            dims,
            seed,
            positive,
        });
        self.vals.push(Val { dims, positive });
        Some(self.vals.len() - 1)
    }

    /// Adds an op with the shape and positivity the builder worked out for it.
    fn push_op(&mut self, op: Op, dims: Dims, positive: bool) -> usize {
        self.steps.push(Step::Op(op));
        self.vals.push(Val { dims, positive });
        self.vals.len() - 1
    }

    /// An operand: the latest value a third of the time, otherwise any value. Picking from
    /// all earlier values is what makes diamonds and shared leaves.
    fn pick(&mut self) -> usize {
        let n = self.vals.len();
        if self.ch.next(3) == 0 {
            n - 1
        } else {
            self.ch.next(n)
        }
    }

    /// An existing value that broadcasts with `dims` (and is positive, if asked), chosen by a
    /// roll.
    fn compatible(&mut self, dims: Dims, positive_only: bool) -> Option<usize> {
        let ok: Vec<usize> = (0..self.vals.len())
            .filter(|&i| {
                let v = fetch(&self.vals, i);
                (!positive_only || v.positive) && broadcast(dims, v.dims).is_some()
            })
            .collect();
        (!ok.is_empty()).then(|| fetch(&ok, self.ch.next(ok.len())))
    }

    /// The shape of a fresh operand that broadcasts with `a`: some leading dimensions dropped
    /// (a rank mismatch), some of the rest set to 1 (a stretched dimension).
    fn partner_dims(&mut self, a: Dims) -> Dims {
        let drop = self.ch.next(a.rank);
        let dims: Vec<usize> = (drop..a.rank)
            .map(|i| {
                if self.ch.next(3) == 0 {
                    1
                } else {
                    fetch(&a.d, i)
                }
            })
            .collect();
        Dims::new(&dims)
    }

    /// `add`, `sub`, `mul` or `div` of a picked value with a partner: an existing value
    /// (a diamond) or a fresh leaf, possibly broadcasting.
    fn binary(&mut self, kind: usize) -> Option<()> {
        let div = kind == 3;
        let a = self.pick();
        let va = fetch(&self.vals, a);
        let reuse = self.ch.next(3) == 0;
        let existing = if reuse {
            self.compatible(va.dims, div)
        } else {
            None
        };
        let b = if let Some(b) = existing {
            b
        } else {
            let dims = self.partner_dims(va.dims);
            self.push_leaf(dims, div)?
        };
        let vb = fetch(&self.vals, b);
        // A denominator stays on the right; the other ops put the partner on either side, so
        // the lower-rank operand is sometimes the left one.
        let (l, r) = if !div && self.ch.next(2) == 0 {
            (b, a)
        } else {
            (a, b)
        };
        let (vl, vr) = (fetch(&self.vals, l), fetch(&self.vals, r));
        let dims = broadcast(vl.dims, vr.dims)?;
        let both = va.positive && vb.positive;
        let (op, positive) = match kind {
            0 => (Op::Add(l, r), both),
            1 => (Op::Sub(l, r), false),
            2 => (Op::Mul(l, r), both),
            _ => (Op::Div(l, r), both),
        };
        self.push_op(op, dims, positive);
        Some(())
    }

    /// A transposed fresh leaf: a leaf of shape `dims` reversed in its last two axes, then
    /// permuted back, so the matmul consumes a permute output (a pair of transpose bugs that
    /// cancel on square inputs shows up on the non-square ones).
    fn permuted_leaf(&mut self, dims: Dims) -> Option<usize> {
        let mut raw = dims.dims().to_vec();
        let rank = raw.len();
        raw.swap(rank - 2, rank - 1);
        let leaf = self.push_leaf(Dims::new(&raw), false)?;
        let mut axes = [0, 1, 2];
        axes.swap(rank - 2, rank - 1);
        self.push_op(
            Op::Permute {
                x: leaf,
                axes,
                rank,
            },
            dims,
            false,
        );
        Some(self.vals.len() - 1)
    }

    /// `matmul` of a picked value with an existing value or a fresh operand.
    fn matmul(&mut self) -> Option<()> {
        let a = self.pick();
        let va = fetch(&self.vals, a);
        let reuse = self.ch.next(3) == 0;
        let existing = if reuse {
            let ok: Vec<usize> = (0..self.vals.len())
                .filter(|&i| matmul_dims(va.dims, fetch(&self.vals, i).dims).is_some())
                .collect();
            (!ok.is_empty()).then(|| fetch(&ok, self.ch.next(ok.len())))
        } else {
            None
        };
        let (l, r) = match existing {
            Some(b) => (a, b),
            None => self.fresh_matmul_operand(a, va)?,
        };
        let dims = matmul_dims(fetch(&self.vals, l).dims, fetch(&self.vals, r).dims)?;
        let both = fetch(&self.vals, l).positive && fetch(&self.vals, r).positive;
        self.push_op(Op::Matmul(l, r), dims, both);
        Some(())
    }

    /// The `(lhs, rhs)` of a matmul that has `a` on one side and a fresh operand on the other.
    fn fresh_matmul_operand(&mut self, a: usize, va: Val) -> Option<(usize, usize)> {
        let n = fetch(&DIM_MENU, self.ch.next(DIM_MENU.len()));
        // The batch dimensions `a` brings, which the fresh matrix must broadcast with; none
        // for a matrix or a vector.
        let a_batch = if va.dims.rank > 2 {
            span(va.dims.dims(), 0..va.dims.rank - 2).to_vec()
        } else {
            Vec::new()
        };
        match self.ch.next(4) {
            // `a @ fresh [.., k, n]`, or `a @ permute(fresh [.., n, k])`
            variant @ (0 | 1) => {
                let mut dims = self.broadcastable_batch(&a_batch);
                dims.extend([va.dims.last(), n]);
                let dims = Dims::new(&dims);
                let b = if variant == 0 {
                    self.push_leaf(dims, false)?
                } else {
                    self.permuted_leaf(dims)?
                };
                Some((a, b))
            }
            // `[m, k] @ fresh [k]`
            2 => {
                if va.dims.rank != 2 {
                    return None;
                }
                Some((a, self.push_leaf(Dims::new(&[va.dims.last()]), false)?))
            }
            // `fresh [.., n, rows] @ a`, plain or permuted: `a` is the rhs, and its second
            // to last dimension (its only one, for a vector) is the contraction.
            _ => {
                let rows = if va.dims.rank == 1 {
                    va.dims.last()
                } else {
                    fetch(&va.dims.d, va.dims.rank - 2)
                };
                let mut dims = self.broadcastable_batch(&a_batch);
                dims.extend([n, rows]);
                let dims = Dims::new(&dims);
                let lhs = if self.ch.next(2) == 0 {
                    self.push_leaf(dims, false)?
                } else {
                    self.permuted_leaf(dims)?
                };
                Some((lhs, a))
            }
        }
    }

    /// Batch dimensions for a fresh matrix that must broadcast with `batch`: each dimension is
    /// 1, or `batch`'s own (any size where that is 1).
    fn broadcastable_batch(&mut self, batch: &[usize]) -> Vec<usize> {
        batch
            .iter()
            .map(|&b| {
                if b == 1 {
                    fetch(&DIM_MENU, self.ch.next(DIM_MENU.len()))
                } else if self.ch.next(2) == 0 {
                    1
                } else {
                    b
                }
            })
            .collect()
    }

    /// `relu`, `exp` or `tanh` of a picked value.
    fn unary(&mut self, kind: usize) {
        let x = self.pick();
        let vx = fetch(&self.vals, x);
        let (op, positive) = match kind % 3 {
            0 => (Op::Relu(x), vx.positive),
            1 => (Op::Exp(x), true),
            _ => (Op::Tanh(x), false),
        };
        self.push_op(op, vx.dims, positive);
    }

    /// `sum_dim` of a picked value over a rolled axis.
    fn sum_dim(&mut self) {
        let x = self.pick();
        let vx = fetch(&self.vals, x);
        let axis = self.ch.next(vx.dims.rank);
        // A rank-1 value keeps its axis: a scalar result is not a case this suite generates.
        let keepdim = vx.dims.rank == 1 || self.ch.next(2) == 0;
        let mut dims = vx.dims.dims().to_vec();
        if keepdim {
            *slot(&mut dims, axis) = 1;
        } else {
            dims.remove(axis);
        }
        self.push_op(
            Op::SumDim { x, axis, keepdim },
            Dims::new(&dims),
            vx.positive,
        );
    }

    /// `permute` of a picked value of rank 2 or more by a rolled permutation.
    fn permute(&mut self) -> Option<()> {
        let x = self.pick();
        let vx = fetch(&self.vals, x);
        let rank = vx.dims.rank;
        if rank < 2 {
            return None;
        }
        let mut axes = [0, 1, 2];
        for i in (1..rank).rev() {
            let j = self.ch.next(i + 1);
            axes.swap(i, j);
        }
        let dims: Vec<usize> = span(&axes, 0..rank)
            .iter()
            .map(|&a| fetch(&vx.dims.d, a))
            .collect();
        self.push_op(Op::Permute { x, axes, rank }, Dims::new(&dims), vx.positive);
        Some(())
    }

    /// `reshape` of a picked value to a rolled factorization of its element count.
    fn reshape(&mut self) {
        let x = self.pick();
        let vx = fetch(&self.vals, x);
        let parts = 1 + self.ch.next(MAX_RANK);
        let mut remaining = vx.dims.numel();
        let mut dims = Vec::with_capacity(parts);
        for _ in 1..parts {
            let divisors: Vec<usize> = (1..=remaining)
                .filter(|&d| remaining.is_multiple_of(d))
                .collect();
            let d = fetch(&divisors, self.ch.next(divisors.len()));
            dims.push(d);
            remaining /= d;
        }
        dims.push(remaining);
        let to = Dims::new(&dims);
        self.push_op(Op::Reshape { x, to }, to, vx.positive);
    }

    /// One rolled op; one that cannot apply (no compatible operand, leaf budget spent) falls
    /// back to a unary op, which always can.
    fn step(&mut self) {
        let kind = self.ch.next(KINDS);
        let done = match kind {
            0..=3 => self.binary(kind),
            4 => self.matmul(),
            5..=7 => {
                self.unary(kind);
                Some(())
            }
            8 => {
                self.sum_dim();
                Some(())
            }
            9 => self.permute(),
            _ => {
                self.reshape();
                Some(())
            }
        };
        if done.is_none() {
            let kind = self.ch.next(3);
            self.unary(kind);
        }
    }
}

impl Dag {
    /// The DAG the entropy stream describes.
    fn build(entropy: &[u32]) -> Self {
        let mut b = Builder {
            ch: Choices { v: entropy, pos: 0 },
            steps: Vec::new(),
            vals: Vec::new(),
            leaf_elems: 0,
        };
        let dims = b.random_dims();
        b.push_leaf(dims, false)
            .expect("one leaf fits the element budget");
        for _ in 0..=b.ch.next(MAX_OPS) {
            b.step();
        }
        Self {
            steps: b.steps,
            vals: b.vals,
        }
    }

    /// The leaves, in step order, as f64 tensors that require grad.
    fn make_leaves(&self) -> Vec<Tensor> {
        self.steps
            .iter()
            .filter_map(|s| match *s {
                Step::Leaf {
                    dims,
                    seed,
                    positive,
                } => {
                    let shape: fn(f64) -> f64 = if positive { |v| 0.5 + v.abs() } else { |v| v };
                    Some(leaf_from(dims.dims(), seed, shape))
                }
                Step::Op(_) => None,
            })
            .collect()
    }

    /// Every step's value, computed from `leaves` (in step order), checking each op's output
    /// shape against the one the builder predicted.
    fn run(&self, leaves: &[Tensor]) -> Vec<Tensor> {
        let mut out: Vec<Tensor> = Vec::with_capacity(self.steps.len());
        let mut next_leaf = 0;
        for (i, step) in self.steps.iter().enumerate() {
            let t = match *step {
                Step::Leaf { .. } => {
                    next_leaf += 1;
                    at(leaves, next_leaf - 1)
                }
                Step::Op(op) => op.apply(&out),
            };
            assert_eq!(
                t.shape().truncate_rank().to_vec(),
                fetch(&self.vals, i).dims.dims().to_vec(),
                "step {i} ({step:?}) has a different shape from the one the builder predicted"
            );
            out.push(t);
        }
        out
    }

    /// The loss: a randomly weighted sum of the last step, so a gradient wrong by a
    /// permutation or an offset does not cancel.
    fn loss(&self, leaves: &[Tensor]) -> Tensor {
        let out = self.run(leaves);
        weighted(&at(&out, out.len() - 1), LOSS_SEED)
    }

    /// Why this DAG's inputs would mislead a finite difference, if they would: a `relu` input
    /// near its kink, a small denominator, or a value that is not finite or too large.
    fn reject_reason(&self, leaves: &[Tensor]) -> Option<String> {
        let out = self.run(leaves);
        let min_abs = |id: usize| {
            values(&at(&out, id))
                .iter()
                .fold(f64::INFINITY, |m, v| m.min(v.abs()))
        };
        let loss = values(&self.loss(leaves));
        if loss.iter().any(|v| !v.is_finite() || v.abs() > MAX_LOSS) {
            return Some(format!("the loss is not finite or exceeds {MAX_LOSS}"));
        }
        for (i, step) in self.steps.iter().enumerate() {
            if values(&at(&out, i))
                .iter()
                .any(|v| !v.is_finite() || v.abs() > MAX_ABS)
            {
                return Some(format!("step {i} is not finite or exceeds {MAX_ABS}"));
            }
            match *step {
                Step::Op(Op::Relu(x)) if min_abs(x) < RELU_MARGIN => {
                    return Some(format!(
                        "step {i}: a relu input is within {RELU_MARGIN} of 0"
                    ));
                }
                Step::Op(Op::Div(_, d)) if min_abs(d) < DENOM_MIN => {
                    return Some(format!(
                        "step {i}: a denominator is within {DENOM_MIN} of 0"
                    ));
                }
                Step::Leaf { .. } | Step::Op(_) => {}
            }
        }
        None
    }
}

/// Prints the DAG as a test body: build the leaves, define the loss over `l`, and check every
/// leaf. Paste it into a `#[test]` to pin a shrunk case.
impl fmt::Display for Dag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "let leaves = vec![")?;
        for step in &self.steps {
            match *step {
                Step::Leaf {
                    dims,
                    seed,
                    positive,
                } => {
                    let map = if positive {
                        "|v| 0.5 + v.abs()"
                    } else {
                        "|v| v"
                    };
                    writeln!(f, "    leaf_from(&{:?}, {seed}, {map}),", dims.dims())?;
                }
                Step::Op(_) => {}
            }
        }
        writeln!(f, "];")?;
        writeln!(f, "let errs = gradcheck_all(&leaves, |l| {{")?;
        let mut next_leaf = 0;
        for (i, step) in self.steps.iter().enumerate() {
            let rhs = match *step {
                Step::Leaf { .. } => {
                    next_leaf += 1;
                    format!("l[{}].clone()", next_leaf - 1)
                }
                Step::Op(op) => match op {
                    Op::Add(a, b) => format!("v{a}.add(&v{b})"),
                    Op::Sub(a, b) => format!("v{a}.sub(&v{b})"),
                    Op::Mul(a, b) => format!("v{a}.elem_mul(&v{b})"),
                    Op::Div(a, b) => format!("v{a}.div(&v{b})"),
                    Op::Matmul(a, b) => format!("v{a}.matmul(&v{b})"),
                    Op::Relu(x) => format!("v{x}.relu()"),
                    Op::Exp(x) => format!("v{x}.exp()"),
                    Op::Tanh(x) => format!("v{x}.tanh()"),
                    Op::SumDim { x, axis, keepdim } => {
                        format!("v{x}.sum_dim(Axis({axis}), {keepdim})")
                    }
                    Op::Permute { x, axes, rank } => {
                        let axes: Vec<String> = span(&axes, 0..rank)
                            .iter()
                            .map(|a| format!("Axis({a})"))
                            .collect();
                        format!("v{x}.permute(&[{}])", axes.join(", "))
                    }
                    Op::Reshape { x, to } => format!("v{x}.reshape(&{:?})", to.dims()),
                },
            };
            writeln!(f, "    let v{i} = {rhs};")?;
        }
        writeln!(f, "    weighted(&v{}, {LOSS_SEED})", self.steps.len() - 1)?;
        writeln!(f, "}});")
    }
}

/// The proptest configuration: `DEFAULT_CASES` cases unless `PROPTEST_CASES` says otherwise
/// (the scheduled run sets it high). No failure-persistence file: a shrunk failure is pinned
/// as a plain `#[test]` from the `Display` output instead.
///
/// proptest aborts after a fixed number of discarded cases (1024 by default), however many it
/// ran. About one DAG in nine is discarded, so a 20000-case run would hit that limit with no
/// failure. The limit scales with the case count instead, to half of it:
/// `guards_reject_a_minority_of_generated_dags` is what keeps the real rate far below that.
fn config() -> ProptestConfig {
    let cases: u32 = std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_CASES);
    ProptestConfig {
        cases,
        max_global_rejects: (cases / 2).max(1024),
        failure_persistence: None,
        ..ProptestConfig::default()
    }
}

proptest! {
    #![proptest_config(config())]

    /// The gradient of every leaf of a random DAG matches a central difference, in f64.
    #[test]
    fn random_dags_gradcheck_in_f64(
        entropy in prop::collection::vec(any::<u32>(), 1..=MAX_ENTROPY)
    ) {
        let dag = Dag::build(&entropy);
        let leaves = dag.make_leaves();
        prop_assume!(dag.reject_reason(&leaves).is_none());
        // A panic (a gradient of the wrong length, a shape check) is a failure too, and it
        // reports the DAG like a wrong gradient does: a bare panic message would not.
        let checked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            gradcheck_all(&leaves, |l| dag.loss(l))
        }));
        let Ok(errs) = checked else {
            return Err(TestCaseError::fail(format!("panicked on\n{dag}")));
        };
        prop_assert!(
            errs.iter().all(|&e| e < TOL),
            "worst relative error per leaf {errs:?} (tolerance {TOL:e}) for\n{dag}"
        );
    }
}

/// A matmul bigger than one 16x16 GPU tile, non-square and not a multiple of 16, with both
/// operands checked: plain, and with both arriving through `permute` (a transpose bug on
/// either side that cancels on square inputs does not cancel here).
#[test]
fn matmul_larger_than_a_gpu_tile_gradchecks_in_f64() {
    let (m, k, n) = (17, 23, 19);
    let plain = gradcheck_all(&[leaf(&[m, k], 201), leaf(&[k, n], 202)], |l| {
        weighted(&l[0].matmul(&l[1]), 203)
    });
    let permuted = gradcheck_all(&[leaf(&[k, m], 204), leaf(&[n, k], 205)], |l| {
        let a = l[0].permute(&[Axis(1), Axis(0)]);
        let b = l[1].permute(&[Axis(1), Axis(0)]);
        weighted(&a.matmul(&b), 206)
    });
    for (name, errs) in [("plain", plain), ("permuted", permuted)] {
        assert_eq!(errs.len(), 2, "{name}: one error per operand");
        for (side, e) in ["lhs", "rhs"].into_iter().zip(errs) {
            assert!(e < TOL, "{name} {side}: relative error {e:.3e}");
        }
    }
}

/// How many pseudo-random entropy vectors the generator self-tests draw.
const SAMPLES: usize = 1500;

/// `SAMPLES` DAGs built from a fixed xorshift stream, so the self-tests are deterministic.
fn sample_dags() -> Vec<Dag> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    (0..SAMPLES)
        .map(|_| {
            let entropy: Vec<u32> = (0..MAX_ENTROPY)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    u32::try_from(state >> 32).expect("the top 32 bits of a u64")
                })
                .collect();
            Dag::build(&entropy)
        })
        .collect()
}

/// A generated DAG can contain every op and structural feature #532 asks for: each op, a
/// diamond (an op consumed twice), a leaf feeding several ops, rank-mismatched and stretched
/// broadcasts, a permute feeding a matmul (square and non-square), batched and vector
/// matmuls, and a reduction that drops its axis. Without this a generator change could
/// quietly stop producing one of them and the suite would keep passing.
#[test]
fn generator_reaches_every_op_and_structure() {
    use std::collections::BTreeSet;
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for dag in sample_dags() {
        let dims = |id: usize| fetch(&dag.vals, id).dims;
        let is_permute = |id: usize| matches!(fetch(&dag.steps, id), Step::Op(Op::Permute { .. }));
        let mut uses = vec![0usize; dag.steps.len()];
        for step in &dag.steps {
            let Step::Op(op) = *step else { continue };
            for x in op.operands() {
                *slot(&mut uses, x) += 1;
            }
            seen.insert(match op {
                Op::Add(..) => "add",
                Op::Sub(..) => "sub",
                Op::Mul(..) => "mul",
                Op::Div(..) => "div",
                Op::Matmul(..) => "matmul",
                Op::Relu(_) => "relu",
                Op::Exp(_) => "exp",
                Op::Tanh(_) => "tanh",
                Op::SumDim { .. } => "sum_dim",
                Op::Permute { .. } => "permute",
                Op::Reshape { .. } => "reshape",
            });
            match op {
                Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) | Op::Div(a, b) => {
                    if dims(a).rank != dims(b).rank {
                        seen.insert("rank-mismatched broadcast");
                    } else if dims(a) != dims(b) {
                        seen.insert("stretched-dimension broadcast");
                    }
                }
                Op::Matmul(a, b) => {
                    if dims(a).rank == 1 || dims(b).rank == 1 {
                        seen.insert("vector matmul");
                    }
                    if dims(a).rank == 3 {
                        seen.insert("batched matmul");
                    }
                    for x in [a, b].into_iter().filter(|&x| is_permute(x)) {
                        let d = dims(x);
                        seen.insert(if fetch(&d.d, d.rank - 1) == fetch(&d.d, d.rank - 2) {
                            "permute feeds matmul, square"
                        } else {
                            "permute feeds matmul, non-square"
                        });
                    }
                }
                Op::SumDim { keepdim: false, .. } => {
                    seen.insert("sum_dim drops its axis");
                }
                Op::Relu(_)
                | Op::Exp(_)
                | Op::Tanh(_)
                | Op::SumDim { .. }
                | Op::Permute { .. }
                | Op::Reshape { .. } => {}
            }
        }
        for (i, &n) in uses.iter().enumerate() {
            if n >= 2 {
                seen.insert(match fetch(&dag.steps, i) {
                    Step::Op(_) => "diamond: an op consumed twice",
                    Step::Leaf { .. } => "a leaf feeding several ops",
                });
            }
        }
    }
    let wanted = [
        "add",
        "sub",
        "mul",
        "div",
        "matmul",
        "relu",
        "exp",
        "tanh",
        "sum_dim",
        "permute",
        "reshape",
        "rank-mismatched broadcast",
        "stretched-dimension broadcast",
        "vector matmul",
        "batched matmul",
        "permute feeds matmul, square",
        "permute feeds matmul, non-square",
        "sum_dim drops its axis",
        "diamond: an op consumed twice",
        "a leaf feeding several ops",
    ];
    let missing: Vec<_> = wanted.iter().filter(|w| !seen.contains(*w)).collect();
    assert!(
        missing.is_empty(),
        "{SAMPLES} generated DAGs never contained: {missing:?}"
    );
}

/// The kink, denominator and magnitude guards discard a minority of generated DAGs. A guard
/// that rejected most of them would leave the suite checking little while still passing
/// (proptest only aborts at its own, much higher, reject limit).
#[test]
fn guards_reject_a_minority_of_generated_dags() {
    let rejected = sample_dags()
        .iter()
        .filter(|dag| dag.reject_reason(&dag.make_leaves()).is_some())
        .count();
    assert!(
        rejected * 4 < SAMPLES,
        "the guards rejected {rejected} of {SAMPLES} generated DAGs"
    );
}

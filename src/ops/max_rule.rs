//! Tie and NaN rules of the max family (S2, S3, S4; DECISIONS.md, "Max ties and NaN follow
//! torch, op by op").
//!
//! PyTorch has no single rule: `maximum` and `max()` split a tied gradient evenly and
//! propagate NaN, while `max(dim)` and `max_pool2d` send the whole gradient to one index.
//! Each function names the torch op it mirrors, and `tests/torch_ref_ops.rs` and
//! `tests/torch_ref_layers.rs` check it against fixtures.

use crate::index::fetch;
use crate::real::Real;

/// `torch.maximum`: NaN if either operand is NaN.
pub(crate) fn maximum<T: Real>(a: T, b: T) -> T {
    if a.is_nan() {
        a
    } else if b.is_nan() || b > a {
        b
    } else {
        a
    }
}

/// `torch.maximum`'s gradient `(∂a, ∂b)`: the larger operand gets all of `g`, a tie splits
/// it in half, and a NaN operand (no comparison holds) gives both sides all of `g`.
pub(crate) fn maximum_backward<T: Real>(g: T, a: T, b: T) -> (T, T) {
    let half = g / T::from_f64(2.0);
    let ga = if a < b {
        T::ZERO
    } else if a == b {
        half
    } else {
        g
    };
    let gb = if a > b {
        T::ZERO
    } else if a == b {
        half
    } else {
        g
    };
    (ga, gb)
}

/// Whether `x` is a maximum of a tensor whose maximum is `max`: equal to it, or NaN when
/// the maximum is NaN. These are the elements `max()` splits its gradient over.
pub(crate) fn is_max<T: Real>(x: T, max: T) -> bool {
    x == max || (x.is_nan() && max.is_nan())
}

/// `torch.max()` over a whole slice: the maximum (NaN if any element is NaN) and how many
/// elements are a maximum ([`is_max`]).
///
/// # Panics
/// If `xs` is empty (tensors never are).
pub(crate) fn max_and_count<T: Real>(xs: &[T]) -> (T, usize) {
    let mut best = fetch(xs, 0);
    let mut count = 1;
    for &x in xs.iter().skip(1) {
        if best.is_nan() {
            count += usize::from(x.is_nan());
        } else if x.is_nan() || x > best {
            best = x;
            count = 1;
        } else if x == best {
            count += 1;
        }
    }
    (best, count)
}

/// `torch.max(x, dim)`'s scan: whether `cand` replaces the running best. A larger value
/// does, and so does the first NaN; ties and later NaNs keep the earlier index.
pub(crate) fn dim_scan_replaces<T: Real>(cand: T, best: T) -> bool {
    cand > best || (cand.is_nan() && !best.is_nan())
}

/// `max_pool2d`'s scan: whether `cand` replaces the running best. A larger value does, and
/// so does every NaN, so the last NaN of a window wins; ties keep the earlier element.
pub(crate) fn pool_scan_replaces<T: Real>(cand: T, best: T) -> bool {
    cand > best || cand.is_nan()
}

#[cfg(test)]
mod tests {
    use super::{
        dim_scan_replaces, is_max, max_and_count, maximum, maximum_backward, pool_scan_replaces,
    };

    #[test]
    fn maximum_propagates_nan_from_either_side() {
        assert!(maximum(f32::NAN, 1.0).is_nan());
        assert!(maximum(1.0, f32::NAN).is_nan());
        assert_eq!(maximum(1.0, 2.0), 2.0);
        assert_eq!(maximum(2.0_f64, 1.0), 2.0);
    }

    #[test]
    fn maximum_backward_splits_a_tie_and_gives_nan_both_sides() {
        assert_eq!(maximum_backward(1.0, 3.0, 3.0), (0.5, 0.5));
        assert_eq!(maximum_backward(1.0, 4.0, 3.0), (1.0, 0.0));
        assert_eq!(maximum_backward(1.0, 3.0, 4.0), (0.0, 1.0));
        assert_eq!(maximum_backward(1.0, f32::NAN, 1.0), (1.0, 1.0));
    }

    #[test]
    fn max_and_count_counts_ties_and_every_nan() {
        assert_eq!(max_and_count(&[2.0, 5.0, 5.0, 1.0]), (5.0, 2));
        assert_eq!(
            max_and_count(&[-f32::INFINITY, -f32::INFINITY]),
            (f32::NEG_INFINITY, 2)
        );
        let (m, n) = max_and_count(&[f32::NAN, 2.0, f32::NAN, 1.0]);
        assert!(m.is_nan());
        assert_eq!(n, 2);
        let (m, n) = max_and_count(&[1.0, f32::NAN]);
        assert!(m.is_nan());
        assert_eq!(n, 1);
        assert!(is_max(f32::NAN, f32::NAN));
        assert!(!is_max(1.0, f32::NAN));
    }

    #[test]
    fn scan_rules_differ_in_which_nan_wins() {
        // max(dim): the first NaN stays. max_pool2d: a later NaN replaces it.
        assert!(!dim_scan_replaces(f32::NAN, f32::NAN));
        assert!(pool_scan_replaces(f32::NAN, f32::NAN));
        assert!(dim_scan_replaces(f32::NAN, 1.0));
        assert!(!dim_scan_replaces(1.0, f32::NAN));
        assert!(!dim_scan_replaces(2.0, 2.0));
        assert!(!pool_scan_replaces(2.0, 2.0));
    }
}

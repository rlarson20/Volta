//! Tie and NaN rules of the max family (S2, S3, S4; DECISIONS.md, "Max ties and NaN follow
//! torch, op by op").
//!
//! PyTorch has no single rule: `maximum` and `max()` split a tied gradient evenly and
//! propagate NaN, while `max(dim)` and `max_pool2d` send the whole gradient to one index.
//! Each function names the torch op it mirrors, and `tests/torch_ref_ops.rs` and
//! `tests/torch_ref_layers.rs` check it against fixtures.

use crate::real::Real;

/// `torch.max(x, dim)`'s scan: whether `cand` replaces the running best. A larger value
/// does, and so does the first NaN; ties and later NaNs keep the earlier index.
pub(crate) fn dim_scan_replaces<T: Real>(cand: T, best: T) -> bool {
    cand > best || (cand.is_nan() && !best.is_nan())
}

#[cfg(test)]
mod tests {
    use super::dim_scan_replaces;

    #[test]
    fn scan_rules_differ_in_which_nan_wins() {
        // max(dim): ties and the first NaN stay.
        assert!(!dim_scan_replaces(f32::NAN, f32::NAN));
        assert!(dim_scan_replaces(f32::NAN, 1.0));
        assert!(!dim_scan_replaces(1.0, f32::NAN));
        assert!(!dim_scan_replaces(2.0, 2.0));
    }
}

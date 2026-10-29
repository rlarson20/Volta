//! Loud indexed access for kernel code (ADDENDUM §8).
//!
//! `clippy::indexing_slicing` is denied, so kernels cannot write `v[i]`. The escape
//! hatch used to be `.get(i).unwrap_or(0.0)` for reads and `if let Some(slot) =
//! v.get_mut(i)` for writes. Both turn an index-mapping bug into a plausible zero or
//! a write that never happens, which is how B4 stayed silent.
//!
//! These helpers panic on an out-of-bounds index in every build profile. An index that
//! is out of bounds in a kernel is a mapping bug, not data. (§8 proposed NaN in release;
//! a skipped write cannot be made loud with NaN, so one rule covers reads and writes.
//! See DECISIONS.md, "Indexing policy".) `#[track_caller]` makes the panic name the
//! kernel site, not this file.
//!
//! Where zero is the *meaning* (padding), check the coordinate explicitly and read with
//! [`fetch`] only once it is known to be in range.

use std::ops::Range;

/// Reads `s[i]`, panicking if `i` is out of bounds.
///
/// # Panics
///
/// If `i >= s.len()`.
#[inline]
#[track_caller]
pub(crate) fn fetch<T: Copy>(s: &[T], i: usize) -> T {
    match s.get(i) {
        Some(&v) => v,
        None => panic!(
            "index {i} out of bounds for len {} (index-mapping bug)",
            s.len()
        ),
    }
}

/// Returns a mutable reference to `s[i]`, panicking if `i` is out of bounds.
///
/// # Panics
///
/// If `i >= s.len()`.
#[inline]
#[track_caller]
pub(crate) fn slot<T>(s: &mut [T], i: usize) -> &mut T {
    let len = s.len();
    match s.get_mut(i) {
        Some(v) => v,
        None => panic!("index {i} out of bounds for len {len} (index-mapping bug)"),
    }
}

/// Returns `s[r]`, panicking if the range is out of bounds or reversed.
///
/// # Panics
///
/// If `r` is not contained in `0..s.len()`.
#[inline]
#[track_caller]
pub(crate) fn span<T>(s: &[T], r: Range<usize>) -> &[T] {
    match s.get(r.clone()) {
        Some(v) => v,
        None => panic!(
            "range {}..{} out of bounds for len {} (index-mapping bug)",
            r.start,
            r.end,
            s.len()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// In-bounds reads return the element.
    #[test]
    fn fetch_in_bounds() {
        let v = [1.0_f32, 2.0, 3.0];
        assert_eq!(fetch(&v, 0), 1.0);
        assert_eq!(fetch(&v, 2), 3.0);
    }

    /// One past the end panics instead of defaulting.
    #[test]
    #[should_panic(expected = "index-mapping bug")]
    fn fetch_out_of_bounds_panics() {
        let v = [1.0_f32, 2.0, 3.0];
        let _ = fetch(&v, 3);
    }

    /// An empty slice panics too (the old `first().unwrap_or(0.0)` case).
    #[test]
    #[should_panic(expected = "index-mapping bug")]
    fn fetch_empty_panics() {
        let v: [f32; 0] = [];
        let _ = fetch(&v, 0);
    }

    /// Writing through `slot` mutates the element.
    #[test]
    fn slot_in_bounds_writes() {
        let mut v = [0.0_f32; 3];
        *slot(&mut v, 1) = 5.0;
        assert_eq!(v, [0.0, 5.0, 0.0]);
    }

    /// A write past the end panics instead of being skipped.
    #[test]
    #[should_panic(expected = "index-mapping bug")]
    fn slot_out_of_bounds_panics() {
        let mut v = [0.0_f32; 3];
        *slot(&mut v, 3) = 1.0;
    }

    /// In-bounds ranges return the sub-slice, including the empty and full ranges.
    #[test]
    fn span_in_bounds() {
        let v = [1.0_f32, 2.0, 3.0, 4.0];
        assert_eq!(span(&v, 1..3), &[2.0, 3.0]);
        assert_eq!(span(&v, 4..4), &[] as &[f32]);
        assert_eq!(span(&v, 0..4), &v);
    }

    /// A range that runs past the end panics.
    #[test]
    #[should_panic(expected = "index-mapping bug")]
    fn span_out_of_bounds_panics() {
        let v = [1.0_f32, 2.0, 3.0, 4.0];
        let _ = span(&v, 2..5);
    }
}

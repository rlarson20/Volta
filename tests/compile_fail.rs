//! Compile-fail tests for the "this misuse no longer compiles" guarantees
//! (ADDENDUM §10.3). Each case lives in `tests/compile_fail/` with a pinned `.stderr`.
//!
//! The pinned text is rustc's diagnostic output, which changes between compiler
//! releases, so the test only runs on toolchains whose wording matches (1.93 and
//! later; 1.89-1.92 render the trait-bound help with source snippets). When a new
//! stable changes the wording, re-bless on stable with `TRYBUILD=overwrite` under
//! both feature sets (`cargo test --test compile_fail` and the same with
//! `--no-default-features`), check the two agree, and commit the `.stderr` files.

/// Every case in `tests/compile_fail/` fails to compile with its pinned error.
// The version gate must come before `#[test]`: the test harness reads `ignore`
// when `#[test]` expands, so an `ignore` added after it is never seen.
#[rustversion::attr(
    before(1.93),
    ignore = "trybuild .stderr is pinned to rustc >= 1.93 diagnostics; older compilers word the trait-bound help differently"
)]
#[test]
fn compile_fail() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/*.rs");
}

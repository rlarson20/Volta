//! Compile-fail tests for the "this misuse no longer compiles" guarantees
//! (ADDENDUM §10.3). Each case lives in `tests/compile_fail/` with a pinned `.stderr`.

#[test]
fn compile_fail() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/*.rs");
}

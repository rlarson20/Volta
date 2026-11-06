//! Compile-fail tests for the "this misuse no longer compiles" guarantees
//! (ADDENDUM §10.3). Each case lives in `tests/compile_fail/` with a pinned `.stderr`.
//!
//! The pinned text is rustc's diagnostic output, which changes between compiler
//! releases, so the test only runs on toolchains whose wording matches (1.93 and
//! later; 1.89-1.92 render the trait-bound help with source snippets). When a new
//! stable changes the wording, re-bless on stable with `TRYBUILD=overwrite` under
//! both feature sets (`cargo test --test compile_fail` and the same with
//! `--no-default-features`), check the two agree, and commit the `.stderr` files.
//!
//! rustc quotes libcore's source in some diagnostics (the `PhantomData` note in
//! `guards_not_send`) only when the `rust-src` component is installed, so the pinned
//! text assumes it. Bless and run with `rust-src` (`rustup component add rust-src`);
//! the test refuses to run without it instead of printing a misleading diff.

/// Every case in `tests/compile_fail/` fails to compile with its pinned error.
// The version gate must come before `#[test]`: the test harness reads `ignore`
// when `#[test]` expands, so an `ignore` added after it is never seen.
#[rustversion::attr(
    before(1.93),
    ignore = "trybuild .stderr is pinned to rustc >= 1.93 diagnostics; older compilers word the trait-bound help differently"
)]
#[test]
fn compile_fail() {
    require_rust_src();
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/*.rs");
}

/// Panics unless libcore's source is installed, since rustc's output (and so the pinned
/// `.stderr`) changes with it.
fn require_rust_src() {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let out = std::process::Command::new(&rustc)
        .args(["--print", "sysroot"])
        .output()
        .expect("run `rustc --print sysroot`");
    let sysroot = String::from_utf8(out.stdout).expect("sysroot is UTF-8");
    let marker = std::path::Path::new(sysroot.trim())
        .join("lib/rustlib/src/rust/library/core/src/marker.rs");
    assert!(
        marker.exists(),
        "the pinned compile_fail .stderr files quote libcore source, which rustc prints only \
         with the `rust-src` component; run `rustup component add rust-src` ({} not found)",
        marker.display()
    );
}

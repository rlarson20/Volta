//! Build script: links Apple's Accelerate framework for the `cblas_sgemm` fast path in
//! `ops/matmul.rs`. Only runs when the `accelerate` feature is on and the target is macOS.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // Cargo exposes features and the target to build scripts as env vars; `cfg!` here would
    // describe the host, not the target.
    let accelerate = std::env::var_os("CARGO_FEATURE_ACCELERATE").is_some();
    let macos = std::env::var("CARGO_CFG_TARGET_OS").is_ok_and(|os| os == "macos");
    if accelerate && macos {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}

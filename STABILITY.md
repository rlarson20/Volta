# Stability

This file states what Volta promises across releases. Users and reviewers can use it to tell a bug from a policy. Volta is pre-1.0 today. Until 1.0, any minor release (`0.x` → `0.(x+1)`) may break the API, as [Cargo's semver rules](https://doc.rust-lang.org/cargo/reference/semver.html) allow, and each break is listed in the changelog. The sections below are the policy that takes effect at 1.0. Where a section already holds today, it says so.

## Stable surface

At 1.0 the semver promise covers the public API with default features (a CPU-only build), with these exceptions:

- **Experimental:** the `gpu` feature and everything behind it, `volta::amp` (autocast, `GradScaler`), and the `Muon` optimizer. These can change in a minor release. How they are marked is #400.
- **Not API:** anything `#[doc(hidden)]`, the exact text of error and panic messages, and `Debug` output. Match on `VoltaError` variants (and, once #377 lands, their structured details), not on strings.

Adding a variant to a `#[non_exhaustive]` enum, or a field to a `#[non_exhaustive]` struct, is not a breaking change.

## Public dependencies

[`half`](https://crates.io/crates/half) is a public dependency: the `Element` impls for `half::f16` and `half::bf16` (`volta::storage::Element`) are public API, and so are `HostBuf`'s `F16`/`BF16` variants until #356 makes `HostBuf` crate-private. A major-version bump of `half` is therefore a major release of Volta. No other dependency's types appear in the public API. wgpu types stay inside `volta::gpu`'s private modules. Native `f16` in Rust would remove part of this once it is stable. `bf16` has no native equivalent, so `half` remains a public dependency either way.

## MSRV

The minimum supported Rust version is `rust-version` in `Cargo.toml`, currently 1.89. The `cpu-only (1.89)` CI job lints and tests the default (CPU-only) build at that version.

- **Raising the MSRV is a minor release, never a patch.** The changelog names the new version. A wgpu upgrade is the usual reason for a bump (`DECISIONS.md`, dependency updates).
- **Gap:** the `gpu` and `accelerate` code is not compiled at the MSRV. Linux CI builds `gpu` on stable only, and the macOS job runs stable only (`DECISIONS.md`, "The `macos` job runs stable only"). An MSRV break in that code is not caught by CI, so at 1.0 the MSRV promise covers the default build until a feature lane runs at the MSRV.

## Numerical results

Volta does not promise bitwise-identical results across releases. That matches [PyTorch's reproducibility notes](https://pytorch.org/docs/stable/notes/randomness.html). A minor release may change summation order, kernels, or algorithm selection (conv's auto-selection, for example). Changes stay within the tolerances the test suite states against PyTorch fixtures and finite differences.

- **Same release, platform and backend:** a seeded run (`manual_seed`) is reproducible. Across `rand` versions it may not be (#401).
- **Across backends:** CPU and GPU results agree within the parity tolerances in `tests/parity_*.rs`, not bitwise.
- **A behavior change that moves a result outside a stated tolerance is a bug,** unless the changelog calls it out as a fix.

## Deprecation

At 1.0 and after, a public item is removed only after at least one minor release under `#[deprecated(since = "..", note = "use .. instead")]`. Removing it, or changing its signature, then waits for the next major release. Before 1.0, items may be removed without a deprecation release.

## Platforms and backends

| Tier | Platform or backend | What CI runs |
| --- | --- | --- |
| Verified | Linux x86_64, CPU | Full test suite at MSRV and stable (`cpu-only`) |
| Verified | macOS aarch64, CPU + Accelerate + Metal | Full test suite on an Apple GPU (`macos (stable)`), after each merge into `improvements` and on the PR into `main`. Linted on every PR (`macos-cross`) |
| Software only | Vulkan | lavapipe, Mesa's software Vulkan (`linux-lavapipe`); no Vulkan hardware |
| Builds only | `wasm32-unknown-unknown`, CPU | `cargo clippy --lib` (`wasm32`); nothing runs |
| Never run | Windows, DX12, WebGPU in a browser, real Vulkan or other GPUs | — (Windows CI is #403) |

A bug report on a "never run" platform is welcome. It is fixed as far as it can be without that hardware.

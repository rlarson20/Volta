# Cloud sessions

Read this first when you run in a Claude Code cloud session (Linux, `CLAUDE_CODE_REMOTE` set, no `rtk`). Everything in `CLAUDE.md` still applies: the invariants, the scope rule, and "stop and ask". This file only covers what the cloud environment changes.

The cloud is not a restricted scope. The limits below come from the hardware, not from the kind of work.

## How cloud differs from local

|        | Local (usual)                                         | Cloud                                                                                              |
| ------ | ----------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| OS     | macOS, Apple Silicon, zsh                             | Linux VM, fresh clone of the pushed branch                                                         |
| Matmul | Accelerate (`cblas_sgemm`), linked by `build.rs`      | `accelerate` is a no-op off macOS, so `matrixmultiply`                                             |
| GPU    | Real Metal adapter; GPU tests run                     | wgpu compiles but finds no adapter; GPU tests skip silently (`src/gpu/mod.rs`, `is_gpu_available`) |
| Shell  | `rtk` rewrites commands; BSD `sed`; zsh gotchas       | No `rtk`; GNU tools; the "Shell gotchas" in `CLAUDE.md` do not apply                               |
| Tools  | `just`, `pre-commit`, `naga`, `nextest` installed     | Any of them may be missing; install or run the steps directly                                      |
| Config | `~/.claude` (global `CLAUDE.md`, memory, user skills) | None of it exists; only what is committed to the repo                                              |
| User   | At the keyboard                                       | Often on a phone; you can ping them, but replies may lag                                           |
| CPU    | Dedicated                                             | Shared and noisy                                                                                   |

A cloud run is the GitHub `cpu-only` lane in `.github/workflows/ci.yml`, not the `macos` lane. "Tests pass" there means CPU-verified only.

## Valid cloud work

Work that a Linux CPU can verify, and that reviews as a PR. Current candidates from `notes/TODO.md` (check it for live status):

- The CPU-only clippy errors (`cargo clippy --lib --no-default-features -- -D warnings`). Most are the missing not-gpu stub pattern.
- Seeding the flaky gradient tests (`linear_tests::backward_gradcheck_input` and the conv list). Use the recipe in `notes/TODO.md`: seed, `eps=0.5`, confirm 0/400 seeds fail, and confirm a 1.1x-scaled analytic gradient still fails.
- Docs toward the merge gate (the undocumented items behind the `lib.rs` allow). Read the code for every claim you write. Don't remove the allow until nothing is left.
- CPU defects: B2 (LSTM `slice_gate`), B3 (`PositionalEncoding`), `check_gradients` not zeroing grads, the greedy `Sequential::load_state_dict` prefix, a diamond-accumulation regression test, the CPU parts of the GEMM test fixes, a reference triple-loop matmul.
- Comment-only fixes (for example the `BackwardParams` comments in `matmul_backward.wgsl`).
- Investigations that end in a `notes/TODO.md` or `DECISIONS.md` entry.
- Phase 3 work (storage, the `Shape` flip) and later architecture, on the same terms as locally: ask before choosing a design. Any GPU-touching part can't be verified here (below).

## Not valid in cloud

- **Anything whose evidence needs a GPU:** H5, GPU broadcast backward, the GPU part of H3, GPU parity claims, the GPU side of `DevicePolicy`. A passing GPU test here was skipped. Writing GPU code is fine once the user agrees; marking it verified, or `[x]` in `notes/TODO.md`, is not. Hand it off as "needs `just test-gpu` on the Mac".
- **Performance and the Accelerate path:** benches, speed claims, and timing thresholds. `bench_matmul_speedup` (`src/lib.rs`) uses the non-Accelerate threshold and may flake on a shared VM. Report it; don't loosen it.
- **Shader validation:** `just validate-shaders` needs `naga`. Install it, or say the check did not run.
- **Git:** never push to `main` or `improvements`, never force-push, never `--no-verify` or `SKIP=`. Work on a branch off `improvements` and open the PR against `improvements`.

## How to work there

- Gate, in order: `cargo fmt --check`, `./scripts/check-fallbacks.sh`, `cargo clippy --all-targets -- -D warnings`, `cargo test`, `RUSTDOCFLAGS=-D warnings cargo doc --no-deps`, `cargo build --examples`, then `cargo test --no-default-features`. This is `just ci` plus the CPU-only test; run the steps directly if `just` is missing.
- Ask when you'd ask locally. Make each question answerable from a phone: short, options listed, your recommendation first. Batch them, and keep working on whatever doesn't depend on the answer. Never guess past an invariant or a design decision to avoid waiting. If no answer arrives before the session ends, put the open question in the PR description.
- The PR description must say "CPU-only, Linux, GPU tests skipped" and list what was not verified.
- One task per session and per PR. Keep diffs reviewable.

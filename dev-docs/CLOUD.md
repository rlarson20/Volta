# Cloud sessions

Read this first when you run in a Claude Code cloud session (Linux, `CLAUDE_CODE_REMOTE` set, no `rtk`). Everything in `CLAUDE.md` still applies: the invariants, the scope rule, and "stop and ask". This file only covers what the cloud environment changes.

The cloud is not a restricted scope. The limits below come from the hardware, not from the kind of work.

## How cloud differs from local

|        | Local (usual)                                         | Cloud                                                                                              |
| ------ | ----------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| OS     | macOS, Apple Silicon, zsh                             | Linux VM, fresh clone of the pushed branch                                                         |
| Matmul | Accelerate (`cblas_sgemm`), linked by `build.rs`      | `accelerate` is a no-op off macOS, so `matrixmultiply`                                             |
| GPU    | Real Metal adapter; GPU tests run                     | No adapter by default: GPU tests skip silently. `apt-get install -y mesa-vulkan-drivers` adds lavapipe (software Vulkan, `shaderFloat16`), which runs them (see below) |
| Shell  | `rtk` rewrites commands; BSD `sed`; zsh gotchas       | No `rtk`; GNU tools; the "Shell gotchas" in `CLAUDE.md` do not apply                               |
| Tools  | `just`, `pre-commit`, `naga`, `nextest` installed     | Any of them may be missing; install or run the steps directly                                      |
| Config | `~/.claude` (global `CLAUDE.md`, memory, user skills) | None of it exists; only what is committed to the repo                                              |
| User   | At the keyboard                                       | Often on a phone; you can ping them, but replies may lag                                           |
| CPU    | Dedicated                                             | Shared and noisy                                                                                   |

A cloud run is the GitHub `cpu-only` lane in `.github/workflows/ci.yml`, not the `macos` lane. "Tests pass" there means CPU-verified only.

The macOS build still *compiles* here: `rustup target add aarch64-apple-darwin`, then `just macos-cross` (clippy, rustdoc and the feature powerset for that target; needs clang and cargo-hack). That is CI's required `macos-cross` job. It checks the macOS-only code type-checks and lints, not that it runs; `just ci` runs it when the target is installed.

## Valid cloud work

Work that a Linux CPU can verify, and that reviews as a PR. Candidates are open issues without the `needs-mac` label; check the issue for live status. Typical shapes:

- Test and CPU-side correctness fixes (labels `kind:test`, `kind:bug` without `area:gpu`).
- Docs toward the merge gate (the undocumented items behind the `lib.rs` allow). Read the code for every claim you write. Don't remove the allow until nothing is left.
- Comment-only fixes (for example the `BackwardParams` comments in `matmul_backward.wgsl`).
- Investigations that end in an issue comment or a `DECISIONS.md` entry.
- Phase 3 work (storage, the `Shape` flip) and later architecture, on the same terms as locally: ask before choosing a design. Any GPU-touching part can't be verified here (below).

## Not valid in cloud

- **Metal-specific evidence:** H5, GPU parity on Apple hardware, GPU speed. Without lavapipe a passing GPU test here was skipped. With lavapipe the GPU tests run on a software Vulkan device: that verifies the kernels' logic and the device-policy paths, not Metal's precision, limits or driver. Say "GPU tests ran on lavapipe" and still hand Metal off as "needs `just test-gpu` on the Mac"; don't close a `needs-mac` issue from the cloud.
- **Performance and the Accelerate path:** benches, speed claims, and timing thresholds. `bench_matmul_speedup` (`src/lib.rs`) uses the non-Accelerate threshold and may flake on a shared VM. Report it; don't loosen it.
- **Shader validation** runs anywhere: `just validate-shaders` is `cargo test --features gpu,accelerate --lib shader_template` (naga is a dev-dependency).
- **Git:** never push to `main`, never force-push, never `--no-verify` or `SKIP=`. Branch and target PRs as in "Pull requests" below.

## How to work there

- Gate: GitHub Actions, on the PR. Locally, run clippy (`-D warnings`) and the tests the change touches, then push; don't run the full `just ci` too (CLAUDE.md, "Commands"). The full gate is 10–15 minutes per branch here, and one `target/` per worktree for several branches fills the container's disk. To reproduce a CI failure locally, run the failing recipe, or `just ci` for all of it. If `just` is missing, its steps are, in order: `cargo fmt --check`, `./scripts/check-fallbacks.sh`, `./scripts/test-check-ignores.sh`, `./scripts/check-ignores.sh` (its issue lookup works here: `gh api` REST goes through the proxy), `cargo clippy --no-default-features --all-targets -- -D warnings`, `cargo clippy --features gpu,accelerate --all-targets -- -D warnings`, `cargo test --features gpu,accelerate`, `cargo test --no-default-features`, `RUSTDOCFLAGS=-D warnings cargo doc --no-deps --features gpu,accelerate`, the same with `--no-default-features`, `cargo build --examples --features gpu,accelerate`, `./scripts/examples-smoke.sh`, `just bench-smoke`'s three `cargo bench --no-default-features --bench <name> -- --test` runs, then `cargo hack check --feature-powerset --no-dev-deps --exclude-features gpu` when `cargo-hack` is installed (`just ci` skips it with a notice otherwise). `just ci` also runs `just macos-cross` and `just wasm-check` when their targets are installed (`rustup target add aarch64-apple-darwin wasm32-unknown-unknown`).
- Run `rustup component add rust-src` first. The pinned `tests/compile_fail/*.stderr` quote libcore source, which rustc prints only with that component, and the test panics without it. Bless with it installed too.
- Ask when you'd ask locally. Make each question answerable from a phone: short, options listed, your recommendation first. Batch them, and keep working on whatever doesn't depend on the answer. Never guess past an invariant or a design decision to avoid waiting. If no answer arrives before the session ends, put the open question in the PR description.
- The PR description must say what ran: "CPU-only, Linux, GPU tests skipped", or "GPU tests ran on lavapipe" (with `mesa-vulkan-drivers` installed), and list what was not verified (always Metal).
- One task per session and per PR. Keep diffs reviewable.

## Pull requests

Each PR branches from `main` and targets it. Open PRs are independent and can be reviewed and merged in any order. `improvements` is retired; don't target it (`DECISIONS.md`, "One branch: `main`"). The local `just pr` / `just sync` commands and the reasoning behind these rules are in `dev-docs/WORKFLOW.md`.

- **Branch from `main`,** from its current tip, with one branch per task. Don't branch from another open PR's branch.
- **One task per PR.** Keep the diff reviewable. A follow-up to an open PR is a new PR once that one has merged; before then, push to its branch.
- **Describe it.** Say what changed and why, write `Fixes #N` for each issue it closes, and list what ran and what was not verified (see above).
- **Squash-merge** (one commit per PR). The ruleset on `main` allows no other merge method. Write the squash commit message as a commit message, not a list of the branch's commits.
- **Conflicts.** If another PR merged first and yours conflicts, bring your branch up to date with `main` and resolve it there. In a cloud session, merge `main` into your branch; force-pushing is ruled out above. A local session may rebase instead and push with `--force-with-lease`. Never force-push `main`.
- **Dependent work.** If a task needs another open PR's change, wait for it to merge, or say so in the description and ask before basing a branch on it. A PR whose base isn't `main` needs the user's say-so.

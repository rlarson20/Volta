# CLAUDE.md — Volta

Educational PyTorch-style autograd engine in pure Rust. Correctness and defensibility over speed: every claim the code, docs, or tests make must hold up under review.

Area-specific rules live in `.claude/rules/` and load automatically when you read or edit matching files. This file holds only what applies everywhere.

In a cloud session (Linux, `CLAUDE_CODE_REMOTE` set, no `rtk`, no GPU), read `dev-docs/CLOUD.md` first.

## Current state

Update this section when a phase lands or the merge gate changes. Keep it short; what landed is in the commit log and `DECISIONS.md`.

- **Roadmap:** `docs/plans/ADDENDUM.md` § Sequencing defines the phases. Phases 0–3 and 5 (dtypes and mixed precision, PRs #19–#28) are done. Phase 4 (arena graph, sealed `Backend` trait, proptest DAG checks) is optional and comes last.
- **Open work lives in [GitHub issues](https://github.com/rlarson20/Volta/issues), nowhere else.** Each defect ID (`S<n>`, `A<n>`) has one issue naming its `#[ignore = "S<n>: ..."]` repro and its audit section. Pick work with `is:open label:P0 -label:blocked` (then P1, ...). The audits in `docs/audits/` are dated, append-only evidence, not status: correct them by appending, and tell the user (`.claude/rules/audits.md`). PyTorch fixtures are regenerated with `just torch-fixtures` (needs `uv`).
- **Merge gate:** `improvements` merges into `main` when every [`merge-gate`](https://github.com/rlarson20/Volta/issues?q=is%3Aopen+label%3Amerge-gate) issue is closed (fixed, or closed as not planned with a reason). `just ci` passes and the library documents every item. Rulesets in `.github/rulesets/` (applied with `just rulesets`) enforce the PR and required CI checks on both branches (`DECISIONS.md`, "Rulesets enforce the merge gate"). PRs into `improvements` gate on Linux, including `macos-cross`. Real macOS (Metal, Accelerate) runs after each merge and on the PR into `main`; a post-merge failure opens a `merge-gate` issue (`DECISIONS.md`, "macOS checks move off the PR gate").

## Invariants

Never violate these. If a task seems to require it, stop and ask.

1. `autograd::edge()` is the only code that attaches `parents`/`grad_fn`. Never build an op or layer output by copying data into `RawTensor::new` (that is how B2/B3 shipped).
2. A `borrow_mut()` outside `src/autograd.rs` and `src/ops/` means `Tensor` is missing a method. Add the method instead.
3. Only `accumulate_grad` and `set_grad` write `grad`, and only `zero_grad` (`RawTensor::zero_grad`, which `Tensor`, `Module` and the optimizers call) clears it.
4. Library code never prints: use `log::` (lint-enforced).
5. Numeric code fails loudly (ADDENDUM §8). Library code reads and writes through `crate::index::{fetch, slot, span}`, which panic on an out-of-bounds index in every profile; `#[cfg(test)]` code in `src/` uses `fetch` too, `tests/` and `#[test]` fns index with `v[i]` (`clippy.toml` allows `indexing_slicing` in tests), or zip expected against actual after asserting equal lengths. `examples/` and `benches/` (public API only, not covered by `clippy.toml`) read a scalar with `Tensor::item()` and anything else with `.expect(..)`. Every target reads a scalar with `Tensor::item()`. Never use `.get(i).unwrap_or(0.0)`, and never write an `if let Some` that silently skips a read or write. A zero that means padding gets an explicit coordinate check. Silent zeros are how wrong gradients pass tests. `just check-fallbacks` (in `just ci` and pre-commit) enforces it; opt out per line with `// index-ok: <reason>`.
6. Never write a test that asserts a known bug's behavior. Use `#[ignore = "<defect ID>: <why>"]` instead, where the ID is one an open issue's title starts with (`S4`, as in "S2/S3/S4: ...") or the issue number itself (`#60`). `just check-ignores` (in `just ci`, CI, and, without the issue lookup, pre-commit) enforces both.
7. Don't overclaim. A layer or feature is "working" only when a test exercises its gradients with non-degenerate inputs.
8. Errors follow where the data comes from (DECISIONS.md, "Panic inside the program, `Result` where data comes from outside"; `dev-docs/DEVELOPMENT.md`). An argument a user can supply (shape, axis, range, label, bytes, state dict) is validated, before any allocation, and returned as a `VoltaError`; the panicking form wraps the `try_` one. A contract violation between tensors the program built panics, with `# Panics`.

## Commands

- Default features are `gpu` + `accelerate`, so a plain `cargo build` compiles the GPU backend. For CPU-only, add `--no-default-features`.
- Fast loop: `cargo test --no-default-features --lib <name-filter>`.
- Before committing, run `just ci` (fmt, `clippy --all-targets -D warnings`, test and `RUSTDOCFLAGS=-D warnings cargo doc`, each with default features and with `--no-default-features`, then examples (built, and the quick ones run by `just examples-smoke`) and the cargo-hack feature powerset when installed). `ci.yml` calls the same justfile recipes. `pre-commit run` is the faster subset (fmt, fallbacks, ignore IDs, both clippy runs, `cargo test`). Both should pass on a clean `HEAD`; if a hook fails, fix the cause rather than skipping it.
- Committing past the hooks: `SKIP=clippy,cargo-test git commit ...` (ask the user first; never `--no-verify`). The `fmt` hook runs `cargo fmt --all` and fails whenever it rewrites a file, so make the tree `cargo fmt --check`-clean first. Stage everything you want and leave nothing unstaged: pre-commit stashes unstaged changes, and that conflicts with its own fixes (also with `git commit -- <paths>`).
- `cargo test <filter>` matches test *names*. There are no category modules called `core`, `neural`, or similar.

## Shell gotchas

The shell is zsh on macOS, and a Claude Code hook rewrites commands through `rtk`.

- `grep` becomes `rtk grep`, which rejects `-n`, `-E`, and `--include`. Use `rtk proxy grep ...` for a standalone search. In a pipeline, `rtk proxy grep` can silently print nothing, so use `command grep` there.
- Quote globs (`--include='*.rs'`); zsh errors on unmatched ones. A bare `===` is also an error in zsh (`=` expansion), so don't use it as a separator in `echo`.
- BSD `sed`: `sed -i '2i text'` fails. Use the Edit tool or a short `python3` script.
- Don't chain a `VAR=...` assignment and its use across `&&` where an earlier command can fail; the next command runs with `VAR` unset (this happened with a scratchpad path). Set it inside the same command.
- `rtk` filters `cargo` output and can drop compiler error bodies (you get only `could not compile`). Use `rtk proxy cargo ...` to see the real error.
- A `cargo test ... | grep` pipeline can come out empty (the hook rewrites the `cargo` side). To capture `--list` output or anything you will diff, redirect: `rtk proxy cargo test --lib -- --list > file`, then grep the file. Don't hide stderr with `2>/dev/null` while checking that the build worked.
- **A command that may run longer than 3 minutes goes in the background from the start** (`run_in_background`, output redirected to a file in the scratchpad), then do other work while it runs. This covers examples that train on MNIST (`mnist_cnn` is far over 3 minutes on the CPU), `just test-gpu`, benches, and full `just ci`. Run each such command as its own task rather than in one `for` loop, so one slow run doesn't hold up or hide the rest. Output redirected through `rtk` is buffered, so a quiet log does not mean a hang: check CPU time with `ps` and use a short run (fewer epochs or batches) to judge speed before a full one.
- Format single files with `rustfmt --edition 2024 <file>`. `cargo fmt` rewrites unrelated, already-dirty files.
- For a mechanical signature change that touches many call sites, change the signature and wrap the spans from `cargo check --message-format=json` (E0308 spans) with a script instead of hand-editing each site.

## Where knowledge lives

Read these when the need comes up; don't preload them.

| Need | Read |
| --- | --- |
| Why something is designed the way it is | `DECISIONS.md` |
| What's open; what's next | [GitHub issues](https://github.com/rlarson20/Volta/issues) (labels `P0`–`P3`, `merge-gate`, `blocked`, `needs-mac`) |
| Evidence and repro for a defect ID | `docs/audits/` |
| Feature ideas | Issues labeled `kind:feature` (don't start these unprompted) |
| Step-by-step: adding an op or layer | `dev-docs/DEVELOPMENT.md` |
| Conv algorithms and selection | `dev-docs/CONV2D_IMPLEMENTATION.md` |
| What's valid in a cloud session; Linux/no-GPU differences | `dev-docs/CLOUD.md` |
| Local PR workflow: `just pr`, `just sync`, `just prs`, and why | `dev-docs/WORKFLOW.md` |

## Filing what you learn

When you're corrected or discover something non-obvious, file it in exactly one place:

- **A defect, task, or idea** goes in a GitHub issue (one per defect ID, with labels `kind:`, `area:`, `P0`–`P3`, `size:`), never in a file. Before filing, run `just issues "<two or three key terms>"` (an op name, a file, the symptom; it searches closed issues too). If something matches, comment on it or add a `Related: #N` line instead of filing a second one.
- **A correction to an audit** is appended under that file's `## Corrections` with an inline pointer, never made by editing the original text, and you tell the user (`.claude/rules/audits.md`).
- **A rule for one area** goes in the matching `.claude/rules/*.md`.
- **A repo-wide rule** goes under Invariants above. Keep that list at 10 or fewer and merge items rather than append.
- **Why a decision was made, or what was tried and rejected** goes in a `DECISIONS.md` entry.
- **What was done** goes in the commit message, not here. Never add "COMPLETED" or changelog sections to this file or to the rules.

## Scope

Stay inside the current phase. Backlog features, refactors from later phases, and "while I'm here" cleanups get proposed as a new issue, not implemented.

## Working an issue

- A PR that fixes an issue says `Fixes #N` in its body; merging closes the issue. One PR may close several, each with its own `Fixes`.
- PR bodies follow `.github/PULL_REQUEST_TEMPLATE.md` (`gh pr create --body` skips it): a short opener the user approved, `Fixes`, the checklist, then the agent's full write-up inside `<details><summary>Model (effort)</summary>`.
- Finished without a PR (an investigation, a refuted defect): comment the result on the issue and close it (`completed` or `not planned`, with the reason).
- Found something new mid-task: open an issue for it and keep it out of the current PR.
- Skip `needs-mac` issues in a cloud session; they need the M2's Metal backend to verify.
- When a fix removes an `#[ignore = "S<n>: ..."]`, the PR closes the issue that names `S<n>`.

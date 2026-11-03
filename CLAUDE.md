# CLAUDE.md — Volta

Educational PyTorch-style autograd engine in pure Rust. Correctness and defensibility over speed: every claim the code, docs, or tests make must hold up under review.

Area-specific rules live in `.claude/rules/` and load automatically when you read or edit matching files. This file holds only what applies everywhere.

In a cloud session (Linux, `CLAUDE_CODE_REMOTE` set, no `rtk`, no GPU), read `dev-docs/CLOUD.md` first.

## Current state

Update this section when a phase lands or the merge gate changes. Keep it short; what landed is in the commit log and `DECISIONS.md`.

- **Roadmap:** `docs/plans/ADDENDUM.md` § Sequencing is the only plan. Phases 0–3 are done. Phase 5 (dtypes and mixed precision, ADDENDUM §11, PRs P0–P8) is implemented as a stack of open PRs (#19–#28), merged bottom-up; it comes before Phase 4 and before the merge. Phase 4 (arena graph, sealed `Backend` trait, proptest DAG checks) is optional and comes last.
- **Open work:** `notes/TODO.md` lists only what is open, including the 2026-11 sweep (`docs/audits/2026-11-sweep.md`): defects found by checking against PyTorch, each with an `#[ignore = "S<n>: ..."]` test. PyTorch fixtures are regenerated with `just torch-fixtures` (needs `uv`). The defect IDs B1–B5 and H1–H8 (`docs/audits/OPUS_5_EVAL_AUDIT.md`) are all fixed; the audits are frozen records.
- **Merge gate:** cleared. The library documents every item (`missing_docs` and `missing_docs_in_private_items` are `deny`), and `just ci` passes. `improvements` can merge into `main` once the other TODO items are done, including Phase 5 and the correctness items from the 2026-11 sweep (`notes/TODO.md`, "Correctness: found by the 2026-11 sweep"). Those are open defects, not features, so the gate is not cleared until they are fixed or deliberately deferred.

## Invariants

Never violate these. If a task seems to require it, stop and ask.

1. `autograd::edge()` is the only code that attaches `parents`/`grad_fn`. Never build an op or layer output by copying data into `RawTensor::new` (that is how B2/B3 shipped).
2. A `borrow_mut()` outside `src/autograd.rs` and `src/ops/` means `Tensor` is missing a method. Add the method instead.
3. Only `accumulate_grad` and `set_grad` write `grad`.
4. Library code never prints: use `log::` (lint-enforced).
5. Numeric code fails loudly (ADDENDUM §8). Library code reads and writes through `crate::index::{fetch, slot, span}`, which panic on an out-of-bounds index in every profile; `#[cfg(test)]` code in `src/` uses `fetch` too, `tests/` and `#[test]` fns index with `v[i]` (`clippy.toml` allows `indexing_slicing` in tests), or zip expected against actual after asserting equal lengths. `examples/` and `benches/` (public API only, not covered by `clippy.toml`) read a scalar with `Tensor::item()` and anything else with `.expect(..)`. Every target reads a scalar with `Tensor::item()`. Never use `.get(i).unwrap_or(0.0)`, and never write an `if let Some` that silently skips a read or write. A zero that means padding gets an explicit coordinate check. Silent zeros are how wrong gradients pass tests. `just check-fallbacks` (in `just ci` and pre-commit) enforces it; opt out per line with `// index-ok: <reason>`.
6. Never write a test that asserts a known bug's behavior. Use `#[ignore = "<defect ID>: <why>"]` instead.
7. Don't overclaim. A layer or feature is "working" only when a test exercises its gradients with non-degenerate inputs.
8. Errors follow where the data comes from (DECISIONS.md, "Panic inside the program, `Result` where data comes from outside"; `dev-docs/DEVELOPMENT.md`). An argument a user can supply (shape, axis, range, label, bytes, state dict) is validated, before any allocation, and returned as a `VoltaError`; the panicking form wraps the `try_` one. A contract violation between tensors the program built panics, with `# Panics`.

## Commands

- Default features are `gpu` + `accelerate`, so a plain `cargo build` compiles the GPU backend. For CPU-only, add `--no-default-features`.
- Fast loop: `cargo test --no-default-features --lib <name-filter>`.
- Before committing, run `just ci` (fmt, `clippy --all-targets -D warnings` with default features and with `--no-default-features`, test, doc, examples) or `pre-commit run`. Both should pass on a clean `HEAD`; if a hook fails, fix the cause rather than skipping it.
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
| What's in scope now; what's next | ADDENDUM § Sequencing + `notes/TODO.md` |
| Evidence and repro for a defect ID | `docs/audits/` |
| Feature ideas after Phase 2 | `docs/backlog.md` (don't start these unprompted) |
| Step-by-step: adding an op or layer | `dev-docs/DEVELOPMENT.md` |
| Conv algorithms and selection | `dev-docs/CONV2D_IMPLEMENTATION.md` |
| What's valid in a cloud session; Linux/no-GPU differences | `dev-docs/CLOUD.md` |

## Filing what you learn

When you're corrected or discover something non-obvious, file it in exactly one place:

- **A rule for one area** goes in the matching `.claude/rules/*.md`.
- **A repo-wide rule** goes under Invariants above. Keep that list at 10 or fewer and merge items rather than append.
- **Why a decision was made, or what was tried and rejected** goes in a `DECISIONS.md` entry.
- **What was done** goes in the commit message, not here. Never add "COMPLETED" or changelog sections to this file or to the rules.

## Scope

Stay inside the current phase. Backlog features, refactors from later phases, and "while I'm here" cleanups get proposed as a line in `notes/TODO.md`, not implemented.

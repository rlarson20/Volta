# CLAUDE.md — Volta

Educational PyTorch-style autograd engine in pure Rust. Correctness and defensibility over speed: every claim the code, docs, or tests make must hold up under review.

Area-specific rules live in `.claude/rules/` and load automatically when you read or edit matching files. This file holds only what applies everywhere.

## Current state

Update this section when a phase lands or the merge gate changes. Keep it short.

- **Roadmap:** `docs/plans/ADDENDUM.md` § Sequencing is the only plan. Phases 0–1 are done. Phase 2 (types for dims and ops) is in progress: `Shape`/`Axis`/`Strides` are defined in `src/shape.rs` but not yet threaded through signatures.
- **Open defects:** IDs B1–B5 and H1–H8 come from `docs/audits/OPUS_5_EVAL_AUDIT.md`. Live status is in `notes/TODO.md`.
- **Merge gate:** `src/lib.rs` carries a temporary `#![allow(missing_docs)]` (FIXME). `improvements` does not merge into `main` until that line is removed and the docs pass.

## Invariants

Never violate these. If a task seems to require it, stop and ask.

1. `autograd::edge()` is the only code that attaches `parents`/`grad_fn`. Never build an op or layer output by copying data into `RawTensor::new` (that is how B2/B3 shipped).
2. A `borrow_mut()` outside `src/autograd.rs` and `src/ops/` means `Tensor` is missing a method. Add the method instead.
3. Only `accumulate_grad` and `set_grad` write `grad`.
4. Library code never prints: use `log::` (lint-enforced).
5. Numeric code fails loudly. Never use `.get(i).unwrap_or(0.0)`, and never write an `if let Some` that silently skips a write (ADDENDUM §8). Silent zeros are how wrong gradients pass tests.
6. Never write a test that asserts a known bug's behavior. Use `#[ignore = "<defect ID>: <why>"]` instead.
7. Don't overclaim. A layer or feature is "working" only when a test exercises its gradients with non-degenerate inputs.

## Commands

- Default features are `gpu` + `accelerate`, so a plain `cargo build` compiles the GPU backend. For CPU-only, add `--no-default-features`.
- Until the merge gate clears, run functional checks with `RUSTFLAGS="-A missing_docs"`. `just test-all` does this through nextest.
- Fast loop: `RUSTFLAGS="-A missing_docs" cargo test --no-default-features --lib <name-filter>`.
- Before committing, run `just ci` (fmt, `clippy --all-targets -D warnings`, test, doc, examples) or `pre-commit run`.
- `cargo test <filter>` matches test *names*. There are no category modules called `core`, `neural`, or similar.

## Where knowledge lives

Read these when the need comes up; don't preload them.

| Need | Read |
| --- | --- |
| Why something is designed the way it is | `DECISIONS.md` |
| What's in scope now; what's next | ADDENDUM § Sequencing + `notes/TODO.md` |
| Evidence and repro for a defect ID | `docs/audits/` |
| Feature ideas after Phase 2 | `docs/backlog.md` (don't start these unprompted) |
| Step-by-step: adding an op or layer | `dev-docs/DEVELOPMENT.md` (pre-Phase-1 in places; the invariants above win) |
| Conv algorithms and selection | `dev-docs/CONV2D_IMPLEMENTATION.md` |

## Filing what you learn

When you're corrected or discover something non-obvious, file it in exactly one place:

- **A rule for one area** goes in the matching `.claude/rules/*.md`.
- **A repo-wide rule** goes under Invariants above. Keep that list at 10 or fewer and merge items rather than append.
- **Why a decision was made, or what was tried and rejected** goes in a `DECISIONS.md` entry.
- **What was done** goes in the commit message, not here. Never add "COMPLETED" or changelog sections to this file or to the rules.

## Scope

Stay inside the current phase. Backlog features, refactors from later phases, and "while I'm here" cleanups get proposed as a line in `notes/TODO.md`, not implemented.

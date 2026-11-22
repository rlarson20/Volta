# Finishing locally

RJ's runbook for the steps that need his Mac, his `gh` admin rights, or his decision. It was written on 2026-11-20, after the serial tail. Status lives in the linked issues, not here. Each section says when it is done, and the file is deleted once every section is.

## 1. The open PR

#448 (#398) STABILITY.md stays open until you've refined it. Its policies are proposals: an MSRV bump is a minor release, deprecations last one minor, and there is no bitwise numerical promise. The serial tail and the other independent PRs from 2026-11-20 are merged.

## 2. Apply the rulesets

`just rulesets` pushes `.github/rulesets/*.json` to GitHub. It needs admin `gh` auth, and agents don't run it unattended. Run it once now that the section 1 PRs have merged, so the rulesets match what they changed:

```sh
git switch improvements && git pull
just rulesets            # prints "updated: <name> (<id>)" per ruleset
```

Done when it prints `updated` for both `improvements` and `main`.

## 3. On the M2

These are the `needs-mac` issues, plus the ones that make speed claims. Each issue's "Done when" is the bar. The commands are where to start.

| Issue | Start with |
| --- | --- |
| #312 Metal fast-math probes | Write one CPU-vs-GPU probe per bullet at the extreme inputs, then `VOLTA_REQUIRE_GPU=1 just test-gpu` |
| #327 ASan over Accelerate | `RUSTFLAGS="-Zsanitizer=address" cargo +nightly-2026-11-01 test -Zbuild-std --target aarch64-apple-darwin --features accelerate --lib --tests -- --skip compile_fail` |
| #172 readback timeout panics | Decide the error shape first (it fits #66). #437 already makes each poll wait on its own submission, but the timeout still panics |
| #188 autocast fixture drift | `just torch-fixtures`, then `cargo test --test torch_ref_autocast`, measured with the test's own metric |
| #214 pairwise vs sequential sum | `just bench-cpu` at `6d0fe09` and on `improvements` (sizes are in the issue) |
| #79 GPU per-op overhead | `just bench-gpu`, then profile one synced op |

Done when each issue is closed, or has its numbers in a comment.

## 4. Decisions only you can make

Each of these issues has a research comment that ends in a recommendation. Reply on the issue with your choice, and an agent can do the work.

- **Research posted:** #355 (exact GPU `fmod`), #321 (f64 optimizer hyperparameters), #66 (`GpuError` shape), #144 (layer field visibility), #105 (`Storage` equality), #170 (`Device::Gpu` id), #175 (GPU context singleton), #177 (`forward(&self)` and BatchNorm), #83 (autocast weight cache), #128 (staging pool), #278 (per-channel normalize).
- **Only you can answer these:**
  - #406: which reporting channel SECURITY.md should give.
  - #400: whether GPU, AMP and Muon get an explicit experimental tier. STABILITY.md assumes they do.
  - #217: unblock it when you want the write-up.

Done when every listed issue has your reply.

## 5. Cutting 0.4.0

Nothing goes to `main` until you say so. When you do, open the PR from `improvements` into `main`. It runs `macos (stable)` and the semver workflow. A break the semver check reports is expected for 0.x, but the changelog must list it. The steps are the checklist on #455, and the tag and changelog research is in a comment there.

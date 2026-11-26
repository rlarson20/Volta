# Finishing locally

RJ's runbook for the steps that need his Mac, his `gh` admin rights, or his decision. It was written on 2026-11-20, after the serial tail. Status lives in the linked issues, not here. Each section says when it is done, and the file is deleted once every section is.

## 1. The open PRs

#448 (#398) STABILITY.md stays open until you've refined it. Its policies are proposals: an MSRV bump is a minor release, deprecations last one minor, and there is no bitwise numerical promise. It and #297 (the audit reference docs) still target `improvements`, so retarget both before section 2 deletes that branch; deleting a PR's base branch closes the PR. The two Dependabot PRs (#477, #478) retarget the same way; after that Dependabot follows `main` (`.github/dependabot.yml` no longer names a branch).

```sh
for n in 448 297 477 478; do gh pr edit "$n" --base main; done
```

Done when `gh pr list --base improvements` prints nothing.

## 2. Apply the ruleset and retire `improvements`

`just rulesets` pushes `.github/rulesets/*.json` to GitHub. It needs admin `gh` auth, and agents don't run it unattended. Run it once the PR that retires `improvements` has merged. The `improvements` ruleset has no JSON any more, so `just rulesets` leaves it in place, and it blocks deleting the branch: delete the ruleset first, then the branch.

```sh
git switch main && git pull --ff-only
just rulesets            # prints "updated: Merge gate: main (<id>)"
id=$(gh api repos/rlarson20/Volta/rulesets --jq '.[] | select(.name == "Merge gate: improvements") | .id')
gh api -X DELETE "repos/rlarson20/Volta/rulesets/$id"
git push origin --delete improvements
git branch -D improvements   # the local copy, if you have one
```

Done when `gh api repos/rlarson20/Volta/rulesets --jq '.[].name'` prints only `Merge gate: main` and `git ls-remote --heads origin improvements` prints nothing.

## 3. On the M2

These are the `needs-mac` issues, plus the ones that make speed claims. Each issue's "Done when" is the bar. The commands are where to start.

| Issue | Start with |
| --- | --- |
| #312 Metal fast-math probes | Write one CPU-vs-GPU probe per bullet at the extreme inputs, then `VOLTA_REQUIRE_GPU=1 just test-gpu` |
| #327 ASan over Accelerate | `RUSTFLAGS="-Zsanitizer=address" cargo +nightly-2026-11-01 test -Zbuild-std --target aarch64-apple-darwin --features accelerate --lib --tests -- --skip compile_fail` |
| #172 readback timeout panics | Decide the error shape first (it fits #66). #437 already makes each poll wait on its own submission, but the timeout still panics |
| #188 autocast fixture drift | `just torch-fixtures`, then `cargo test --test torch_ref_autocast`, measured with the test's own metric |
| #214 pairwise vs sequential sum | `just bench-cpu` at `6d0fe09` and on `main` (sizes are in the issue) |
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

## 5. Set up automated releases (#481)

It needs repository admin and your crates.io login, so agents can't do it. Follow "One-time setup" in `dev-docs/RELEASING.md`: create the `release` environment with you as required reviewer, add the crates.io trusted publisher (`release.yml`, environment `release`), and install git-cliff.

Done when `gh api repos/rlarson20/Volta/environments/release --jq '[.protection_rules[].type]'` lists `required_reviewers` and crates.io shows the trusted publisher.

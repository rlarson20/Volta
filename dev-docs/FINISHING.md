# Finishing locally

RJ's runbook for the steps that need his Mac, his `gh` admin rights, or his decision. It was written on 2026-11-20, after the serial tail. Status lives in the linked issues, not here. Each section says when it is done, and the file is deleted once every section is.

## 1. Merge the open PRs

### The serial-tail stack (#432 → #442)

There are nine stacked PRs. Each one is based on the branch before it, and only #432 targets `improvements`. `improvements` allows squash merges only (`.github/rulesets/improvements.json`), so after each squash the next branch still carries the original commits. Rebasing it onto the squashed commit drops them. Do that once per PR, from the bottom up.

Metal is not on the PR gate (`DECISIONS.md`, "macOS checks move off the PR gate"). Before the first merge, run the top of the stack on the M2 once. It contains every PR below it:

```sh
git fetch origin && git switch --detach origin/fix/116-debug
VOLTA_REQUIRE_GPU=1 just test-full
```

Then repeat this loop for each PR, from the bottom up. Don't `git fetch --prune` until the stack is merged: the rebase needs the old tip of each merged branch, and GitHub deletes it on merge.

```sh
# The stack, bottom first: PR number and branch.
stack=(
  "432 fix/356-opaque-storage"     "434 fix/268-backward-storages"
  "435 fix/377-structured-errors"  "436 fix/394-io-volta-result"
  "438 fix/100-rank-zero"          "439 fix/65-atomic-load"
  "440 fix/133-layer-try-new"      "441 fix/142-panic-messages"
  "442 fix/116-debug"
)
git fetch origin                   # once, before the first merge

# For i = 0, 1, ...: squash-merge stack[i] on GitHub (CI green), then:
i=0
read -r _ merged <<<"${stack[$i]}"; read -r next_pr next <<<"${stack[$((i+1))]}"
git fetch origin improvements
git rebase --onto origin/improvements "origin/$merged" "$next"   # replays only next's commits
git push --force-with-lease origin "$next"
gh pr edit "$next_pr" --base improvements
```

Each `rebase --onto` replays commits whose base tree is identical to the squashed commit, so it should apply cleanly. If one stops on a conflict, the squash differs from the branch: run `git rebase --abort` and ask an agent session to look. An agent can also run this loop for you after each merge. Merging is the only step that needs you.

Done when #442 is merged.

### The independent PRs

These are based on `improvements`, not on the stack. Merge them after the stack and update each one from `improvements` first: #399 and #398 touch the README, and the stack edits it too.

| PR for | What to check before merging |
| --- | --- |
| #399 README claims | The caution banner ("not in a spot that I'm proud of") is left for you. Replace it when you decide to |
| #398 STABILITY.md | The policies are proposals: MSRV bump = minor, one-minor deprecation, no bitwise numerical promise. Edit what you disagree with |
| #443 minimal-versions | Nothing; `just minimal-versions` passed in the cloud |
| #433 GPU placeholders | `VOLTA_REQUIRE_GPU=1 just test-repeat-threads "--test gpu_optimizer_threads" 30` on the M2 |
| #437 readback polls | `VOLTA_REQUIRE_GPU=1 just test-repeat-threads "--test gpu_readback_threads" 30` on the M2. The race never reproduced in the cloud, so Metal is the better test |
| #397 semver job | Runs only on PRs into `main` and by hand: try it once with `gh workflow run semver.yml --ref <branch>` |

Done when all six are merged or closed.

## 2. Apply the rulesets

`just rulesets` pushes `.github/rulesets/*.json` to GitHub. It needs admin `gh` auth, and agents don't run it unattended. Run it once after section 1, so that the rulesets match whatever the merged PRs changed:

```sh
git switch improvements && git pull
just rulesets            # prints "updated: <name> (<id>)" per ruleset
```

Done when it prints `updated` for both `improvements` and `main`.

## 3. On the M2

These are the `needs-mac` issues, plus the ones that make speed claims. Each issue's "Done when" is the bar. The commands are where to start.

| Issue | Start with |
| --- | --- |
| #202 naga dev-dependency to wgpu 30's version | Bump `naga` in `[dev-dependencies]`, check that `cargo tree -d` shows one naga, then `VOLTA_REQUIRE_GPU=1 just test-gpu` |
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

Nothing goes to `main` until you say so. When you do, open the PR from `improvements` into `main`. It runs `macos (stable)` and the semver workflow. A break the semver check reports is expected for 0.x, but the changelog must list it. The tag and changelog plan is in the project's `volta-tracking/release-plan.md`.

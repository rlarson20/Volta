# Pull request workflow (local)

How a change gets from your machine into `improvements`, why the repo works that way, and the three `just` recipes that make it easy. `dev-docs/CLOUD.md` § Pull requests holds the rules for agent sessions; this explains the reasoning.

## Why every change goes through a PR

The rulesets in `.github/rulesets/` (applied with `just rulesets`) make a pull request the only way to change `improvements` or `main`: no direct push, no force push, no deletion. Even working alone, that buys four things:

- **CI runs before the change lands.** The four required checks must pass before the merge button works. A direct push puts a broken commit on the branch before anyone knows.
- **A record.** The description and `Fixes #N` tie the change to its issue, and merging closes the issue. `git blame` leads to a commit, and the commit to the PR that says why.
- **A review point.** Reading your own diff on GitHub catches things your editor hides. It is also where you review anyone else's work.
- **One unit to undo.** `improvements` is squash-only, so a bad PR is one `git revert`.

The ruleset lets the repo admin bypass the checks, but `bypass_mode: "pull_request"` means only by merging a PR, never by pushing. That is why a push from `improvements` is refused.

## The loop

```
git switch -c type/short-desc origin/improvements   # branch first, from the remote tip
# ... edit, git add, git commit (hooks run) ...
just pr          # push the branch and open the PR (prefilled, opens your editor)
just prs         # your open PRs; checks on this branch's PR
# CI green -> squash-merge on GitHub
just sync        # fast-forward improvements, delete branches it already holds
```

Treat local `improvements` as a read-only mirror of the remote: only ever `git pull --ff-only` it, and never commit on it.

### When you committed on `improvements` anyway

`just pr` handles it, from staged changes or from commits already made. It commits anything staged, then does by hand what you would otherwise type:

```
git branch chore/my-change             # the commits now also live on a new branch
git reset --keep origin/improvements   # local improvements goes back to the remote tip
git switch chore/my-change
git push -u origin chore/my-change
gh pr create --base improvements       # fill in the template
```

The reset is safe only because the branch already holds the commits, and `--keep` (unlike `--hard`) refuses to run if it would discard uncommitted work. The script checks everything first and changes nothing if a check fails. It names the branch from the commit subject (`chore: fix x` becomes `chore/fix-x`); pass a name to choose your own: `just pr fix/my-branch`. `Fixes #N` lines come from `Fixes`/`Closes`/`Resolves #N` in the commit messages. The `<details>` block is for agent write-ups: delete it for a PR you wrote yourself.

### Without an editor

`just pr` opens your editor twice by default: for the commit message of staged changes, and for the PR description. For a script or an agent session, flags replace both (`scripts/pr.sh --help`):

```
just pr -m "docs: fix x" --body-file /path/body.md docs/fix-x
```

- `-m, --message MSG`: commit message for staged changes.
- `--body-file FILE`: use FILE as the PR description, verbatim. The template prefill is skipped, so the file should follow `.github/PULL_REQUEST_TEMPLATE.md` itself.
- `--no-edit`: keep the prefilled description (opener, `Fixes` lines, template) without opening the editor. Check it first: the checklist boxes are unchecked and the opener is the commit body or subjects.
- `--title TITLE`: override the PR title (default: the first commit's subject).

All arguments are checked before anything changes, so a missing `--body-file` leaves the branch and `improvements` as they were.

## Why `just sync` asks GitHub

Squash merging replaces your branch's commits with one new commit, so nothing links your branch to `improvements` and `git branch --merged` reports none of them as merged. `just sync` (`scripts/sync.sh`) deletes a local branch when one of these holds:

- **ancestor:** its tip is already in `origin/improvements` (merge commit, fast-forward, or a stacked PR whose lower branch merged), with or without a PR.
- **squash:** a PR from it merged into `improvements` and that PR's final commit is exactly the local tip.
- **content:** a PR from it merged and reached `improvements`, the branch's files are identical to that PR's final commit, and every non-merge commit the branch has beyond it matches a commit in `improvements` by patch. This covers a local `Merge branch 'lower' into upper` that moved the tip without changing the content.

Everything else is kept, with the reason printed: no PR and unmerged commits, a merged PR but commits or changes beyond it, an open PR, or a PR closed without merging. Commits you added after the merge are never lost. Run `just sync dry-run` first to see what it would delete, and `just test-sync` after changing the script. After a squash merge, start the next change from a new branch; don't keep working on the merged one.

## Updating a PR branch

If another PR merged first and yours conflicts, update your branch from `improvements`. Locally, rebase and push with the lease:

```
git fetch origin && git rebase origin/improvements
git push --force-with-lease
```

`--force-with-lease` overwrites the remote branch only if it is still what you last fetched, so you can't wipe out a push you haven't seen (plain `--force` would). Only ever force-push your own PR branch; the rulesets block it for `improvements` and `main`.

## Two branches, and a trade-off

`improvements` is the integration branch, where finished PRs collect (squash: one line of history per PR). `main` is the release branch: `improvements` merges into it when the merge gate in `CLAUDE.md` closes, with a merge commit so a phase keeps its history.

The checks are not "strict" (`strict_required_status_checks_policy: false`): a PR need not be up to date with `improvements` to merge. Two PRs that are each green can still break once both land. Strict mode prevents that at the cost of updating and re-running CI on every PR after each merge. Turn it on in the rulesets (and `just rulesets`) if it ever bites.

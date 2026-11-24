#!/usr/bin/env bash
# Checks scripts/restack.sh against a scratch repo and a stub `gh`: a three-branch stack,
# squash-merged one level at a time, with each merged branch deleted on the remote and
# pruned locally (as GitHub's delete-on-merge and `git fetch --prune` do). Changes nothing
# outside a temporary directory.
#
#   scripts/test-restack.sh
set -euo pipefail

script=$(cd "$(dirname "$0")" && pwd)/restack.sh
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

export GIT_CONFIG_GLOBAL=$tmp/gitconfig GIT_CONFIG_NOSYSTEM=1
git config --global user.name test
git config --global user.email test@example.invalid
git config --global init.defaultBranch main
git config --global advice.detachedHead false

# Stub gh: PR state per head branch in $tmp/gh/state/<branch>, number in .../num/<branch>;
# `pr edit` appends to $tmp/gh/edits.
mkdir -p "$tmp/bin" "$tmp/gh/state" "$tmp/gh/num"
cat >"$tmp/bin/gh" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
d=$GH_STUB_DIR
head=""; state=all; edit=""
while [[ $# -gt 0 ]]; do
	case $1 in
	--head) head=$2; shift ;;
	--state) state=$2; shift ;;
	edit) edit=$2; shift ;;
	esac
	shift
done
if [[ -n $edit ]]; then echo "$edit" >>"$d/edits"; exit 0; fi
s=$(cat "$d/state/$head" 2>/dev/null || echo NONE)
if [[ $state == open ]]; then
	[[ $s == OPEN ]] && cat "$d/num/$head"
	exit 0
fi
echo "$s"
STUB
chmod +x "$tmp/bin/gh"
export PATH="$tmp/bin:$PATH" GH_STUB_DIR="$tmp/gh"

git init --quiet --bare "$tmp/origin.git"
git clone --quiet "$tmp/origin.git" "$tmp/work" 2>/dev/null
cd "$tmp/work"

n=0
commit() { # file
	n=$((n + 1))
	echo "$n" >"$1"
	git add "$1"
	git commit --quiet -m "$1 $n"
}
fail() {
	echo "FAIL: $*" >&2
	exit 1
}
pr() { # branch state number
	echo "$2" >"$tmp/gh/state/$1"
	echo "$3" >"$tmp/gh/num/$1"
}
# Squash-merge $1 into main on the "remote", delete it there, prune locally.
squash_merge() {
	git switch --quiet main
	git pull --quiet --ff-only origin main
	git merge --quiet --squash "origin/$1" >/dev/null
	git commit --quiet -m "$1 (squashed)"
	git push --quiet origin main
	git push --quiet origin --delete "$1"
	git fetch --quiet --prune origin
	git branch --quiet -D "$1" 2>/dev/null || true
	echo MERGED >"$tmp/gh/state/$1"
}

commit base
git push --quiet -u origin main
git switch --quiet -c a && commit a1 && commit a2
git switch --quiet -c b && commit b1
git switch --quiet -c c && commit c1 && commit c2
git push --quiet origin a b c
git switch --quiet main
# The script lives on main only, as in the real repo, so `next` switching to a stack
# branch removes it from the working tree mid-run.
mkdir scripts && cp "$script" scripts/restack.sh
git add scripts && git commit --quiet -m "add restack" && git push --quiet origin main
script=$PWD/scripts/restack.sh
pr a OPEN 1 && pr b OPEN 2 && pr c OPEN 3

bash "$script" save a b c >/dev/null

# Nothing merged yet: next does nothing and says to merge a.
out=$(bash "$script" next)
[[ $out == *"squash-merge the PR for a"* ]] || fail "before any merge: $out"

squash_merge a

# Signing is on and gpg fails: next stops before touching anything.
printf '#!/bin/sh\nexit 2\n' >"$tmp/bin/badgpg" && chmod +x "$tmp/bin/badgpg"
git config commit.gpgsign true && git config gpg.program "$tmp/bin/badgpg"
before=$(git rev-parse origin/b)
bash "$script" next >/dev/null 2>&1 && fail "next should refuse when signing fails"
[[ $(git rev-parse origin/b) == "$before" ]] || fail "b was pushed despite the signing failure"
git config --unset commit.gpgsign && git config --unset gpg.program
git switch --quiet main

git show-ref --verify --quiet refs/remotes/origin/a && fail "origin/a should be pruned"
bash "$script" next >/dev/null
# b now holds exactly its own commit on top of the squashed a.
[[ $(git rev-list --count origin/main..origin/b) == 1 ]] || fail "b should carry 1 commit"
[[ $(git log -1 --format=%s origin/b) == "b1 4" ]] || fail "b's commit"
[[ $(cat "$tmp/gh/edits") == 2 ]] || fail "PR 2 should be retargeted"

# b is not merged yet: next waits.
out=$(bash "$script" next)
[[ $out == *"squash-merge the PR for b"* ]] || fail "waiting on b: $out"

# A conflict: main changes c1's file before b merges. next aborts the rebase,
# leaves no rebase in progress, pushes nothing, then succeeds once the conflict is gone.
git switch --quiet main && git pull --quiet --ff-only origin main
echo conflicting >c1 && git add c1 && git commit --quiet -m "conflict" && git push --quiet origin main
squash_merge b
before=$(git rev-parse origin/c)
bash "$script" next >/dev/null 2>&1 && fail "next should fail on the conflict"
[[ -d $(git rev-parse --git-path rebase-merge) || -d $(git rev-parse --git-path rebase-apply) ]] && fail "left mid-rebase"
git fetch --quiet origin
[[ $(git rev-parse origin/c) == "$before" ]] || fail "c was pushed despite the conflict"
git switch --quiet main && git rm --quiet c1 && git commit --quiet -m "drop conflict" && git push --quiet origin main
bash "$script" next >/dev/null 2>&1
[[ $(git rev-list --count origin/main..origin/c) == 2 ]] || fail "c should carry 2 commits"
# The rebased c has the original stack top's files, plus only what main added.
git diff --quiet "refs/restack/tip/c" origin/c -- . ':!scripts' || fail "c's content changed"

out=$(bash "$script" next)
[[ $out == *"stack done"* ]] || fail "after the top: $out"

bash "$script" clean >/dev/null
[[ -z $(git for-each-ref refs/restack/) ]] || fail "clean left refs"
echo "restack: all checks passed"

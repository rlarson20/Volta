#!/usr/bin/env bash
# Rebases a stack of PRs after each squash merge into improvements (dev-docs/FINISHING.md).
# `improvements` takes squash merges only, so once the bottom PR merges, the next branch
# still carries the bottom's original commits. This replays only its own commits onto
# improvements, force-pushes it (with lease) and retargets its PR.
#
#   scripts/restack.sh save <branch>...   once, before the first merge, bottom branch first
#   scripts/restack.sh next               after each squash merge on GitHub
#   scripts/restack.sh status             what `next` would do
#   scripts/restack.sh clean              forget the stack (refs/restack/*)
#
# `save` records each branch's current tip as refs/restack/tip/<branch>. GitHub deletes a
# merged branch, and `git fetch --prune` (or `just sync`) would then drop origin/<branch>,
# but the saved ref keeps the old tip, which is what the next branch was built on. Re-run
# `save` if a stack branch gets new commits before it merges.
#
# Runs under bash whatever your login shell is (zsh arrays start at 1, bash's at 0). Plain
# bash 3.2 (macOS /bin/bash) is enough. Needs `gh`, and a clean working tree for `next`.
set -euo pipefail

# `next` switches the working tree to a stack branch, which may not contain this script (or
# holds an older one), and bash reads a script as it runs. Run from a private copy instead.
if [[ -z ${RESTACK_RUNNING_COPY:-} ]]; then
	copy=$(mktemp "${TMPDIR:-/tmp}/restack.XXXXXX")
	cp "$0" "$copy"
	RESTACK_RUNNING_COPY=$copy exec bash "$copy" "$@"
fi
trap 'rm -f "$RESTACK_RUNNING_COPY"' EXIT

die() {
	echo "restack: $*" >&2
	exit 1
}

gitdir=$(git rev-parse --git-common-dir)
order_file="$gitdir/restack-order"
base=improvements

# The saved stack, bottom first, into the array `stack`.
load() {
	[[ -f $order_file ]] || die "no saved stack; run: scripts/restack.sh save <branch>... (bottom first)"
	stack=()
	while IFS= read -r line; do
		[[ -n $line ]] && stack+=("$line")
	done <"$order_file"
	[[ ${#stack[@]} -ge 2 ]] || die "the saved stack has fewer than two branches"
}

# The state (OPEN, MERGED, CLOSED) of the newest PR whose head is $1, or NONE.
pr_state() {
	local s
	s=$(gh pr list --state all --head "$1" --json state --jq '.[0].state // "NONE"')
	echo "${s:-NONE}"
}

# The number of the open PR whose head is $1.
pr_number() {
	gh pr list --state open --head "$1" --json number --jq '.[0].number // empty'
}

# Sets `i` to the index of the next branch to rebase (whose parent stack[i-1] is merged and
# which is not done yet), or prints why there is none and returns 1.
find_next() {
	local k parent child state
	for ((k = 1; k < ${#stack[@]}; k++)); do
		child=${stack[$k]}
		parent=${stack[$((k - 1))]}
		git show-ref --verify --quiet "refs/restack/done/$child" && continue
		state=$(pr_state "$parent")
		if [[ $state != MERGED ]]; then
			echo "next: squash-merge the PR for $parent on GitHub (it is $state), then run next again"
			return 1
		fi
		i=$k
		return 0
	done
	echo "stack done: every branch is rebased. Clean up with: scripts/restack.sh clean"
	return 1
}

case "${1:-}" in
save)
	shift
	[[ $# -ge 2 ]] || die "save needs at least two branches, bottom first"
	git fetch --quiet origin "$base" "$@"
	for b in "$@"; do
		git update-ref "refs/restack/tip/$b" "refs/remotes/origin/$b"
		git update-ref -d "refs/restack/done/$b" 2>/dev/null || true
	done
	printf '%s\n' "$@" >"$order_file"
	for b in "$@"; do
		printf '  %s %s\n' "$(git rev-parse --short "refs/restack/tip/$b")" "$b"
	done
	echo "saved ${#} branches. Squash-merge the bottom PR, then run: scripts/restack.sh next"
	;;
status)
	load
	find_next && echo "next: rebase ${stack[$i]} off ${stack[$((i - 1))]} onto $base"
	;;
next)
	load
	find_next || exit 0
	child=${stack[$i]}
	parent=${stack[$((i - 1))]}
	[[ -z $(git status --porcelain --untracked-files=no) ]] || die "the working tree has changes; commit or stash them first"

	# A rebase re-signs every commit it replays. Check signing works first, so a bad passphrase
	# fails here instead of leaving a half-done rebase.
	if [[ $(git config --bool commit.gpgsign || true) == true ]]; then
		fmt=$(git config gpg.format || echo openpgp)
		if [[ $fmt == openpgp ]]; then
			key=$(git config user.signingkey || true)
			echo restack-sign-check | "$(git config gpg.program || echo gpg)" ${key:+-u "$key"} --clearsign >/dev/null ||
				die "commit.gpgsign is on and gpg could not sign (passphrase? pinentry?); fix that and run next again"
		fi
	fi

	git fetch --quiet origin "$base" "$child"
	old=$(git rev-parse "refs/restack/tip/$parent")
	child_tip=$(git rev-parse "refs/remotes/origin/$child")
	git merge-base --is-ancestor "$old" "$child_tip" ||
		die "$child does not contain $parent's saved tip $(git rev-parse --short "$old"); was $parent changed after save?"

	echo "replaying $child's own commits onto origin/$base:"
	git log --oneline "$old..$child_tip"
	git switch --quiet -C "$child" "$child_tip"
	if ! git rebase --onto "refs/remotes/origin/$base" "$old"; then
		# Never leave the repo mid-rebase: undo it, and leave the branch at its remote tip.
		git rebase --abort 2>/dev/null || true
		git switch --quiet -C "$child" "$child_tip"
		echo "restack: the rebase of $child failed (a conflict, or a commit that could not be written or signed) and was aborted; nothing was pushed." >&2
		echo "  To do it by hand: git switch $child && git rebase --onto origin/$base $(git rev-parse --short "$old")" >&2
		echo "  then: git push --force-with-lease=$child:$child_tip origin $child && gh pr edit <PR> --base $base" >&2
		exit 1
	fi
	git push --quiet --force-with-lease="$child:$child_tip" origin "$child"
	num=$(pr_number "$child")
	if [[ -n $num ]]; then
		gh pr edit "$num" --base "$base" >/dev/null
		echo "pushed $child and retargeted PR #$num to $base"
	else
		echo "pushed $child (no open PR found for it to retarget)"
	fi
	git update-ref "refs/restack/done/$child" HEAD
	if ((i + 1 < ${#stack[@]})); then
		echo "next: once CI is green, squash-merge the PR for $child, then run next again"
	else
		echo "that was the top of the stack: squash-merge the PR for $child, then run: scripts/restack.sh clean"
	fi
	;;
clean)
	git for-each-ref --format='%(refname)' refs/restack/ | while IFS= read -r r; do git update-ref -d "$r"; done
	rm -f "$order_file"
	echo "forgot the saved stack"
	;;
*)
	sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
	exit 2
	;;
esac

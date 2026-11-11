#!/usr/bin/env bash
# After PRs merge: fast-forward `improvements` and delete local branches it already holds
# (see dev-docs/WORKFLOW.md, "Why `just sync` asks GitHub").
#
#   scripts/sync.sh [dry-run]
#
# A branch is deleted when one of these holds, checked in order:
#   ancestor  its tip is an ancestor of origin/improvements (merge commit, fast-forward, or a
#             stacked PR whose lower branch merged), whether or not it had a PR;
#   squash    a PR from it merged into improvements and that PR's final commit is the local tip;
#   content   a PR from it merged and its content reached improvements (base improvements, or
#             its final commit is an ancestor of origin/improvements), the branch's tree equals
#             that final commit, and every non-merge commit the branch has beyond it matches a
#             commit in improvements by patch (`git patch-id`). This is the stacked-PR case where
#             a local `Merge branch 'lower' into upper` moved the tip but not the content.
# Every other branch is kept, and the message says why.
set -euo pipefail

base=improvements
remote=origin/$base

die() {
	echo "sync: $*" >&2
	exit 1
}

dry=false
case "${1:-}" in
"") ;;
dry-run) dry=true ;;
*) die "usage: just sync [dry-run]" ;;
esac

command -v gh >/dev/null || die "the GitHub CLI (gh) is not installed"
command -v jq >/dev/null || die "jq is not installed"
cd "$(git rev-parse --show-toplevel)"

git switch --quiet "$base"
git pull --quiet --ff-only || die "local $base has commits origin lacks; run 'just pr' to move them to a PR branch"
git fetch --prune --quiet

# Squash merging leaves no ancestor link, so git alone cannot see those merges: ask GitHub.
# All states and bases: an open or unmerged PR explains a kept branch, and a stacked PR merged
# into a lower branch, not into improvements.
prs=$(gh pr list --state all --limit 1000 --json number,state,baseRefName,headRefName,headRefOid)

delete() { # branch rule
	if $dry; then echo "would delete: $1 ($2)"; else git branch -D --quiet "$1" && echo "deleted: $1 ($2)"; fi
}

# Patch ids of every non-merge commit in improvements, computed once, on first use.
landed=""
landed_ids() {
	[[ -n $landed ]] || landed=$(git log --no-merges -p "$remote" | git patch-id --stable | cut -d' ' -f1 | sort -u)
	echo "$landed"
}

# True when each non-merge commit in from..to has a patch-equivalent commit in improvements.
# A commit with no patch (empty) has no id and counts as missing.
extras_landed() { # from to
	local c id
	for c in $(git rev-list --no-merges "$1..$2"); do
		id=$(git show "$c" | git patch-id --stable | cut -d' ' -f1)
		[[ -n $id ]] && command grep -qxF "$id" <<<"$(landed_ids)" || return 1
	done
}

# Make a PR's final commit available locally; it may be gone from the remote branch.
have_commit() { # oid pr-number
	git cat-file -e "$1^{commit}" 2>/dev/null ||
		{ git fetch --quiet origin "refs/pull/$2/head" 2>/dev/null && git cat-file -e "$1^{commit}" 2>/dev/null; }
}

for b in $(git for-each-ref --format='%(refname:short)' refs/heads); do
	[[ $b == "$base" || $b == main ]] && continue
	tip=$(git rev-parse "refs/heads/$b")

	if git merge-base --is-ancestor "$tip" "$remote"; then
		delete "$b" "already in $remote"
		continue
	fi

	mine=$(jq -c --arg b "$b" '[.[] | select(.headRefName == $b)]' <<<"$prs")
	if [[ $(jq length <<<"$mine") -eq 0 ]]; then
		echo "kept: $b (no PR, and it has commits $remote lacks)"
		continue
	fi

	squash=$(jq -r --arg t "$tip" --arg base "$base" \
		'[.[] | select(.state == "MERGED" and .baseRefName == $base and .headRefOid == $t)][0].number // empty' <<<"$mine")
	if [[ -n $squash ]]; then
		delete "$b" "PR #$squash merged at this commit"
		continue
	fi

	matched=""
	while read -r n into oid; do
		[[ -n $n ]] || continue
		have_commit "$oid" "$n" || continue
		[[ $into == "$base" ]] || git merge-base --is-ancestor "$oid" "$remote" || continue
		if git diff --quiet "$oid" "$tip" && extras_landed "$oid" "$tip"; then
			matched=$n
			break
		fi
	done < <(jq -r '.[] | select(.state == "MERGED") | "\(.number) \(.baseRefName) \(.headRefOid)"' <<<"$mine")
	if [[ -n $matched ]]; then
		delete "$b" "same content as merged PR #$matched, and every commit beyond it is in $remote"
		continue
	fi

	open=$(jq -r '[.[] | select(.state == "OPEN") | "#\(.number)"] | join(", ")' <<<"$mine")
	merged=$(jq -r '[.[] | select(.state == "MERGED") | "#\(.number)"] | join(", ")' <<<"$mine")
	if [[ -n $open ]]; then
		echo "kept: $b (PR $open still open)"
	elif [[ -n $merged ]]; then
		echo "kept: $b (PR $merged merged, but the branch has commits or changes beyond it)"
	else
		echo "kept: $b (PR closed without merging, and it has commits $remote lacks)"
	fi
done

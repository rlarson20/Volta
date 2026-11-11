#!/usr/bin/env bash
# Turn finished work into a pull request against `improvements` (see dev-docs/WORKFLOW.md).
#
#   scripts/pr.sh [branch-name]
#
# Run it on local `improvements` with commits ahead of origin (the ones the ruleset won't let you
# push) and/or staged changes: the staged changes are committed, then the commits move to a new
# branch and local `improvements` goes back to origin's tip. Run it on a feature branch and it
# just pushes that branch and opens the PR. Every check runs before anything changes.
set -euo pipefail

base=improvements
remote=origin/$base

die() {
	echo "pr: $*" >&2
	exit 1
}

command -v gh >/dev/null || die "the GitHub CLI (gh) is not installed"
cd "$(git rev-parse --show-toplevel)"

branch=$(git symbolic-ref --quiet --short HEAD) || die "HEAD is detached; switch to a branch first"
[[ $branch == main ]] && die "refusing to open a PR from main"
git fetch --quiet origin "$base"

# Unstaged edits to tracked files make pre-commit's stash and the branch move unsafe.
git diff --quiet || die "unstaged changes to tracked files; stage them (git add) or stash them first"

if ! git diff --cached --quiet; then
	echo "pr: committing the staged changes"
	git commit
fi

if [[ $branch == "$base" ]]; then
	git merge-base --is-ancestor "$remote" HEAD ||
		die "local $base has diverged from $remote; rebase onto it first (git rebase $remote)"
else
	[[ $# -eq 0 ]] || die "already on branch '$branch'; it is the PR branch, so a name argument makes no sense"
fi

range=$remote..HEAD
count=$(git rev-list --count "$range")
((count > 0)) || die "nothing to open a PR for: $branch has no commits ahead of $remote"

subjects=$(git log --reverse --format=%s "$range")
title=${subjects%%$'\n'*}

slugify() {
	printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//' |
		cut -c1-50 | sed -E 's/-+$//'
}

# `chore(lints): drop clones (#171)` -> `chore/drop-clones-171`; no `type:` prefix -> `change/...`
branch_name_for() {
	local re='^([a-z]+)(\([^)]*\))?!?: (.+)$'
	if [[ $1 =~ $re ]]; then
		echo "${BASH_REMATCH[1]}/$(slugify "${BASH_REMATCH[3]}")"
	else
		echo "change/$(slugify "$1")"
	fi
}

if [[ $branch == "$base" ]]; then
	name=${1:-$(branch_name_for "$title")}
	git check-ref-format --branch "$name" >/dev/null 2>&1 || die "'$name' is not a valid branch name; pass one: just pr type/short-desc"
	git show-ref --verify --quiet "refs/heads/$name" && die "branch '$name' already exists locally"
	git ls-remote --exit-code --heads origin "$name" >/dev/null 2>&1 && die "branch '$name' already exists on origin"

	echo "pr: moving $count commit(s) from $base to $name"
	git branch "$name"
	# --keep, unlike --hard, refuses rather than discard anything uncommitted.
	git reset --keep "$remote"
	git switch "$name"
else
	name=$branch
fi

git push -u origin "$name"

if existing=$(gh pr view "$name" --json url --jq .url 2>/dev/null); then
	echo "pr: $name already has a pull request: $existing (pushed the new commits to it)"
	exit 0
fi

# Prefill the template: the commit body (or the subjects) as the opener, a `Fixes #N` line per
# issue the commit messages close, then the template from its `Fixes #` line down.
template=.github/PULL_REQUEST_TEMPLATE.md
body=$(mktemp -t pr-body.XXXXXX)
trap 'rm -f "$body"' EXIT

fixes=$(git log --format=%B "$range" | grep -ioE '(fix(es|ed)?|close[sd]?|resolve[sd]?):? #[0-9]+' |
	grep -oE '#[0-9]+' | sort -u || true)
{
	if ((count == 1)); then
		opener=$(git log -1 --format=%b "$range" |
			grep -viE '^(co-authored-by|signed-off-by|(fix(es|ed)?|close[sd]?|resolve[sd]?):? #[0-9]+)' || true)
		[[ -n ${opener//[[:space:]]/} ]] && printf '%s\n' "$opener" || printf '%s\n' "$title"
	else
		while IFS= read -r s; do printf -- '- %s\n' "$s"; done <<<"$subjects"
	fi
	echo
	if [[ -f $template ]]; then
		while IFS= read -r line; do
			if [[ $line == 'Fixes #' ]]; then
				if [[ -n $fixes ]]; then
					while IFS= read -r n; do echo "Fixes $n"; done <<<"$fixes"
				else
					echo 'Fixes #'
				fi
			else
				printf '%s\n' "$line"
			fi
		done < <(sed -n '/^Fixes #$/,$p' "$template")
	fi
} >"$body"

# Same editor resolution git uses for commit messages.
sh -c "$(git var GIT_EDITOR) \"\$@\"" _ "$body"
if [[ -z $(tr -d '[:space:]' <"$body") ]]; then
	echo "pr: empty description, not creating the PR; $name is pushed. Retry with:" >&2
	echo "    gh pr create --base $base --head $name --fill" >&2
	exit 1
fi

gh pr create --base "$base" --head "$name" --title "$title" --body-file "$body"

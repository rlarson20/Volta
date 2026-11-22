#!/usr/bin/env bash
# Print a "State of Volta" digest in Markdown, for a comment on the State of Volta issue (#456)
# (CLAUDE.md, "Review and status"). Read-only: it queries GitHub's search API and the local
# git log, and changes nothing.
#
#   scripts/status.sh [YYYY-MM-DD]
#
# The date is where the digest starts (default: 7 days ago). Needs `gh` signed in, and a
# checkout whose `origin/improvements` is fetched.
set -euo pipefail

repo=rlarson20/Volta
since=${1:-$(date -u -d '7 days ago' +%F 2>/dev/null || date -u -v-7d +%F)}
[[ $since =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || {
	echo "status: the date must be YYYY-MM-DD, got '$since'" >&2
	exit 1
}

# One search query, printed as a Markdown list; "- none" when it matches nothing.
list() {
	local out
	out=$(gh api -X GET search/issues -f q="repo:$repo $1" -f per_page=100 \
		--jq '.items[] | "- #\(.number) \(.title)"')
	if [[ -n $out ]]; then echo "$out"; else echo "- none"; fi
}

# Number of results for one search query.
count() {
	gh api -X GET search/issues -f q="repo:$repo $1" --jq .total_count
}

echo "## State of Volta, $(date -u +%F) (since $since)"
echo
echo "### Merged into improvements"
echo
list "is:pr is:merged base:improvements merged:>=$since"
echo
echo "### Design PRs (wait for RJ to read the diff)"
echo
list "is:pr is:open \"Review tier: design\" in:body"
echo
echo "### Open PRs"
echo
list "is:pr is:open"
echo
echo "### Blocked, or needs the Mac"
echo
list "is:issue is:open label:blocked"
list "is:issue is:open label:needs-mac"
echo
echo "### Milestones (open / closed issues)"
echo
for m in v0.4.0 v0.5.0 v1.0.0 Backlog; do
	echo "- $m: $(count "is:issue is:open milestone:$m") / $(count "is:issue is:closed milestone:$m")"
done
echo
echo "### Decisions and public surface touched since $since"
echo
git log --since="$since" --format='- %h %s' origin/improvements -- DECISIONS.md src/lib.rs Cargo.toml |
	grep . || echo "- none"
echo
echo "### Filed since $since"
echo
list "is:issue created:>=$since"

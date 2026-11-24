#!/usr/bin/env bash
# Checks scripts/sync.sh against a scratch repo and a stub `gh`: one branch per delete rule,
# and one per reason to keep a branch. Changes nothing outside a temporary directory.
#
#   scripts/test-sync.sh
set -euo pipefail

script=$(cd "$(dirname "$0")" && pwd)/sync.sh
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

export GIT_CONFIG_GLOBAL=$tmp/gitconfig GIT_CONFIG_NOSYSTEM=1
git config --global user.name test
git config --global user.email test@example.invalid
git config --global init.defaultBranch main
git config --global advice.detachedHead false

git init --quiet --bare "$tmp/origin.git"
git clone --quiet "$tmp/origin.git" "$tmp/work" 2>/dev/null
cd "$tmp/work"

n=0
commit() { # file [message]
	n=$((n + 1))
	echo "$n" >"$1"
	git add "$1"
	GIT_COMMITTER_DATE="2026-01-01T00:00:$(printf %02d "$n")" git commit --quiet -m "${2:-$1 $n}"
}
oid() { git rev-parse "$1"; }

commit base
git push --quiet -u origin main

# ancestor: fast-forwarded into main, no PR.
git switch --quiet -c ff-no-pr
commit ff
git switch --quiet main
git merge --quiet --ff-only ff-no-pr

# ancestor: stacked PR merged into a lower branch, which then merged into main.
git switch --quiet -c stack-lower
commit lower
git switch --quiet -c stack-upper
commit upper
git switch --quiet stack-lower
git merge --quiet --no-ff -m "merge #2" stack-upper
git switch --quiet main
git merge --quiet --no-ff -m "merge #1" stack-lower

# squash: PR squash-merged at the local tip.
git switch --quiet -c squashed
commit sq1
commit sq2
squashed_tip=$(oid squashed)
git switch --quiet main
git merge --quiet --squash squashed >/dev/null
git commit --quiet -m "squashed (#3)"

# kept: squash-merged, then a commit after the merge.
git switch --quiet -c post-merge
commit pm1
post_merge_head=$(oid post-merge)
git switch --quiet main
git merge --quiet --squash post-merge >/dev/null
git commit --quiet -m "post-merge (#4)"
git switch --quiet post-merge
commit pm2

# content: a local merge commit moved the tip off the PR head without changing the tree.
# The PR carried a rebased copy (dtype-lower-copy) of the lower branch's commit; the local
# branch also merged the original, so its tip and history differ but its tree does not.
git switch --quiet main
git switch --quiet -c dtype-lower
commit dl
git switch --quiet main
git switch --quiet -c dtype-upper
git cherry-pick --quiet dtype-lower >/dev/null
commit du
local_merge_head=$(oid dtype-upper)
git switch --quiet main
git merge --quiet --no-ff -m "dtype-upper (#5)" dtype-upper
lower_original=$(oid dtype-lower)
git branch --quiet -D dtype-lower # stands in for a lower branch already gone locally
git switch --quiet dtype-upper
git merge --quiet --no-ff -m "Merge branch 'dtype-lower' into dtype-upper" "$lower_original"
[[ $(oid dtype-upper) != "$local_merge_head" ]] || { echo "test setup: tip did not move" >&2; exit 1; }
git diff --quiet "$local_merge_head" dtype-upper || { echo "test setup: tree changed" >&2; exit 1; }

# kept: PR merged into a lower branch that never reached main (tip matches its head).
git switch --quiet main
git switch --quiet -c orphan-lower
commit ol
git switch --quiet -c orphan-upper
commit ou
orphan_upper_head=$(oid orphan-upper)
git switch --quiet orphan-lower
git merge --quiet --no-ff -m "merge #6" orphan-upper
git switch --quiet orphan-upper

# kept: open PR; no PR; closed without merging.
git switch --quiet main
git switch --quiet -c open-pr
commit op
open_head=$(oid open-pr)
git switch --quiet main
git switch --quiet -c no-pr
commit np
git switch --quiet main
git switch --quiet -c closed-pr
commit cp
closed_head=$(oid closed-pr)

git switch --quiet main
git push --quiet origin main
git branch --quiet -D orphan-lower # its PR is still open; not under test here

# Stub gh: answers `gh pr list` from a fixed list.
mkdir "$tmp/bin"
cat >"$tmp/bin/gh" <<EOF
#!/usr/bin/env bash
[[ "\$1 \$2" == "pr list" ]] || { echo "stub gh: unexpected \$*" >&2; exit 1; }
cat "$tmp/prs.json"
EOF
chmod +x "$tmp/bin/gh"
pr() { printf '{"number":%s,"state":"%s","baseRefName":"%s","headRefName":"%s","headRefOid":"%s"}' "$@"; }
{
	echo "["
	pr 1 MERGED main stack-lower "$(oid stack-lower)"
	echo ","
	pr 2 MERGED stack-lower stack-upper "$(oid stack-upper)"
	echo ","
	pr 3 MERGED main squashed "$squashed_tip"
	echo ","
	pr 4 MERGED main post-merge "$post_merge_head"
	echo ","
	pr 5 MERGED main dtype-upper "$local_merge_head"
	echo ","
	pr 6 MERGED orphan-lower orphan-upper "$orphan_upper_head"
	echo ","
	pr 7 OPEN main open-pr "$open_head"
	echo ","
	pr 8 CLOSED main closed-pr "$closed_head"
	echo "]"
} >"$tmp/prs.json"

export PATH=$tmp/bin:$PATH
fail=0
expect() { # output-file line
	if command grep -qxF "$2" "$1"; then echo "ok:   $2"; else echo "FAIL: missing '$2'"; fail=1; fi
}

before=$(git for-each-ref --format='%(refname:short)' refs/heads | sort)
bash "$script" dry-run >"$tmp/dry.out"
after=$(git for-each-ref --format='%(refname:short)' refs/heads | sort)
[[ $before == "$after" ]] || { echo "FAIL: dry-run deleted branches"; fail=1; }
expect "$tmp/dry.out" "would delete: ff-no-pr (already in origin/main)"

bash "$script" >"$tmp/run.out"
cat "$tmp/run.out"
echo
for line in \
	"deleted: ff-no-pr (already in origin/main)" \
	"deleted: stack-lower (already in origin/main)" \
	"deleted: stack-upper (already in origin/main)" \
	"deleted: squashed (PR #3 merged at this commit)" \
	"deleted: dtype-upper (same content as merged PR #5, and every commit beyond it is in origin/main)" \
	"kept: post-merge (PR #4 merged, but the branch has commits or changes beyond it)" \
	"kept: orphan-upper (PR #6 merged, but the branch has commits or changes beyond it)" \
	"kept: open-pr (PR #7 still open)" \
	"kept: no-pr (no PR, and it has commits origin/main lacks)" \
	"kept: closed-pr (PR closed without merging, and it has commits origin/main lacks)"; do
	expect "$tmp/run.out" "$line"
done

remaining=$(git for-each-ref --format='%(refname:short)' refs/heads | sort | tr '\n' ' ')
[[ $remaining == "closed-pr main no-pr open-pr orphan-upper post-merge " ]] ||
	{ echo "FAIL: remaining branches: $remaining"; fail=1; }
[[ $(wc -l <"$tmp/run.out") -eq 10 ]] || { echo "FAIL: expected 10 lines of output"; fail=1; }

exit $fail

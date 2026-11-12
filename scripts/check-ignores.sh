#!/usr/bin/env bash
# Checks the `#[ignore = "<defect ID>: <why>"]` convention (CLAUDE.md invariant 6):
#   1. every `#[ignore]` in src/, tests/, examples/, benches/ starts its reason with a defect ID
#      (`S21: `, `A7: `, `D6: `) or an issue number (`#60: `). A bare `#[ignore]` or a reason
#      without an ID fails.
#   2. every ID has an open issue: one whose title starts with it (`S2/S3/S4: ...`, the IDs
#      before the first ": "), or, for `#N`, issue N itself. A live ignore whose issue is closed
#      is stale: the fix landed and left the guard behind, or the issue closed with the defect
#      still there. Either way, remove the ignore or reopen the issue.
#
# Step 2 reads the issues with `gh api` (REST) and runs whenever that works. `--offline` skips
# it (pre-commit); otherwise a missing or unauthenticated `gh` skips it with a notice, and
# VOLTA_REQUIRE_GH=1 turns that skip into a failure (CI). The repository is $GITHUB_REPOSITORY
# when set (CI), else the owner/name in the `origin` URL.
#
# `--list` also prints each ignored test as `file:line  test  ID  issues` (issues are `-` when
# the issue check is skipped).
#
# Not caught: an ignore reason that starts on the line after `#[ignore =` (it fails step 1,
# loudly), and an `ignore` spelled through a macro.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

offline=0
list=0
for arg in "$@"; do
    case "$arg" in
    --offline) offline=1 ;;
    --list) list=1 ;;
    *)
        echo "usage: $0 [--offline] [--list]" >&2
        exit 2
        ;;
    esac
done

dirs=(src tests examples benches)
id_re='#[0-9]+|[A-Z]+[0-9]+'

# `file:line:text` for each `#[ignore ...]` or `cfg_attr(.., ignore ..)`, comment lines excluded.
hits=$(
    git grep -nE --untracked '#\[ignore([^A-Za-z0-9_]|$)|cfg_attr\(.*[^A-Za-z0-9_]ignore([^A-Za-z0-9_]|$)' -- "${dirs[@]}" |
        grep -vE '^[^:]+:[0-9]+:[[:space:]]*//' || true
)

bad=()
ids=() # "ID<TAB>file:line"
while IFS= read -r hit; do
    [ -n "$hit" ] || continue
    loc=$(printf '%s' "$hit" | cut -d: -f1,2)
    id=$(printf '%s' "$hit" | sed -nE "s/.*ignore[[:space:]]*=[[:space:]]*\"($id_re): .*/\\1/p")
    if [ -z "$id" ]; then
        bad+=("$hit")
    else
        ids+=("$id"$'\t'"$loc")
    fi
done <<<"$hits"

# `--list`: `file:line  test  ID  issues` for each ignored test. The test is the first `fn` at or
# after the attribute; the issues are filled in by check_issues, or "-" when it does not run.
test_name() {
    awk -v n="${1#*:}" 'NR >= n && match($0, /fn [A-Za-z0-9_]+/) { print substr($0, RSTART + 3, RLENGTH - 3); exit }' "${1%%:*}"
}
print_entry() { # loc id issues
    printf '  %s  %s  %s  %s\n' "$1" "$(test_name "$1")" "$2" "$3"
}

status=0
if [ ${#bad[@]} -gt 0 ]; then
    echo "#[ignore] without a defect ID (write #[ignore = \"S<n>: <why>\"] or #[ignore = \"#<issue>: <why>\"]):" >&2
    printf '  %s\n' "${bad[@]}" >&2
    status=1
fi

skip() {
    if [ "${VOLTA_REQUIRE_GH:-0}" = 1 ]; then
        echo "check-ignores: $1, and VOLTA_REQUIRE_GH=1" >&2
        exit 1
    fi
    echo "check-ignores: issue check skipped ($1)"
    list_unresolved
}

check_issues() {
    local repo=${GITHUB_REPOSITORY:-}
    if [ -z "$repo" ]; then
        repo=$(git remote get-url origin 2>/dev/null |
            sed -nE 's#^(https://|ssh://git@|git@)github\.com[/:]([^/]+/[^/]+)$#\2#p' | sed -E 's#(\.git)?/?$##')
    fi
    [ -n "$repo" ] || {
        skip "no GITHUB_REPOSITORY and no github.com origin"
        return
    }
    command -v gh >/dev/null || {
        skip "gh not installed"
        return
    }

    # number<TAB>state<TAB>kind<TAB>title, every issue and PR (the endpoint lists both). Pages
    # are walked by hand: an empty page ends the list.
    local rows="" page=1 out
    while :; do
        if ! out=$(gh api "repos/$repo/issues?state=all&per_page=100&page=$page" \
            --jq '.[] | [.number, .state, (if .pull_request then "pr" else "issue" end), .title] | @tsv' 2>&1); then
            skip "gh api failed: $(printf '%s' "$out" | head -n1)"
            return
        fi
        [ -n "$out" ] || break
        rows+="$out"$'\n'
        page=$((page + 1))
    done

    local missing=() stale=() entry id loc open closed
    for entry in "${ids[@]}"; do
        id=${entry%%$'\t'*}
        loc=${entry#*$'\t'}
        # Issues that carry the ID: the title's ID list (before the first ": ", split on "/"),
        # or the issue number for `#N`.
        matches=$(printf '%s' "$rows" | awk -F'\t' -v id="$id" '
            $3 != "issue" { next }
            {
                hit = 0
                if (id ~ /^#/) hit = ("#" $1 == id)
                else if (index($4, ": ") > 0) {
                    n = split(substr($4, 1, index($4, ": ") - 1), part, "/")
                    for (i = 1; i <= n; i++) if (part[i] == id) hit = 1
                }
                if (hit) print $1 "\t" $2
            }')
        open=$(printf '%s' "$matches" | awk -F'\t' '$2 == "open" { printf "%s#%s", sep, $1; sep = ", " }')
        closed=$(printf '%s' "$matches" | awk -F'\t' '$2 == "closed" { printf "%s#%s", sep, $1; sep = ", " }')
        if [ "$list" = 1 ]; then
            print_entry "$loc" "$id" "$(
                [ -n "$open" ] && printf 'open %s' "$open"
                [ -n "$open" ] && [ -n "$closed" ] && printf '; '
                [ -n "$closed" ] && printf 'closed %s' "$closed"
                [ -z "$open" ] && [ -z "$closed" ] && printf 'no issue'
                true
            )"
        fi
        if [ -n "$open" ]; then
            continue
        elif [ -n "$closed" ]; then
            stale+=("$loc: $id (closed: $closed)")
        else
            missing+=("$loc: $id")
        fi
    done

    if [ ${#stale[@]} -gt 0 ]; then
        echo "Stale #[ignore]: its issue is closed (remove the ignore if the fix landed, else reopen the issue):" >&2
        printf '  %s\n' "${stale[@]}" >&2
        status=1
    fi
    if [ ${#missing[@]} -gt 0 ]; then
        echo "#[ignore] ID with no issue (file one titled \"<ID>: ...\", or cite the issue as #<n>):" >&2
        printf '  %s\n' "${missing[@]}" >&2
        status=1
    fi
    [ ${#stale[@]} -gt 0 ] || [ ${#missing[@]} -gt 0 ] ||
        echo "check-ignores: ${#ids[@]} ignored tests, each names an open issue"
}

list_unresolved() {
    [ "$list" = 1 ] || return 0
    local entry
    for entry in "${ids[@]}"; do
        print_entry "${entry#*$'\t'}" "${entry%%$'\t'*}" "-"
    done
}

if [ "$offline" = 1 ]; then
    echo "check-ignores: issue check skipped (--offline)"
    list_unresolved
elif [ ${#ids[@]} -gt 0 ]; then
    check_issues
fi

[ "$status" = 0 ] || exit 1
echo "check-ignores: ok"

#!/usr/bin/env bash
# Checks scripts/check-ignores.sh against a scratch repo and a stub `gh`: one case per way an
# `#[ignore]` passes or fails. Changes nothing outside a temporary directory.
#
#   scripts/test-check-ignores.sh
set -euo pipefail

script=$(cd "$(dirname "$0")" && pwd)/check-ignores.sh
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

export GIT_CONFIG_GLOBAL=$tmp/gitconfig GIT_CONFIG_NOSYSTEM=1
unset GITHUB_REPOSITORY VOLTA_REQUIRE_GH
git init --quiet "$tmp/repo"
git -C "$tmp/repo" remote add origin https://github.com/owner/name.git

# Stub gh: page 1 of the issue list (pull requests included, as the REST endpoint returns
# them), then an empty page. Fails when GH_STUB_FAIL is set, like an unauthenticated gh.
mkdir "$tmp/bin"
cat >"$tmp/bin/gh" <<'STUB'
#!/usr/bin/env bash
[ -z "${GH_STUB_FAIL:-}" ] || { echo "gh: not logged in" >&2; exit 1; }
case "$2" in
repos/owner/name/issues\?*page=1) ;;
repos/owner/name/issues\?*) exit 0 ;;
*) echo "stub gh: unexpected $*" >&2; exit 2 ;;
esac
printf '%s\t%s\t%s\t%s\n' \
    1 open issue "S2/S3: max ties and NaN" \
    2 closed issue "S1: second backward" \
    3 open issue "Not yet run: S5 effect on something" \
    4 open issue "S10: unrelated" \
    5 open pr "S9: fix the optimizer" \
    60 open issue "GPU broadcast backward panics" \
    61 closed issue "a closed one"
STUB
chmod +x "$tmp/bin/gh"

fails=0
# case_ NAME EXPECTED_EXIT EXPECTED_OUTPUT_REGEX [ARGS...]: run the check on tests/t.rs; the regex
# matches the output with newlines turned into spaces.
case_() {
    local name=$1 want=$2 pattern=$3 got out
    shift 3
    out=$(cd "$tmp/repo" && PATH="$tmp/bin:$PATH" "$script" "$@" 2>&1) && got=0 || got=$?
    if [ "$got" != "$want" ] || ! printf '%s' "$out" | tr '\n' ' ' | grep -qE "$pattern"; then
        echo "FAIL $name: exit $got (want $want), output:" >&2
        printf '%s\n' "$out" | sed 's/^/    /' >&2
        fails=$((fails + 1))
    else
        echo "ok   $name"
    fi
}
src() { # write tests/t.rs from stdin
    mkdir -p "$tmp/repo/tests"
    cat >"$tmp/repo/tests/t.rs"
}

src <<'RS'
#[test]
#[ignore = "S3: ties"]
fn a() {}
#[test]
#[ignore = "#60: GPU broadcast backward: a reason \
            that continues"]
fn b() {}
//! `#[ignore = "S<n>: ..."]` in a doc comment is not an ignore.
RS
case_ "id in a multi-ID title, and #N" 0 '2 ignored tests, each names an open issue'

src <<'RS'
#[ignore = "S1: fixed long ago"]
fn a() {}
RS
# Issue 4 ("S10: ...") is open; S1 must not match it.
case_ "closed issue is stale" 1 'Stale.*tests/t.rs:1: S1 \(closed: #2\)'

src <<'RS'
#[ignore = "S5: named only after the title's colon"]
fn a() {}
RS
case_ "id after the colon does not count" 1 'with no issue.*tests/t.rs:1: S5'

src <<'RS'
#[ignore = "S9: only a PR names it"]
fn a() {}
RS
case_ "a PR is not an issue" 1 'with no issue.*tests/t.rs:1: S9'

src <<'RS'
#[ignore = "S0: no such issue"]
fn a() {}
RS
case_ "unknown id" 1 'with no issue.*tests/t.rs:1: S0'

src <<'RS'
#[ignore = "#61: closed"]
fn a() {}
RS
case_ "closed #N is stale" 1 'tests/t.rs:1: #61 \(closed: #61\)'

src <<'RS'
#[ignore]
fn a() {}
#[ignore = "slow"]
fn b() {}
#[cfg_attr(miri, ignore)]
fn c() {}
RS
case_ "ignore without an id" 1 'without a defect ID.*t.rs:1:.*t.rs:3:.*t.rs:5:'

src <<'RS'
#[ignore = "S0: no such issue"]
fn a() {}
RS
case_ "offline skips the issue check" 0 'skipped \(--offline\)' --offline
GH_STUB_FAIL=1 case_ "gh failure skips locally" 0 'skipped \(gh api failed'
GH_STUB_FAIL=1 VOLTA_REQUIRE_GH=1 case_ "gh failure fails under VOLTA_REQUIRE_GH" 1 'VOLTA_REQUIRE_GH=1'

if [ "$fails" -gt 0 ]; then
    echo "test-check-ignores: $fails failed" >&2
    exit 1
fi
echo "test-check-ignores: ok"

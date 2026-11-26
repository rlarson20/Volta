#!/usr/bin/env bash
# Release steps and the checks that guard them (#481; dev-docs/RELEASING.md). The same checks
# run on RJ's machine (`just release-*`) and in CI (release-check.yml, release.yml), so a
# release can't pass one and fail the other.
#
#   scripts/release.sh prepare <X.Y.Z>     bump Cargo.toml, start the CHANGELOG section (local)
#   scripts/release.sh check <X.Y.Z>       package checks: version, CHANGELOG, metadata, dry run
#   scripts/release.sh gate <X.Y.Z>        check + HEAD is on main, CI and macOS green, no
#                                          open merge-gate issue (before tagging)
#   scripts/release.sh tag <X.Y.Z>         gate, then `git tag -s vX.Y.Z` and push it (local)
#   scripts/release.sh verify-tag <X.Y.Z>  gate + the pushed tag is annotated, GPG-signed and
#                                          verified by GitHub, on HEAD (release.yml)
#   scripts/release.sh notes <X.Y.Z>       print the CHANGELOG section body (GitHub Release)
#
# Nothing here publishes: only release.yml's `publish` job does, after RJ approves it in the
# `release` environment. Needs `jq` and `curl`; `gate`, `tag` and `verify-tag` need `gh`
# signed in; `prepare` needs `git-cliff`.
set -euo pipefail

repo=rlarson20/Volta
crate=volta
changelog=CHANGELOG.md
# Left in the section by `prepare`; `check` fails while it is there, so the git-cliff draft
# can't ship without the human edit pass.
draft_marker='<!-- release-draft:'
# crates.io asks API clients for a User-Agent that identifies them.
ua="volta-release-check (https://github.com/$repo)"

die() {
	echo "release: $*" >&2
	exit 1
}
ok() { echo "ok: $*"; }

usage() {
	sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
	exit 2
}

[[ $# -ge 2 ]] || usage
cmd=$1
version=$2
[[ $version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] ||
	die "the version must be X.Y.Z with no 'v' and no pre-release suffix, got '$version'"
tag=v$version
cd "$(git rev-parse --show-toplevel)"

manifest_version() {
	cargo metadata --no-deps --format-version 1 --offline |
		jq -r --arg c "$crate" '.packages[] | select(.name == $c) | .version'
}

# The section for $version: the lines after its heading, up to the next `## [` heading or
# the link references at the bottom.
section_body() {
	awk -v h="## [$version]" '
		index($0, h) == 1 { found = 1; next }
		found && (/^## \[/ || /^\[[^]]+\]: /) { exit }
		found { print }
	' "$changelog"
}

check_version() {
	local v
	v=$(manifest_version)
	[[ $v == "$version" ]] || die "Cargo.toml says $v, not $version"
	ok "Cargo.toml version is $version"
	awk -v c="$crate" -v v="$version" '
		$0 == "name = \"" c "\"" { getline; if ($0 == "version = \"" v "\"") f = 1 }
		END { exit !f }' Cargo.lock ||
		die "Cargo.lock still has the old $crate version (run \`cargo update --workspace\`)"
	ok "Cargo.lock matches"
}

check_changelog() {
	local heading date body
	heading=$(grep -E "^## \[$version\]" "$changelog" || true)
	[[ -n $heading ]] || die "$changelog has no '## [$version] - YYYY-MM-DD' section"
	[[ $(grep -cE "^## \[$version\]" "$changelog") == 1 ]] ||
		die "$changelog has more than one [$version] section"
	date=$(sed -nE "s/^## \[$version\] - ([0-9]{4}-[0-9]{2}-[0-9]{2})\$/\1/p" <<<"$heading")
	[[ -n $date ]] || die "the heading must read '## [$version] - YYYY-MM-DD', got '$heading'"
	[[ $(date -u -d "$date" +%F 2>/dev/null || date -u -j -f %F "$date" +%F 2>/dev/null) == "$date" ]] ||
		die "'$date' in the $version heading is not a real date"
	ok "$changelog has a dated [$version] section ($date)"
	body=$(section_body)
	[[ -n ${body//[[:space:]]/} ]] || die "the [$version] section is empty"
	if grep -qF "$draft_marker" <<<"$body"; then
		die "the [$version] section still has the git-cliff draft marker: edit the draft, then delete the '$draft_marker ...' line"
	fi
	ok "[$version] section is filled in and edited"
	grep -qE "^\[$version\]: https://github.com/$repo/compare/v[0-9.]+\.\.\.$tag\$" "$changelog" ||
		die "$changelog has no '[$version]: https://github.com/$repo/compare/vPREV...$tag' link"
	grep -qE "^\[Unreleased\]: https://github.com/$repo/compare/$tag\.\.\.HEAD\$" "$changelog" ||
		die "the [Unreleased] link must compare $tag...HEAD"
	ok "compare links point at $tag"
}

# crates.io validates categories and keywords only on a real upload, not in `--dry-run`
# (that is how `science::ml` reached the v0.4.0 tag), so check them against its rules here.
check_metadata() {
	local meta cats kws c k code
	meta=$(cargo metadata --no-deps --format-version 1 --offline |
		jq -c --arg c "$crate" '.packages[] | select(.name == $c)')
	cats=$(jq -r '.categories[]' <<<"$meta")
	kws=$(jq -r '.keywords[]' <<<"$meta")
	[[ $(jq '.categories | length' <<<"$meta") -le 5 ]] || die "crates.io allows at most 5 categories"
	[[ $(jq '.keywords | length' <<<"$meta") -le 5 ]] || die "crates.io allows at most 5 keywords"
	while IFS= read -r c; do
		[[ -n $c ]] || continue
		[[ $c =~ ^[a-z0-9-]+(::[a-z0-9-]+)*$ ]] ||
			die "category '$c' is not shaped like a crates.io slug (one slug per string, e.g. \"mathematics\"; see https://crates.io/category_slugs)"
		code=$(curl -sS -o /dev/null -w '%{http_code}' -A "$ua" "https://crates.io/api/v1/categories/$c" || true)
		[[ $code == 200 ]] || die "category '$c' is not a crates.io category slug (HTTP $code; see https://crates.io/category_slugs)"
	done <<<"$cats"
	ok "categories exist on crates.io: $(tr '\n' ' ' <<<"$cats")"
	while IFS= read -r k; do
		[[ -n $k ]] || continue
		[[ ${#k} -le 20 && $k =~ ^[A-Za-z0-9][A-Za-z0-9_+-]*$ ]] ||
			die "keyword '$k' breaks crates.io's rule (ASCII alphanumeric, '_', '-' or '+', starts alphanumeric, at most 20 characters)"
	done <<<"$kws"
	ok "keywords are valid: $(tr '\n' ' ' <<<"$kws")"
	for k in description license repository; do
		[[ $(jq -r --arg k "$k" '.[$k] // empty' <<<"$meta") != "" ]] || die "Cargo.toml has no $k"
	done
	ok "description, license and repository are set"
	code=$(curl -sS -o /dev/null -w '%{http_code}' -A "$ua" "https://crates.io/api/v1/crates/$crate/$version" || true)
	case $code in
	404) ok "$crate $version is not on crates.io yet" ;;
	200) die "$crate $version is already on crates.io; a published version can't be replaced" ;;
	*) die "crates.io answered HTTP $code for $crate $version" ;;
	esac
}

check_package() {
	local dirty=()
	[[ ${allow_dirty:-} == 1 ]] && dirty=(--allow-dirty)
	cargo package --list --locked ${dirty[@]+"${dirty[@]}"} >"${TMPDIR:-/tmp}/volta-package-list" ||
		die "cargo package --list failed"
	grep -qx "$changelog" "${TMPDIR:-/tmp}/volta-package-list" || die "the package leaves out $changelog"
	ok "the package includes $changelog"
	cargo publish --dry-run --locked ${dirty[@]+"${dirty[@]}"} || die "cargo publish --dry-run failed"
	ok "cargo publish --dry-run passed"
}

check() {
	check_version
	check_changelog
	check_metadata
	check_package
}

# The latest run of $1 for a push of $sha; prints its conclusion, or nothing if there is none.
run_conclusion() {
	gh api -X GET "repos/$repo/actions/workflows/$1/runs" -f head_sha="$sha" -f event=push \
		-f per_page=1 --jq '.workflow_runs[0] | select(.) | (.conclusion // .status)'
}

gate() {
	[[ -z $(git status --porcelain) ]] || die "the working tree is not clean"
	check
	sha=$(git rev-parse HEAD)
	git fetch -q origin main
	git merge-base --is-ancestor "$sha" origin/main || die "HEAD ($sha) is not on origin/main"
	ok "HEAD $sha is on main"
	local wf c
	for wf in ci.yml macos.yml; do
		c=$(run_conclusion "$wf")
		[[ $c == success ]] ||
			die "$wf on $sha after the push to main is '${c:-not run}', not success (https://github.com/$repo/actions/workflows/$wf)"
		ok "$wf passed on $sha"
	done
	c=$(gh api -X GET "repos/$repo/issues" -f labels=merge-gate -f state=open --jq 'map(select(.pull_request | not)) | length')
	[[ $c == 0 ]] ||
		die "$c merge-gate issue(s) open; each must be fixed or closed with a reason first (https://github.com/$repo/issues?q=is%3Aopen+label%3Amerge-gate)"
	ok "no merge-gate issue is open"
}

verify_tag() {
	gate
	local type obj tsha verified reason
	# Ask GitHub, not the checkout: actions/checkout can fetch an annotated tag as a
	# lightweight one, and the runner has no copy of RJ's public key.
	tsha=$(gh api "repos/$repo/git/ref/tags/$tag" --jq .object.sha)
	type=$(gh api "repos/$repo/git/ref/tags/$tag" --jq .object.type)
	[[ $type == tag ]] || die "GitHub sees $tag as a $type, not an annotated tag"
	obj=$(gh api "repos/$repo/git/tags/$tsha")
	verified=$(jq -r .verification.verified <<<"$obj")
	reason=$(jq -r .verification.reason <<<"$obj")
	[[ $(jq -r .object.sha <<<"$obj") == "$sha" ]] || die "$tag does not point at HEAD ($sha)"
	[[ $verified == true ]] || die "GitHub does not verify the signature on $tag ($reason)"
	ok "$tag is annotated, signed and verified by GitHub"
}

prepare() {
	local current prev today draft
	command -v git-cliff >/dev/null || die "needs git-cliff (\`cargo install git-cliff --locked\`, \`brew install git-cliff\` or nix)"
	[[ -z $(git status --porcelain) ]] || die "the working tree is not clean"
	git fetch -q origin main --tags
	[[ $(git rev-parse HEAD) == $(git rev-parse origin/main) ]] ||
		die "start from origin/main's tip (git switch -c release-$tag origin/main)"
	current=$(manifest_version)
	[[ $(printf '%s\n%s\n' "$current" "$version" | sort -V | tail -1) == "$version" && $current != "$version" ]] ||
		die "$version is not newer than Cargo.toml's $current"
	prev=$(git describe --tags --abbrev=0 --match 'v[0-9]*' origin/main)
	git rev-parse -q --verify "refs/tags/$tag" >/dev/null && die "tag $tag already exists"
	grep -q '^## \[Unreleased\]$' "$changelog" || die "$changelog has no '## [Unreleased]' heading"

	git switch -c "release-$tag"
	# The first `version = ` line is [package]'s. (python3, not `sed -i`: BSD and GNU differ.)
	CUR=$current NEW=$version python3 -c 'import os, re; e = os.environ; t = open("Cargo.toml").read()
t = re.sub("(?m)^version = \"" + re.escape(e["CUR"]) + "\"$", "version = \"" + e["NEW"] + "\"", t, count=1)
open("Cargo.toml", "w").write(t)'
	[[ $(manifest_version) == "$version" ]] || die "could not set the version in Cargo.toml"
	cargo update --workspace --offline -q || cargo update --workspace -q

	today=$(date -u +%F)
	# The draft groups, without git-cliff's own heading line.
	draft=$(git cliff "$prev.." --tag "$tag" --strip all 2>/dev/null | sed '1{/^## /d}')
	VERSION=$version TAG=$tag PREV=$prev TODAY=$today DRAFT=$draft REPO=$repo MARK=$draft_marker \
		python3 - "$changelog" <<-'PY'
			import os, re, sys
			path = sys.argv[1]
			e = os.environ
			text = open(path).read()
			head, rest = text.split("## [Unreleased]\n", 1)
			# Notes already written under [Unreleased] move into the new section.
			m = re.search(r"^(## \[|\[[^\]]+\]: )", rest, re.M)
			pending, tail = (rest[: m.start()], rest[m.start():]) if m else (rest, "")
			section = (
				f"## [Unreleased]\n\n## [{e['VERSION']}] - {e['TODAY']}\n\n"
				f"{e['MARK']} delete this line once the section below is edited (scripts/release.sh check fails while it is here) -->\n\n"
				+ (pending.strip() + "\n\n" if pending.strip() else "")
				+ (e["DRAFT"].strip() + "\n\n" if e["DRAFT"].strip() else "")
			)
			old_link = f"[Unreleased]: https://github.com/{e['REPO']}/compare/{e['PREV']}...HEAD\n"
			new_links = (
				f"[Unreleased]: https://github.com/{e['REPO']}/compare/{e['TAG']}...HEAD\n"
				f"[{e['VERSION']}]: https://github.com/{e['REPO']}/compare/{e['PREV']}...{e['TAG']}\n"
			)
			old_link in tail or sys.exit(f"release: {path} has no line '{old_link.strip()}'")
			open(path, "w").write(head + section + tail.replace(old_link, new_links, 1))
		PY

	git add Cargo.toml Cargo.lock "$changelog"
	git commit -q -m "chore(release): $tag"
	cat <<-EOF
		Made branch release-$tag with commit "chore(release): $tag".
		Next:
		  1. Edit the [$version] section of $changelog (git-cliff drafted it from $prev..HEAD) and
		     delete the '$draft_marker' line. Amend or add commits.
		  2. just release-check $version
		  3. just pr        (release-check.yml and semver.yml run on the PR)
		  4. Merge it, then: git switch main && git pull && just release-tag $version
	EOF
}

do_tag() {
	gate
	git rev-parse -q --verify "refs/tags/$tag" >/dev/null && die "tag $tag already exists"
	[[ $(git rev-parse HEAD) == $(git rev-parse origin/main) ]] ||
		echo "note: HEAD is on main but behind its tip; tagging HEAD ($sha)"
	git tag -s "$tag" -m "$tag" "$sha"
	git push origin "refs/tags/$tag"
	cat <<-EOF
		Pushed $tag. release.yml now re-runs these checks, then waits for your approval in the
		\`release\` environment before it publishes: https://github.com/$repo/actions/workflows/release.yml
	EOF
}

case $cmd in
prepare) prepare ;;
check) allow_dirty=1 check ;;
gate) gate ;;
tag) do_tag ;;
verify-tag) verify_tag ;;
notes)
	body=$(section_body)
	[[ -n ${body//[[:space:]]/} ]] || die "no [$version] section in $changelog"
	printf '%s\n' "$body" | sed -e '/./,$!d'
	;;
*) usage ;;
esac

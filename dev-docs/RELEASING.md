# Releasing Volta

How a release is cut (#481; why it works this way: `DECISIONS.md`, "Releases: a signed local tag, CI publishes"). Three steps are RJ's: editing the changelog, signing the tag, and approving the publish. Everything else is scripted in `scripts/release.sh`, and CI repeats each check.

Versioning: before 1.0, a breaking change bumps the minor version (0.4.x to 0.5.0), and anything else bumps the patch. `semver.yml` runs on the release PR and reports breaks against the last crates.io release.

## One-time setup (repository admin)

Do these once, before the first automated release. `release.yml` fails until both are in place.

1. **`release` environment.** Settings > Environments > New environment, named `release`. Add RJ as a required reviewer and enable "Prevent self-review" only if someone else will approve. Under deployment branches and tags, allow tags matching `v*`. The verify job fails if this environment has no required reviewer.
2. **crates.io trusted publisher.** On [crates.io](https://crates.io/crates/volta), go to Settings > Trusted Publishing > Add. Enter owner `rlarson20`, repository `Volta`, workflow `release.yml` and environment `release` ([docs](https://crates.io/docs/trusted-publishing)). After the first release through it succeeds, you can turn on "require trusted publishing" and revoke any old API tokens.
3. **git-cliff** on the Mac: `brew install git-cliff`, or `cargo install git-cliff --locked`.

## Cutting X.Y.Z

```sh
git switch main && git pull
just release-prepare X.Y.Z      # branch release-vX.Y.Z: version bump, Cargo.lock, CHANGELOG draft
$EDITOR CHANGELOG.md            # edit the [X.Y.Z] section; delete the `<!-- release-draft:` line
git commit -a --amend --no-edit
just release-check X.Y.Z        # same package checks as CI (needs network)
just pr                         # release-check.yml, semver.yml and macOS run on the PR
```

Merge the PR once it is green. Then wait for the post-merge `ci.yml` and `macos.yml` runs on `main` to pass. If the macOS run was cancelled, re-run it.

```sh
git switch main && git pull
just release-tag X.Y.Z          # gate checks, then `git tag -s vX.Y.Z` and push
```

Pushing the tag starts [release.yml](../.github/workflows/release.yml):

1. **verify** checks that the `release` environment has a reviewer, that the tag is signed and verified by GitHub, the release gate, and every package check.
2. **publish** waits for your approval in the Actions run ("Review deployments"), then runs `cargo publish` with a short-lived OIDC token.
3. **github-release** creates the GitHub Release, with the `[X.Y.Z]` CHANGELOG section as its notes.

## What each check catches

| Check | Where | Catches |
| --- | --- | --- |
| Cargo.toml and Cargo.lock say X.Y.Z | check | a tag that doesn't match the package |
| one `## [X.Y.Z] - YYYY-MM-DD` section, not empty, no draft marker, compare links | check | an unedited git-cliff draft, a missing date |
| each category exists on crates.io; keywords follow its rules | check | `science::ml`, `"mathematics, artificial-intelligence"` (v0.4.0) |
| X.Y.Z isn't on crates.io yet | check | re-tagging a published version |
| `cargo package` includes CHANGELOG.md; `cargo publish --dry-run --locked` | check | packaging and build errors |
| HEAD is on `origin/main`; `ci.yml` and `macos.yml` passed on it after the push | gate | tagging a commit that isn't on main or isn't green, including a cancelled macOS run |
| no open `merge-gate` issue | gate | the release gate (`CLAUDE.md`) |
| tag is annotated, signed, verified by GitHub, on HEAD | verify-tag | a tag that RJ didn't sign |

`check` runs on the release PR, `gate` runs before tagging, and `verify-tag` runs in `release.yml`. Each one includes the checks before it.

## When something fails

- **Before `publish`:** nothing was uploaded. Delete the tag (`git push origin :refs/tags/vX.Y.Z && git tag -d vX.Y.Z`), fix `main` through a PR, and tag again.
- **During or after `publish`:** a crates.io version can't be replaced. If the upload went through, only the GitHub Release is missing: re-run that job. If the crate itself is wrong, fix it in a patch release, and `cargo yank` the bad version if it is harmful.
- **Dry run of the whole flow:** run `release-check.yml` by hand (Actions > Release check > Run workflow) on a release branch. The tag-triggered part can only really run on a real release, and the environment approval is where you can still stop it.

<!-- One or two sentences a human has read and approved: what this changes and why it's
     worth a reviewer's time. Add a screenshot or command output if that shows it faster. -->

Fixes #

<!-- One `Fixes #N` line per issue this closes. -->

## Checklist

- [ ] `just ci` exits 0 on this branch
- [ ] GPU: N/A, or touches `src/gpu/` and the tests ran with `VOLTA_REQUIRE_GPU=1` on: <!-- Apple GPU (local / CI macOS), lavapipe -->
- [ ] Removed `#[ignore = "<ID>: ..."]`: none, or <!-- defect ID --> and its issue is in `Fixes` above
- [ ] Out-of-scope findings filed as issues: none, or <!-- links -->
- [ ] Audit corrections appended under `## Corrections`, not edited in place (`.claude/rules/audits.md`): N/A, or done

<details>
<summary><!-- Model and effort, e.g. Claude Opus 5.5 (high). Delete this block for a human-written PR. --></summary>

<!-- The full write-up: summary of changes, test plan, what isn't covered. The blank lines
     around this text are required, or GitHub won't render Markdown inside <details>. -->

</details>

<!-- Claude-authored PRs keep the attribution footer as the last line. -->

---
paths:
  - "docs/audits/**"
---

# Audits (`docs/audits/`)

An audit is a dated record of what was true when it was written. It stays readable as history, and it must not mislead a reader who trusts it today. Both hold only if corrections are appended, never made by editing.

- **Never delete or rewrite original text.** Add to it. The only edits to existing lines are the inline pointers below.
- **A correction has two parts, and both go in.**
  1. An entry under `## Corrections` at the end of that file, numbered `C1`, `C2`, ... with a dated heading, then: what the text said (quote it, with its section or ID), what is true, why, how you know (command, measurement, commit), and the PR that fixes it, if any. Create the section if the file has none.
  2. An inline pointer right after the stale claim, `[corrected <YYYY-MM-DD>: see C<n>]`, so a reader sees the correction where the claim is and not only at the bottom.
- **A correction is for a claim of fact that was wrong or incomplete** (a threshold, a cause, a count, a path, a version). Status ("fixed", "still open") is not one; it lives in the GitHub issue. A claim that the audit made correctly about the code as it was needs no correction because the code has changed since.
- **Tell the user.** List each correction in your reply and in the PR description, with the file and its `C<n>`, so the user knows the audits changed. Never correct an audit as a side effect of other work without saying so. A correction made in a commit message alone is not enough.
- **A cross-file correction** (the same wrong claim in two audits) gets an entry in each file; the second can point to the first.
- **A new audit file** opens with: "Dated record. Original text is never edited or removed; corrections are appended under `## Corrections` (`.claude/rules/audits.md`)."

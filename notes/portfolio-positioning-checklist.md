# Portfolio Positioning Checklist

Objective evaluation criteria for resume, GitHub, LinkedIn, and Volta.
Each item is binary: ✅ pass or ❌ fail. No subjective judgment.

---

## 1. Resume

### Problem-First Framing

| #   | Criterion                                                                                                   | Check |
| --- | ----------------------------------------------------------------------------------------------------------- | ----- |
| 1.1 | Every bullet starts with a verb (Built, Architected, Developed, Designed, Implemented)                      | ☐     |
| 1.2 | Zero bullets contain "to learn", "for learning", or "educational"                                           | ☐     |
| 1.3 | Every project bullet includes at least one quantified metric (tests, coverage, lines, users, performance)   | ☐     |
| 1.4 | Volta bullet mentions the _problem_ (debugging opacity, visualization, memory safety) before the _solution_ | ☐     |

### Signal Density

| #   | Criterion                                                                | Check |
| --- | ------------------------------------------------------------------------ | ----- |
| 1.5 | Resume is exactly 1 page (not 0.8, not 1.2)                              | ☐     |
| 1.6 | No orphan lines or widows (no single-line sections, no dangling bullets) | ☐     |
| 1.7 | Every listed technology appears in at least one project bullet           | ☐     |
| 1.8 | Skills section has ≤15 items total (ruthless curation)                   | ☐     |

### Narrative Thread

| #    | Criterion                                                                              | Check |
| ---- | -------------------------------------------------------------------------------------- | ----- |
| 1.9  | Summary (if present) is ≤3 sentences and names target role explicitly                  | ☐     |
| 1.10 | Projects section appears before "Skills" or "Technologies" section                     | ☐     |
| 1.11 | Volta appears in top 2 projects listed                                                 | ☐     |
| 1.12 | Mathematical/computational background connects to ML work explicitly (not just listed) | ☐     |

---

## 2. GitHub Profile

### First Impression (30-second scan)

| #   | Criterion                                                    | Check |
| --- | ------------------------------------------------------------ | ----- |
| 2.1 | Profile has a bio ≤160 characters that states what you build | ☐     |
| 2.2 | Exactly 4-6 pinned repositories (not 0-3, not 7+)            | ✅    |
| 2.3 | Volta is pinned in position 1 or 2                           | ✅    |
| 2.4 | Every pinned repo has a description ≤100 chars               | ✅    |
| 2.5 | Profile README exists and loads in <2 seconds                | ✅    |
| 2.6 | No pinned repos with 0 stars AND 0 forks AND <10 commits     | ☐     |

### README Quality (per pinned repo)

| #    | Criterion                                                               | Check |
| ---- | ----------------------------------------------------------------------- | ----- |
| 2.7  | README has a visual (GIF, screenshot, diagram) above the fold           | ☐     |
| 2.8  | First paragraph is ≤3 sentences and states what it does                 | ☐     |
| 2.9  | "Installation" or "Quick Start" section exists with copy-paste commands | ☐     |
| 2.10 | Badge row exists showing: build status, test coverage, or version       | ☐     |
| 2.11 | No default "Create React App" or framework boilerplate README           | ☐     |

### Code Quality Signals

| #    | Criterion                                                                  | Check |
| ---- | -------------------------------------------------------------------------- | ----- |
| 2.12 | CI/CD configuration file exists (.github/workflows/, .gitlab-ci.yml, etc.) | ☐     |
| 2.13 | Test directory exists with ≥10 test files or ≥50 test functions            | ☐     |
| 2.14 | Commits in last 6 months on at least 1 pinned repo                         | ☐     |
| 2.15 | No secrets, API keys, or .env files in commit history                      | ☐     |

(run `git log -p \| grep -i "api_key\|secret\|password"`)

---

## 3. LinkedIn

### Headline & Summary

| #   | Criterion                                                                | Check |
| --- | ------------------------------------------------------------------------ | ----- |
| 3.1 | Headline contains target role (ML Engineer, AI Engineer, etc.)           | ☐     |
| 3.2 | Headline does NOT contain "Seeking opportunities" or "Open to work" text | ☐     |
| 3.3 | Summary (About) section is 150-300 words                                 | ☐     |
| 3.4 | Summary mentions 1-2 specific projects by name                           | ☐     |
| 3.5 | Summary ends with what you're looking for (specific role type)           | ☐     |

### Experience Section

| #   | Criterion                                                                               | Check |
| --- | --------------------------------------------------------------------------------------- | ----- |
| 3.6 | Each role has 3-5 bullets (not 1-2, not 6+)                                             | ☐     |
| 3.7 | Freelance/contract work has client count or project count ("8+ clients", "12 projects") | ☐     |
| 3.8 | No bullets are copy-pasted from resume (rephrase for LinkedIn's tone)                   | ☐     |

### Projects Section

| #    | Criterion                                                               | Check |
| ---- | ----------------------------------------------------------------------- | ----- |
| 3.9  | Volta appears in Featured section with link and media (image/video)     | ☐     |
| 3.10 | Each featured project has a 1-2 sentence description visible in preview | ☐     |
| 3.11 | Projects link to GitHub, not to dead/broken URLs                        | ☐     |

### Network Signals

| #    | Criterion                                                            | Check |
| ---- | -------------------------------------------------------------------- | ----- |
| 3.12 | ≥100 connections (signals active use)                                | ☐     |
| 3.13 | Profile photo is professional headshot (face visible, good lighting) | ☐     |
| 3.14 | Custom URL set (linkedin.com/in/yourname, not random characters)     | ☐     |

---

## 4. Volta (Critical Path)

### README Structure (exact order matters)

| #   | Criterion                                                                | Check |
| --- | ------------------------------------------------------------------------ | ----- |
| 4.1 | **Line 1**: Project name + emoji/icon + one-line description (≤15 words) | X     |
| 4.2 | **Lines 2-5**: GIF or screenshot of visualization (not below fold)       | ☐     |
| 4.3 | **Before scroll**: Feature list with emoji bullets (≤6 items)            | ☐     |
| 4.4 | Explicit "Why Volta?" or "Motivation" section exists                     | ☐     |
| 4.5 | "Why Volta?" section states the _problem_ before mentioning learning     | ☐     |

### Visual Proof

| #    | Criterion                                                             | Check |
| ---- | --------------------------------------------------------------------- | ----- |
| 4.6  | GIF exists showing the visualization in action                        | ☐     |
| 4.7  | GIF is <5MB and loads in <3 seconds on GitHub                         | ☐     |
| 4.8  | GIF demonstrates something non-trivial (not just "hello world" graph) | ☐     |
| 4.9  | Static architecture diagram exists (computation graph structure)      | ☐     |
| 4.10 | Live demo link works and loads in <10 seconds (if deployed)           | ☐     |

### Quantified Claims

| #    | Criterion                                                               | Check |
| ---- | ----------------------------------------------------------------------- | ----- |
| 4.11 | Test count appears in README ("94 tests" or badge)                      | ☐     |
| 4.12 | "Validated against PyTorch" claim is verifiable (test shows comparison) | ☐     |
| 4.13 | Supported operations listed explicitly (matmul, conv2d, relu, etc.)     | ☐     |
| 4.14 | What it does NOT support is stated (honest scoping)                     | ☐     |

### Technical Credibility

| #    | Criterion                                                         | Check |
| ---- | ----------------------------------------------------------------- | ----- |
| 4.15 | CI badge shows passing status (green)                             | X     |
| 4.16 | `cargo test` passes locally with 0 failures                       | X     |
| 4.17 | `cargo clippy` produces 0 warnings (or warnings are acknowledged) | X     |
| 4.18 | Code has doc comments on public APIs (`///` comments)             | X     |
| 4.19 | Architecture or design doc exists (in /docs or README section)    | ☐     |

### Installation & Usage

| #    | Criterion                                                             | Check |
| ---- | --------------------------------------------------------------------- | ----- |
| 4.20 | "Quick Start" has ≤5 copy-paste commands to run an example            | ☐     |
| 4.21 | Example code in README compiles without modification                  | ☐     |
| 4.22 | Example demonstrates core value prop (building + visualizing a graph) | ☐     |
| 4.23 | WASM demo has separate clear instructions (or is one-click deploy)    | ☐     |

### Interview Prep (verify you can answer)

| #    | Criterion                                                      | Check |
| ---- | -------------------------------------------------------------- | ----- |
| 4.24 | Can explain in 60 seconds what Volta does and why you built it | ☐     |
| 4.25 | Can name 3 specific architectural decisions and defend them    | ☐     |
| 4.26 | Can describe the hardest bug you fixed and how you debugged it | ☐     |
| 4.27 | Can explain one thing you'd do differently with more time      | ☐     |
| 4.28 | Can draw the computation graph structure on a whiteboard       | ☐     |

---

## Quick Reference: The Three Questions

Before any interview, verify you can answer these for Volta:

**Q1: "Why did you build this?"**

> Template: "I was [doing X] and realized [problem Y]. Rather than [obvious solution], I [what you did] to [outcome]."

Your answer (write it out):

```
_____________________________________________________________
_____________________________________________________________
```

**Q2: "What was the hardest part?"**

> Template: "[Specific technical challenge] because [why it was hard]. I solved it by [approach]."

Your answer (write it out):

```
_____________________________________________________________
_____________________________________________________________
```

**Q3: "What would you do differently?"**

> Template: "With more time, I'd [improvement] because [reasoning]. The current approach [tradeoff you accepted]."

Your answer (write it out):

```
_____________________________________________________________
_____________________________________________________________
```

---

## Scoring

| Section   | Items  | Your Score  |
| --------- | ------ | ----------- |
| Resume    | 12     | \_\_/12     |
| GitHub    | 15     | \_\_/15     |
| LinkedIn  | 14     | \_\_/14     |
| Volta     | 28     | \_\_/28     |
| **Total** | **69** | **\_\_/69** |

### Thresholds

- **<50**: Significant gaps. Prioritize Volta (highest leverage).
- **50-60**: Competitive. Focus on weak sections.
- **60+**: Strong. Polish and iterate.

---

## Priority Order (if time-constrained)

1. **Volta GIF** (4.6-4.8) — highest ROI, 30-second scan differentiator
2. **Volta README structure** (4.1-4.5) — frames everything else
3. **GitHub pinned repos** (2.2-2.4) — first thing recruiters see
4. **Resume project bullets** (1.1-1.4) — problem-first framing
5. **LinkedIn Featured** (3.9-3.11) — passive discovery channel
6. **Everything else** — iterate after above are solid

# Development Experience Cleanup Checklist

This checklist focuses on improving the development experience of the Volta project - making it easier to work with, cleaner, and more maintainable. This is **not** about refactors or feature improvements, but about hygiene and developer experience.

## 🗂️ Repository Organization

### Root Directory Cleanup
- [ ] **Consolidate documentation files**
  - [ ] Move or remove `AGENTS.md` (only contains `@CLAUDE.md`)
  - [ ] Consider moving `scrapboard.md` to `docs/` or remove if obsolete
  - [ ] Review `REFACTOR_SUGGESTIONS.md` vs `docs/` content for duplication
  - [ ] Review `VOLTA_ANALYSIS_REPORT.md` - consolidate with other docs or archive

- [ ] **Clean up LLM-specific files** (development artifacts)
  - [ ] Move or remove `.aider*` files (chat history, input history, tags cache)
  - [ ] Consider moving `responses/` directory outside of repo or gitignore it
  - [ ] Consider moving `sys-prompts/` directory outside of repo or gitignore it
  - [ ] Remove or move `.claude/` directory

- [ ] **Organize or remove node_modules**
  - [ ] Check if `node_modules/` is needed - appears to be leftover from web builds
  - [ ] If not needed, remove directory
  - [ ] If needed, document why in README

### .gitignore Improvements
- [ ] **Update .gitignore to be more comprehensive**
  - [ ] Change `*.txt` to specific patterns (too broad, catches legitimate docs)
  - [ ] Add `.DS_Store` explicitly
  - [ ] Add `model.bin` pattern for test models
  - [ ] Add `.claude/` directory
  - [ ] Add `responses/` directory (or make it explicit which files to track)
  - [ ] Add `sys-prompts/` directory (or make it explicit which files to track)
  - [ ] Consider adding `.aider*` patterns
  - [ ] Add `coverage.txt`, `err.txt`, `tests.txt`, `cmd.txt` explicitly instead of `*.txt`

## 📝 Configuration Files

### Cargo.toml Cleanup
- [ ] **Address TODOs in Cargo.toml**
  - [ ] Line 31: "TODO: add the other examples" - Add missing examples or remove comment
  - [ ] Line 77: "TODO: go over and properly organize lints" - Review and organize lints

- [ ] **Review and document default features**
  - [ ] Document why `gpu` and `accelerate` are default features in comments
  - [ ] Consider if both should really be default (gpu might fail on systems without GPU)

### bacon.toml Cleanup
- [ ] **Address TODOs in bacon.toml**
  - [ ] Line 33: "TODO: organize these" - Organize clippy pedantic lints
  - [ ] Line 92: "TODO: check if `map_to`s should be `map_to_else`" - Review and decide
  - [ ] Line 237: "TODO: add examples here" - Add example shortcuts or remove comment

- [ ] **Review temporary lint exceptions**
  - [ ] Lines 47-58: Multiple "TEMP IMP TO FIX!!!" comments in pedantic job
  - [ ] Document why each is temporarily allowed
  - [ ] Create issues to track fixing these

### rustfmt.toml Enhancement
- [ ] **Expand rustfmt.toml with project preferences**
  - Currently only has `edition = "2024"`
  - Add common formatting preferences for consistency

### CI/CD Configuration
- [ ] **Review CI matrix**
  - Currently tests on `[1.89, stable]` - is 1.89 still needed?
  - Document MSRV (Minimum Supported Rust Version) policy

- [ ] **Document known CI limitations**
  - GPU tests can't run in CI (no GPU in GitHub Actions)
  - Consider adding documentation about this

## 🧪 Testing & Development Tools

### Test Organization
- [ ] **Review test file structure in tests/**
  - Several `gpu_*` test files - consolidate if possible
  - `improvements_test.rs` - vague name, rename to be more specific
  - `coverage_fill.rs` - document purpose

### Benchmarking
- [ ] **Document unsafe benchmarks**
  - `gpu_comparison` benchmark is documented as unsafe in CLAUDE.md
  - Add warnings in the benchmark file itself
  - Consider adding runtime memory checks

- [ ] **Review bench.sh script**
  - Document what it does
  - Consider moving logic to justfile for consistency

### Development Scripts
- [ ] **Review justfile organization**
  - LLM commands (lines 57-162) - mark clearly as maintainer-only
  - Consider separating into `justfile.local` that's gitignored
  - Add comment headers to organize sections better

## 🔧 Source Code Hygiene

### TODOs in Source Code
- [ ] **Address or document TODOs**
  - `src/nn/mod.rs:77` - "TODO: implement direct GPU-to-GPU transfer"
  - `src/gpu/context.rs:214` - "TODO: inspect this deeper"
  - `src/gpu/monitor.rs:131` - "TODO: Could use sysinfo crate"
  - `src/gpu/kernels.rs:1909` - "TEMP!!!" comment
  - `src/lib.rs:2708` - Review test organization comment

- [ ] **Create GitHub issues for TODOs**
  - Convert inline TODOs to tracked issues
  - Reference issue numbers in code comments

### Documentation
- [ ] **Add missing module-level documentation**
  - Ensure all public modules have doc comments
  - Run `cargo doc` and review for missing docs

- [ ] **Review and update CLAUDE.md**
  - Very comprehensive, but verify all information is current
  - Consider breaking into smaller files in `.claude/` directory

## 📦 Dependencies

### Dependency Audit
- [ ] **Review unused dependencies**
  - `scrapboard.md` mentions "cargo machete says cblas and serde-json are no longer used"
  - Run `cargo machete` or `cargo-udeps` to find unused deps
  - Remove any truly unused dependencies

- [ ] **Review dev dependencies**
  - Not listed in Cargo.toml analysis - ensure they're properly categorized

## 🎯 Developer Experience

### Documentation
- [ ] **Create CONTRIBUTING.md**
  - Guide for new contributors
  - Testing guidelines
  - PR process
  - Code style preferences

- [ ] **Create DEVELOPMENT.md**
  - Quick start guide
  - Common development tasks
  - Troubleshooting common issues
  - How to run different test suites

- [ ] **Update README.md**
  - Verify all examples listed actually exist
  - Ensure installation instructions are current
  - Add badges for build status, coverage, etc.

### Code Organization
- [ ] **Review examples/ directory**
  - Ensure all examples in Cargo.toml exist
  - Ensure all examples work
  - Add README in examples/ explaining each

- [ ] **Review docs/ directory structure**
  - Many analysis docs - consider archiving old ones
  - Create clear hierarchy of current vs archived docs

### Pre-commit Hooks
- [ ] **Document pre-commit setup**
  - Add installation instructions to README or DEVELOPMENT.md
  - Test that hooks work on fresh clone

## 🔍 Code Quality Tools

### Linting Setup
- [ ] **Review clippy configuration**
  - Many lints configured in both Cargo.toml and bacon.toml
  - Ensure consistency between files
  - Document why each strict lint is enabled

- [ ] **Consider cargo-deny**
  - Add dependency licensing checks
  - Add security advisory checks

- [ ] **Consider cargo-audit integration**
  - Already in CI (audit.yml)
  - Document how to run locally

## 📊 Metrics & Monitoring

### Project Health
- [ ] **Set up code coverage tracking**
  - `coverage.txt` exists but no clear workflow
  - Document how to generate and interpret

- [ ] **Document benchmarking workflow**
  - When to run benchmarks
  - How to interpret results
  - How to avoid memory issues

## Priority Levels

### High Priority (Do First)
1. Clean up root directory temporary files
2. Fix .gitignore to be more specific
3. Address TODOs in configuration files
4. Create CONTRIBUTING.md and DEVELOPMENT.md

### Medium Priority (Do Soon)
5. Consolidate documentation files
6. Review and organize test files
7. Audit and remove unused dependencies
8. Add module-level documentation

### Low Priority (Nice to Have)
9. Enhance CI/CD with more checks
10. Set up additional code quality tools
11. Improve benchmark organization

## Notes

- This checklist focuses on **developer experience** and **repository hygiene**
- Does not include feature work or architectural refactors (see REFACTOR_SUGGESTIONS.md)
- Many items can be done in parallel
- Each checkbox represents a discrete, completable task

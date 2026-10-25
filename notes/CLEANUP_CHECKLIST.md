# Development Experience Cleanup Checklist

## 📝 Configuration Files

### Cargo.toml Cleanup

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

- [ ] **Review dev dependencies**
  - Not listed in Cargo.toml analysis - ensure they're properly categorized

## 🎯 Developer Experience

### Documentation

- [ ] **Update README.md**
  - Verify all examples listed actually exist
  - Ensure installation instructions are current
  - Add badges for build status, coverage, etc.

### Code Organization

- [ ] **Review examples/ directory**
  - Ensure all examples in Cargo.toml exist
  - Ensure all examples work
  - Add README in examples/ explaining each

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

### Medium Priority (Do Soon)

5. Consolidate documentation files
6. Review and organize test files
7. Audit and remove unused dependencies
8. Add module-level documentation

### Low Priority (Nice to Have)

9. Enhance CI/CD with more checks
10. Set up additional code quality tools
11. Improve benchmark organization

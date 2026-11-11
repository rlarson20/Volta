md:
	fd . --extension md | xargs nvim

test-all:
	cargo nextest run --no-fail-fast

# ===== CODE HEALTH & ANALYSIS =====

# Run coverage with llvm-cov and open HTML report
coverage:
	cargo llvm-cov --html
	open target/llvm-cov/html/index.html

# Terminal-only coverage summary
coverage-summary:
	cargo llvm-cov

# Per-file coverage table, no default features (the CPU lane; coverage.yml runs this)
coverage-cpu:
	cargo llvm-cov --no-default-features --summary-only

# Fast targeted mutation testing (e.g. just mutants src/tensor.rs)
mutants target="":
	cargo mutants -- {{target}}

# Full mutation testing (WARNING: slow)
mutants-all:
	cargo mutants

# Mutation testing of the autograd core on the CPU lane (mutants.yml runs this in shards, e.g.
# `just mutants-core --shard 0/8`). gpu_ops.rs is excluded: it is compiled out without `gpu`,
# so every mutant there would survive. About 400 mutants, ~20 s each on 4 cores.
mutants-core *flags:
	cargo mutants --no-default-features --file src/autograd.rs --file 'src/ops/**' --exclude src/ops/gpu_ops.rs {{flags}}

# Find unused dependencies
dead-deps:
	cargo machete

# Analyze binary bloat for an example (e.g. just bloat showcase)
bloat name:
	cargo bloat --example {{name}}

# Show macro-expanded code for a module (e.g. just expand nn::layers::conv)
expand mod:
	cargo expand {{mod}}

# ===== GPU & SHADER VALIDATION =====

# Validate all WGSL shaders in src/gpu/shaders/
validate-shaders:
	cargo test --lib shader_template

# Run the GPU tests: the gpu_* integration targets and the GPU lib tests (gpu_tests, gpu_half_tests)
test-gpu:
	cargo test --test 'gpu_*' -- --nocapture
	cargo test --lib gpu_ -- --nocapture

# Regenerate the PyTorch reference fixtures in tests/fixtures/torch/ (needs uv; pinned torch)
torch-fixtures:
	uv run scripts/torch_ref/gen_fixtures.py

# Download MNIST into data/mnist (ossci mirror) and check the files' SHA-256
mnist-data:
	#!/usr/bin/env bash
	set -euo pipefail
	mkdir -p data/mnist
	cd data/mnist
	for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
		[ -f "$f" ] || { curl -sSfL -O "https://ossci-datasets.s3.amazonaws.com/mnist/$f.gz" && gunzip -f "$f.gz"; }
	done
	sha256sum -c - <<'SUMS'
	ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db  train-images-idx3-ubyte
	65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5  train-labels-idx1-ubyte
	0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7  t10k-images-idx3-ubyte
	ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2  t10k-labels-idx1-ubyte
	SUMS

# MNIST under bf16 autocast vs f32 (Phase 5 P7): 5 seeds, fails if the mean gap > 0.2 points
mnist-autocast: mnist-data
	cargo run --release --no-default-features --example mnist_autocast

# Run all tests without GPU/Accelerate features
test-cpu:
	cargo test --no-default-features

# ===== EXAMPLE & INTEGRATION QUALITY =====

# Compile-check every example (catches API drift)
examples-check:
	cargo build --examples

# Run a single example by name
example name:
	cargo run --example {{name}}

# ===== PROJECT METRICS & HYGIENE =====

# Lines of code by language
loc:
	tokei src/

# Dependency tree (root level)
deps-tree:
	cargo tree --depth 1

# Check for outdated dependencies
outdated:
	cargo outdated --root-deps-only

# Nuke all build artifacts and mutation-testing output
clean:
	cargo clean
	-rm -rf mutants.out
	@echo "Cleaned target and mutants output."

# ===== AGGREGATE WORKFLOWS =====

# Clippy over every target with default features, then without; the single definition of the
# lint gate (ci.yml and pre-commit call this too)
lint: lint-cpu
	cargo clippy --all-targets -- -D warnings

# Clippy over every target with no default features (the CPU-only lane)
lint-cpu:
	cargo clippy --no-default-features --all-targets -- -D warnings

# Rustdoc with warnings denied (a broken intra-doc link fails), default features
doc:
	RUSTDOCFLAGS="-D warnings" cargo doc --no-deps

# Rustdoc with warnings denied, no default features
doc-cpu:
	RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --no-default-features

# Feature-combination rot check (e.g. `just hack --exclude-features gpu`); needs cargo-hack
hack *flags:
	cargo hack check --feature-powerset --no-dev-deps {{flags}}

# `hack`, skipped with a notice when cargo-hack is not installed (`cargo install cargo-hack`)
hack-if-installed:
	#!/usr/bin/env bash
	set -euo pipefail
	if command -v cargo-hack >/dev/null; then
		just hack
	else
		echo "skipped: cargo-hack not installed (cargo +stable install cargo-hack --locked)"
	fi

# Fail on silent numeric fallbacks (`unwrap_or(0.0)`, `if let Some(slot) = v.get_mut(i)`)
check-fallbacks:
	./scripts/check-fallbacks.sh

# Fail on an `#[ignore]` without a defect ID, or whose ID names no open issue (the issue half
# needs `gh` and skips with a notice without it; `just check-ignores --offline` skips it)
check-ignores *flags:
	./scripts/check-ignores.sh {{flags}}

# Check scripts/check-ignores.sh against a scratch repo and a stub gh
test-check-ignores:
	./scripts/test-check-ignores.sh

# CI-like pipeline (fmt -> fallbacks -> ignores -> lint, both feature sets -> tests, both feature
# sets -> docs with -D warnings, both feature sets -> examples -> feature powerset when cargo-hack
# is installed)
ci:
	cargo fmt --check
	just check-fallbacks
	just test-check-ignores
	just check-ignores
	just lint
	cargo test
	just test-cpu
	just doc
	just doc-cpu
	cargo build --examples
	just hack-if-installed
	@echo "CI check passed!"

# Apply .github/rulesets/*.json to GitHub (create or update by name; needs admin `gh` auth)
rulesets:
	#!/usr/bin/env bash
	set -euo pipefail
	repo=$(gh repo view --json nameWithOwner --jq .nameWithOwner)
	for f in .github/rulesets/*.json; do
		name=$(jq -r .name "$f")
		id=$(gh api "repos/$repo/rulesets" --jq ".[] | select(.name == \"$name\") | .id")
		if [[ -n "$id" ]]; then
			gh api -X PUT "repos/$repo/rulesets/$id" --input "$f" --silent
			echo "updated: $name ($id)"
		else
			gh api -X POST "repos/$repo/rulesets" --input "$f" --silent
			echo "created: $name"
		fi
	done

# ===== GIT & PR WORKFLOW (dev-docs/WORKFLOW.md) =====

# Turn staged changes and/or commits ahead of origin/improvements into a pull request (e.g. `just pr`, `just pr fix/my-branch`)
pr name="":
	./scripts/pr.sh {{name}}

# After merges: fast-forward improvements and delete local branches it already holds, saying why each kept one stays (`just sync dry-run` lists them first)
sync mode="":
	./scripts/sync.sh {{mode}}

# Check `just sync`'s delete and keep rules against a scratch repo and a stub gh
test-sync:
	./scripts/test-sync.sh

# Your open PRs and the checks on the current branch's PR
prs:
	#!/usr/bin/env bash
	set -euo pipefail
	gh pr status
	b=$(git branch --show-current)
	if [[ -n "$b" ]] && gh pr view "$b" >/dev/null 2>&1; then
		echo
		gh pr checks "$b" || true
	fi

# Licenses, RustSec advisories, duplicate versions, and sources (needs network for the advisory db)
deny:
	cargo deny check

# Comprehensive pre-release checklist
pre-release: ci validate-shaders dead-deps coverage-summary
	@echo "Pre-release checks complete."


# ===== BENCHMARKING =====
# Run all benchmarks
# Run the CPU benchmarks (gpu_comparison is slow and needs a GPU: use `just bench-gpu`)
bench:
	cargo bench --bench tensor_ops --
	cargo bench --bench neural_networks --
	cargo bench --bench conv_algorithms --

# Run convolution algorithm comparison benchmarks
bench-conv:
	cargo bench --bench conv_algorithms --

# Run specific benchmark
bench-name name:
	cargo bench --bench {{name}} --

# Benchmark CPU only (no features)
bench-cpu:
	cargo bench --bench tensor_ops --no-default-features --
	cargo bench --bench neural_networks --no-default-features --
	cargo bench --bench conv_algorithms --no-default-features --

# Benchmark with Accelerate (macOS BLAS)
bench-accel:
	cargo bench --features accelerate --no-default-features --bench tensor_ops --
	cargo bench --features accelerate --no-default-features --bench neural_networks --
	cargo bench --features accelerate --no-default-features --bench conv_algorithms --

# Benchmark GPU comparison
bench-gpu:
	cargo bench --features gpu --bench gpu_comparison --

# Save benchmark results for comparison
bench-save:
	cargo bench --bench tensor_ops -- --save-baseline main
	cargo bench --bench neural_networks -- --save-baseline main
	cargo bench --bench conv_algorithms -- --save-baseline main

# Compare against baseline saved by `just bench-save`
bench-compare:
	cargo bench --bench tensor_ops -- --baseline main
	cargo bench --bench neural_networks -- --baseline main
	cargo bench --bench conv_algorithms -- --baseline main

# Time the 256x256 `matmul_raw` smoke test N times in release and average (was bench.sh).
# A quick regression check, not a benchmark: use `just bench` for criterion numbers.
bench-matmul runs="10" *flags="":
	#!/usr/bin/env bash
	set -euo pipefail
	total=0
	for ((i = 1; i <= {{runs}}; i++)); do
		line=$(cargo test --release --lib {{flags}} bench_matmul_speedup -- --nocapture 2>&1 | grep '256x256 matmul:' || true)
		ms=$(printf '%s' "$line" | sed -nE 's/.*matmul: ([0-9.]+)ms.*/\1/p')
		if [[ -z "$ms" ]]; then echo "run $i: no timing line (did the test build or fail?)" >&2; exit 1; fi
		echo "run $i: ${ms}ms"
		total=$(awk -v t="$total" -v m="$ms" 'BEGIN {print t + m}')
	done
	awk -v t="$total" -v n="{{runs}}" 'BEGIN {printf "average: %.3fms over %d runs\n", t / n, n}'

bench-report:
	open './target/criterion/report/index.html'

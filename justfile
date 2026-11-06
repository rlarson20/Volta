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

# Fast targeted mutation testing (e.g. just mutants src/tensor.rs)
mutants target="":
	cargo mutants -- {{target}}

# Full mutation testing (WARNING: slow)
mutants-all:
	cargo mutants

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
	@echo "Validating shaders..."
	@for f in src/gpu/shaders/*.wgsl; do \
		echo "Checking $f..."; \
		naga $f || (echo "Failed: $f" && exit 1); \
	done
	@echo "All shaders valid."

# Run only GPU integration tests
test-gpu:
	cargo test --test 'gpu_*' -- --nocapture

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
	cargo test --no-default-features -- --nocapture

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

# Clippy over every target; the single definition of the lint gate (pre-commit calls this too)
lint:
	cargo clippy --all-targets -- -D warnings

# Fail on silent numeric fallbacks (`unwrap_or(0.0)`, `if let Some(slot) = v.get_mut(i)`)
check-fallbacks:
	./scripts/check-fallbacks.sh

# CI-like pipeline (fmt -> fallbacks -> lint, both feature sets -> test -> docs -> examples)
ci:
	cargo fmt --check
	just check-fallbacks
	just lint
	cargo clippy --no-default-features --all-targets -- -D warnings
	cargo test
	cargo doc --no-deps
	cargo build --examples
	@echo "CI check passed!"

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

bench-report:
	open './target/criterion/report/index.html'

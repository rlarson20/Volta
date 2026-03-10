check:
	cargo check
	cargo build
	cargo test

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
	cargo test --features gpu --test 'gpu_*' -- --nocapture

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

# Nuke all build artifacts and temporary files
clean:
	cargo clean
	-rm mutants.out/
	-rm *.txt responses/context.md
	@echo "Cleaned target, mutants output, and temporary text files."

# ===== AGGREGATE WORKFLOWS =====

# Fast CI-like pipeline (fmt -> lint -> test -> docs -> examples)
ci:
	cargo fmt --check
	cargo clippy --all-targets -- -D warnings
	cargo test
	cargo doc --no-deps
	cargo build --examples
	@echo "CI check passed!"

# Comprehensive pre-release checklist
pre-release: ci validate-shaders dead-deps coverage-summary examples-check
	@echo "Pre-release checks complete."


# ===== BENCHMARKING =====
# Run all benchmarks
bench: # NOT FOR USE, RUNS THE GPU_COMPARISON
	cargo bench --bench tensor_ops --
	cargo bench --bench neural_networks --
	cargo bench --bench gpu_comparison --
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
	cargo bench --bench gpu_comparison --no-default-features --
	cargo bench --bench conv_algorithms --no-default-features --

# Benchmark with Accelerate (macOS BLAS)
bench-accel:
	cargo bench --features accelerate --no-default-features --bench tensor_ops --
	cargo bench --features accelerate --no-default-features --bench neural_networks --
	cargo bench --features accelerate --no-default-features --bench gpu_comparison --
	cargo bench --features accelerate --no-default-features --bench conv_algorithms --

# Benchmark GPU comparison
bench-gpu:
	cargo bench --features gpu --bench gpu_comparison --

# Save benchmark results for comparison
bench-save:
	cargo bench --bench tensor_ops -- --save-baseline main
	cargo bench --bench neural_networks -- --save-baseline main
	cargo bench --bench gpu_comparison -- --save-baseline main
	cargo bench --bench conv_algorithms -- --save-baseline main

# Compare against saved baseline (DO NOT USE, RUNNING)
bench-compare:
	cargo bench --bench tensor_ops -- --baseline main
	cargo bench --bench neural_networks -- --baseline main
	cargo bench --bench gpu_comparison -- --baseline main
	cargo bench --bench conv_algorithms -- --baseline main

bench-report:
	open './target/criterion/report/index.html'

# ===== LLM ASSISTED DEV =====

ask-gpu model:
	@echo "Cleaning!"
	-rm err.txt tests.txt
	-rm responses/context.md
	@echo "Asking!"
	-cargo build &> err.txt
	-cargo test &> tests.txt
	files-to-prompt Cargo.toml src/ tests/ err.txt tests.txt responses/claude-on-gpu-integration.md responses/status-report.md README.md > responses/context.md
	cat responses/context.md | llm --model "openrouter/{{model}}" --system "`cat sys-prompts/gpu-sys-prompt.md`" -o reasoning_effort high > responses/gpu-recommendations.md
	nvim responses/gpu-recommendations.md

ask-codex:
	@echo "Cleaning!"
	-rm err.txt tests.txt
	-rm responses/context.md
	@echo "Asking!"
	-cargo build &> err.txt
	-cargo test &> tests.txt
	files-to-prompt Cargo.toml src/ tests/ err.txt tests.txt  responses/status-report.md README.md > responses/context.md
	cat responses/context.md | llm --model "openrouter/openai/gpt-5.1-codex" --system "`cat sys-prompts/sys-prompt.md`" -o reasoning_effort high > responses/recommendations.md
	@echo "Finished!"

ask-codex-mini:
	@echo "Cleaning!"
	-rm err.txt tests.txt
	-rm responses/context.md
	@echo "Asking!"
	-cargo build &> err.txt
	-cargo test &> tests.txt
	files-to-prompt Cargo.toml src/ tests/ err.txt tests.txt  responses/status-report.md README.md > responses/context.md
	cat responses/context.md | llm --model "openrouter/openai/gpt-5.1-codex-mini" --system "`cat sys-prompts/sys-prompt.md`" -o reasoning_effort high > responses/recommendations.md
	@echo "Finished!"


ask-readme model:
	@echo "Cleaning!"
	-rm err.txt tests.txt
	-rm responses/context.md
	@echo "Asking!"
	-cargo build &> err.txt
	-cargo test &> tests.txt
	files-to-prompt Cargo.toml src/ tests/ err.txt tests.txt responses/status-report.md README.md > responses/context.md
	cat responses/context.md | llm --model "openrouter/{{model}}" --system "`cat sys-prompts/readme-sys-prompt.md`" > responses/updates-to-readme.md
	@echo "Finished!"


ask-custom model prompt filename:
	@echo "Cleaning!"
	-rm err.txt tests.txt
	-rm responses/context.md
	@echo "Asking!"
	-cargo build &> err.txt
	-cargo test &> tests.txt
	files-to-prompt Cargo.toml src/ tests/ err.txt tests.txt responses/status-report.md README.md > responses/context.md
	cat responses/context.md | llm --model "openrouter/{{model}}" --system "{{prompt}}" > responses/{{filename}}.md
	@echo "Finished!"


ask-err-file model file:
	@echo "Cleaning!"
	-rm err.txt tests.txt
	-rm responses/context.md
	-rm err-recommendations.md
	@echo "Asking!"
	-cargo build &> err.txt
	-cargo test &> tests.txt
	files-to-prompt Cargo.toml src/{{file}} tests/ err.txt tests.txt responses/status-report.md README.md > responses/context.md
	cat responses/context.md | llm --model "openrouter/{{model}}" --system "`cat sys-prompts/err-sys-prompt.md`" > responses/err-recommendations.md
	@echo "Finished!"

ask-err model:
	@echo "Cleaning!"
	-rm err.txt tests.txt
	-rm responses/context.md
	-rm responses/err-recommendations.md
	@echo "Asking!"
	-cargo build &> err.txt
	-cargo test &> tests.txt
	files-to-prompt Cargo.toml src/ tests/ err.txt tests.txt responses/status-report.md README.md > responses/context.md
	cat responses/context.md | llm --model "openrouter/{{model}}" --system "`cat sys-prompts/err-sys-prompt.md`" > responses/err-recommendations.md
	@echo "Finished!"


ask-status model:
	@echo "Cleaning!"
	-rm err.txt tests.txt
	-rm responses/context.md
	@echo "Asking!"
	-cargo build &> err.txt
	-cargo test &> tests.txt
	files-to-prompt Cargo.toml src/ tests/ err.txt tests.txt  responses/status-report.md README.md > responses/context.md
	cat responses/context.md | llm --model "openrouter/{{model}}" --system "`cat sys-prompts/status-sys-prompt.md`" > responses/status-report.md
	@echo "Finished!"

ask model:
	@echo "Cleaning!"
	-rm err.txt tests.txt
	-rm responses/context.md
	@echo "Asking!"
	-cargo build &> err.txt
	-cargo test &> tests.txt
	files-to-prompt Cargo.toml src/ tests/ err.txt tests.txt  responses/status-report.md README.md > responses/context.md
	cat responses/context.md | llm --model "openrouter/{{model}}" --system "`cat sys-prompts/sys-prompt.md`" > responses/recommendations.md
	@echo "Finished!"

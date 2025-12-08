check:
	cargo check
	cargo build
	cargo test


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

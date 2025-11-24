check:
	cargo check
	cargo build
	cargo test

ask-readme model:
	@echo "Cleaning!"
	-rm err.txt tests.txt
	-rm responses/context.md
	@echo "Asking!"
	-cargo build &> err.txt
	-cargo test &> tests.txt
	files-to-prompt Cargo.toml src/ err.txt tests.txt responses/status-report.md README.md > responses/context.md
	cat responses/context.md | llm --model "openrouter/{{model}}" --system "`cat sys-prompts/readme-sys-prompt.md`" > responses/updates-to-readme.md
	@echo "Finished!"


ask-custom model prompt:
	@echo "Cleaning!"
	-rm err.txt tests.txt
	-rm responses/context.md
	@echo "Asking!"
	-cargo build &> err.txt
	-cargo test &> tests.txt
	files-to-prompt Cargo.toml src/ err.txt tests.txt responses/status-report.md README.md > responses/context.md
	cat responses/context.md | llm --model "openrouter/{{model}}" --system "{{prompt}}" > responses/custom-resp.md
	@echo "Finished!"


ask-err-file model file:
	@echo "Cleaning!"
	-rm err.txt tests.txt
	-rm responses/context.md
	-rm err-recommendations.md
	@echo "Asking!"
	-cargo build &> err.txt
	-cargo test &> tests.txt
	files-to-prompt Cargo.toml src/{{file}} err.txt tests.txt responses/status-report.md README.md > responses/context.md
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
	files-to-prompt Cargo.toml src/ err.txt tests.txt responses/status-report.md README.md > responses/context.md
	cat responses/context.md | llm --model "openrouter/{{model}}" --system "`cat sys-prompts/err-sys-prompt.md`" > responses/err-recommendations.md
	@echo "Finished!"


ask-status model:
	@echo "Cleaning!"
	-rm err.txt tests.txt
	-rm responses/context.md
	@echo "Asking!"
	-cargo build &> err.txt
	-cargo test &> tests.txt
	files-to-prompt Cargo.toml src/ err.txt tests.txt  responses/status-report.md README.md > responses/context.md
	cat responses/context.md | llm --model "openrouter/{{model}}" --system "`cat sys-prompts/status-sys-prompt.md`" > responses/status-report.md
	@echo "Finished!"

ask model:
	@echo "Cleaning!"
	-rm err.txt tests.txt
	-rm responses/context.md
	@echo "Asking!"
	-cargo build &> err.txt
	-cargo test &> tests.txt
	files-to-prompt Cargo.toml src/ err.txt tests.txt  responses/status-report.md README.md > responses/context.md
	cat responses/context.md | llm --model "openrouter/{{model}}" --system "`cat sys-prompts/sys-prompt.md`" > responses/recommendations.md
	@echo "Finished!"

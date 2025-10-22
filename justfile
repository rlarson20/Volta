check:
	cargo check
	cargo build
	cargo test

ask-custom model prompt:
	@echo "Cleaning!"
	-rm err.txt tests.txt
	-rm context.md
	-rm err-recommendations.md
	-rm recommendations.md
	@echo "Asking!"
	-cargo build &> err.txt
	-cargo test &> tests.txt
	files-to-prompt Cargo.toml src/lib.rs err.txt tests.txt status-report.md > context.md
	cat context.md | llm --model "openrouter/{{model}}" --system "{{prompt}}" > custom-resp.md
	@echo "Finished!"

ask-err model:
	@echo "Cleaning!"
	-rm err.txt tests.txt
	-rm context.md
	-rm err-recommendations.md
	-rm recommendations.md
	@echo "Asking!"
	-cargo build &> err.txt
	-cargo test &> tests.txt
	files-to-prompt Cargo.toml src/lib.rs err.txt tests.txt status-report.md > context.md
	cat context.md | llm --model "openrouter/{{model}}" --system "`cat err-sys-prompt.md`" > err-recommendations.md
	@echo "Finished!"


ask-status model:
	@echo "Cleaning!"
	-rm err.txt tests.txt
	-rm context.md
	-rm err-recommendations.md
	-rm recommendations.md
	@echo "Asking!"
	-cargo build &> err.txt
	-cargo test &> tests.txt
	files-to-prompt Cargo.toml src/lib.rs src/notes.rs err.txt tests.txt o3-deep-research-plan.md status-report.md > context.md
	cat context.md | llm --model "openrouter/{{model}}" --system "`cat status-sys-prompt.md`" > status-report.md
	@echo "Finished!"

ask model:
	@echo "Cleaning!"
	-rm err.txt tests.txt
	-rm context.md
	-rm err-recommendations.md
	-rm recommendations.md
	@echo "Asking!"
	-cargo build &> err.txt
	-cargo test &> tests.txt
	files-to-prompt Cargo.toml src/lib.rs src/notes.rs err.txt tests.txt o3-deep-research-plan.md status-report.md > context.md
	cat context.md | llm --model "openrouter/{{model}}" --system "`cat sys-prompt.md`" > recommendations.md
	@echo "Finished!"

jumpin:
	nvim src/lib.rs err-recommendations.md recommendations.md status-report.md

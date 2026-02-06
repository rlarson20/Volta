main task:
go through pytorch examples (~/src/forks/pytorch-examples/README.md is a good start)
for each:
determine if i can run them in Volta or not right now
if i can
run a side by side comparison on CPU
else
figure out minimal set of changes to make to get them runnable
add tests for each example

most of the basic examples are done

subtask that's probably needed to finish main task:
figure out how to load torch models
how to load other model formats (onnx, candle, ggml, huggingface etc)
go over save/load mechanics, could be unsafe, could accidentally allow loading arbitrary code or smth
add tests for each format

- Full HuggingFace Transformers compatibility
- ONNX support
- Tokenizer integration
- Model hub integration
- Automatic architecture inference from weights

memory usage does not go down
actual memory usage of the matmul check ~128MB

when I run the matmul itself `cargo bench --features gpu -- matmul_cpu_vs_gpu`, it's fine
when I run it in the series of benchmarks `just bench-gpu`, then there's the problem (blows up to expecting it to take like 500s)
it's always the last of the benches in the series
crashes similar to when I open way too many tabs in firefox

if it gets to gpu run on the last bench it crashes
it crashes after running the cpu_mul
but before completing the gpu mul
after the biggest mem transfer
also actually read criterion docs

```markdown
Put yourself in the role of a senior machine learning engineer, with a focus on Rust and WebGPU/wgpu programming.
I need you to investigate deeper architectural issues that are revealed by the following behaviors.

I run individual benchmarks (ex: `cargo bench --features gpu -- matmul_cpu_vs_gpu`) and everything works fine.
I run benchmarks as a whole ( `just bench-gpu`) and then once it reaches the last set of tests, it explodes (blows up to expecting it to take like 500s), crashing my computer similar to when I open way too many browser tabs.
Recent changes have slightly improved the situation, but they have only been sufficient in going from fully crashing, with a crash report from MacOS, to the crashing described before.

Write a comprehensive report on the issues revealed in your analysis and save it to GPU_BENCH_REPORT.md
```

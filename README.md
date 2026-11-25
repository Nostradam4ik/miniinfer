# MiniInfer

A small LLM inference engine written from scratch in C++/CUDA: hand-written kernels, no cuDNN, no
PyTorch at runtime. It runs [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
(Llama architecture, GQA, tied embeddings) end to end on the GPU, and it was built as a learning project
to understand where the time goes in LLM decoding.

**Result:** decode speed went from **59 to 455 tokens/s (17.0 to 2.2 ms/token, 7.7x)** on an RTX 4060 Laptop,
and every step was checked against PyTorch (logits match to ~1e-5 relative error, greedy generation matches
token for token).

| | |
|---|---|
| Model | SmolLM2-135M, 30 layers, 9 query / 3 KV heads, bf16 checkpoint |
| GPU | NVIDIA GeForce RTX 4060 Laptop (Ada, sm_89, 8 GB, ~256 GB/s) |
| Decode speed | 2.2 ms/token, ~455 tok/s (bf16 weights + CUDA Graph) |
| Correctness | logits within 1.05e-5 of PyTorch; CUDA Graph path is bitwise identical to the eager path |
| Tests | 153 checks (`tests.exe`), including comparison against PyTorch activations |

**Against llama.cpp** (same model, bf16, same GPU): single-token decode is **1.12x faster** than llama.cpp
(437 vs 390 tok/s), but prefill is **1.9x to 4.5x slower**. Details and caveats are
[below](#comparison-with-llamacpp).

## Comparison with llama.cpp

llama.cpp build `b12728` (official Windows CUDA 13.5 binary) with GGUF files from
[ggml-org/SmolLM2-135M-GGUF](https://huggingface.co/ggml-org/SmolLM2-135M-GGUF). Median of 7 rounds in
tok/s, min to max in parentheses. Decode is 128 single-token steps.

| Engine | Decode | Prefill, 128 tokens | Prefill, 512 tokens |
|---|---|---|---|
| **MiniInfer bf16** (CUDA Graph for decode) | **437** (405-446) | 6,622 (6,595-6,639) | 5,049 (4,756-5,972) |
| MiniInfer bf16, no graph | 339 (314-357) | | |
| llama.cpp BF16 | 390 (377-399) | **12,870** (3,404-13,025) | **22,777** (15,599-26,007) |
| llama.cpp Q8_0 | 454 (429-469) | 12,315 (8,995-12,576) | 25,661 (23,883-26,453) |
| llama.cpp Q4_K_M | 450 (418-475) | 10,608 (9,850-11,226) | 23,887 (22,926-25,381) |

**Method.** [tools/compare_llamacpp.py](tools/compare_llamacpp.py) alternates the engines, rotating the order
each round, so thermal and clock drift of the laptop GPU affects all of them. Both sides report the *mean*
per-token time of a run (`llama-bench` reports a mean). Neither side times sampling; MiniInfer's decode time
includes copying the logits to the host. Greedy output of MiniInfer and llama.cpp BF16 was identical on 3
prompts of 60 tokens each.

**How to read it.**

- The decode lead is **not** evidence that this engine is better than llama.cpp. MiniInfer is hard-wired for
  one architecture and replays one CUDA Graph; llama.cpp is a general engine that supports many
  architectures and backends. Without the graph, MiniInfer is slower than llama.cpp (339 vs 390).
- Prefill is the clear weakness: decode got the optimization work, large-batch matmul and attention did not.
  The cause has not been profiled yet.
- Not identical settings: llama.cpp ran with its defaults, which store the KV cache in fp16 (MiniInfer uses
  fp32) and keep the token embedding lookup on the CPU.
- Quantized llama.cpp is only ~15% faster than its own bf16 in decode here, although the weights are much
  smaller. For a model this small, memory bandwidth is not the only limit.
- One laptop GPU, Windows, one tiny model. Do not extrapolate to larger models or batched serving.

To reproduce:

```
.venv\Scripts\python.exe tools\compare_llamacpp.py --llama-bench <path>\llama-bench.exe --gguf <path>\SmolLM2-135M-BF16.gguf
```

## What is implemented

- **Tensor and memory:** a small `Tensor` type, a pooled GPU allocator (no `cudaMalloc` per op) and an arena
  allocator for CUDA Graph buffers.
- **Matmul kernels:** naive, shared-memory tiled, and 4x4 register-tiled (benchmarked against cuBLAS),
  plus a bandwidth-oriented **GEMV** kernel for decoding (M <= 4) with vectorized loads.
- **Transformer ops:** RMSNorm, RoPE, causal attention with GQA and KV cache, SiLU-gated MLP, embedding.
- **Fused kernels for decoding:** rmsnorm+QKV, RoPE+KV-cache write, o_proj+residual, rmsnorm+gate+up+SiLU,
  down+residual. 455 kernel launches per token became 184.
- **Low-precision weights:** fp16 and bf16 storage with fp32 accumulation. bf16 is lossless for this
  checkpoint (it is stored in bf16); fp16 flushes 24 of 163M weights to zero.
- **CUDA Graph decode:** the whole decode step is captured once and replayed with one launch. Token and
  position live in device memory so one graph serves every position.
- **Generation:** KV cache, greedy / temperature / top-p sampling, CLI (`generate.exe`) and a Python wrapper
  for tokenization (`tools/chat.py`).
- **Test harness:** own minimal framework; every kernel has a CPU reference, and the model is compared with
  PyTorch activations layer by layer.

## Performance journey

Measured with `bench_decode.exe` (median of 64 steps, including the logits copy to the host).
Memory-bandwidth floor for fp32 weights: ~540 MB per token at ~256 GB/s = ~2.1 ms.

| Step | ms/token | tok/s | What changed |
|---|---|---|---|
| Baseline | 17.0 | 59 | `cudaMalloc`/`cudaFree` for each of 455 ops per token |
| + memory pool, no needless zeroing | 12.0 | 84 | 0 `cudaMalloc` per token; the implicit sync from `cudaFree` is gone |
| + GEMV kernel for M <= 4 | 5.7 | 174 | M=1 matmul: 9.0 to 2.3 ms per token (`bench_ops.exe`) |
| + kernel fusion | 3.6 | 278 | 455 to 184 launches per token |
| + fp16 weights | 2.9 | 347 | weights 651 to 326 MB |
| + bf16 weights | 2.7 | 373 | same size, but lossless |
| + CUDA Graph (bf16) | **2.2** | **455** | one launch per step; logits bitwise identical |

Total: 17.0 to 2.2 ms/token = 7.7x. Run-to-run spread is about +-0.2 ms (laptop GPU clocks vary with
temperature).

### How each step was found

I profiled before optimizing each time:

- `bench_ops.exe` showed a 64x64-tile matmul taking 44 us at M=1 regardless of size, even for a raw launch
  with no allocation. The problem was the kernel shape (tiles built for large M), not memory or launches.
  That led to the GEMV kernel.
- `bench_layer_ops.exe` showed most layer ops (QKV, RoPE, attention, o_proj) cost 9-12 us against ~7.5 us for
  an empty launch, i.e. they were launch-bound. That motivated fusion and then CUDA Graphs.
- Inside the graph (`bench_layer_ops.exe --graph`, bf16) the pure graph time is ~2.05 ms: ~1.1 ms of weight
  streaming from DRAM plus ~0.9 ms of accumulated latency from ~180 small kernels. `lm_head` already runs at
  ~243 GB/s, close to the memory limit. Launches are no longer the bottleneck.
- GEMV vector width (1, 2 or 4 columns per thread) made no measurable difference, even inside the graph.

### What is left on the table

- Prefill: 1.9x to 4.5x behind llama.cpp, the largest known gap.
- Device-side argmax for greedy decoding (skips the 196 KB logits copy every step, est. 0.1-0.3 ms).
- Bigger fusions to cut the ~0.9 ms of small-kernel latency.
- int8/int4 weights (the quantized llama.cpp results suggest a modest decode gain on a model this small).

## Matmul benchmark (fp32, GFLOPS)

| N | naive | tiled | reg (4x4) | cuBLAS |
|---|---|---|---|---|
| 512 | 484 | 632 | 2704 | 4024 |
| 1024 | 533 | 726 | 3713 | 5278 |
| 2048 | 697 | 1019 | 4821 | 6497 |

RTX 4060 Laptop, mean of 20 runs, `matmul.exe <N>`. After generalizing the kernels to arbitrary M, N, K the
`reg` kernel became ~20% slower than the square-only version (4668 GFLOPS at N=1024); the cause has not been
investigated.

## Build

Requirements: Windows, Visual Studio 2022 Build Tools (C++), CUDA Toolkit 12+, CMake 3.24+, Ninja.
The CUDA architecture is set to `89` in [CMakeLists.txt](CMakeLists.txt); change
`CMAKE_CUDA_ARCHITECTURES` for another GPU.

From an "x64 Native Tools Command Prompt for VS 2022":

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

[build.bat](build.bat) does the same with the author's local paths (CUDA in `C:\Code\cuda-13.5`); edit the
paths at the top for your machine.

## Run

Set up the model and the PyTorch reference once (downloads SmolLM2-135M, writes `model/` and `reference/`):

```
py -m venv .venv
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install transformers safetensors numpy huggingface_hub
.venv\Scripts\python.exe tools\export_reference.py
```

Generate text (bf16 weights and CUDA Graph are on by default in `chat.py`):

```
.venv\Scripts\python.exe tools\chat.py "Once upon a time" --max 60
.venv\Scripts\python.exe tools\chat.py "The best way to learn programming is" --max 60 --temp 0.8 --top-p 0.95 --seed 3
```

This is a base model: it continues text, it does not answer questions. Tokenization happens in Python;
`chat.py` passes token ids to `build\generate.exe` (`generate.exe --help`, flags `--dtype f32|f16|bf16`,
`--graph`).

Benchmarks: `bench_decode.exe [--dtype bf16] [--graph]`, `bench_ops.exe`, `bench_layer_ops.exe [--graph]`,
`matmul.exe <N>`.

## Tests

```
build\tests.exe            # all tests
build\tests.exe matmul     # only tests with "matmul" in the name
```

The `ref_*` tests need `model/` and `reference/` from the step above. Without them they print a "skipped"
message and still pass, so check that the message is absent.

How correctness is established:

- every kernel has a CPU reference implementation;
- activations are compared with PyTorch layer by layer, then logits and greedy generation end to end;
- buffers are surrounded by NaN-poisoned padding, because out-of-bounds reads normally hit zeroed memory and
  go unnoticed;
- the difference helper treats NaN as a failure (`std::fmax` silently ignores NaN);
- the CUDA Graph path is checked for bitwise equality with the eager path in fp32, fp16 and bf16;
- bugs were injected on purpose (mutation checks) to confirm that the tests actually fail.

## Project layout

```
src/        kernels and engine (ops.cu, gemv.cu, ops_fused.cu, lowp.cu, model.cuh, graph.cuh, gpu_alloc.cuh, ...)
            bench_*.cu, generate_cli.cu, matmul.cu: executables
tests/      test framework and tests
tools/      export_reference.py (model + PyTorch reference), chat.py (tokenization + CLI wrapper),
            compare_llamacpp.py (benchmark against llama.cpp)
```

Source comments are written in Ukrainian.

## Limitations

- Single model family (Llama-style, small), single GPU, batch size 1 in the optimized decode path.
- Attention keeps scores in shared memory, so the context is limited to 12,288 tokens.
- Tokenization is done in Python, not in C++.
- Tuned and measured on one GPU (RTX 4060 Laptop) only.

## License

MIT, see [LICENSE](LICENSE).

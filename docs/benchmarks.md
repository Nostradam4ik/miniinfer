# Benchmarks

Details behind the numbers in the [README](../README.md). GPU: NVIDIA GeForce RTX 4060 Laptop.

## Comparison with llama.cpp

llama.cpp build `b12728` (official Windows CUDA 13.5 binary) with GGUF files from
[ggml-org/SmolLM2-135M-GGUF](https://huggingface.co/ggml-org/SmolLM2-135M-GGUF). Median of 7 rounds in
tok/s, min to max in parentheses. Decode is 128 single-token steps. Measured before the
[shared attention change](#shared-k-and-v-loads-in-attention).

| Engine | Decode | Prefill, 128 tokens | Prefill, 512 tokens |
|---|---|---|---|
| **MiniInfer bf16** (CUDA Graph for decode) | **407** (388-441) | 8,460 (8,226-8,482) | 9,294 (9,149-9,329) |
| MiniInfer bf16, no graph | 342 (188-360) | | |
| MiniInfer bf16, `wmma` prefill (rounds activations to bf16) | | 12,561 (10,847-12,673) | 13,502 (13,152-13,534) |
| llama.cpp BF16 | 369 (336-378) | 11,282 (6,932-13,075) | **23,132** (21,930-23,782) |
| llama.cpp Q8_0 | 429 (417-461) | 11,366 (9,219-11,925) | 23,868 (22,968-24,514) |
| llama.cpp Q4_K_M | 446 (403-462) | 10,265 (9,879-11,036) | 20,429 (18,413-23,020) |

**Method.** [tools/compare_llamacpp.py](../tools/compare_llamacpp.py) alternates the engines, rotating the order
each round, so thermal and clock drift of the laptop GPU affects all of them. Both sides report the *mean*
per-token time of a run (`llama-bench` reports a mean). Neither side times sampling; MiniInfer's decode time
includes copying the logits to the host. Greedy output of MiniInfer (default mode, an earlier build) and
llama.cpp BF16 was identical on 3 prompts of 60 tokens each.

**How to read it.**

- The decode lead is **not** evidence that this engine is better than llama.cpp. MiniInfer is hard-wired for
  one architecture and replays one CUDA Graph; llama.cpp is a general engine that supports many
  architectures and backends. Without the graph, MiniInfer is slower than llama.cpp BF16 (342 vs 369), and
  quantized llama.cpp is faster than MiniInfer even with the graph.
- Prefill is the clear weakness. At 512 tokens llama.cpp BF16 is 2.5x faster than the default mode and still
  1.7x faster than `wmma`.
- At 128 tokens llama.cpp BF16 ranged from 6,932 to 13,075 tok/s in this run, so do not rely on that column.
  Its median here gives llama.cpp a 1.3x lead over the default mode; in two earlier runs its medians (12,122 and
  14,468) gave 1.5x and 1.7x. The `wmma` median is above this run's llama.cpp BF16 median, but inside its range
  and below both earlier medians, so it does not show which engine is faster.
- The `wmma` row is not the same computation as the default: it rounds activations to bf16 (see
  [tensor-core-prefill.md](tensor-core-prefill.md) for what that changes). How llama.cpp rounds its activations was not
  checked.
- Compare numbers only within one run. With the same llama.cpp build, three runs of the script gave llama.cpp
  BF16 prefill medians of 23,017, 26,873 and 23,132 tok/s at 512 tokens. The cause is not known.
- Not identical settings: llama.cpp ran with its defaults, which store the KV cache in fp16 (MiniInfer uses
  fp32) and keep the token embedding lookup on the CPU.
- Quantized llama.cpp is only 16-21% faster than its own bf16 in decode here, although the weights are much
  smaller. For a model this small, memory bandwidth is not the only limit.
- One laptop GPU, Windows, one tiny model. Do not extrapolate to larger models or batched serving.

To reproduce:

```
.venv\Scripts\python.exe tools\compare_llamacpp.py --mini-wmma --llama-bench <path>\llama-bench.exe --gguf <path>\SmolLM2-135M-BF16.gguf
```

## Prefill

Time of one prefill in ms, `bench_decode.exe --dtype bf16`, median of 7 rounds that alternated three builds:
with the scalar dot product in the attention kernel, after the attention change, and after the later matmul
change.

| Prompt tokens | Scalar | + attention | + matmul |
|---|---|---|---|
| 32 | 11.9 | 11.7 | 11.1 |
| 128 | 19.3 | 15.9 | 15.1 |
| 512 | 107.5 | 57.4 | 55.0 |
| 1024 | 250.2 | 136.9 | 135.2 |

At 1024 tokens the last step is within the spread (126-147 ms before it, 119-137 ms after). Decode time per
token did not change beyond the run-to-run spread, and after each of the two changes the prefill logits for a
512-token prompt were bitwise identical in bf16 and fp32.

- Tracing the prefill with Nsight Systems showed that attention dominated long prompts: 74 of 108 ms at 512
  tokens (that traced run was slower than the untraced ones). GPU idle time was under 1 ms, so kernel launches
  were not the problem.
- Variants of the attention kernel showed that the Q.K dot product took about 95% of its time. Loading it as
  `float4` instead of single floats made the kernel about 3x faster at 512 keys. Keeping Q in registers did not
  help, and neither did using more threads for the weighted sum of V. The likely reason is the number of load
  instructions (neighboring threads read different keys, so their loads cannot be coalesced); it was not
  checked with hardware counters.
- Traced again after the attention change, the tiled matmul was the largest cost: 12.1 of 15.8 ms at 128
  tokens and 30.4 of 57.5 ms at 512. Attention took 2.0 and 23.9 ms.
- The matmul kernel now loads four elements of A and of B per instruction when K and N are multiples of 4. In a
  standalone harness this was 22-24% faster, but inside the model the gain is much smaller (table above). The
  cause of the difference is not known.

### Shared K and V loads in attention

Three query heads share each KV head. For prefill, one attention block per (token, KV head) now serves all
three, so K and V are loaded once instead of three times. It runs when there are more than 4 query rows, the
position comes from the host and the scores of three heads fit into the default shared memory (at most 4085
keys). Decoding uses the per-head kernel as before.

Time of one prefill in ms, `bench_decode.exe --dtype bf16`, median (min-max) of 7 rounds that alternated the
build before the change and the build with it:

| Prompt tokens | `reg` before | `reg` after | `wmma` before | `wmma` after |
|---|---|---|---|---|
| 5 | 10.77 (10.71-11.31) | 10.83 (10.74-11.38) | 7.28 (7.11-7.60) | 7.21 (7.16-7.67) |
| 16 | 10.94 (10.83-11.39) | 10.93 (10.86-11.40) | 7.33 (7.29-7.42) | 7.42 (7.31-7.86) |
| 32 | 11.15 (11.07-11.71) | 11.11 (11.01-11.50) | 7.61 (7.59-8.14) | 7.59 (7.51-8.08) |
| 128 | 15.17 (15.10-15.72) | 14.31 (14.27-14.34) | 10.12 (10.08-10.51) | 9.58 (9.29-10.65) |
| 512 | 55.53 (54.72-55.72) | 42.40 (42.29-43.19) | 37.97 (37.81-38.33) | 25.42 (25.27-26.13) |
| 1024 | 130.13 (127.09-140.61) | 105.97 (100.25-107.57) | 107.56 (101.52-120.84) | 74.60 (72.09-74.67) |

- The medians fell by 24% (`reg`) and 33% (`wmma`) at 512 tokens, and by 19% and 31% at 1024. At 128 tokens
  they fell by 5-6%; in `wmma` the ranges overlap there. At 5 to 32 tokens the differences are within the spread.
- Decode time per token (mean of 128 steps, median of the same 7 rounds) stayed within the spread: 2.40
  (2.23-2.74) before and 2.46 (2.26-2.81) after with the CUDA Graph, 3.07 (2.93-3.61) and 2.97 (2.76-3.02)
  without it.
- All logits of a 512-token prefill in the default mode are bitwise identical to the build before the change,
  in bf16 and fp32.
- In a standalone harness the attention kernel itself took 32-50% less time at 32 to 1024 tokens. For one token
  with 1024 keys it took 17% more, which is why decoding keeps the per-head kernel.
- The llama.cpp comparison and the `reg` against `wmma` tables were measured before this change.

## Matmul benchmark (fp32, GFLOPS)

| N | naive | tiled | reg (4x4) | cuBLAS |
|---|---|---|---|---|
| 512 | 484 | 632 | 2704 | 4024 |
| 1024 | 533 | 726 | 3713 | 5278 |
| 2048 | 697 | 1019 | 4821 | 6497 |

RTX 4060 Laptop, mean of 20 runs, `matmul.exe <N>`. After generalizing the kernels to arbitrary M, N, K the
`reg` kernel became ~20% slower than the square-only version (4668 GFLOPS at N=1024); the cause has not been
investigated.

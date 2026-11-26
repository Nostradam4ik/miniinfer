# Tensor core prefill (opt-in)

`--prefill-matmul wmma` (`generate.exe`, `bench_decode.exe`, `tools/chat.py`) runs the large matmuls of the
prefill on tensor cores with a WMMA kernel ([src/wmma.cu](../src/wmma.cu)). It needs bf16 weights and a GPU of
compute capability 8.0 or newer. The default is `reg`, whose results this mode does not change. Single-token
decoding does not use it.

The price is accuracy: the tensor cores take bf16 inputs, so the activations are rounded from fp32 to bf16
(the weights are bf16 already). Speed and accuracy belong together:

**Speed.** One prefill in ms, `bench_decode.exe --dtype bf16`, median of 7 rounds that alternated the
variants (measured before the [shared attention change](benchmarks.md#shared-k-and-v-loads-in-attention)):

| Prompt tokens | `reg` | `wmma` | Time saved |
|---|---|---|---|
| 32 | 11.16 | 7.72 | 31% |
| 128 | 15.13 | 10.12 | 33% |
| 512 | 54.89 | 37.89 | 31% |
| 1024 | 141.5 | 109.6 | unreliable, see below |

At 1024 tokens the timings were unstable (`reg` 120.7-142.6 ms, `wmma` 97.7-120.6 ms, and the previous build,
which runs the same code as `reg`, had a median of 130.3 ms). Decode time per token did not change beyond the
spread.

**Accuracy.** A 497-token English text, written for this project, compared with PyTorch (fp32):

| | `reg` | `wmma` |
|---|---|---|
| Positions with the same most likely next token | 497 of 497 | 492 of 497 (99.0%) |
| Largest logit difference | 2.8e-4 | 0.79 |
| Mean logit difference | 1.7e-5 | 0.056 |
| Largest difference relative to the largest logit | 7.2e-6 | 2.1e-2 |

Logit differences are measured at every 32nd position and the last one (17 positions). For this comparison the
logits of every position are computed, so `lm_head` also runs on tensor cores; in generation it does not. On 3
short prompts (5, 15 and 7 tokens), 60 greedy tokens were the same with `reg` and `wmma`. This is one text and
three prompts: it does not show that the output stays the same on other input.

The kernel is tested against a double-precision CPU product of the bf16-rounded inputs, which separates kernel
bugs from the cost of the format. All logits of a 512-token prefill in the default mode are bitwise identical
to the build before the kernel was added.

See also the [README](../README.md) and [benchmarks.md](benchmarks.md).

# Attention that reads K and V once per KV head

## Intent

Make prefill faster with code the author can explain line by line. A moderate gain with simple code is preferred
over a larger gain with code that is hard to explain.

## Findings

One traced prefill per configuration (bf16 weights): at 512 tokens with `--prefill-matmul wmma`, attention takes
23.9 of 37.3 ms of GPU time; in the default mode 23.9 of 54.4 ms; at 128 tokens about 2.0 ms in both modes.

The model has 9 query heads and 3 KV heads. The current kernel runs one block per (token, query head), so three
blocks load the same K and V.

A throwaway harness compared it with one block per (token, KV head) serving the 3 query heads, 128 threads in
both (random data, ms per call, median of 7 rotated rounds, one run):

| Tokens / keys | Current | Shared | Change in time |
|---|---|---|---|
| 32 / 32 | 0.0144 | 0.0098 | -32% |
| 128 / 128 | 0.0646 | 0.0409 | -37% |
| 512 / 512 | 0.671 | 0.336 | -50% |
| 1024 / 1024 | 2.730 | 1.502 | -45% |
| 1 / 128 (decode) | 0.0140 | 0.0136 | within the spread |
| 1 / 1024 (decode) | 0.0349 | 0.0407 | +17% (slower) |

On this data every output value of the 128-thread shared kernel was bitwise identical to the current one. A
256-thread variant was not (differences up to 9e-8) and was slower at 32 to 512 tokens.

The harness does not show the gain inside the model. An earlier matmul change gained much less in the model than
in its harness.

## Goals

- Prefill time in the model goes down, shown by alternated measurements before and after.
- Logits are bitwise identical to the current build in fp32 and bf16.
- Decode time does not get worse.

## Non-goals

Tiling the keys (FlashAttention style), matmul changes, a longer context.

## Approaches

| Approach | Gain | Cost |
|---|---|---|
| **A. One block per (token, KV head)** (chosen) | Measured above in a harness | Small change; the same arithmetic per head |
| B. Tiled attention over blocks of keys | Probably the largest | Much more complex; results would differ |
| C. Faster matmul | Helps short prompts most | PTX; only the `wmma` mode |

The author chose A.

## Design

- A second kernel in `src/ops.cu` serves all query heads that share a KV head. The softmax and the weighted sum
  keep the order of the current kernel.
- `attention()` uses it only for prefill, and only when the scores of the whole group fit into the default
  48 KB of dynamic shared memory (at most 4096 keys with 3 heads per KV head). Otherwise the current kernel
  runs. So decode and long contexts behave exactly as today. This is the builder's call, open to correction.
- The group size is a template argument; a ratio of heads without an instantiation uses the current kernel.

## Checks

- A test of the new path against `cpu_attention`, including shapes around both switching limits, with NaN
  padding after the inputs. It is shown to fail with an injected bug.
- A bitwise comparison of the two kernels on the same input.
- `build/tests` passes with no `skipped` line; model logits are compared bitwise with a snapshot of the current
  build.
- Prefill and decode are measured with alternated old and new builds, several rounds, the same statistic.
- The smallest prefill sizes above the decode limit are measured too; the harness did not cover them. If the new
  kernel is slower there, the switching rule changes.
- If the gain in the model is within the spread, the change is not merged.

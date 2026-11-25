# Design: opt-in tensor core matmul for prefill

Status: approved on 2026-11-25, including the items marked **(proposal)**.

## Why

Prefill is the largest known gap against llama.cpp BF16 (1.7x at 128 tokens, 2.9x at 512), and the engine
does not use tensor cores. The project is meant to show GPU engineering, so the kernel is our own.

## Goals

- A matmul kernel on tensor cores (WMMA, bf16 inputs, fp32 accumulation) that the model can use for prefill.
- A measured answer to two questions: how much faster is prefill in the model, and how much do the results
  change.

## Non-goals

- Single-token decoding. It keeps the GEMV kernels and the CUDA Graph.
- TF32, hand-written PTX (`mma.sync`, `ldmatrix`), fp32 or fp16 weights, other models.
- cuBLAS inside the engine. It stays a reference point outside the repository.

## Anti-goals

- Changing the results of the default configuration.
- Publishing a speedup without the accuracy numbers next to it.

## Experiment so far

A throwaway harness ran the five large matmuls of all 30 layers (150 calls) with the real weights.
It measures matmul only, not the model. RTX 4060 Laptop, ms, median of 7 rounds:

| Kernel | M=32 | M=128 | M=512 |
|---|---|---|---|
| Library register-tiled (current) | 8.81 | 8.93 | 23.74 |
| WMMA bf16 | 5.66 | 5.24 | 9.58 |
| WMMA TF32 | 10.97 | 10.01 | 18.80 |
| cuBLAS, bf16 inputs | 2.13 | 2.56 | 8.29 |

Relative L2 error against a double-precision CPU product (worst of 5 layers, M=512, inputs built from real
token embeddings): current kernel 4e-7 to 7e-7, TF32 about 2e-4, bf16 about 1.8e-3. WMMA bf16 and cuBLAS bf16
had the same error to the printed precision.

An earlier matmul change gained 22-24% in a harness and much less in the model, for unknown reasons. The
table above therefore does not predict the speed of the model.

## Approaches

| Approach | For | Against |
|---|---|---|
| **WMMA (chosen)** | Short, explainable C++ | Less control than PTX; behind cuBLAS at M=32 and 128 |
| `mma.sync` and `ldmatrix` in PTX | More control, likely faster | Much harder to write and explain |
| cuBLAS | Fastest in the harness, little code | New dependency; not our kernel |

Format: **bf16 (chosen)**. TF32 had about 8x lower error, but this TF32 kernel was slower than the current
kernel at M=32 and 128 and only about 21% faster at 512. It was not tuned further.

## Design

### Kernel (new file `src/wmma.cu`)

- `C[M x N] = A[M x K] (fp32) * B[K x N] (bf16)`, any M; K a multiple of 32, N a multiple of 64.
- 64x64 output tile, 128 threads; each of 4 warps computes 32x32 with four 16x16 accumulator fragments.
  K step 32 (two `m16n16k16` steps).
- A is rounded to bf16 (nearest even) while it is staged in shared memory. B is copied unchanged.
- The input tiles and the output tile share one shared-memory buffer.
- A must be aligned to 16 bytes and B to 8 bytes.
- Launched on `exec_stream()`. It writes every element of C and nothing else.
- Requires compute capability 8.0. The project must still compile for older targets; selecting the kernel
  there is a runtime error. **(proposal)**

### Selection

- New `MatmulKind::Wmma`. Unsupported shapes, weight formats or alignment throw `std::invalid_argument`.
  There is no silent fallback, so a benchmark cannot measure a different kernel than it names. **(proposal)**
- `Model` gets a setting for the matmul used when M > `kGemvMaxRows`: `Reg` (default) or `Wmma`. It is passed
  explicitly through the cached forward path and the large-M fallbacks of the fused operations, not through a
  global. Choosing `Wmma` with weights other than bf16 is an error. **(proposal)**
- The uncached reference path (`Model::forward`) does not change.
- `generate`, `bench_decode` and `tools/chat.py` get `--prefill-matmul reg|wmma`. The output format that the
  tools parse does not change.

## Verification

1. **Kernel against the CPU.** The independent reference rounds A to bf16 on the host and multiplies in double
   precision. This separates kernel bugs from the cost of the format. The tolerance of each element is relative
   to the same element of `|A| * |B|` (fp32 accumulation error). **(proposal)** Also: an exact test with small
   integers; M below, at and above the tile size; NaN-poisoned padding around A, B and C; a dirty pool block
   as output; the argument errors.
2. **Mutation checks.** Inject a bug into the A staging, the B staging, the fragment offsets and the output
   guard. Each one must make a test fail.
3. **Default path unchanged.** All prefill logits for 512 tokens are bitwise identical to the build before
   the change, in bf16 and fp32.
4. **Model accuracy.** `tools/export_reference.py` also exports one long natural English text (about 512
   tokens, written for this project) with the PyTorch top-1 token and logits at the positions needed for the
   comparison. For `reg` and `wmma` report: maximum and typical logit error against PyTorch, the share of
   positions with the same top-1 token, and whether 60 greedy tokens are the same with `reg` and `wmma` on
   3 prompts. The model and reference folders must be exported again. **(proposal)**
   Computing logits at every position also runs `lm_head` through the large-M matmul. Generation does not
   (it needs only the last position, which uses GEMV), so this comparison includes a little extra error.
5. All tests pass with no `skipped` line, and `pre-commit run --all-files` passes.

## Performance measurement

`bench_decode --dtype bf16`, prefill at 32, 128, 512 and 1024 tokens, `reg` and `wmma` alternated over
7 rounds, medians with min and max. Decode time per token is checked too. If the gain in the model is far
below the harness, trace both with Nsight Systems and report what is found, including "cause unknown".

## Keep or drop **(proposal)**

Keep the mode if both hold:

- prefill of 512 tokens in the model is at least 20% faster than with `reg`;
- the top-1 token matches PyTorch at 95% or more of the positions of the long text.

If one of them fails, stop and decide together. The numbers are reported in either case. `reg` stays the
default in either case.

## Documentation

README gets a short section on the fast mode with speed and accuracy side by side. Its claims about matching
PyTorch must say that they describe the default mode. The existing llama.cpp rows stay as the default mode.
A `wmma` row is added only from a new run of the whole comparison, labeled as rounding activations to bf16,
with all existing caveats.

## Left to the implementer

Names inside the kernel file, shared-memory padding, and the order of the commits.

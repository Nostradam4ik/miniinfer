# AGENTS.md

MiniInfer: a from-scratch C++/CUDA LLM inference engine that runs SmolLM2-135M (Llama architecture, GQA,
tied embeddings). It is a learning and portfolio project: the author must be able to explain every line.

If `CLAUDE.local.md` exists, read it too: it holds machine-specific paths and developer preferences.

## Rules

**Correctness**

- A kernel change is not done until `build/tests` passes **and** the output contains no `skipped` line.
  The `ref_*` tests compare against PyTorch and silently pass when `model/` or `reference/` is missing.
- A new or changed kernel gets a test against an independent CPU reference (`src/ops_cpu.h` or plain math in
  the test), not against another GPU kernel.
- Check that a new test can fail: inject the bug it targets, see it fail, restore the code. Rewrite the
  restored file rather than copying a backup over it, so that its timestamp changes and Ninja rebuilds.
- Optimizations must not change results. State the tolerance; where the claim is "identical", compare bitwise.

**Performance**

- Measure before and after on the same machine. Do not describe a change as faster without both numbers.
- Profile first (`bench_ops`, `bench_layer_ops`, `bench_layer_ops --graph`), then optimize.
- Laptop GPU clocks drift. For comparisons, alternate the configurations and repeat
  (see `tools/compare_llamacpp.py`). Compare the same statistic (mean with mean, median with median).
- The llama.cpp comparison in the README keeps its caveats. Never shorten it to "faster than llama.cpp":
  prefill is several times slower.

**Style**

- Code, comments, messages and docs are in English.
- Comments explain the code as it is now. History and rationale of a change go into the commit message.
- `pre-commit run --all-files` must pass (clang-format 19, ruff).
- See [CONTRIBUTING.md](CONTRIBUTING.md) for the C++ and CUDA conventions.
- Do not add dependencies to the engine without asking.

## Commands

| Task | Command |
|------|---------|
| Build (Windows, any terminal) | `build.bat` |
| Build (any OS, compiler on PATH) | `cmake --preset release && cmake --build --preset release` |
| All tests | `build/tests` |
| Tests by name substring | `build/tests graph` |
| CPU-only tests (no GPU needed) | `build/tests sampler` |
| Model + PyTorch reference (once, ~650 MB) | `python tools/export_reference.py` |
| Generate text | `python tools/chat.py "Once upon a time" --max 60` |
| Decode benchmark | `build/bench_decode --dtype bf16 --graph` |
| Lint and format | `pre-commit run --all-files` |

Python commands use the virtual environment from the README. On Windows `build.bat` may print
"vswhere.exe is not recognized" from `vcvars64.bat`; judge the build by its exit code.

## Architecture

`tokens -> embedding -> 30 x decoder layer -> RMSNorm + lm_head -> logits -> sampler`

| File | Contents |
|------|----------|
| `src/tensor.cuh` | `Tensor` (owning, move-only, fp32), `DType`, `WeightRef` (non-owning weight matrix) |
| `src/gpu_alloc.cuh` | Pooled allocator and `GpuArena` for CUDA Graph buffers |
| `src/cuda_utils.cuh` | `CUDA_CHECK`, `exec_stream()`, event timer |
| `src/ops.cu`, `src/ops.cuh` | Matmul variants, attention, RoPE, RMSNorm, embedding |
| `src/gemv.cu` | Bandwidth-oriented GEMV kernels for decoding (M <= 4) |
| `src/ops_fused.cu` | Fused decode operations; fall back to separate ops for larger M |
| `src/lowp.cu`, `src/wt_convert.cuh` | fp16/bf16 weights and conversion |
| `src/model.cuh` | Model loading, KV cache, forward passes |
| `src/graph.cuh` | `DecodeGraph`: one decode step as a CUDA Graph |
| `src/generate.h`, `src/sampler.h` | Generation loop and sampling |

## Pitfalls

- Every kernel launch in the library goes to `exec_stream()`. A launch on the default stream is silently left
  out of the captured graph.
- Inside graph capture there must be no host synchronization or host copies, and all allocations come from the
  arena. Values that change between steps (token, position) live in device memory.
- `Tensor::uninitialized` may return a dirty pool block: the kernel must write every element. `test_alloc.cu`
  checks this with NaN-filled blocks.
- Memory past a buffer is often zero, so out-of-bounds reads pass ordinary tests. Use NaN-poisoned padding.
- `std::fmax` ignores NaN. Difference helpers must treat NaN as a failure explicitly.
- Attention keeps scores in shared memory: the number of keys is limited by `kMaxAttentionKeys`.
- `tools/chat.py` and `tools/compare_llamacpp.py` parse the program output; change both sides together.

## Git

- Small commits with an imperative subject; the body explains why and how it was verified.
- Keep formatting-only changes in separate commits and add them to `.git-blame-ignore-revs`.
- Ask before pushing. Never commit `model/`, `reference/` or `build/`.

# Tensor Core Prefill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An opt-in WMMA bf16 matmul for prefill, with its speed and its accuracy cost measured in the model.

**Architecture:** A new kernel in `src/wmma.cu` is reachable as `MatmulKind::Wmma`. `Model` holds the kind used
for large-M matmuls and passes it explicitly down the cached forward path. Decoding and the default
configuration do not change.

**Tech Stack:** C++17, CUDA 13.5, `mma.h` (WMMA), the repository's own test framework, Python for the export.

**Spec:** [2026-11-25-wmma-prefill-design.md](2026-11-25-wmma-prefill-design.md)

## Global Constraints

- The default is `reg`. All 512-token prefill logits in default mode stay bitwise identical, bf16 and fp32.
- bf16 weights only. K a multiple of 32, N a multiple of 64, any M. A aligned to 16 bytes, B to 8 bytes.
- No silent fallback: unsupported shape, format, alignment or GPU throws `std::invalid_argument`.
- Compute capability 8.0 is required to run the kernel; the project still compiles for older targets.
- Every library launch goes to `exec_stream()`.
- No new dependencies. Code, comments and docs in English.
- Done means: `build.bat` exit code 0, `build\tests.exe` with 0 failed and no `skipped` line,
  `pre-commit run --all-files` passes.
- Work on branch `wmma-prefill`. Commit only after the developer has allowed commits on that branch.
  Merging and pushing each need a separate permission.
- Commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Review Focus

1. Prompts of 5 to 63 tokens or not a multiple of 64: results are correct (Task 1, bounds and shape tests).
2. A prompt processed in two chunks with `wmma`: the second chunk sees the cache of the first (Task 2).
3. `--prefill-matmul wmma` with the default `--dtype f32` of `generate`: a clear error, no output (Task 3).
4. Switching the setting between calls must not leave decoding affected (Task 2).
5. Shapes the kernel cannot take (another model, K or N not divisible): an error that names the rule (Task 1).

---

### Task 0: Branch and bitwise baseline (before any source change)

**Files:** none in the repository. Uses `C:\Code\miniinfer-experiments\snapshot.cu` and `snap.bat`.

- [ ] **Step 1:** `git switch -c wmma-prefill`
- [ ] **Step 2:** `build.bat`, then `C:\Code\miniinfer-experiments\snap.bat wmma_base`.
      Expected: `bf16_wmma_base.bin` and `f32_wmma_base.bin`, each "25165824 floats written".
- [ ] **Step 3:** Keep a copy of `build\bench_decode.exe` as
      `C:\Code\miniinfer-experiments\bench_decode_wmma_base.exe`.

### Task 1: Kernel and `MatmulKind::Wmma`

**Files:**
- Create: `src/wmma.cu`, `src/wmma.h`, `tests/test_wmma.cu`
- Modify: `src/ops.cuh` (enum and comments), `src/ops.cu` (`launch_matmul_raw`), `CMakeLists.txt`

**Interfaces:**
- Produces: `MatmulKind::Wmma`; in `src/wmma.h`:
  `bool wmma_supported_by_device();` and
  `void launch_matmul_wmma(const float* A, const void* B, DType dt, float* C, int M, int N, int K);`
  (checks every constraint, launches on `exec_stream()`).
- `matmul(A, W, MatmulKind::Wmma)` works for bf16 `W`. `launch_matmul` takes fp32 weights, so it throws for `Wmma`.
- `Auto` never selects `Wmma`.

- [ ] **Step 1: Write the tests** in `tests/test_wmma.cu`; add the file to the `tests` target.
  Reference: `a_r = round_to_dtype(a, DType::BF16)`, `b_r` likewise; `ref[i,j] = sum_k a_r*b_r` in double,
  `bound[i,j] = sum_k |a_r*b_r|`. An element passes only if `fabs(c - ref) <= 1e-5 * bound` (written so
  that NaN fails).
  - `wmma_matches_rounded_cpu_reference`: random values in [-1, 1]; (M,K,N) = (1,32,64), (5,576,960),
    (31,64,128), (32,576,576), (63,32,192), (64,576,1536), (65,1536,576), (130,96,64), (512,576,576).
  - `wmma_small_integers_are_exact`: `a = i % 7 - 3`, `b = i % 5 - 2`, K=64, N=128, M = 1, 5, 63, 65, 130;
    `CHECK(c == ref)` for every element.
  - `wmma_does_not_touch_memory_out_of_bounds`: raw `launch_matmul_wmma`, 20000 poisoned floats (0xFF)
    after A, B and C; M = 5, 65, 70 with K=64, N=128. Result passes the reference, the tail of C keeps
    its bit pattern.
  - `wmma_overwrites_dirty_pool_block`: fill a tensor of the output size with NaN, destroy it, then
    `matmul(..., Wmma)` with M=65; no NaN in the result.
  - `wmma_zero_inner_dimension_gives_zeros`: `A({3,0})`, bf16 `B({0,64})`; all zeros.
  - `wmma_rejects_unsupported_input`: `CHECK_THROWS` for K=48, N=96, fp32 W, f16 W, A offset by one float,
    B offset by one element, and `launch_matmul(MatmulKind::Wmma, ...)`.
- [ ] **Step 2:** Add the enum value and `src/wmma.h` with throwing stubs; run `build.bat`, then
      `build\tests.exe wmma`. Expected: the correctness tests FAIL.
- [ ] **Step 3: Implement** `src/wmma.cu`. Start from `wmma_kernel<false>` in
      `C:\Code\miniinfer-experiments\mm_tc.cu`: 64x64 tile, 128 threads, K step 32, padding 8, one shared
      buffer for inputs and output. Required changes: no TF32 path; use `src/wt_convert.cuh` conventions;
      a comment that explains the memory access pattern, the warp layout and the synchronization;
      the kernel body guarded by `#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800`
      (with `#include <mma.h>` usable under the same condition);
      `wmma_supported_by_device()` reads `cudaGetDeviceProperties(...).major >= 8`.
      Add `src/wmma.cu` to the `miniinfer` library.
- [ ] **Step 4:** `build.bat`, `build\tests.exe wmma`. Expected: all pass.
- [ ] **Step 5: Mutation checks.** One at a time, rebuild, confirm a `wmma_` test FAILS, restore by editing:
      (a) store two neighbouring A elements swapped in shared memory; (b) load B from `blockCol + c + 4`
      when `blockCol + c + 4 < N`; (c) swap `i` and `j` in the output fragment offsets; (d) remove the row
      guard of the final store. Record which test caught each.
- [ ] **Step 6:** `cmake -S . -B <scratch dir> -G Ninja -DCMAKE_CUDA_ARCHITECTURES=75` and build the
      `miniinfer` target there (inside the same compiler environment as `build.bat`). Expected: it compiles.
- [ ] **Step 7:** Full tests and pre-commit. Commit `Add a tensor core matmul kernel`.

### Task 2: Selection through the model

**Files:**
- Modify: `src/ops.cuh`, `src/ops_fused.cu`, `src/model.cuh`
- Test: `tests/test_fused.cu`, `tests/test_reference.cu`

**Interfaces:**
- Consumes: `MatmulKind::Wmma`, `wmma_supported_by_device()`.
- Produces: a last parameter `MatmulKind large = MatmulKind::Reg` on `rmsnorm_matmul`, `matmul_residual`,
  `rmsnorm_gate_up` and `decoder_layer_cached`; it is used only on the separate-ops path.
  `void Model::set_prefill_matmul(MatmulKind kind);` and `MatmulKind Model::prefill_matmul() const;`.
  The setter accepts `Reg`, or `Wmma` when `dtype == DType::BF16` and the device supports it; anything
  else throws `std::invalid_argument`.
- `forward_cached` passes the setting to every layer and to the final `lm_head` call.
  `forward` and `forward_decode_dev` do not take it.

- [ ] **Step 1: Write the tests.**
  - `fused_large_m_honours_matmul_kind` (`test_fused.cu`): each of the three fused ops, M=70, K=64, N=128,
    bf16 weights, `large = Wmma`: bitwise equal to the same separate ops called with `MatmulKind::Wmma`,
    and not bitwise equal to the `Reg` result.
  - `model_prefill_matmul_defaults_to_reg_and_validates`: default is `Reg`; `Gemv` and `Auto` throw;
    an f32 model throws for `Wmma` and keeps `Reg`.
  - `model_wmma_prefill_differs_slightly_from_reg`: bf16 model, ids `1000 + i*37 % 20000`, 70 tokens,
    all logits: not bitwise equal, `rel_diff <= 0.05` (a sanity bound, not the accuracy report).
  - `model_wmma_chunked_prefill_matches_single_pass`: 40 + 30 tokens against 70 at once, both `wmma`,
    `rel_diff <= 0.05` for the last 30 rows.
  - `model_prefill_setting_does_not_change_decoding`: two caches filled by the same `reg` prefill; one
    single-token step with `Reg`, one with `Wmma`: bitwise equal logits.
- [ ] **Step 2:** Add the signatures without using them; run the tests. Expected: the first and third FAIL.
- [ ] **Step 3: Implement.** Restore the model's setting to `Reg` at the end of every test that uses a
      shared model.
- [ ] **Step 4:** Run the tests. Expected: all pass. Mutation: drop `large` in one fused fallback; a test
      must fail.
- [ ] **Step 5: Default unchanged.** `snap.bat wmma_after`, then compare with `fc /b` against the
      `wmma_base` files. Expected: "no differences" for bf16 and f32.
- [ ] **Step 6:** Full tests and pre-commit. Commit `Let the model use the tensor core matmul for prefill`.

### Task 3: Command line flag

**Files:** Modify: `src/generate_cli.cu`, `src/bench_decode.cu`, `tools/chat.py`

**Interfaces:**
- Produces: `--prefill-matmul reg|wmma` (default `reg`) in all three. The C++ tools call
  `model.set_prefill_matmul` right after `Model::load`. `bench_decode` prints a new line
  `prefill matmul: <reg|wmma>`; existing lines keep their format. `chat.py` passes the flag through.

- [ ] **Step 1: Implement** the parsing; any other value is the error `--prefill-matmul must be reg or wmma`.
- [ ] **Step 2: Check by running.**
  - `build\generate.exe --ids "504 3575 282 4649 314" --max 5 --prefill-matmul wmma` (f32 default):
    exit code 1, a message that bf16 weights are required, no `tok` lines.
  - The same with `--dtype bf16`: exit code 0, five `tok` lines.
  - `--prefill-matmul fast`: exit code 1.
  - `.venv\Scripts\python.exe tools\chat.py "Once upon a time" --max 20 --prefill-matmul wmma`: text.
  - `.venv\Scripts\python.exe tools\compare_llamacpp.py` regexes still match the `bench_decode` output.
- [ ] **Step 3:** Pre-commit. Commit `Add --prefill-matmul to the tools`.

### Task 4: Accuracy in the model

**Files:** Modify: `tools/export_reference.py`, `tests/test_reference.cu`

**Interfaces:**
- Produces in `reference/`: `long.ids` (int32 [T], 450 <= T <= 600), `long.top1` (int32 [T], PyTorch argmax
  per position), `long.positions` (int32: every 32nd position and the last one),
  `long.logits` (float32 [len(positions) x vocab]).

- [ ] **Step 1:** Add `LONG_TEXT` (new natural English prose written for this project, several topics) and
      the export; the exporter asserts the token count range.
- [ ] **Step 2:** `.venv\Scripts\python.exe tools\export_reference.py`. Expected: `DONE`.
- [ ] **Step 3:** Add `ref_long_text_reg_and_wmma_against_pytorch`: bf16 model, both settings, all logits.
      For each print: share of positions whose top-1 equals `long.top1`; on `long.positions` the maximum
      and mean `|diff|`, and the maximum divided by `max(1, max|ref|)`. Assert for `reg` the relative
      maximum `<= 1e-3`, for `wmma` `<= 0.05`. Missing `long.*` entries print a `skipped` line.
- [ ] **Step 4:** Run it. Save the printed numbers.
- [ ] **Step 5: Greedy comparison.** For the 3 prompts "The capital of France is",
      "Once upon a time, in a small village near the sea, there lived" and
      "The best way to learn programming is": `chat.py ... --max 60` with `reg` and with `wmma`.
      Record for each whether the text is the same and, if not, the first differing token.
- [ ] **Step 6:** Full tests and pre-commit. Commit `Measure the accuracy of tensor core prefill`.

### Task 5: Speed in the model, then keep or drop

**Files:** none in the repository. Script in `C:\Code\miniinfer-experiments\ab_wmma.py`.

- [ ] **Step 1:** Check that the GPU is idle (`nvidia-smi`).
- [ ] **Step 2:** Script: the same `bench_decode.exe --dtype bf16`, `reg` and `wmma` in alternating order,
      7 rounds, 2 s pause; prefill 32, 128, 512, 1024 (`--steps 1`, ids as in Task 2) and decode per token
      with and without `--graph` (`--steps 128`); median, min, max.
- [ ] **Step 3:** Also compare `bench_decode_wmma_base.exe` with the new build in `reg` mode (same script,
      two executables): the default must not be slower beyond the spread.
- [ ] **Step 4:** If the gain at 512 is far below what 23.7 -> 9.6 ms of matmul suggests, trace both with
      Nsight Systems and report per-kernel times (`window.py`). "Cause unknown" is an acceptable finding.
- [ ] **Step 5: Gate.** Keep if prefill at 512 is at least 20% faster with `wmma` AND top-1 agreement with
      PyTorch is at least 95% of positions. Report all numbers to the developer. If either fails, stop here
      and decide together.

### Task 6: Documentation (after the gate)

**Files:** Modify: `README.md`, `AGENTS.md` (architecture table: `src/wmma.cu`), `tools/compare_llamacpp.py`

- [ ] **Step 1:** `compare_llamacpp.py` gets `--mini-wmma`, which adds the prefill configuration
      `MiniInfer bf16 wmma`.
- [ ] **Step 2:** One new run of the whole comparison (7 rounds, BF16, Q8_0, Q4_K_M). All rows of the table
      come from this run.
- [ ] **Step 3:** README: a section on the fast mode with the speed table and the accuracy table next to
      each other; "matches PyTorch" statements say they describe the default mode; the `wmma` row is
      labeled as rounding activations to bf16; all caveats stay; the test count comes from the test run.
      Say only what Tasks 4 and 5 measured.
- [ ] **Step 4:** Full tests and pre-commit. Commit `Document the tensor core prefill mode`.
- [ ] **Step 5:** Ask about merging into `main`, and separately about pushing.

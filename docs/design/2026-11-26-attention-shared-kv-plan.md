# Shared K/V Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for
> tracking.

**Goal:** Faster prefill: one attention block per (token, KV head) loads K and V once for all query heads that
share the KV head.

**Architecture:** A second kernel beside `attention_kernel` in `src/ops.cu`. `attention()` picks it through one
host predicate. Decode keeps the current kernel.

**Tech Stack:** C++17, CUDA 13.5, the project's test framework.

**Spec:** [2026-11-26-attention-shared-kv-design.md](2026-11-26-attention-shared-kv-design.md)

## Global Constraints

- Logits are bitwise identical to the build at `7aa8866` in fp32 and bf16.
- Decode time does not get worse.
- The author must be able to explain every line: no tiling, no PTX, no new dependencies.
- Launches go to `exec_stream()`. Comments in English, describing the code as it is.
- `build/tests` passes with no `skipped` line; `pre-commit run --all-files` passes before each commit.
- Branch `attention-shared-kv`. Do not push.
- If the gain in the model is within the spread, the change is not merged.

## Review Focus

1. Prefill of a new chunk on a filled cache (`pos_offset > 0`): same bits as the per-head kernel.
2. Key capacity just above 4096: falls back, no launch failure from too much shared memory.
3. Head ratios other than 3, and head widths that are not a multiple of 4: fall back and stay correct.
4. Cache rows after the last valid key hold garbage: never read into the result.
5. A few rows above the decode limit (T = 5 to 31): correct, and not slower than before.

---

### Task 1: Save the baseline

**Files:** none in the repository. Outputs go to `C:\Code\miniinfer-experiments`.

- [ ] **Step 1:** On the unchanged sources run `build.bat`; expected: exit code 0.
- [ ] **Step 2:** Copy `build\bench_decode.exe` to `bench_decode_attn_base.exe`.
- [ ] **Step 3:** Run `snap.bat attn_base`; expected: `bf16_attn_base.bin` and `f32_attn_base.bin` written.

### Task 2: The kernel and its tests

**Files:**
- Modify: `src/ops.cuh` (next to `attention`), `src/ops.cu` (after `attention_kernel`, and `attention()`)
- Test: `tests/test_ops.cu` (attention section)

**Interfaces:**
- Produces:
  `bool attention_uses_shared_kv(int T, int key_capacity, int num_heads, int num_kv_heads, int head_dim, bool device_pos);`
  true only if `T > kGemvMaxRows`, `!device_pos`, `num_heads / num_kv_heads == 3`, `head_dim % 4 == 0`, and
  `3 * key_capacity * sizeof(float) <= 48 * 1024`. `key_capacity` is `pos_offset + T`.
- Produces: `template <int G> attention_shared_kv_kernel`, grid `(T, num_kv_heads)`, 128 threads, dynamic shared
  memory `float scores[G][nvalid]`. `attention()` keeps its signature.

- [ ] **Step 1: Write the failing tests**

`attention_shared_kv_selection`:

```cpp
CHECK(attention_uses_shared_kv(5, 5, 9, 3, 64, false));
CHECK(attention_uses_shared_kv(512, 4096, 9, 3, 64, false));
CHECK(!attention_uses_shared_kv(4, 4, 9, 3, 64, false));       // decode sizes
CHECK(!attention_uses_shared_kv(512, 4097, 9, 3, 64, false));  // scores do not fit
CHECK(!attention_uses_shared_kv(8, 8, 9, 3, 64, true));        // position on the device
CHECK(!attention_uses_shared_kv(8, 8, 4, 2, 64, false));       // ratio 2
CHECK(!attention_uses_shared_kv(8, 8, 9, 3, 6, false));        // width not a multiple of 4
```

`attention_shared_kv_is_bitwise_equal_to_per_head`: for (T, pos) in (5,0), (8,0), (31,7), (129,0), (200,300),
(5,4091), (5,4092), with H=9, KV=3, hd=64 and `S = pos + T + 3`: the result of one call is compared with
`bitwise_equal` to the concatenation of calls on at most 4 query rows each (`pos_offset = pos + first row`),
which use the per-head kernel. Rows of `k` and `v` from `pos + T` on are NaN. Every output value is finite.

`attention_shared_kv_matches_cpu`: for (T, pos) in (5,0), (33,12), (130,70): `max_abs_diff` against
`cpu_attention` below the tolerance that `attention_matches_cpu_for_many_shapes` uses.

- [ ] **Step 2:** Run `build.bat`. Expected: it fails to compile, `attention_uses_shared_kv` is undeclared.
- [ ] **Step 3:** Implement the predicate, the kernel and the selection. Start from `attn_shared` in
  `C:\Code\miniinfer-experiments\attn_share.cu`. Per head, keep the order of the current kernel: dot product
  terms in index order, then `block_reduce_max`, `exp`, `block_reduce_sum`, then the sum over keys in index
  order. Reuse `block_reduce_max` and `block_reduce_sum`. The comment states the access pattern and the
  shared memory limit.
- [ ] **Step 4:** Run `build.bat`, then `build\tests.exe`. Expected: 169 passed, 0 failed, no `skipped`.
- [ ] **Step 5: Check that the tests can fail.** One at a time, rebuild and run `build\tests.exe attention`:
  (a) the kernel writes head `g` to the place of head `(g + 1) % G`; expected: both result tests fail;
  (b) the weighted sum runs one key past `nvalid`; expected: the bitwise test fails.
  Restore by rewriting the lines, rebuild, run all tests again: 169 passed.
- [ ] **Step 6: Commit** `Load K and V once per KV head in prefill attention`

### Task 3: Measure in the model

**Files:**
- Modify: `docs/benchmarks.md` (section `## Prefill`), `README.md` (the prefill paragraph)
- Temporary: `C:\Code\miniinfer-experiments\ab_attn.py`

- [ ] **Step 1:** Run `snap.bat attn_after` and compare both files with the baseline using `fc /b`.
  Expected: no differences in bf16 and in f32.
- [ ] **Step 2:** Write `ab_attn.py` from `ab_wmma.py`: variants old and new, each in `reg` and `wmma`, in
  rotated order over 7 rounds; prefill at 5, 16, 32, 128, 512, 1024 tokens; decode with and without the graph.
- [ ] **Step 3:** Run it with no other GPU load. Report median, min and max for every cell.
- [ ] **Step 4:** Apply the spec: decode must be within the spread. If a small prefill size is slower, raise the
  switching threshold, update the tests and measure again. If the gain at 128 and 512 tokens is within the
  spread, stop, report, and do not merge.
- [ ] **Step 5:** Add the new before/after table, with its command and statistic, to `docs/benchmarks.md`, and
  one sentence with those conditions to the README. Leave the llama.cpp table unchanged and say that it was
  measured before this change: its numbers are comparable only within one run.
- [ ] **Step 6: Commit** `Record the prefill gain of shared K and V loads`

### Task 4: Walk through the kernel with the author

- [ ] Go through `attention_shared_kv_kernel` piece by piece, one question at a time. The author explains each
  piece in their own words. Simplify or comment whatever cannot be explained.

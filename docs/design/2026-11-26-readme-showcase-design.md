# README as a showcase

## Intent

The repository is a portfolio piece. A reader from NVIDIA who spends one minute on it should conclude that the
author understands the project and worked in a methodical way: measured, found the cause, fixed it, verified
the result, and stated where the evidence ends.

## Goals

- The reasoning is visible without following links.
- The first screen says what the project is and gives the main result with its conditions.
- A diagram shows how the engine works, and a chart shows the decode speedup.
- The README is clearly shorter than the current one (about 320 lines; aim for about 180).

## Non-goals

- No changes to engine code, tests or tools. Removing `hello.cu` and `vector_add.cu` is a separate change.
- No article, no recorded demo, and no change to the repository visibility.
- No new measurements to obtain better numbers.

## Anti-goals

- A short showcase that hides how the results were obtained.
- Any claim stronger than what was measured. In particular, the llama.cpp comparison is never reduced to
  "faster than llama.cpp".

## Approaches

| Approach | Gain | Cost |
|---|---|---|
| A. Trim the current README, add a chart and a badge | Quick | The reasoning stays buried in numbers |
| **B. A story of decisions, details moved to `docs/`** (chosen) | The reasoning and the limits are visible | More work; numbers must be moved with care |
| C. One-screen showcase, everything else in `docs/` | Read in a minute | The evidence of understanding sits behind links |

The author chose B because it serves the intent directly.

## README structure

1. What the project is, the main result with its conditions, the CI badge.
2. How the engine works: a diagram of `tokens -> embedding -> 30 decoder layers -> RMSNorm + lm_head ->
   logits -> sampler`.
3. The performance path, as measured -> cause -> fix -> check, with the chart and a short table. It includes
   what stays unexplained.
4. How correctness is established, including the opt-in tensor core mode and its accuracy cost.
5. Comparison with llama.cpp: a short table with decode and prefill and the main caveats.
6. What is still weak, and the limitations.
7. Build, run, tests.
8. Development (including the statement about AI assistance), project layout, license.

## Moved to `docs/`

- `docs/benchmarks.md`: method and all caveats of the llama.cpp comparison, the prefill history, the matmul
  benchmark.
- `docs/tensor-core-prefill.md`: speed and accuracy tables of the `wmma` mode.

Nothing measured is deleted: what leaves the README goes to one of these files.

## Caveats that stay in the README

- Prefill is clearly slower than llama.cpp; the table shows it.
- The decode lead depends on the CUDA Graph and a single hard-wired architecture; without the graph MiniInfer is
  slower than llama.cpp BF16, and quantized llama.cpp is faster than MiniInfer.
- The 128-token prefill column was unstable, and ratios differ between runs.
- The KV cache format and the embedding lookup differ between the two engines.
- `wmma` rounds activations to bf16, and its accuracy was measured on one text and three prompts only.
- One laptop GPU, Windows, one tiny model.

## Builder's calls (open to correction)

- The diagram is a Mermaid block, which GitHub renders. The chart is a static SVG committed under `docs/img/`,
  legible in light and dark themes. Neither adds a dependency.
- 455 and 407 tok/s: the README states next to the numbers which tool, statistic and run produced each. Before
  writing this, check both in the code. Do not claim that this accounts for the whole gap unless it is shown.
- A short text sample of real output from `tools/chat.py`, produced by running it, stands in for a demo.
- Section wording and the exact length.

## Checks

- Every number in the new README and the two new files matches the current README or a command run for this
  change.
- All relative links and anchors resolve.
- `pre-commit run --all-files` passes.
- The author reads each section and can explain its claims.

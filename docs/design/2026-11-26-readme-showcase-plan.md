# README Showcase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for
> tracking.

**Goal:** Turn the README into a story of engineering decisions that shows the author understands the project,
with details moved to `docs/`.

**Architecture:** First move the long measurement material into two files under `docs/`, then add the chart,
then rewrite the README on top of them. A temporary script, kept outside the repository, checks numbers and
links against the README at commit `b14a68d`.

**Tech Stack:** Markdown, Mermaid (rendered by GitHub), hand-written SVG, Python standard library for checks.

**Spec:** [2026-11-26-readme-showcase-design.md](2026-11-26-readme-showcase-design.md)

## Global Constraints

- Everything in the repository is in English.
- No changes to `src/`, `tests/`, `tools/`, CMake or CI files. No new dependencies.
- No new benchmark measurements. Every number matches the README at `b14a68d` or a command run for this change.
- The llama.cpp comparison is never reduced to "faster than llama.cpp". Its table stays in the README and
  shows that prefill is clearly slower (`AGENTS.md` relies on this).
- README length: about 180 lines.
- Work on branch `readme-showcase`, created from `main`. Do not push.
- `pre-commit run --all-files` passes before each commit.
- Commit messages: imperative subject; the body says why and how it was verified.

## Review Focus

1. A number moved without its conditions (tool, statistic, run, weight format): a reader must be able to tell
   what each number means where it stands.
2. The first screen quoted alone, for example in a screenshot: it must not suggest a general win over
   llama.cpp, and "matches PyTorch" must be limited to the default mode.
3. The chart on GitHub's dark theme: text and bars stay legible.
4. The CI badge read as "GPU tests pass": the text next to it says that CI has no GPU.
5. The text sample read as typical quality: it shows the exact command, is unedited, and says that this is a
   base model that continues text.

---

### Task 1: Move the long material to `docs/`

**Files:**
- Create: `docs/benchmarks.md`
- Create: `docs/tensor-core-prefill.md`
- Temporary, not committed: `check_readme.py` in the session scratchpad directory

**Interfaces:**
- Produces: `docs/benchmarks.md` with the headings `## Comparison with llama.cpp`, `## Prefill`,
  `## Matmul benchmark (fp32, GFLOPS)`; `docs/tensor-core-prefill.md` with the title
  `# Tensor core prefill (opt-in)`. Task 3 links to these.

- [ ] **Step 1: Create the branch**

Run: `git switch -c readme-showcase`

- [ ] **Step 2: Write the check script**

It is run from the repository root.

```python
import pathlib, re, subprocess, sys

NUM = re.compile(r"\d+(?:[.,]\d+)*(?:e-?\d+)?")
old = subprocess.run(["git", "show", "b14a68d:README.md"], capture_output=True, text=True,
                     encoding="utf-8", check=True).stdout
files = [pathlib.Path(p) for p in ("README.md", "docs/benchmarks.md", "docs/tensor-core-prefill.md")]
texts = {f: f.read_text(encoding="utf-8") for f in files if f.exists()}
new = "\n".join(texts.values())
print("lost:", sorted(set(NUM.findall(old)) - set(NUM.findall(new))))
print("added:", sorted(set(NUM.findall(new)) - set(NUM.findall(old))))


def anchors(text):
    prose = re.sub(r"```.*?```", "", text, flags=re.S)
    return {re.sub(r"[^\w\- ]", "", h.strip().lower()).replace(" ", "-")
            for h in re.findall(r"^#{1,6} (.+)$", prose, flags=re.M)}


bad = []
for f, text in texts.items():
    for link in re.findall(r"\]\(([^)\s]+)\)", text):
        if link.startswith("http"):
            continue
        path, _, frag = link.partition("#")
        target = (f.parent / path) if path else f
        if not target.exists() or (frag and frag not in anchors(target.read_text(encoding="utf-8"))):
            bad.append(f"{f}: {link}")
print("broken links:", bad)
sys.exit(1 if bad else 0)
```

- [ ] **Step 3: Check that the script can fail**

Temporarily add `[x](docs/no-such-file.md)` to `README.md` and run
`.venv\Scripts\python.exe <scratchpad>\check_readme.py`.
Expected: exit code 1 and the link listed. Remove the line again; `git diff README.md` is then empty.

- [ ] **Step 4: Write `docs/benchmarks.md`**

Move, with the wording kept: the whole "Comparison with llama.cpp" section (table, method, every "How to read
it" item, reproduce command) from `README.md:27-75`; the "Prefill" subsection with its Nsight findings from
`README.md:129-158`; "Matmul benchmark" from `README.md:212-222`. Rewrite relative links for the new location
(`../tools/...`, `../src/...`, `tensor-core-prefill.md`).

- [ ] **Step 5: Write `docs/tensor-core-prefill.md`**

Move `README.md:160-200` with the wording kept, including the speed table, the accuracy table and what the
comparison does not show. Rewrite relative links.

- [ ] **Step 6: Verify and commit**

Run the check script and `pre-commit run --all-files`.
Expected: `broken links: []`, nothing under `lost:` (the README is still complete), hooks pass.
Commit the two new files, the design and this plan:
`git commit -m "Move the benchmark details into docs"`

### Task 2: Decode speedup chart

**Files:**
- Create: `docs/img/decode-speedup.svg`

**Interfaces:**
- Produces: `docs/img/decode-speedup.svg`, embedded by Task 3.

- [ ] **Step 1: Write the SVG**

Horizontal bars of decode time in ms per token, one per step, in this order and with these values:
Baseline 17.0; Memory pool 12.0; GEMV kernel 5.7; Kernel fusion 3.6; fp16 weights 2.9; bf16 weights 2.7;
CUDA Graph 2.2. Each bar is labeled with its value; the title says "Decode time, ms per token (lower is
better)"; a footnote says "RTX 4060 Laptop, bench_decode, median of 64 steps". Bar length is proportional to
the value and the axis starts at zero. The SVG draws its own opaque background, so it reads the same on light
and dark pages.

- [ ] **Step 2: Verify**

Open the SVG in a browser on a dark and on a light page, take a screenshot of each and look at them: no
clipped or overlapping text. Check the seven bar widths against value x scale.

- [ ] **Step 3: Commit**

`git commit -m "Add a chart of the decode speedup"`

### Task 3: Rewrite the README

**Files:**
- Modify: `README.md` (whole file)

**Interfaces:**
- Consumes: the files and headings of Task 1, the chart of Task 2.

- [ ] **Step 1: Collect the real text sample**

Run: `.venv\Scripts\python.exe tools\chat.py "Once upon a time" --max 60`
Run it once. Use the generated text unedited, whatever its quality.

- [ ] **Step 2: Confirm how 455 and 407 were measured**

455: `bench_decode.cu` prints "decode speed" from the median step; the README states 64 steps.
407: `compare_llamacpp.py` takes the mean time per token of a 128-step run and reports the median over 7
rounds, alternated with the other configurations.
State both next to the numbers, and say that the two are separate runs measured differently and that how much
of the gap each difference explains was not measured.

- [ ] **Step 3: Write the README in this order**

1. Title; CI badge
   `https://github.com/Nostradam4ik/miniinfer/actions/workflows/ci.yml/badge.svg` linking to the workflow,
   followed by: CI has no GPU; it checks formatting, builds and runs the CPU-only tests.
2. What it is; the main result (59 to 455 tok/s, 7.7x) with GPU, weight format and "default mode"; the text
   sample with its command and the base-model note.
3. "How it works": a Mermaid `flowchart` of the pipeline named in the spec, plus one decoder layer. Take the
   order of the operations inside the layer from `src/model.cuh`, not from memory. Then the implemented parts
   as short bullets.
4. "From 17.0 to 2.2 ms per token": the chart, the step table, and for each finding measured -> cause -> fix
   -> check. Keep what is unexplained (the matmul gain difference). Summarize prefill in a few lines and link
   to `docs/benchmarks.md#prefill`.
5. "How correctness is established": the list from `README.md:282-290`, the 166 tests, the "skipped" warning,
   and the `wmma` accuracy cost (492 of 497, largest logit difference 0.79, one text and three prompts) with a
   link to `docs/tensor-core-prefill.md`.
6. "Comparison with llama.cpp": rows MiniInfer bf16, no graph, `wmma`, llama.cpp BF16, Q8_0 and Q4_K_M with
   the min-to-max ranges; every caveat listed in the spec; a link to
   `docs/benchmarks.md#comparison-with-llamacpp`.
7. "What is left" and "Limitations".
8. "Build", "Run", "Tests".
9. "Development" (keep the AI assistance sentence), "Project layout", "License".

- [ ] **Step 4: Verify**

Run the check script, `pre-commit run --all-files` and `(Get-Content README.md).Count`.
Expected: `broken links: []`; every entry under `lost:` is explained (nothing measured is lost); every entry
under `added:` comes from the badge, the sample, or a command run for this change; about 180 lines.
Go through the five Review Focus lines against the new text. Report the Mermaid diagram as not verified on
GitHub until the branch is pushed, unless it was rendered locally.

- [ ] **Step 5: Read-through with the author**

Go through the README with the author section by section: they say in their own words what each claim means.
Fix or remove what they cannot explain or find unclear.

- [ ] **Step 6: Commit**

`git commit -m "Rewrite the README around the engineering decisions"`

# Contributing

## Setup

Build and model setup are described in the [README](README.md). For the checks:

```
pip install pre-commit
pre-commit install
git config blame.ignoreRevsFile .git-blame-ignore-revs
```

## Before a change is merged

1. `pre-commit run --all-files` passes.
2. `build/tests` passes on a machine with a GPU, with `model/` and `reference/` present (no `skipped` lines).
3. A performance change reports numbers before and after, with the command, GPU and weight format.

CI runs on GitHub-hosted runners, which have no GPU. It checks formatting, compiles everything and runs the
CPU-only sampler tests. **A green CI does not mean the kernels are correct**; item 2 is the real check.

## C++ and CUDA conventions

Formatting is defined by [.clang-format](.clang-format). Beyond that:

- C++17. Types are `PascalCase`, functions and variables `snake_case`, constants `kCamelCase`, macros and
  kernel block dimensions `UPPER_CASE`. Private members end with `_`.
- Kernel function names end with `_kernel`.
- GPU resources are owned by RAII types that are move-only.
- Wrap every CUDA runtime call in `CUDA_CHECK`. It terminates the process, which is acceptable for this
  project; invalid arguments from the caller throw `std::invalid_argument` instead.
- Launch kernels on `exec_stream()`, never implicitly on the default stream (CUDA Graph capture relies on it).
- State the shapes of matrix arguments in the declaration comment, e.g. `C[M x N] = A[M x K] * B[K x N]`.
- Explain the memory access pattern and the synchronization of non-trivial kernels in a comment.
- All activations and accumulation are fp32; only weights may be stored in 16 bits.

## Tests

The framework is [tests/test_framework.h](tests/test_framework.h) (`TEST`, `CHECK`, `CHECK_NEAR`,
`CHECK_THROWS`). Add new files to the `tests` target in [CMakeLists.txt](CMakeLists.txt). Test kernels
against an independent CPU reference, include shapes that are not multiples of the tile size, and cover the
argument checks.

cmake_minimum_required(VERSION 3.24)

# GPU architecture. Must be set before project(). Default: compute capability 8.9 (Ada Lovelace, e.g. RTX 4060).
# Override with -DCMAKE_CUDA_ARCHITECTURES=<cc>, e.g. 86 for Ampere or "native" for the local GPU.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures to build for")
endif()

project(MiniInfer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)
include_directories(${CMAKE_SOURCE_DIR}/src)

# Library of kernels and tensors (used by the benchmarks, the tests and the engine)
add_library(miniinfer STATIC src/ops.cu src/ops_fused.cu src/gemv.cu src/lowp.cu src/wmma.cu)
target_link_libraries(miniinfer PUBLIC CUDA::cudart)

# First steps
add_executable(hello src/hello.cu)
add_executable(vector_add src/vector_add.cu)

# Matmul benchmark
add_executable(matmul src/matmul.cu)
target_link_libraries(matmul miniinfer CUDA::cublas)

# Command-line generation (see tools/chat.py)
add_executable(generate src/generate_cli.cu)
target_link_libraries(generate miniinfer)

# Decode speed
add_executable(bench_decode src/bench_decode.cu)
target_link_libraries(bench_decode miniinfer)

add_executable(bench_ops src/bench_ops.cu)
target_link_libraries(bench_ops miniinfer)

add_executable(bench_layer_ops src/bench_layer_ops.cu)
target_link_libraries(bench_layer_ops miniinfer)

# Tests: tests.exe or ctest --test-dir build
enable_testing()
add_executable(tests
    tests/test_main.cu
    tests/test_tensor.cu
    tests/test_matmul.cu
    tests/test_ops.cu
    tests/test_layer.cu
    tests/test_sampler.cu
    tests/test_alloc.cu
    tests/test_fused.cu
    tests/test_lowp.cu
    tests/test_graph.cu
    tests/test_wmma.cu
    tests/test_reference.cu)
target_include_directories(tests PRIVATE tests)
target_compile_definitions(tests PRIVATE MINIINFER_ROOT="${CMAKE_SOURCE_DIR}")
target_link_libraries(tests miniinfer)
add_test(NAME all_tests COMMAND tests)

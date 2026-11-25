#pragma once
// Matmul on tensor cores (WMMA): C[M x N] = A[M x K] (fp32) * B[K x N] (bf16). A is rounded to bf16 inside
// the kernel and the products are accumulated in fp32, so the result differs from the other matmul kernels.
#include "tensor.cuh"

// True if the current GPU has bf16 tensor cores (compute capability 8.0 or newer).
bool wmma_supported_by_device();

// Raw launch on exec_stream(). K must be a multiple of 32 and N of 64, dt must be BF16, A aligned to
// 16 bytes and B to 8 bytes. Throws std::invalid_argument otherwise.
void launch_matmul_wmma(const float* A, const void* B, DType dt, float* C, int M, int N, int K);

#pragma once
// Internal interface of the GEMV kernels (used by ops.cu and ops_fused.cu).
//
// C[m, :] = P(A[m, :]) * B (+ R[m, :]),  where P = RMSNorm with weight norm_w if norm_w != nullptr,
// B: [K x N] in format dt (fp32/fp16/bf16), R is an optional residual [M x N].
// A, R, C are fp32. Accumulation is always in fp32.
#include "tensor.cuh"

void gemv_launch(const float* A, const float* norm_w, float eps, const void* B, DType dt, const float* R, float* C,
                 int M, int N, int K);

// out = silu(P(A) * Wg) * (P(A) * Wu), P = RMSNorm. Both matrices use the same format dt.
void gemv_gate_up_launch(const float* A, const float* norm_w, float eps, const void* Wg, const void* Wu, DType dt,
                         float* C, int M, int N, int K);

// How many columns one thread handles: 1, 2 or 4 (vectorized loads). N must be divisible by it.
// The automatic choice depends on N and the format; override_vec != 0 forces a value
// (only for tests and tuning, 0 enables the automatic choice).
int gemv_choose_vec(int N, DType dt);
void gemv_set_vec_override(int vec);

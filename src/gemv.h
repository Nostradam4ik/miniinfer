#pragma once
// Внутрішній інтерфейс GEMV-ядер (використовують ops.cu і ops_fused.cu).
//
// C[m, :] = P(A[m, :]) * B (+ R[m, :]),  де P = RMSNorm з вагою norm_w, якщо norm_w != nullptr,
// B: [K x N] у форматі dt (fp32/fp16/bf16), R - необов'язковий residual [M x N].
// A, R, C - fp32. Накопичення завжди в fp32.
#include "tensor.cuh"

void gemv_launch(const float* A, const float* norm_w, float eps, const void* B, DType dt, const float* R,
                 float* C, int M, int N, int K);

// out = silu(P(A) * Wg) * (P(A) * Wu), P = RMSNorm. Обидві матриці одного формату dt.
void gemv_gate_up_launch(const float* A, const float* norm_w, float eps, const void* Wg, const void* Wu,
                         DType dt, float* C, int M, int N, int K);

// Скільки стовпців обробляє один потік: 1, 2 або 4 (векторизоване завантаження). N має ділитись на це.
// Автоматичний вибір залежить від N і формату; override_vec != 0 примусово задає значення
// (лише для тестів і налаштування, 0 вмикає автоматичний вибір).
int gemv_choose_vec(int N, DType dt);
void gemv_set_vec_override(int vec);

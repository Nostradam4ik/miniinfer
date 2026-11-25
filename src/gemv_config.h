#pragma once
// Спільна конфігурація GEMV-ядер (ops.cu і ops_fused.cu). Блок: GEMV_COLS x GEMV_KSPLIT потоків.
// GEMV_COLS сусідніх стовпців B читаються послідовно (coalesced); GEMV_KSPLIT груп потоків
// ділять вісь K між собою. Злиті ядра покладаються на GEMV_COLS * GEMV_KSPLIT == 512 (16 warp-ів).
#define GEMV_COLS 32
#define GEMV_KSPLIT 16
static_assert(GEMV_COLS * GEMV_KSPLIT == 512, "fused GEMV kernels assume 512 threads per block");

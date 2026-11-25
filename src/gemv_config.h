#pragma once
// Shared configuration of the GEMV kernels (ops.cu and ops_fused.cu). Block: GEMV_COLS x GEMV_KSPLIT threads.
// GEMV_COLS adjacent columns of B are read sequentially (coalesced); GEMV_KSPLIT thread groups split
// the K axis between them. The fused kernels rely on GEMV_COLS * GEMV_KSPLIT == 512 (16 warps).
#define GEMV_COLS 32
#define GEMV_KSPLIT 16
static_assert(GEMV_COLS * GEMV_KSPLIT == 512, "fused GEMV kernels assume 512 threads per block");

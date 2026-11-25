// Matmul on tensor cores through WMMA.
//
// One block computes a 64 x 64 tile of C with 128 threads (4 warps). The loop over K takes steps of 32:
//   1. All 128 threads copy the next tiles of A [64 x 32] and B [32 x 64] to shared memory. Every load
//      instruction brings four adjacent elements of a row (float4 for A, 4 bf16 weights for B). A is rounded
//      to bf16 here; B is copied bit for bit.
//   2. __syncthreads().
//   3. Each warp owns a 32 x 32 quarter of the output tile as four 16 x 16 accumulator fragments. It loads
//      two A fragments (its rows) and two B fragments (its columns) from shared memory and runs four
//      m16n16k16 multiply-adds, twice per step of 32. A warp's load and mma calls are collective: all
//      32 lanes take part.
//   4. __syncthreads() before the tiles are overwritten by the next step.
// At the end the warps store their fragments to shared memory (the same buffer, no longer holding A and B),
// and the threads copy the tile to C, skipping rows past M.
//
// K is a multiple of 32 and N of 64, so only the rows need a bounds check: rows of A past M are staged as
// zeros and rows of C past M are not written.
#include <cuda_bf16.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include "cuda_utils.cuh"
#include "wmma.h"

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#include <mma.h>
#endif

namespace {

constexpr int kTile = 64;      // a block computes kTile x kTile elements of C
constexpr int kStepK = 32;     // columns of A (rows of B) staged at a time
constexpr int kThreads = 128;  // 4 warps
// Extra columns of the tiles in shared memory: fewer bank conflicts, and the fragments stay aligned to
// 32 bytes, which load_matrix_sync requires.
constexpr int kPad = 8;
[[maybe_unused]] constexpr int kLdA = kStepK + kPad;
[[maybe_unused]] constexpr int kLdB = kTile + kPad;

}  // namespace

__global__ void matmul_wmma_kernel(const float* A, const __nv_bfloat16* B, float* C, int M, int N, int K) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    namespace wmma = nvcuda::wmma;
    // The input tiles and the output tile are never live at the same time, so they share one buffer.
    constexpr int kInputFloats = (kTile * kLdA + kStepK * kLdB) / 2;  // bf16: two elements per float
    constexpr int kSharedFloats = kInputFloats > kTile * kTile ? kInputFloats : kTile * kTile;
    __shared__ __align__(32) float shared[kSharedFloats];
    __nv_bfloat16* As = reinterpret_cast<__nv_bfloat16*>(shared);  // [kTile x kLdA]
    __nv_bfloat16* Bs = As + kTile * kLdA;                         // [kStepK x kLdB]

    const int tid = threadIdx.x;
    const int block_row = blockIdx.y * kTile, block_col = blockIdx.x * kTile;
    const int warp = tid / 32;
    const int warp_row = warp / 2 * 32, warp_col = warp % 2 * 32;  // the warp's quarter of the tile

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.f);

    for (int t = 0; t < K; t += kStepK) {
        // A: 64 rows of 8 chunks of 4 elements; 4 chunks per thread.
        for (int idx = tid; idx < kTile * kStepK / 4; idx += kThreads) {
            const int r = idx / (kStepK / 4), c = idx % (kStepK / 4) * 4;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (block_row + r < M)
                v = *reinterpret_cast<const float4*>(A + static_cast<size_t>(block_row + r) * K + t + c);
            __nv_bfloat16* dst = As + r * kLdA + c;
            dst[0] = __float2bfloat16_rn(v.x);
            dst[1] = __float2bfloat16_rn(v.y);
            dst[2] = __float2bfloat16_rn(v.z);
            dst[3] = __float2bfloat16_rn(v.w);
        }
        // B: 32 rows of 16 chunks of 4 weights (8 bytes); 4 chunks per thread.
        for (int idx = tid; idx < kStepK * kTile / 4; idx += kThreads) {
            const int r = idx / (kTile / 4), c = idx % (kTile / 4) * 4;
            *reinterpret_cast<uint2*>(Bs + r * kLdB + c) =
                *reinterpret_cast<const uint2*>(B + static_cast<size_t>(t + r) * N + block_col + c);
        }
        __syncthreads();

        for (int k = 0; k < kStepK; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b[2];
            for (int i = 0; i < 2; ++i) {
                wmma::load_matrix_sync(a[i], As + (warp_row + i * 16) * kLdA + k, kLdA);
                wmma::load_matrix_sync(b[i], Bs + k * kLdB + warp_col + i * 16, kLdB);
            }
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(shared + (warp_row + i * 16) * kTile + warp_col + j * 16, acc[i][j], kTile,
                                    wmma::mem_row_major);
    __syncthreads();
    for (int idx = tid; idx < kTile * kTile / 4; idx += kThreads) {
        const int r = idx / (kTile / 4), c = idx % (kTile / 4) * 4;
        if (block_row + r < M)
            *reinterpret_cast<float4*>(C + static_cast<size_t>(block_row + r) * N + block_col + c) =
                *reinterpret_cast<const float4*>(shared + r * kTile + c);
    }
#else
    // Code built for an older target can still be compiled by the driver for a newer GPU, which passes
    // wmma_supported_by_device(). The launch must then fail instead of leaving C unwritten.
    __trap();
#endif
}

bool wmma_supported_by_device() {
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    return prop.major >= 8;
}

void launch_matmul_wmma(const float* A, const void* B, DType dt, float* C, int M, int N, int K) {
    if (dt != DType::BF16) throw std::invalid_argument("matmul wmma: the weights must be bf16");
    if (K % kStepK != 0 || N % kTile != 0)
        throw std::invalid_argument("matmul wmma: K must be a multiple of 32 and N of 64 (K=" + std::to_string(K) +
                                    ", N=" + std::to_string(N) + ")");
    if (reinterpret_cast<uintptr_t>(A) % sizeof(float4) != 0 || reinterpret_cast<uintptr_t>(B) % sizeof(uint2) != 0)
        throw std::invalid_argument("matmul wmma: A must be aligned to 16 bytes and B to 8 bytes");
    if (!wmma_supported_by_device())
        throw std::invalid_argument("matmul wmma: the GPU has no bf16 tensor cores (compute capability 8.0 needed)");
    if (M <= 0 || N <= 0) return;
    const dim3 grid(N / kTile, (M + kTile - 1) / kTile);
    matmul_wmma_kernel<<<grid, kThreads, 0, exec_stream()>>>(A, static_cast<const __nv_bfloat16*>(B), C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

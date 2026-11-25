// GEMV для малих M (декодування): швидкість упирається в пропускну здатність пам'яті, бо кожен
// елемент ваг читається рівно один раз. Тому ядро влаштоване так, щоб читати їх якомога ширшими
// послідовними запитами.
//
// Блок: GEMV_COLS x GEMV_KSPLIT потоків. Потік (tx, ty) обробляє VEC сусідніх стовпців
// (tx * VEC .. tx * VEC + VEC - 1 всередині блоку) для елементів k = ty, ty + KSPLIT, ...
// Warp (32 потоки з однаковим ty) за одну інструкцію читає 32 * VEC * sizeof(WT) послідовних байтів.
// Потім GEMV_KSPLIT часткових сум додаються в shared memory.
#include <atomic>
#include <stdexcept>
#include <string>
#include <type_traits>
#include "gemv.h"
#include "gemv_config.h"
#include "wt_convert.cuh"

namespace {
std::atomic<int> g_vec_override{0};
}

void gemv_set_vec_override(int vec) {
    if (vec != 0 && vec != 1 && vec != 2 && vec != 4) throw std::invalid_argument("gemv_set_vec_override: 0, 1, 2 or 4");
    g_vec_override = vec;
}

int gemv_choose_vec(int N, DType dt) {
    int v = g_vec_override.load();
    if (v == 0) v = 4;  // (налаштовується вимірюванням, див. bench_ops)
    (void)dt;
    while (v > 1 && N % v != 0) v >>= 1;  // N має ділитись на VEC, інакше беремо менше
    return v;
}

// Сума квадратів рядка A (K елементів) по всьому блоку; результат rstd у s_rstd.
__device__ __forceinline__ float block_rstd(const float* a, int K, float eps, float* wsum, float* s_rstd) {
    const int tid = threadIdx.y * GEMV_COLS + threadIdx.x;
    float ss = 0.f;
    for (int k = tid; k < K; k += GEMV_COLS * GEMV_KSPLIT) ss += a[k] * a[k];
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, off);
    if ((tid & 31) == 0) wsum[tid >> 5] = ss;
    __syncthreads();
    if (tid == 0) {
        float t = 0.f;
        for (int w = 0; w < GEMV_KSPLIT; ++w) t += wsum[w];
        *s_rstd = rsqrtf(t / K + eps);
    }
    __syncthreads();
    return *s_rstd;
}

template <typename WT, int VEC, bool NORM, bool RESID>
__global__ void gemv_kernel(const float* A, const float* norm_w, float eps, const WT* B, const float* R,
                            float* C, int M, int N, int K) {
    constexpr int BLOCK_COLS = GEMV_COLS * VEC;
    __shared__ float part[GEMV_KSPLIT][BLOCK_COLS];
    __shared__ float wsum[GEMV_KSPLIT];  // 512 потоків = 16 warp-ів
    __shared__ float s_rstd;

    const int m = blockIdx.y;
    const float* a = A + static_cast<size_t>(m) * K;
    const int tid = threadIdx.y * GEMV_COLS + threadIdx.x;

    float rstd = 1.f;
    if (NORM) rstd = block_rstd(a, K, eps, wsum, &s_rstd);

    const int col0 = blockIdx.x * BLOCK_COLS + threadIdx.x * VEC;
    float acc[VEC];
#pragma unroll
    for (int j = 0; j < VEC; ++j) acc[j] = 0.f;

    if (col0 < N)  // N кратне VEC (запускач це гарантує), тож весь вектор у межах
        for (int k = threadIdx.y; k < K; k += GEMV_KSPLIT) {
            const float av = NORM ? a[k] * rstd * norm_w[k] : a[k];
            float w[VEC];
            load_weights<WT, VEC>(B + static_cast<size_t>(k) * N + col0, w);
#pragma unroll
            for (int j = 0; j < VEC; ++j) acc[j] += av * w[j];
        }
#pragma unroll
    for (int j = 0; j < VEC; ++j) part[threadIdx.y][threadIdx.x * VEC + j] = acc[j];
    __syncthreads();

    if (tid < BLOCK_COLS) {  // перші BLOCK_COLS потоків додають часткові суми своїх стовпців
        const int col = blockIdx.x * BLOCK_COLS + tid;
        if (col < N) {
            float s = 0.f;
            for (int i = 0; i < GEMV_KSPLIT; ++i) s += part[i][tid];
            const size_t o = static_cast<size_t>(m) * N + col;
            C[o] = RESID ? s + R[o] : s;
        }
    }
}

template <typename WT, int VEC>
__global__ void gemv_gate_up_kernel(const float* A, const float* norm_w, float eps, const WT* Wg, const WT* Wu,
                                    float* C, int M, int N, int K) {
    constexpr int BLOCK_COLS = GEMV_COLS * VEC;
    __shared__ float part_g[GEMV_KSPLIT][BLOCK_COLS];
    __shared__ float part_u[GEMV_KSPLIT][BLOCK_COLS];
    __shared__ float wsum[GEMV_KSPLIT];
    __shared__ float s_rstd;

    const int m = blockIdx.y;
    const float* a = A + static_cast<size_t>(m) * K;
    const int tid = threadIdx.y * GEMV_COLS + threadIdx.x;
    const float rstd = block_rstd(a, K, eps, wsum, &s_rstd);

    const int col0 = blockIdx.x * BLOCK_COLS + threadIdx.x * VEC;
    float acc_g[VEC], acc_u[VEC];
#pragma unroll
    for (int j = 0; j < VEC; ++j) acc_g[j] = acc_u[j] = 0.f;

    if (col0 < N)
        for (int k = threadIdx.y; k < K; k += GEMV_KSPLIT) {
            const float av = a[k] * rstd * norm_w[k];
            float g[VEC], u[VEC];
            const size_t o = static_cast<size_t>(k) * N + col0;
            load_weights<WT, VEC>(Wg + o, g);
            load_weights<WT, VEC>(Wu + o, u);
#pragma unroll
            for (int j = 0; j < VEC; ++j) {
                acc_g[j] += av * g[j];
                acc_u[j] += av * u[j];
            }
        }
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
        part_g[threadIdx.y][threadIdx.x * VEC + j] = acc_g[j];
        part_u[threadIdx.y][threadIdx.x * VEC + j] = acc_u[j];
    }
    __syncthreads();

    if (tid < BLOCK_COLS) {
        const int col = blockIdx.x * BLOCK_COLS + tid;
        if (col < N) {
            float g = 0.f, u = 0.f;
            for (int i = 0; i < GEMV_KSPLIT; ++i) {
                g += part_g[i][tid];
                u += part_u[i][tid];
            }
            C[static_cast<size_t>(m) * N + col] = g / (1.f + expf(-g)) * u;
        }
    }
}

namespace {

template <typename WT, int VEC>
void launch_vec(const float* A, const float* nw, float eps, const void* Bv, const float* R, float* C, int M, int N, int K) {
    const WT* B = static_cast<const WT*>(Bv);
    const dim3 grid((N + GEMV_COLS * VEC - 1) / (GEMV_COLS * VEC), M), block(GEMV_COLS, GEMV_KSPLIT);
    if (nw && R) gemv_kernel<WT, VEC, true, true><<<grid, block, 0, exec_stream()>>>(A, nw, eps, B, R, C, M, N, K);
    else if (nw) gemv_kernel<WT, VEC, true, false><<<grid, block, 0, exec_stream()>>>(A, nw, eps, B, R, C, M, N, K);
    else if (R) gemv_kernel<WT, VEC, false, true><<<grid, block, 0, exec_stream()>>>(A, nw, eps, B, R, C, M, N, K);
    else gemv_kernel<WT, VEC, false, false><<<grid, block, 0, exec_stream()>>>(A, nw, eps, B, R, C, M, N, K);
}

template <typename WT>
void launch_wt(const float* A, const float* nw, float eps, const void* B, const float* R, float* C, int M, int N, int K) {
    switch (gemv_choose_vec(N, std::is_same<WT, float>::value ? DType::F32 : DType::F16)) {
        case 4: launch_vec<WT, 4>(A, nw, eps, B, R, C, M, N, K); break;
        case 2: launch_vec<WT, 2>(A, nw, eps, B, R, C, M, N, K); break;
        default: launch_vec<WT, 1>(A, nw, eps, B, R, C, M, N, K); break;
    }
}

template <typename WT, int VEC>
void launch_gu_vec(const float* A, const float* nw, float eps, const void* Wg, const void* Wu, float* C, int M, int N, int K) {
    const dim3 grid((N + GEMV_COLS * VEC - 1) / (GEMV_COLS * VEC), M), block(GEMV_COLS, GEMV_KSPLIT);
    gemv_gate_up_kernel<WT, VEC><<<grid, block, 0, exec_stream()>>>(A, nw, eps, static_cast<const WT*>(Wg), static_cast<const WT*>(Wu), C, M, N, K);
}

template <typename WT>
void launch_gu_wt(const float* A, const float* nw, float eps, const void* Wg, const void* Wu, float* C, int M, int N, int K) {
    switch (gemv_choose_vec(N, std::is_same<WT, float>::value ? DType::F32 : DType::F16)) {
        case 4: launch_gu_vec<WT, 4>(A, nw, eps, Wg, Wu, C, M, N, K); break;
        case 2: launch_gu_vec<WT, 2>(A, nw, eps, Wg, Wu, C, M, N, K); break;
        default: launch_gu_vec<WT, 1>(A, nw, eps, Wg, Wu, C, M, N, K); break;
    }
}

void check_launch_dims(int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0) throw std::invalid_argument("gemv: M, N, K must be positive");
    if (M > 65535) throw std::invalid_argument("gemv: at most 65535 rows");
}

}  // namespace

void gemv_launch(const float* A, const float* norm_w, float eps, const void* B, DType dt, const float* R,
                 float* C, int M, int N, int K) {
    check_launch_dims(M, N, K);
    switch (dt) {
        case DType::F32: launch_wt<float>(A, norm_w, eps, B, R, C, M, N, K); break;
        case DType::F16: launch_wt<__half>(A, norm_w, eps, B, R, C, M, N, K); break;
        case DType::BF16: launch_wt<__nv_bfloat16>(A, norm_w, eps, B, R, C, M, N, K); break;
    }
    CUDA_CHECK(cudaGetLastError());
}

void gemv_gate_up_launch(const float* A, const float* norm_w, float eps, const void* Wg, const void* Wu,
                         DType dt, float* C, int M, int N, int K) {
    check_launch_dims(M, N, K);
    switch (dt) {
        case DType::F32: launch_gu_wt<float>(A, norm_w, eps, Wg, Wu, C, M, N, K); break;
        case DType::F16: launch_gu_wt<__half>(A, norm_w, eps, Wg, Wu, C, M, N, K); break;
        case DType::BF16: launch_gu_wt<__nv_bfloat16>(A, norm_w, eps, Wg, Wu, C, M, N, K); break;
    }
    CUDA_CHECK(cudaGetLastError());
}

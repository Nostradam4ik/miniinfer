// Fused operations: several model steps in one kernel launch.
// Why: in single-token decoding the kernels are short (a few microseconds) and the cost of a launch
// (~7 us) is comparable to the work. Fewer launches = less GPU idle time.
#include <stdexcept>
#include <string>
#include "gemv.h"
#include "gemv_config.h"
#include "ops.cuh"

static void check_fused_shapes(const Tensor& x, const WeightRef& W, const char* op) {
    if (x.ndim() != 2 || W.ndim != 2 || x.dim(1) != W.rows)
        throw std::invalid_argument(std::string(op) + ": expected x[M x K] and W[K x N]");
}

Tensor rmsnorm_matmul(const Tensor& x, const Tensor& norm_w, float eps, WeightRef W, MatmulKind large) {
    check_fused_shapes(x, W, "rmsnorm_matmul");
    if (norm_w.ndim() != 1 || norm_w.dim(0) != x.dim(1))
        throw std::invalid_argument("rmsnorm_matmul: norm_w must be [K]");
    const int M = x.dim(0), K = x.dim(1), N = W.cols;
    if (M == 0 || N == 0 || K == 0 || M > kGemvMaxRows) return matmul(rmsnorm(x, norm_w, eps), W, large);
    Tensor out = Tensor::uninitialized({M, N});
    gemv_launch(x.data(), norm_w.data(), eps, W.data, W.dtype, nullptr, out.data(), M, N, K);
    return out;
}

Tensor matmul_residual(const Tensor& x, WeightRef W, const Tensor& residual, MatmulKind large) {
    check_fused_shapes(x, W, "matmul_residual");
    const int M = x.dim(0), K = x.dim(1), N = W.cols;
    if (residual.ndim() != 2 || residual.dim(0) != M || residual.dim(1) != N)
        throw std::invalid_argument("matmul_residual: residual must be [M x N]");
    if (M == 0 || N == 0 || K == 0 || M > kGemvMaxRows) return add(matmul(x, W, large), residual);
    Tensor out = Tensor::uninitialized({M, N});
    gemv_launch(x.data(), nullptr, 0.f, W.data, W.dtype, residual.data(), out.data(), M, N, K);
    return out;
}

Tensor rmsnorm_gate_up(const Tensor& x, const Tensor& norm_w, float eps, WeightRef Wg, WeightRef Wu, MatmulKind large) {
    check_fused_shapes(x, Wg, "rmsnorm_gate_up");
    if (Wu.ndim != 2 || Wu.rows != Wg.rows || Wu.cols != Wg.cols)
        throw std::invalid_argument("rmsnorm_gate_up: Wg and Wu shapes differ");
    if (Wu.dtype != Wg.dtype) throw std::invalid_argument("rmsnorm_gate_up: Wg and Wu must have the same dtype");
    if (norm_w.ndim() != 1 || norm_w.dim(0) != x.dim(1))
        throw std::invalid_argument("rmsnorm_gate_up: norm_w must be [K]");
    const int M = x.dim(0), K = x.dim(1), N = Wg.cols;
    if (M == 0 || N == 0 || K == 0 || M > kGemvMaxRows) {
        Tensor h = rmsnorm(x, norm_w, eps);
        return silu_mul(matmul(h, Wg, large), matmul(h, Wu, large));
    }
    Tensor out = Tensor::uninitialized({M, N});
    gemv_gate_up_launch(x.data(), norm_w.data(), eps, Wg.data, Wu.data, Wg.dtype, out.data(), M, N, K);
    return out;
}

// RoPE for Q and K plus the write of K and V to the cache, all in one launch. The index space is
// [Q pairs | K pairs | V elements]; each thread handles one element of its section.
// The position of the first token comes from pos_ptr (device memory, for the CUDA graph) if not nullptr,
// otherwise from pos_offset. Cache rows outside [0, cache_rows) are not written (guards a device position).
__global__ void rope_qkv_cache_kernel(const float* qkv, float* q_out, float* ck, float* cv, int T, int H, int KV,
                                      int hd, float theta, int pos_offset, const int* pos_ptr, int cache_rows) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int pos0 = pos_ptr ? *pos_ptr : pos_offset;
    const int half = hd / 2;
    const long long nq = static_cast<long long>(T) * H * half;
    const long long nk = static_cast<long long>(T) * KV * half;
    const long long nv = static_cast<long long>(T) * KV * hd;
    if (idx >= nq + nk + nv) return;

    const int kv_dim = KV * hd;
    const size_t row_stride = static_cast<size_t>(H + 2 * KV) * hd;

    if (idx < nq + nk) {
        const bool is_q = idx < nq;
        const long long j = is_q ? idx : idx - nq;
        const int heads = is_q ? H : KV;
        const int i = static_cast<int>(j % half);
        const long long ht = j / half;  // t * heads + h
        const int h = static_cast<int>(ht % heads);
        const int t = static_cast<int>(ht / heads);

        if (!is_q && (pos0 < 0 || pos0 + t >= cache_rows)) return;  // cache row out of range: do not write

        const float* src =
            qkv + t * row_stride + (is_q ? 0 : static_cast<size_t>(H) * hd) + static_cast<size_t>(h) * hd;
        float* dst = is_q ? q_out + (static_cast<size_t>(t) * H + h) * hd
                          : ck + static_cast<size_t>(pos0 + t) * kv_dim + static_cast<size_t>(h) * hd;

        const float inv_freq = powf(theta, -2.0f * i / hd);
        const float ang = static_cast<float>(pos0 + t) * inv_freq;
        const float c = cosf(ang), s = sinf(ang);
        const float a = src[i], b = src[i + half];
        dst[i] = a * c - b * s;
        dst[i + half] = b * c + a * s;
    } else {
        const long long j = idx - nq - nk;
        const int t = static_cast<int>(j / kv_dim), d = static_cast<int>(j % kv_dim);
        if (pos0 < 0 || pos0 + t >= cache_rows) return;
        cv[static_cast<size_t>(pos0 + t) * kv_dim + d] = qkv[t * row_stride + static_cast<size_t>(H + KV) * hd + d];
    }
}

void launch_rope_qkv_cache(const float* qkv, float* q_out, float* cache_k, float* cache_v, int T, int num_heads,
                           int num_kv_heads, int head_dim, float theta, int pos_offset, const int* d_pos,
                           int cache_rows) {
    // Q pairs + K pairs + V elements
    const long long items =
        static_cast<long long>(T) * ((num_heads + num_kv_heads) * (head_dim / 2) + num_kv_heads * head_dim);
    if (items <= 0) return;
    const int block = 256;
    rope_qkv_cache_kernel<<<static_cast<unsigned>((items + block - 1) / block), block, 0, exec_stream()>>>(
        qkv, q_out, cache_k, cache_v, T, num_heads, num_kv_heads, head_dim, theta, pos_offset, d_pos, cache_rows);
    CUDA_CHECK(cudaGetLastError());
}

Tensor rope_qkv_to_cache(const Tensor& qkv, int num_heads, int num_kv_heads, int head_dim, float theta, int pos_offset,
                         Tensor& cache_k, Tensor& cache_v, const int* d_pos) {
    if (head_dim <= 0 || head_dim % 2 != 0)
        throw std::invalid_argument("rope_qkv_to_cache: head_dim must be positive and even");
    if (num_heads <= 0 || num_kv_heads <= 0) throw std::invalid_argument("rope_qkv_to_cache: bad head counts");
    if (pos_offset < 0) throw std::invalid_argument("rope_qkv_to_cache: negative pos_offset");
    const int width = (num_heads + 2 * num_kv_heads) * head_dim, kv_dim = num_kv_heads * head_dim;
    if (qkv.ndim() != 2 || qkv.dim(1) != width)
        throw std::invalid_argument("rope_qkv_to_cache: qkv must be [T x (H + 2*KV)*head_dim]");
    if (cache_k.ndim() != 2 || cache_v.ndim() != 2 || cache_k.dim(1) != kv_dim || cache_v.dim(1) != kv_dim ||
        cache_k.dim(0) != cache_v.dim(0))
        throw std::invalid_argument("rope_qkv_to_cache: cache shape mismatch");
    const int T = qkv.dim(0);
    if (!d_pos &&
        static_cast<long long>(pos_offset) + T > cache_k.dim(0))  // with d_pos the host does not know the position
        throw std::invalid_argument("rope_qkv_to_cache: rows exceed cache size");

    Tensor q = Tensor::uninitialized({T, num_heads * head_dim});
    launch_rope_qkv_cache(qkv.data(), q.data(), cache_k.data(), cache_v.data(), T, num_heads, num_kv_heads, head_dim,
                          theta, pos_offset, d_pos, cache_k.dim(0));
    return q;
}

Tensor concat_cols(const std::vector<const Tensor*>& parts) {
    if (parts.empty()) throw std::invalid_argument("concat_cols: nothing to concatenate");
    const int R = parts[0]->ndim() == 2 ? parts[0]->dim(0) : -1;
    int total = 0;
    for (const Tensor* p : parts) {
        if (p->ndim() != 2 || p->dim(0) != R) throw std::invalid_argument("concat_cols: row counts differ");
        total += p->dim(1);
    }
    Tensor out = Tensor::uninitialized({R, total});
    int offset = 0;
    for (const Tensor* p : parts) {
        const int c = p->dim(1);
        if (R > 0 && c > 0)
            CUDA_CHECK(cudaMemcpy2D(out.data() + offset, static_cast<size_t>(total) * sizeof(float), p->data(),
                                    static_cast<size_t>(c) * sizeof(float), static_cast<size_t>(c) * sizeof(float),
                                    static_cast<size_t>(R), cudaMemcpyDeviceToDevice));
        offset += c;
    }
    return out;
}

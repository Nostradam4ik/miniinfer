#include "ops.cuh"
#include "gemv.h"
#include "gemv_config.h"
#include "wt_convert.cuh"
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// All three kernels work for arbitrary M, N, K (not only square matrices).
// A[M x K], B[K x N], C[M x N], row-major.
// ---------------------------------------------------------------------------

// Naive version: one thread = one element of C.
__global__ void matmul_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float s = 0.f;
        for (int k = 0; k < K; ++k) s += A[row * K + k] * B[k * N + col];
        C[row * N + col] = s;
    }
}

// Tiled: 16x16 tiles in shared memory.
#define TILE 16
__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float s = 0.f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int aCol = t * TILE + tx;
        int bRow = t * TILE + ty;
        As[ty][tx] = (row < M && aCol < K) ? A[row * K + aCol] : 0.f;
        Bs[ty][tx] = (bRow < K && col < N) ? B[bRow * N + col] : 0.f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k) s += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = s;
}

// Register tiling: a 64x64 tile of C per block, 4x4 elements per thread.
#define BM 64
#define BK 16
#define TM 4
#define TN 4
// B may be fp32, fp16 or bf16: each element is converted to float while the tile is loaded
// into shared memory, and the computation is in fp32. This gives prefill with 16-bit weights without an fp32 copy.
template <typename WT>
__global__ void matmul_reg_kernel(const float* A, const WT* B, float* C, int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BM];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;  // 0..255
    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BM;

    float acc[TM][TN] = {};

    for (int t = 0; t < K; t += BK) {
        for (int i = 0; i < 4; ++i) {
            int idx = tid + i * 256;

            int ar = idx / BK, ac = idx % BK;
            int gr = blockRow + ar, gc = t + ac;
            As[ar][ac] = (gr < M && gc < K) ? A[gr * K + gc] : 0.f;

            int br = idx / BM, bc = idx % BM;
            int hr = t + br, hc = blockCol + bc;
            Bs[br][bc] = (hr < K && hc < N) ? wt_to_float<WT>(B[static_cast<size_t>(hr) * N + hc]) : 0.f;
        }
        __syncthreads();

        for (int k = 0; k < BK; ++k) {
            float a[TM], b[TN];
            for (int i = 0; i < TM; ++i) a[i] = As[ty * TM + i][k];
            for (int j = 0; j < TN; ++j) b[j] = Bs[k][tx * TN + j];
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) {
            int r = blockRow + ty * TM + i;
            int c = blockCol + tx * TN + j;
            if (r < M && c < N) C[r * N + c] = acc[i][j];
        }
}

// B may be in any of the weight formats. Naive and Tiled work only with fp32 (teaching versions);
// Reg and Gemv work with all of them.
static void launch_matmul_raw(MatmulKind kind, const float* A, const void* B, DType dt, float* C, int M, int N, int K) {
    if (M <= 0 || N <= 0) return;
    if (K <= 0) {  // empty sum: the result is zeros
        CUDA_CHECK(cudaMemset(C, 0, static_cast<size_t>(M) * N * sizeof(float)));
        return;
    }
    if (kind == MatmulKind::Auto) kind = M <= kGemvMaxRows ? MatmulKind::Gemv : MatmulKind::Reg;
    if (kind == MatmulKind::Gemv) {
        gemv_launch(A, nullptr, 0.f, B, dt, nullptr, C, M, N, K);
        return;
    }
    dim3 block(16, 16);
    switch (kind) {
        case MatmulKind::Naive:
        case MatmulKind::Tiled: {
            if (dt != DType::F32)
                throw std::invalid_argument("matmul: naive and tiled kernels support only fp32 weights");
            const float* Bf = static_cast<const float*>(B);
            dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
            if (kind == MatmulKind::Naive)
                matmul_naive_kernel<<<grid, block, 0, exec_stream()>>>(A, Bf, C, M, N, K);
            else
                matmul_tiled_kernel<<<grid, block, 0, exec_stream()>>>(A, Bf, C, M, N, K);
            break;
        }
        case MatmulKind::Reg: {
            dim3 grid((N + BM - 1) / BM, (M + BM - 1) / BM);
            if (dt == DType::F32)
                matmul_reg_kernel<float>
                    <<<grid, block, 0, exec_stream()>>>(A, static_cast<const float*>(B), C, M, N, K);
            else if (dt == DType::F16)
                matmul_reg_kernel<__half>
                    <<<grid, block, 0, exec_stream()>>>(A, static_cast<const __half*>(B), C, M, N, K);
            else
                matmul_reg_kernel<__nv_bfloat16>
                    <<<grid, block, 0, exec_stream()>>>(A, static_cast<const __nv_bfloat16*>(B), C, M, N, K);
            break;
        }
        case MatmulKind::Gemv:
        case MatmulKind::Auto:
            break;  // already handled above
    }
    CUDA_CHECK(cudaGetLastError());
}

void launch_matmul(MatmulKind kind, const float* A, const float* B, float* C, int M, int N, int K) {
    launch_matmul_raw(kind, A, B, DType::F32, C, M, N, K);
}

Tensor matmul(const Tensor& A, WeightRef B, MatmulKind kind) {
    if (A.ndim() != 2 || B.ndim != 2) throw std::invalid_argument("matmul: expected 2D tensors");
    if (A.dim(1) != B.rows)
        throw std::invalid_argument("matmul: inner dimensions differ (" + std::to_string(A.dim(1)) + " vs " +
                                    std::to_string(B.rows) + ")");
    int M = A.dim(0), K = A.dim(1), N = B.cols;
    Tensor C = Tensor::uninitialized({M, N});  // launch_matmul_raw writes every element (or zeros when K=0)
    launch_matmul_raw(kind, A.data(), B.data, B.dtype, C.data(), M, N, K);
    return C;
}

// ---------------------------------------------------------------------------
// Embedding: one block = one token.
// ---------------------------------------------------------------------------
template <typename WT>
__global__ void embedding_kernel(const WT* table, const int* ids, float* out, int D) {
    const WT* src = table + static_cast<size_t>(ids[blockIdx.x]) * D;
    float* dst = out + static_cast<size_t>(blockIdx.x) * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) dst[d] = wt_to_float<WT>(src[d]);
}

Tensor embedding(WeightRef table, const std::vector<int>& ids) {
    if (table.ndim != 2) throw std::invalid_argument("embedding: table must be 2D");
    const int V = table.rows, D = table.cols, T = static_cast<int>(ids.size());
    for (int id : ids)
        if (id < 0 || id >= V)
            throw std::invalid_argument("embedding: token id " + std::to_string(id) + " out of range [0, " +
                                        std::to_string(V) + ")");
    Tensor out = Tensor::uninitialized({T, D});
    if (T == 0 || D == 0) return out;
    GpuAllocator& alloc = GpuAllocator::instance();
    const size_t id_bytes = ids.size() * sizeof(int);
    int* d_ids = static_cast<int*>(alloc.allocate(id_bytes));
    CUDA_CHECK(cudaMemcpy(d_ids, ids.data(), id_bytes, cudaMemcpyHostToDevice));
    if (table.dtype == DType::F32)
        embedding_kernel<float>
            <<<T, 128, 0, exec_stream()>>>(static_cast<const float*>(table.data), d_ids, out.data(), D);
    else if (table.dtype == DType::F16)
        embedding_kernel<__half>
            <<<T, 128, 0, exec_stream()>>>(static_cast<const __half*>(table.data), d_ids, out.data(), D);
    else
        embedding_kernel<__nv_bfloat16>
            <<<T, 128, 0, exec_stream()>>>(static_cast<const __nv_bfloat16*>(table.data), d_ids, out.data(), D);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());  // d_ids may return to the pool only after the kernel has finished
    alloc.release(d_ids, id_bytes);
    return out;
}

// Embedding with ids in device memory: no host copy and no synchronization, so it can be captured
// into a CUDA graph. The host never sees the id, so an invalid id (outside [0, V)) yields a row of NaN,
// which the sampler later turns into an error, instead of reading outside the table.
template <typename WT>
__global__ void embedding_dev_kernel(const WT* table, const int* ids, float* out, int V, int D) {
    const int id = ids[blockIdx.x];
    float* dst = out + static_cast<size_t>(blockIdx.x) * D;
    if (id < 0 || id >= V) {
        for (int d = threadIdx.x; d < D; d += blockDim.x) dst[d] = NAN;
        return;
    }
    const WT* src = table + static_cast<size_t>(id) * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) dst[d] = wt_to_float<WT>(src[d]);
}

Tensor embedding_dev(WeightRef table, const int* d_ids, int T) {
    if (table.ndim != 2) throw std::invalid_argument("embedding_dev: table must be 2D");
    if (T < 0 || (T > 0 && !d_ids)) throw std::invalid_argument("embedding_dev: bad ids");
    const int V = table.rows, D = table.cols;
    Tensor out = Tensor::uninitialized({T, D});
    if (T == 0 || D == 0) return out;
    if (table.dtype == DType::F32)
        embedding_dev_kernel<float>
            <<<T, 128, 0, exec_stream()>>>(static_cast<const float*>(table.data), d_ids, out.data(), V, D);
    else if (table.dtype == DType::F16)
        embedding_dev_kernel<__half>
            <<<T, 128, 0, exec_stream()>>>(static_cast<const __half*>(table.data), d_ids, out.data(), V, D);
    else
        embedding_dev_kernel<__nv_bfloat16>
            <<<T, 128, 0, exec_stream()>>>(static_cast<const __nv_bfloat16*>(table.data), d_ids, out.data(), V, D);
    CUDA_CHECK(cudaGetLastError());
    return out;
}

// ---------------------------------------------------------------------------
// Copying rows between tensors (device -> device).
// ---------------------------------------------------------------------------
void write_rows(Tensor& dst, const Tensor& src, int row_offset) {
    if (dst.ndim() != 2 || src.ndim() != 2 || dst.dim(1) != src.dim(1))
        throw std::invalid_argument("write_rows: expected 2D tensors with equal column count");
    if (row_offset < 0 || static_cast<long long>(row_offset) + src.dim(0) > dst.dim(0))
        throw std::invalid_argument("write_rows: rows " + std::to_string(row_offset) + ".." +
                                    std::to_string(static_cast<long long>(row_offset) + src.dim(0)) +
                                    " exceed destination size " + std::to_string(dst.dim(0)));
    if (src.numel() == 0) return;
    CUDA_CHECK(cudaMemcpy(dst.data() + static_cast<size_t>(row_offset) * dst.dim(1), src.data(),
                          src.numel() * sizeof(float), cudaMemcpyDeviceToDevice));
}

Tensor slice_rows(const Tensor& x, int start, int count) {
    if (x.ndim() != 2) throw std::invalid_argument("slice_rows: expected 2D tensor");
    if (start < 0 || count < 0 || static_cast<long long>(start) + count > x.dim(0))
        throw std::invalid_argument("slice_rows: range out of bounds");
    Tensor out = Tensor::uninitialized({count, x.dim(1)});
    if (out.numel() == 0) return out;
    CUDA_CHECK(cudaMemcpy(out.data(), x.data() + static_cast<size_t>(start) * x.dim(1), out.numel() * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    return out;
}

// ---------------------------------------------------------------------------
// Transpose through shared memory: both the reads and the writes are sequential (coalesced).
// The tile row is padded to 33 to avoid bank conflicts.
// ---------------------------------------------------------------------------
#define TT 32
__global__ void transpose_kernel(const float* in, float* out, int R, int C) {
    __shared__ float tile[TT][TT + 1];
    int x = blockIdx.x * TT + threadIdx.x;  // input column
    int y = blockIdx.y * TT + threadIdx.y;  // input row
    for (int j = 0; j < TT; j += 8)
        if (x < C && y + j < R) tile[threadIdx.y + j][threadIdx.x] = in[static_cast<size_t>(y + j) * C + x];
    __syncthreads();
    x = blockIdx.y * TT + threadIdx.x;  // output column = input row
    y = blockIdx.x * TT + threadIdx.y;  // output row = input column
    for (int j = 0; j < TT; j += 8)
        if (x < R && y + j < C) out[static_cast<size_t>(y + j) * R + x] = tile[threadIdx.x][threadIdx.y + j];
}

Tensor transpose(const Tensor& x) {
    if (x.ndim() != 2) throw std::invalid_argument("transpose: expected 2D tensor");
    const int R = x.dim(0), C = x.dim(1);
    Tensor out = Tensor::uninitialized({C, R});
    if (R == 0 || C == 0) return out;
    transpose_kernel<<<dim3((C + TT - 1) / TT, (R + TT - 1) / TT), dim3(TT, 8), 0, exec_stream()>>>(x.data(),
                                                                                                    out.data(), R, C);
    CUDA_CHECK(cudaGetLastError());
    return out;
}

// ---------------------------------------------------------------------------
// Elementwise operations
// ---------------------------------------------------------------------------
__global__ void add_kernel(const float* a, const float* b, float* out, size_t n) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void silu_mul_kernel(const float* g, const float* u, float* out, size_t n) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const float x = g[i];
        out[i] = x / (1.f + expf(-x)) * u[i];  // expf(-x) = inf for large negative x, then x/inf = -0
    }
}

static void check_same_shape(const Tensor& a, const Tensor& b, const char* op) {
    if (a.shape() != b.shape()) throw std::invalid_argument(std::string(op) + ": shapes differ");
}

Tensor add(const Tensor& a, const Tensor& b) {
    check_same_shape(a, b, "add");
    Tensor out = Tensor::uninitialized(a.shape());
    const size_t n = a.numel();
    if (n == 0) return out;
    add_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, exec_stream()>>>(a.data(), b.data(), out.data(), n);
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor silu_mul(const Tensor& gate, const Tensor& up) {
    check_same_shape(gate, up, "silu_mul");
    Tensor out = Tensor::uninitialized(gate.shape());
    const size_t n = gate.numel();
    if (n == 0) return out;
    silu_mul_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, exec_stream()>>>(gate.data(), up.data(),
                                                                                       out.data(), n);
    CUDA_CHECK(cudaGetLastError());
    return out;
}

// ---------------------------------------------------------------------------
// RoPE: one thread = one pair (i, i + hd/2) of one head of one token.
// sinf/cosf (accurate versions, not __sinf), because the angles are large and the fast intrinsics' error grows.
// ---------------------------------------------------------------------------
__global__ void rope_kernel(const float* x, float* y, int total_pairs, int heads, int hd, float theta, int pos_offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) return;
    const int half = hd / 2;
    const int i = idx % half;
    const int head_token = idx / half;  // t * heads + h
    const int t = head_token / heads;
    const size_t base = static_cast<size_t>(head_token) * hd;

    const float inv_freq = powf(theta, -2.0f * i / hd);
    const float ang = static_cast<float>(pos_offset + t) * inv_freq;
    const float c = cosf(ang), s = sinf(ang);
    const float a = x[base + i], b = x[base + i + half];
    y[base + i] = a * c - b * s;
    y[base + i + half] = b * c + a * s;
}

Tensor rope(const Tensor& x, int num_heads, int head_dim, float theta, int pos_offset) {
    if (head_dim <= 0 || head_dim % 2 != 0) throw std::invalid_argument("rope: head_dim must be positive and even");
    if (x.ndim() != 2 || num_heads <= 0 || x.dim(1) != num_heads * head_dim)
        throw std::invalid_argument("rope: expected x[T x (num_heads*head_dim)]");
    if (pos_offset < 0) throw std::invalid_argument("rope: negative pos_offset");
    const int T = x.dim(0);
    Tensor y = Tensor::uninitialized({T, x.dim(1)});
    const long long pairs = static_cast<long long>(T) * num_heads * (head_dim / 2);
    if (pairs == 0) return y;
    const int block = 256;
    rope_kernel<<<static_cast<unsigned>((pairs + block - 1) / block), block, 0, exec_stream()>>>(
        x.data(), y.data(), static_cast<int>(pairs), num_heads, head_dim, theta, pos_offset);
    CUDA_CHECK(cudaGetLastError());
    return y;
}

// ---------------------------------------------------------------------------
// Attention: one block = (token t, head h). The Q.K scores live in shared memory.
// Steps: scores -> maximum -> exp and sum -> weighted sum of V.
// Simple version without flash-style tiling.
// ---------------------------------------------------------------------------
__device__ float block_reduce_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    v = lane < (blockDim.x >> 5) ? sh[lane] : 0.f;
    if (wid == 0)
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    float r = sh[0];
    __syncthreads();  // sh can be reused
    return r;
}

__device__ float block_reduce_max(float v, float* sh) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    v = lane < (blockDim.x >> 5) ? sh[lane] : -INFINITY;
    if (wid == 0)
        for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    float r = sh[0];
    __syncthreads();
    return r;
}

// Dot product of two vectors of length n. If n is a multiple of 4, they are loaded as float4: neighboring
// threads work on different keys, which lie far apart in memory, so their loads cannot be coalesced, and
// the Q.K phase is limited by the number of load instructions. The terms are added one by one in index
// order, so both paths round the same way.
// a and b must be 16-byte aligned when n is a multiple of 4. That holds for heads inside a Tensor: the
// allocation is aligned and every head starts at a multiple of n floats.
__device__ float dot_product(const float* a, const float* b, int n) {
    float sum = 0.f;
    if (n % 4 == 0) {
        for (int i = 0; i < n; i += 4) {
            const float4 x = *reinterpret_cast<const float4*>(a + i);
            const float4 y = *reinterpret_cast<const float4*>(b + i);
            sum += x.x * y.x;
            sum += x.y * y.y;
            sum += x.z * y.z;
            sum += x.w * y.w;
        }
    } else {
        for (int i = 0; i < n; ++i) sum += a[i] * b[i];
    }
    return sum;
}

// The position of the first query is read from pos_ptr (device memory) if it is not nullptr, otherwise
// it is pos_offset. The first form is needed for the CUDA graph: one graph works for any position.
// max_keys is how many keys fit into shared memory (and the cache): if the position from the device is
// out of range, the output is filled with NaN (the error is visible) instead of overrunning shared memory.
__global__ void attention_kernel(const float* q, const float* k, const float* v, float* out, int H, int KV, int hd,
                                 int pos_offset, const int* pos_ptr, int max_keys, float scale) {
    extern __shared__ float scores[];  // [nvalid]
    __shared__ float red[32];

    const int t = blockIdx.x, h = blockIdx.y;
    const int kvh = h / (H / KV);
    const int pos = pos_ptr ? *pos_ptr : pos_offset;
    const int nvalid = pos + t + 1;
    const size_t kv_stride = static_cast<size_t>(KV) * hd;
    const float* qh = q + (static_cast<size_t>(t) * H + h) * hd;

    if (pos < 0 || nvalid > max_keys) {  // same branch for the whole block, so returning is safe
        float* bad = out + (static_cast<size_t>(t) * H + h) * hd;
        for (int d = threadIdx.x; d < hd; d += blockDim.x) bad[d] = NAN;
        return;
    }

    float lmax = -INFINITY;
    for (int j = threadIdx.x; j < nvalid; j += blockDim.x) {
        const float* kj = k + j * kv_stride + static_cast<size_t>(kvh) * hd;
        const float dot = dot_product(qh, kj, hd);
        const float s = dot * scale;
        scores[j] = s;
        lmax = fmaxf(lmax, s);
    }
    const float gmax = block_reduce_max(lmax, red);

    float lsum = 0.f;
    for (int j = threadIdx.x; j < nvalid; j += blockDim.x) {
        const float e = expf(scores[j] - gmax);
        scores[j] = e;
        lsum += e;
    }
    const float inv = 1.f / block_reduce_sum(lsum, red);  // contains __syncthreads

    float* oh = out + (static_cast<size_t>(t) * H + h) * hd;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j < nvalid; ++j) acc += scores[j] * v[j * kv_stride + static_cast<size_t>(kvh) * hd + d];
        oh[d] = acc * inv;
    }
}

Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v, int num_heads, int num_kv_heads, int head_dim,
                 int pos_offset, const int* d_pos) {
    if (num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 || num_heads % num_kv_heads != 0)
        throw std::invalid_argument("attention: num_heads must be a multiple of num_kv_heads");
    if (pos_offset < 0) throw std::invalid_argument("attention: negative pos_offset");
    if (q.ndim() != 2 || k.ndim() != 2 || v.ndim() != 2) throw std::invalid_argument("attention: q, k, v must be 2D");
    if (q.dim(1) != num_heads * head_dim || k.dim(1) != num_kv_heads * head_dim ||
        v.dim(1) != num_kv_heads * head_dim || k.dim(0) != v.dim(0))
        throw std::invalid_argument("attention: shape mismatch");
    const int T = q.dim(0), S = k.dim(0);
    // With the position on the device the host does not know it: size shared memory for the whole cache
    // (S keys); the kernel itself validates the position (see attention_kernel).
    if (!d_pos && S < pos_offset + T) throw std::invalid_argument("attention: need S >= pos_offset + T");
    if (S > kMaxAttentionKeys)
        throw std::invalid_argument("attention: too many keys (" + std::to_string(S) + " > " +
                                    std::to_string(kMaxAttentionKeys) + ")");
    Tensor out = Tensor::uninitialized({T, num_heads * head_dim});
    if (T == 0) return out;
    const int smem = (d_pos ? S : pos_offset + T) * static_cast<int>(sizeof(float));
    attention_kernel<<<dim3(T, num_heads), 128, smem, exec_stream()>>>(
        q.data(), k.data(), v.data(), out.data(), num_heads, num_kv_heads, head_dim, pos_offset, d_pos, S,
        1.0f / sqrtf(static_cast<float>(head_dim)));
    CUDA_CHECK(cudaGetLastError());
    return out;
}

// ---------------------------------------------------------------------------
// RMSNorm: one block (256 threads) = one row. Sum of squares: first every thread computes
// its own part, then a reduction within the warp (shuffle) and across warps.
// ---------------------------------------------------------------------------
__global__ void rmsnorm_kernel(const float* x, const float* w, float* y, int D, float eps) {
    const float* xr = x + static_cast<size_t>(blockIdx.x) * D;
    float* yr = y + static_cast<size_t>(blockIdx.x) * D;

    float ss = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) ss += xr[i] * xr[i];

    for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, off);

    __shared__ float warp_sums[32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) warp_sums[wid] = ss;
    __syncthreads();

    if (wid == 0) {
        ss = lane < (blockDim.x >> 5) ? warp_sums[lane] : 0.f;
        for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, off);
        if (lane == 0) warp_sums[0] = ss;
    }
    __syncthreads();

    const float rstd = rsqrtf(warp_sums[0] / D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) yr[i] = xr[i] * rstd * w[i];
}

Tensor rmsnorm(const Tensor& x, const Tensor& w, float eps) {
    if (x.ndim() != 2 || w.ndim() != 1 || x.dim(1) != w.dim(0))
        throw std::invalid_argument("rmsnorm: expected x[T x D] and w[D]");
    const int T = x.dim(0), D = x.dim(1);
    Tensor y = Tensor::uninitialized({T, D});
    if (T == 0 || D == 0) return y;
    rmsnorm_kernel<<<T, 256, 0, exec_stream()>>>(x.data(), w.data(), y.data(), D, eps);
    CUDA_CHECK(cudaGetLastError());
    return y;
}

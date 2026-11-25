#include <cmath>
#include <cstring>
#include <stdexcept>
#include "lowp.cuh"

namespace {

template <typename H>
__device__ __forceinline__ H float_to(float v);
template <>
__device__ __forceinline__ __half float_to<__half>(float v) { return __float2half_rn(v); }
template <>
__device__ __forceinline__ __nv_bfloat16 float_to<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }
__device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }

struct DevStats {
    unsigned long long overflow;
    unsigned long long flushed;
    unsigned int max_rel_bits;  // non-negative floats are ordered the same way as their bits as unsigned
};

// All threads of a block reach the shuffle (no early return), otherwise ballot/shuffle are invalid.
template <typename H, bool STATS>
__global__ void to_lowp_kernel(const float* in, H* out, size_t n, DevStats* st) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool valid = i < n;
    float v = valid ? in[i] : 0.f;
    H h = float_to<H>(v);
    if (valid) out[i] = h;

    if (STATS) {
        const float back = to_float(h);
        const bool of = valid && isfinite(v) && !isfinite(back);
        const bool fl = valid && v != 0.f && back == 0.f;
        float rel = 0.f;
        if (valid && v != 0.f && isfinite(v) && isfinite(back)) rel = fabsf(back - v) / fabsf(v);

        const unsigned mask = 0xffffffffu;
        const unsigned of_cnt = __popc(__ballot_sync(mask, of));
        const unsigned fl_cnt = __popc(__ballot_sync(mask, fl));
        for (int off = 16; off > 0; off >>= 1) rel = fmaxf(rel, __shfl_down_sync(mask, rel, off));
        if ((threadIdx.x & 31) == 0) {
            if (of_cnt) atomicAdd(&st->overflow, static_cast<unsigned long long>(of_cnt));
            if (fl_cnt) atomicAdd(&st->flushed, static_cast<unsigned long long>(fl_cnt));
            if (rel > 0.f) atomicMax(&st->max_rel_bits, __float_as_uint(rel));
        }
    }
}

template <typename H>
__global__ void from_lowp_kernel(const H* in, float* out, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = to_float(in[i]);
}

template <typename H>
void convert(const Tensor& w, LowpMatrix& out, LowpStats* stats) {
    const size_t n = w.numel();
    if (n == 0) return;
    const int block = 256;
    const unsigned grid = static_cast<unsigned>((n + block - 1) / block);
    if (!stats) {
        to_lowp_kernel<H, false><<<grid, block, 0, exec_stream()>>>(w.data(), static_cast<H*>(out.data()), n, nullptr);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    DevStats* d = static_cast<DevStats*>(GpuAllocator::instance().allocate(sizeof(DevStats)));
    CUDA_CHECK(cudaMemset(d, 0, sizeof(DevStats)));
    to_lowp_kernel<H, true><<<grid, block, 0, exec_stream()>>>(w.data(), static_cast<H*>(out.data()), n, d);
    CUDA_CHECK(cudaGetLastError());
    DevStats h{};
    CUDA_CHECK(cudaMemcpy(&h, d, sizeof(DevStats), cudaMemcpyDeviceToHost));
    GpuAllocator::instance().release(d, sizeof(DevStats));
    float rel;
    std::memcpy(&rel, &h.max_rel_bits, 4);
    stats->elements += n;
    stats->overflow += h.overflow;
    stats->flushed += h.flushed;
    stats->max_rel_error = std::fmax(stats->max_rel_error, static_cast<double>(rel));
}

}  // namespace

LowpMatrix to_lowp(const Tensor& w, DType dtype, LowpStats* stats) {
    if (dtype == DType::F32) throw std::invalid_argument("to_lowp: dtype must be f16 or bf16");
    if (w.ndim() != 2) throw std::invalid_argument("to_lowp: expected a 2D tensor");
    LowpMatrix out(w.shape(), dtype);
    if (dtype == DType::F16)
        convert<__half>(w, out, stats);
    else
        convert<__nv_bfloat16>(w, out, stats);
    return out;
}

Tensor from_lowp(const LowpMatrix& m) {
    Tensor out = Tensor::uninitialized(m.shape());
    const size_t n = m.numel();
    if (n == 0) return out;
    const int block = 256;
    const unsigned grid = static_cast<unsigned>((n + block - 1) / block);
    if (m.dtype() == DType::F16)
        from_lowp_kernel<__half><<<grid, block, 0, exec_stream()>>>(static_cast<const __half*>(m.data()), out.data(), n);
    else
        from_lowp_kernel<__nv_bfloat16><<<grid, block, 0, exec_stream()>>>(static_cast<const __nv_bfloat16*>(m.data()), out.data(), n);
    CUDA_CHECK(cudaGetLastError());
    return out;
}

#pragma once
// Weight matrices in fp16 or bf16 (2 bytes per element) and conversion from fp32.
//
// fp16:  1 sign, 5 exponent bits, 10 mantissa bits. Range up to 65504, normal numbers from 6.1e-5.
// bf16:  1 sign, 8 exponent bits, 7 mantissa bits. Same range as fp32.
// The SmolLM2 checkpoint is stored in bf16, so bf16 reproduces the weights EXACTLY. fp16 is also exact
// for values in its normal range, but very small weights become subnormal (lower precision).
// LowpStats shows how much exactly is lost.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include "tensor.cuh"

struct LowpStats {
    size_t elements = 0;
    size_t overflow = 0;     // a finite value became inf (outside the range of the format)
    size_t flushed = 0;      // a non-zero value became zero
    double max_rel_error = 0;  // largest relative rounding error among finite non-zero values

    void merge(const LowpStats& o) {
        elements += o.elements;
        overflow += o.overflow;
        flushed += o.flushed;
        if (o.max_rel_error > max_rel_error) max_rel_error = o.max_rel_error;
    }
};

// A 2D weight matrix in fp16/bf16 on the GPU. Owns its memory (returns it to the pool), move-only.
class LowpMatrix {
public:
    LowpMatrix() = default;
    LowpMatrix(std::vector<int> shape, DType dtype) : dtype_(dtype), shape_(std::move(shape)) {
        if (dtype == DType::F32) throw std::invalid_argument("LowpMatrix: dtype must be f16 or bf16");
        if (shape_.size() != 2 || shape_[0] < 0 || shape_[1] < 0)
            throw std::invalid_argument("LowpMatrix: expected a 2D shape");
        numel_ = static_cast<size_t>(shape_[0]) * static_cast<size_t>(shape_[1]);
        if (numel_ > 0) data_ = GpuAllocator::instance().allocate(numel_ * 2);
    }
    ~LowpMatrix() { release(); }

    LowpMatrix(const LowpMatrix&) = delete;
    LowpMatrix& operator=(const LowpMatrix&) = delete;
    LowpMatrix(LowpMatrix&& o) noexcept
        : data_(o.data_), numel_(o.numel_), dtype_(o.dtype_), shape_(std::move(o.shape_)) {
        o.data_ = nullptr;
        o.numel_ = 0;
    }
    LowpMatrix& operator=(LowpMatrix&& o) noexcept {
        if (this != &o) {
            release();
            data_ = o.data_;
            numel_ = o.numel_;
            dtype_ = o.dtype_;
            shape_ = std::move(o.shape_);
            o.data_ = nullptr;
            o.numel_ = 0;
        }
        return *this;
    }

    void* data() { return data_; }
    const void* data() const { return data_; }
    size_t numel() const { return numel_; }
    size_t bytes() const { return numel_ * 2; }
    DType dtype() const { return dtype_; }
    const std::vector<int>& shape() const { return shape_; }
    int rows() const { return shape_.empty() ? 0 : shape_[0]; }
    int cols() const { return shape_.size() < 2 ? 0 : shape_[1]; }

    operator WeightRef() const { return WeightRef(data_, dtype_, rows(), cols()); }  // NOLINT: implicit on purpose

private:
    void release() {
        if (data_) GpuAllocator::instance().release(data_, numel_ * 2);
        data_ = nullptr;
    }
    void* data_ = nullptr;
    size_t numel_ = 0;
    DType dtype_ = DType::F16;
    std::vector<int> shape_;
};

// fp32 -> fp16/bf16 on the GPU (round to nearest even). If stats != nullptr, fills in
// the loss statistics (slightly slower). The tensor must be 2D.
LowpMatrix to_lowp(const Tensor& w, DType dtype, LowpStats* stats = nullptr);

// fp16/bf16 -> fp32 on the GPU (for tests and diagnostics).
Tensor from_lowp(const LowpMatrix& m);

// The same rounding on the host: the value an element gets after to_lowp. For test references.
inline float round_to_dtype(float x, DType dt) {
    if (dt == DType::F16) return __half2float(__float2half_rn(x));
    if (dt == DType::BF16) return __bfloat162float(__float2bfloat16_rn(x));
    return x;
}

#pragma once
// Tensor: a float buffer on the GPU with a shape. RAII: the memory returns to the pool by itself.
// No copying (expensive and error-prone), only moving.
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>
#include "cuda_utils.cuh"
#include "gpu_alloc.cuh"

class Tensor {
public:
    Tensor() = default;

    // A zero tensor. For temporary kernel outputs prefer uninitialized(): zeroing something the
    // kernel will overwrite anyway is an extra launch per operation.
    explicit Tensor(std::vector<int> shape) : Tensor(std::move(shape), true) {}

    // Contents are undefined (may be leftovers of a previous tensor from the pool).
    // Use only where the kernel is guaranteed to write every element.
    static Tensor uninitialized(std::vector<int> shape) { return Tensor(std::move(shape), false); }

    static Tensor from_host(const std::vector<float>& host, std::vector<int> shape) {
        Tensor t = uninitialized(std::move(shape));
        if (host.size() != t.numel_)
            throw std::invalid_argument("Tensor::from_host: size does not match shape");
        if (t.numel_ > 0)
            CUDA_CHECK(cudaMemcpy(t.data_, host.data(), t.numel_ * sizeof(float),
                                  cudaMemcpyHostToDevice));
        return t;
    }

    ~Tensor() { release(); }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&& o) noexcept
        : data_(o.data_), numel_(o.numel_), shape_(std::move(o.shape_)) {
        o.data_ = nullptr;
        o.numel_ = 0;
    }
    Tensor& operator=(Tensor&& o) noexcept {
        if (this != &o) {
            release();
            data_ = o.data_;
            numel_ = o.numel_;
            shape_ = std::move(o.shape_);
            o.data_ = nullptr;
            o.numel_ = 0;
        }
        return *this;
    }

    std::vector<float> to_host() const {
        std::vector<float> h(numel_);
        if (numel_ > 0)
            CUDA_CHECK(cudaMemcpy(h.data(), data_, numel_ * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        return h;
    }

    float* data() { return data_; }
    const float* data() const { return data_; }
    size_t numel() const { return numel_; }
    const std::vector<int>& shape() const { return shape_; }
    int ndim() const { return static_cast<int>(shape_.size()); }
    int dim(int i) const { return shape_.at(i); }

private:
    Tensor(std::vector<int> shape, bool zero) : shape_(std::move(shape)) {
        numel_ = 1;
        for (int d : shape_) {
            if (d < 0) throw std::invalid_argument("Tensor: negative dimension");
            numel_ *= static_cast<size_t>(d);
        }
        if (shape_.empty()) numel_ = 0;
        if (numel_ > 0) {
            data_ = static_cast<float*>(GpuAllocator::instance().allocate(numel_ * sizeof(float)));
            if (zero) CUDA_CHECK(cudaMemset(data_, 0, numel_ * sizeof(float)));
        }
    }

    void release() {
        if (data_) GpuAllocator::instance().release(data_, numel_ * sizeof(float));
        data_ = nullptr;
    }

    float* data_ = nullptr;
    size_t numel_ = 0;
    std::vector<int> shape_;
};

// Storage format of the weights. Activations are always fp32; weights may be compressed to 2 bytes
// and are converted to fp32 on the fly during multiplication (accumulation is also in fp32).
enum class DType { F32, F16, BF16 };

inline size_t dtype_size(DType t) { return t == DType::F32 ? 4 : 2; }
inline const char* dtype_name(DType t) { return t == DType::F32 ? "f32" : t == DType::F16 ? "f16" : "bf16"; }

// A lightweight, non-owning description of a weight matrix [rows x cols] in GPU memory. Converts
// implicitly from Tensor (fp32), so operations taking a WeightRef work with old call sites
// unchanged. Also converts from LowpMatrix (fp16/bf16).
struct WeightRef {
    const void* data = nullptr;
    DType dtype = DType::F32;
    int ndim = 0;  // 2 for a valid matrix
    int rows = 0, cols = 0;

    WeightRef() = default;
    WeightRef(const Tensor& t) : data(t.data()), dtype(DType::F32), ndim(t.ndim()) {  // NOLINT: implicit on purpose
        if (ndim == 2) {
            rows = t.dim(0);
            cols = t.dim(1);
        }
    }
    WeightRef(const void* p, DType dt, int r, int c) : data(p), dtype(dt), ndim(2), rows(r), cols(c) {}
};

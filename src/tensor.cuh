#pragma once
// Tensor: float-буфер на GPU з формою. RAII: пам'ять повертається в пул сама.
// Копіювати не можна (дорого й небезпечно), тільки переміщувати.
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>
#include "cuda_utils.cuh"
#include "gpu_alloc.cuh"

class Tensor {
public:
    Tensor() = default;

    // Нульовий тензор. Для тимчасових результатів ядер краще uninitialized(): зануляти
    // те, що ядро все одно перезапише, це зайвий запуск на кожну операцію.
    explicit Tensor(std::vector<int> shape) : Tensor(std::move(shape), true) {}

    // Вміст не визначений (може бути сміття від попереднього тензора з пулу).
    // Використовувати лише там, де ядро гарантовано записує кожен елемент.
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

// Формат зберігання ваг. Активації завжди fp32; ваги можуть бути стиснуті до 2 байтів,
// а при множенні перетворюються на fp32 "на льоту" (накопичення теж у fp32).
enum class DType { F32, F16, BF16 };

inline size_t dtype_size(DType t) { return t == DType::F32 ? 4 : 2; }
inline const char* dtype_name(DType t) { return t == DType::F32 ? "f32" : t == DType::F16 ? "f16" : "bf16"; }

// Легкий опис матриці ваг [rows x cols] у пам'яті GPU, без володіння нею. Із Tensor (fp32)
// перетворюється неявно, тож операції, що приймають WeightRef, працюють зі старими викликами
// без змін. З LowpMatrix (fp16/bf16) теж.
struct WeightRef {
    const void* data = nullptr;
    DType dtype = DType::F32;
    int ndim = 0;  // 2 для коректної матриці
    int rows = 0, cols = 0;

    WeightRef() = default;
    WeightRef(const Tensor& t) : data(t.data()), dtype(DType::F32), ndim(t.ndim()) {  // NOLINT: неявно навмисно
        if (ndim == 2) {
            rows = t.dim(0);
            cols = t.dim(1);
        }
    }
    WeightRef(const void* p, DType dt, int r, int c) : data(p), dtype(dt), ndim(2), rows(r), cols(c) {}
};

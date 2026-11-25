#pragma once
// Матриці ваг у fp16 або bf16 (2 байти на елемент) і конвертація з fp32.
//
// fp16:  1 знак, 5 біт експоненти, 10 біт мантиси. Діапазон до 65504, нормальні числа від 6.1e-5.
// bf16:  1 знак, 8 біт експоненти, 7 біт мантиси. Діапазон як у fp32.
// Чекпойнт SmolLM2 зберігається в bf16, тож bf16 відтворює ваги ТОЧНО, а fp16 теж точно для
// значень у своєму нормальному діапазоні, але дуже малі ваги стають субнормальними (менша точність).
// Скільки саме втрачено, показує LowpStats.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include "tensor.cuh"

struct LowpStats {
    size_t elements = 0;
    size_t overflow = 0;     // скінченне значення стало inf (поза діапазоном формату)
    size_t flushed = 0;      // ненульове значення стало нулем
    double max_rel_error = 0;  // найбільша відносна похибка округлення серед скінченних ненульових

    void merge(const LowpStats& o) {
        elements += o.elements;
        overflow += o.overflow;
        flushed += o.flushed;
        if (o.max_rel_error > max_rel_error) max_rel_error = o.max_rel_error;
    }
};

// Двовимірна матриця ваг у fp16/bf16 на GPU. Володіє пам'яттю (повертає в пул), лише переміщується.
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

    operator WeightRef() const { return WeightRef(data_, dtype_, rows(), cols()); }  // NOLINT: неявно навмисно

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

// fp32 -> fp16/bf16 на GPU (округлення до найближчого парного). Якщо stats != nullptr, заповнює
// статистику втрат (трохи повільніше). Тензор має бути 2D.
LowpMatrix to_lowp(const Tensor& w, DType dtype, LowpStats* stats = nullptr);

// fp16/bf16 -> fp32 на GPU (для тестів і діагностики).
Tensor from_lowp(const LowpMatrix& m);

// Те саме округлення на хості: значення, яке отримає елемент після to_lowp. Для еталонів у тестах.
inline float round_to_dtype(float x, DType dt) {
    if (dt == DType::F16) return __half2float(__float2half_rn(x));
    if (dt == DType::BF16) return __bfloat162float(__float2bfloat16_rn(x));
    return x;
}

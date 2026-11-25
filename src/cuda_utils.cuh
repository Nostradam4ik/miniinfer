#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Без цієї перевірки помилки CUDA мовчать.
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__,         \
                         __LINE__, cudaGetErrorString(err__));               \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

// Потік CUDA, у який бібліотека ставить запуски ядер. За замовчуванням 0 (стандартний потік);
// під час захоплення CUDA-графа DecodeGraph тимчасово підміняє його потоком захоплення.
// Окремий для кожного потоку виконання (thread_local), тож паралельні виклики не заважають.
inline cudaStream_t& exec_stream_ref() {
    static thread_local cudaStream_t s = 0;
    return s;
}
inline cudaStream_t exec_stream() { return exec_stream_ref(); }

// Тимчасово робить s потоком виконання (повертає попередній у деструкторі, навіть при винятку).
class ScopedStream {
public:
    explicit ScopedStream(cudaStream_t s) : prev_(exec_stream_ref()) { exec_stream_ref() = s; }
    ~ScopedStream() { exec_stream_ref() = prev_; }
    ScopedStream(const ScopedStream&) = delete;
    ScopedStream& operator=(const ScopedStream&) = delete;

private:
    cudaStream_t prev_;
};

// Вимірювання часу ядра через cudaEvent (у мілісекундах).
class GpuTimer {
public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    float stop() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
};

#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Without this check CUDA errors are silent.
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__,         \
                         __LINE__, cudaGetErrorString(err__));               \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

// The CUDA stream the library submits kernel launches to. Defaults to 0 (the default stream);
// while a CUDA graph is being captured, DecodeGraph temporarily replaces it with the capture stream.
// One per host thread (thread_local), so parallel callers do not interfere.
inline cudaStream_t& exec_stream_ref() {
    static thread_local cudaStream_t s = 0;
    return s;
}
inline cudaStream_t exec_stream() { return exec_stream_ref(); }

// Temporarily makes s the execution stream (the destructor restores the previous one, even on exceptions).
class ScopedStream {
public:
    explicit ScopedStream(cudaStream_t s) : prev_(exec_stream_ref()) { exec_stream_ref() = s; }
    ~ScopedStream() { exec_stream_ref() = prev_; }
    ScopedStream(const ScopedStream&) = delete;
    ScopedStream& operator=(const ScopedStream&) = delete;

private:
    cudaStream_t prev_;
};

// Kernel timing with cudaEvent (in milliseconds).
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

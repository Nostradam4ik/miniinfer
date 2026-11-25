// Vector addition: check on the CPU and measure the time.
// EXERCISE: vary the block size (64, 128, 256, 512) and record the time.
#include <cmath>
#include <cstdio>
#include <vector>
#include "cuda_utils.cuh"

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int n = 1 << 24;  // ~16M elements
    const size_t bytes = n * sizeof(float);

    std::vector<float> h_a(n), h_b(n), h_c(n);
    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i % 100);
        h_b[i] = 2.0f;
    }

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    for (int block : {64, 128, 256, 512}) {
        int grid = (n + block - 1) / block;
        vector_add<<<grid, block>>>(d_a, d_b, d_c, n);  // warm-up
        GpuTimer timer;
        timer.start();
        vector_add<<<grid, block>>>(d_a, d_b, d_c, n);
        float ms = timer.stop();
        CUDA_CHECK(cudaGetLastError());
        // reads 2 arrays and writes 1: 3 * bytes
        std::printf("block=%4d  %.3f ms  %.1f GB/s\n", block, ms,
                    3.0 * bytes / (ms * 1e-3) / 1e9);
    }

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    int errors = 0;
    for (int i = 0; i < n; ++i)
        if (std::fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5f) ++errors;
    std::printf(errors ? "ERRORS: %d\n" : "OK: the result is correct\n", errors);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return errors != 0;
}

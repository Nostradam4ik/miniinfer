// Бенчмарк множення матриць: naive / tiled / reg / cuBLAS.
// Запуск: matmul.exe [N]   (N x N, за замовчуванням 512)
// Ядра лежать у ops.cu і перевіряються тестами (tests/).
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cublas_v2.h>
#include "cuda_utils.cuh"
#include "ops.cuh"
#include "ops_cpu.h"

// cuBLAS працює в column-major, тому міняємо A і B місцями:
// (A*B)^T = B^T * A^T, і row-major C виходить без транспонування.
static void cublas_matmul(cublasHandle_t h, const float* A, const float* B, float* C,
                          int n) {
    const float alpha = 1.f, beta = 0.f;
    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, B, n, A, n, &beta, C, n);
}

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 512;
    std::printf("N = %d\n", n);
    const size_t bytes = static_cast<size_t>(n) * n * sizeof(float);

    std::vector<float> A(static_cast<size_t>(n) * n), B(A.size());
    for (size_t i = 0; i < A.size(); ++i) {
        A[i] = (i % 7) * 0.1f;
        B[i] = (i % 5) * 0.2f;
    }
    const std::vector<float> ref = cpu_matmul(A, B, n, n, n);
    std::vector<float> out(A.size());

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));

    const double flops = 2.0 * n * n * n;
    GpuTimer timer;

    // Прогрів + середнє за 20 запусків, потім звірка з CPU-еталоном.
    auto bench = [&](const char* name, auto&& launch) {
        launch();
        CUDA_CHECK(cudaDeviceSynchronize());
        const int iters = 20;
        timer.start();
        for (int i = 0; i < iters; ++i) launch();
        float ms = timer.stop() / iters;
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(out.data(), dC, bytes, cudaMemcpyDeviceToHost));
        std::printf("%-7s: %8.3f мс  %8.1f GFLOPS  max_diff=%.2e\n", name, ms,
                    flops / (ms * 1e-3) / 1e9, max_abs_diff(out, ref));
    };

    bench("naive", [&] { launch_matmul(MatmulKind::Naive, dA, dB, dC, n, n, n); });
    bench("tiled", [&] { launch_matmul(MatmulKind::Tiled, dA, dB, dC, n, n, n); });
    bench("reg", [&] { launch_matmul(MatmulKind::Reg, dA, dB, dC, n, n, n); });

    cublasHandle_t handle;
    cublasCreate(&handle);
    bench("cuBLAS", [&] { cublas_matmul(handle, dA, dB, dC, n); });
    cublasDestroy(handle);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}

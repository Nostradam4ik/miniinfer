// bench_ops.exe - where does decode time go: launch overhead or the kernels themselves?
//
// 1) cost of launching an empty kernel (host -> GPU);
// 2) matmul with M = 1 at the real model shapes: GPU time and effective memory bandwidth
//    (weights read / time) against the theoretical ~256 GB/s.
#include <chrono>
#include <cstdio>
#include <vector>
#include "cuda_utils.cuh"
#include "ops.cuh"

__global__ void empty_kernel() {}

int main() {
    using clock = std::chrono::steady_clock;

    // --- 1) launch cost ---
    {
        const int N = 2000;
        empty_kernel<<<1, 32>>>();  // warm-up
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t0 = clock::now();
        for (int i = 0; i < N; ++i) empty_kernel<<<1, 32>>>();
        CUDA_CHECK(cudaDeviceSynchronize());
        double us = std::chrono::duration<double, std::micro>(clock::now() - t0).count() / N;
        std::printf("empty kernel, back-to-back launches: %.2f us per launch\n", us);
        std::printf("  => ~455 launches per token = %.2f ms for launches alone\n\n", us * 455 / 1000);
    }

    // --- 2) matmul with M = 1 at real shapes (K x N), fp32 ---
    struct Shape {
        const char* name;
        int K, N;
        int count_per_token;
    };
    const Shape shapes[] = {{"q_proj    576x576  ", 576, 576, 30},  {"k_proj    576x192  ", 576, 192, 30},
                            {"v_proj    576x192  ", 576, 192, 30},  {"o_proj    576x576  ", 576, 576, 30},
                            {"gate/up   576x1536 ", 576, 1536, 60}, {"down     1536x576  ", 1536, 576, 30},
                            {"lm_head   576x49152", 576, 49152, 1}};
    // "via Tensor" = as in the model (output taken from the pool + launch);
    // "raw launch" = only the kernel launch on ready buffers. If the raw launch is much faster, the time
    // goes to host overhead; if not, the kernel itself is slow at M=1.
    std::printf("%-20s %12s %12s %12s %12s %10s\n", "matmul M=1", "Auto(Tensor)", "reg (raw)", "gemv (raw)",
                "GB/s (gemv)", "calls/tok");
    double total_tensor_ms = 0, total_raw_ms = 0, total_gemv_ms = 0;
    for (const Shape& s : shapes) {
        Tensor x({1, s.K}), w({s.K, s.N}), y({1, s.N});
        Tensor tmp = matmul(x, w);  // warm-up
        launch_matmul(MatmulKind::Reg, x.data(), w.data(), y.data(), 1, s.N, s.K);
        CUDA_CHECK(cudaDeviceSynchronize());
        const int iters = 300;
        GpuTimer timer;

        timer.start();
        for (int i = 0; i < iters; ++i) tmp = matmul(x, w);
        const double us_tensor = timer.stop() * 1000.0 / iters;

        timer.start();
        for (int i = 0; i < iters; ++i) launch_matmul(MatmulKind::Reg, x.data(), w.data(), y.data(), 1, s.N, s.K);
        const double us_raw = timer.stop() * 1000.0 / iters;

        timer.start();
        for (int i = 0; i < iters; ++i) launch_matmul(MatmulKind::Gemv, x.data(), w.data(), y.data(), 1, s.N, s.K);
        const double us_gemv = timer.stop() * 1000.0 / iters;

        const double gbs = (static_cast<double>(s.K) * s.N * 4) / (us_gemv * 1e-6) / 1e9;
        total_tensor_ms += us_tensor * s.count_per_token / 1000.0;
        total_raw_ms += us_raw * s.count_per_token / 1000.0;
        total_gemv_ms += us_gemv * s.count_per_token / 1000.0;
        std::printf("%-20s %9.1f us %9.1f us %9.1f us %12.1f %10d\n", s.name, us_tensor, us_raw, us_gemv, gbs,
                    s.count_per_token);
    }
    std::printf("\ntotal matmul per token: via Tensor (Auto) %.2f ms | reg %.2f ms | gemv %.2f ms\n", total_tensor_ms,
                total_raw_ms, total_gemv_ms);
    std::printf("(theoretical memory floor for fp32: ~2.1 ms for all weights)\n");
    return 0;
}

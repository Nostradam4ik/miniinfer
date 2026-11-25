// bench_ops.exe - де йде час при декодуванні: накладні витрати запуску чи самі ядра?
//
// 1) вартість запуску порожнього ядра (хост -> GPU);
// 2) matmul з M = 1 для реальних форм моделі: час на GPU, ефективна пропускна здатність пам'яті
//    (зчитані ваги / час) проти теоретичних ~256 ГБ/с.
#include <chrono>
#include <cstdio>
#include <vector>
#include "cuda_utils.cuh"
#include "ops.cuh"

__global__ void empty_kernel() {}

int main() {
    using clock = std::chrono::steady_clock;

    // --- 1) вартість запуску ---
    {
        const int N = 2000;
        empty_kernel<<<1, 32>>>();  // прогрів
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t0 = clock::now();
        for (int i = 0; i < N; ++i) empty_kernel<<<1, 32>>>();
        CUDA_CHECK(cudaDeviceSynchronize());
        double us = std::chrono::duration<double, std::micro>(clock::now() - t0).count() / N;
        std::printf("порожнє ядро, запуски підряд: %.2f мкс на запуск\n", us);
        std::printf("  => ~455 запусків на токен = %.2f мс лише на запуски\n\n", us * 455 / 1000);
    }

    // --- 2) matmul M = 1 на реальних формах (K x N), fp32 ---
    struct Shape { const char* name; int K, N; int count_per_token; };
    const Shape shapes[] = {{"q_proj    576x576  ", 576, 576, 30},   {"k_proj    576x192  ", 576, 192, 30},
                            {"v_proj    576x192  ", 576, 192, 30},   {"o_proj    576x576  ", 576, 576, 30},
                            {"gate/up   576x1536 ", 576, 1536, 60},  {"down     1536x576  ", 1536, 576, 30},
                            {"lm_head   576x49152", 576, 49152, 1}};
    // "через Tensor" = як у моделі (виділення виходу з пулу + запуск);
    // "сирий запуск" = лише запуск ядра на готових буферах. Якщо сирий запуск набагато швидший,
    // час іде на хост-накладні витрати; якщо ні - ядро саме повільне при M=1.
    std::printf("%-20s %12s %12s %12s %12s %10s\n", "matmul M=1", "Auto(Tensor)", "reg (сирий)", "gemv (сирий)",
                "ГБ/с (gemv)", "разів/ток");
    double total_tensor_ms = 0, total_raw_ms = 0, total_gemv_ms = 0;
    for (const Shape& s : shapes) {
        Tensor x({1, s.K}), w({s.K, s.N}), y({1, s.N});
        Tensor tmp = matmul(x, w);  // прогрів
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
        std::printf("%-20s %9.1f мкс %9.1f мкс %9.1f мкс %12.1f %10d\n", s.name, us_tensor, us_raw, us_gemv, gbs,
                    s.count_per_token);
    }
    std::printf("\nсума matmul на токен: через Tensor (Auto) %.2f мс | reg %.2f мс | gemv %.2f мс\n",
                total_tensor_ms, total_raw_ms, total_gemv_ms);
    std::printf("(теоретична межа пам'яті для fp32: ~2.1 мс на всі ваги)\n");
    return 0;
}

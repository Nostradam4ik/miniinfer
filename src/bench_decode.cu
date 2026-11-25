// bench_decode.exe - швидкість prefill і декодування по одному токену.
//
//   bench_decode.exe [--steps 64] [--no-pool] [--model model] [--ids "504 3575 282 4649 314"]
//
// Кожен крок декодування міряється окремо (разом з копіюванням логітів на CPU і вибором argmax),
// тож цифри відповідають реальній генерації. Друкує медіану й середнє, а також скільки пам'яті
// запитувалось на токен і скільки з цього були реальні cudaMalloc.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <memory>
#include "gemv.h"
#include "gpu_alloc.cuh"
#include "graph.cuh"
#include "model.cuh"
#include "sampler.h"

int main(int argc, char** argv) {
    try {
        std::string model_dir = "model", ids_str = "504 3575 282 4649 314";
        int steps = 64;
        bool pool = true;
        bool use_graph = false;
        DType dtype = DType::F32;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
                return argv[++i];
            };
            if (a == "--steps") steps = std::stoi(next());
            else if (a == "--no-pool") pool = false;
            else if (a == "--graph") use_graph = true;
            else if (a == "--model") model_dir = next();
            else if (a == "--ids") ids_str = next();
            else if (a == "--dtype") {
                const std::string t = next();
                if (t == "f32") dtype = DType::F32;
                else if (t == "f16") dtype = DType::F16;
                else if (t == "bf16") dtype = DType::BF16;
                else throw std::invalid_argument("--dtype must be f32, f16 or bf16");
            } else if (a == "--vec") gemv_set_vec_override(std::stoi(next()));
            else throw std::invalid_argument("unknown option: " + a);
        }
        std::vector<int> ids;
        {
            std::istringstream is(ids_str);
            int x;
            while (is >> x) ids.push_back(x);
        }
        if (ids.empty() || steps < 1) throw std::invalid_argument("need non-empty --ids and --steps >= 1");

        GpuAllocator::instance().set_pooling(pool);
        Model model = Model::load(model_dir, dtype);
        std::printf("weights: %s, %.0f MB | pool: %s | prompt: %zu tokens | steps: %d\n", dtype_name(dtype),
                    model.weight_bytes() / 1e6, pool ? "on" : "off", ids.size(), steps);
        if (dtype != DType::F32)
            std::printf("conversion: %zu elements, %zu overflow, %zu flushed to zero, max relative error %.2e\n",
                        model.lowp_stats.elements, model.lowp_stats.overflow, model.lowp_stats.flushed,
                        model.lowp_stats.max_rel_error);

        using clock = std::chrono::steady_clock;
        auto ms = [](clock::time_point a, clock::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };

        // Прогрів (перший запуск ядер і заповнення пулу), далі міряємо.
        {
            KVCache warm(model.cfg, static_cast<int>(ids.size()) + 8);
            std::vector<float> l = model.forward_cached(ids, warm, true).to_host();
            for (int i = 0; i < 4; ++i) l = model.forward_cached({Sampler::argmax(l)}, warm, true).to_host();
        }

        KVCache cache(model.cfg, static_cast<int>(ids.size()) + steps);
        GpuAllocator::instance().reset_counters();
        auto t0 = clock::now();
        std::vector<float> logits = model.forward_cached(ids, cache, true).to_host();
        const double prefill_ms = ms(t0, clock::now());
        const GpuAllocator::Stats after_prefill = GpuAllocator::instance().stats();

        std::unique_ptr<DecodeGraph> graph;
        if (use_graph) {
            auto g0 = clock::now();
            graph = std::make_unique<DecodeGraph>(model, cache);
            std::printf("CUDA graph: captured in %.2f ms, arena %.2f of %.0f MB\n", ms(g0, clock::now()),
                        graph->arena_used() / 1e6, graph->arena_capacity() / 1e6);
        }

        GpuAllocator::instance().reset_counters();
        std::vector<double> times;
        for (int i = 0; i < steps; ++i) {
            const int tok = Sampler::argmax(logits);
            auto s0 = clock::now();
            if (graph)
                graph->step(tok, logits);
            else
                logits = model.forward_cached({tok}, cache, true).to_host();
            times.push_back(ms(s0, clock::now()));
        }
        const GpuAllocator::Stats st = GpuAllocator::instance().stats();
        if (graph) {
            graph->time_replays(20);  // прогрів
            std::printf("graph replay only (no copies, no sync per step): %.3f ms\n", graph->time_replays(200));
        }

        std::vector<double> sorted = times;
        std::sort(sorted.begin(), sorted.end());
        double sum = 0;
        for (double t : times) sum += t;
        const double median = sorted[sorted.size() / 2], mean = sum / times.size();

        std::printf("prefill (%zu tokens): %.2f ms  [%zu allocs, %zu cudaMalloc]\n", ids.size(), prefill_ms,
                    after_prefill.requests, after_prefill.cuda_mallocs);
        std::printf("decode per token: median %.2f ms, mean %.2f ms, min %.2f, max %.2f\n", median, mean,
                    sorted.front(), sorted.back());
        std::printf("decode speed: %.1f tok/s (median)\n", 1000.0 / median);
        std::printf("allocations per token: %.1f requests, %.1f cudaMalloc, %.1f cudaFree, pool hit rate %.1f%%\n",
                    double(st.requests) / steps, double(st.cuda_mallocs) / steps, double(st.cuda_frees) / steps,
                    st.requests ? 100.0 * st.pool_hits / st.requests : 0.0);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}

#pragma once
// Цикл генерації: prefill промпту, потім по одному токену з KV-кешем.
#include <chrono>
#include <functional>
#include <memory>
#include <vector>
#include "graph.cuh"
#include "model.cuh"
#include "sampler.h"

struct GenerateOptions {
    int max_new_tokens = 64;
    SamplerConfig sampler;
    std::vector<int> stop_ids;  // токен із цього списку зупиняє генерацію (сам не додається)
    bool use_graph = false;     // кроки декодування через CUDA-граф (швидше; див. graph.cuh)
};

struct GenerateResult {
    std::vector<int> tokens;  // лише нові токени, без промпту й без стоп-токена
    bool hit_stop = false;
    double prefill_ms = 0;    // обробка промпту
    double decode_ms = 0;     // усі кроки по одному токену
    double graph_capture_ms = 0;  // захоплення графа (одноразово на виклик generate; не входить у decode_ms)
};

// on_token викликається одразу після кожного нового токена (для стрімінгу).
inline GenerateResult generate(const Model& model, const std::vector<int>& prompt,
                               const GenerateOptions& opt,
                               const std::function<void(int)>& on_token = nullptr) {
    if (prompt.empty()) throw std::invalid_argument("generate: empty prompt");
    if (opt.max_new_tokens < 0) throw std::invalid_argument("generate: negative max_new_tokens");
    const long long need = static_cast<long long>(prompt.size()) + opt.max_new_tokens;
    if (need > kMaxAttentionKeys)
        throw std::invalid_argument("generate: prompt + max_new_tokens = " + std::to_string(need) +
                                    " exceeds the limit of " + std::to_string(kMaxAttentionKeys));

    using clock = std::chrono::steady_clock;
    auto ms = [](clock::time_point a, clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    GenerateResult res;
    if (opt.max_new_tokens == 0) return res;

    Sampler sampler(opt.sampler);
    KVCache cache(model.cfg, static_cast<int>(need));
    auto is_stop = [&](int t) {
        return std::find(opt.stop_ids.begin(), opt.stop_ids.end(), t) != opt.stop_ids.end();
    };

    auto t0 = clock::now();
    std::vector<float> logits = model.forward_cached(prompt, cache, true).to_host();  // to_host синхронізує
    auto t1 = clock::now();
    res.prefill_ms = ms(t0, t1);

    // Граф має сенс, лише якщо буде хоча б один крок декодування (останній токен не потребує проходу).
    std::unique_ptr<DecodeGraph> graph;
    if (opt.use_graph && opt.max_new_tokens > 1) {
        auto g0 = clock::now();
        graph = std::make_unique<DecodeGraph>(model, cache);
        res.graph_capture_ms = ms(g0, clock::now());
    }

    for (int i = 0; i < opt.max_new_tokens; ++i) {
        const int tok = sampler.sample(logits);
        if (is_stop(tok)) {
            res.hit_stop = true;
            break;
        }
        res.tokens.push_back(tok);
        if (on_token) on_token(tok);
        if (i + 1 == opt.max_new_tokens) break;  // останній токен: наступний прохід не потрібен
        auto s0 = clock::now();
        if (graph)
            graph->step(tok, logits);
        else
            logits = model.forward_cached({tok}, cache, true).to_host();
        res.decode_ms += ms(s0, clock::now());
    }
    return res;
}

#pragma once
// Generation loop: prefill the prompt, then one token at a time with the KV cache.
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
    std::vector<int> stop_ids;  // a token from this list stops generation (it is not appended)
    bool use_graph = false;     // run decode steps through a CUDA graph (faster; see graph.cuh)
};

struct GenerateResult {
    std::vector<int> tokens;  // new tokens only, without the prompt and the stop token
    bool hit_stop = false;
    double prefill_ms = 0;        // prompt processing
    double decode_ms = 0;         // all single-token steps
    double graph_capture_ms = 0;  // graph capture (once per generate() call; not part of decode_ms)
};

// on_token is called right after each new token (for streaming).
inline GenerateResult generate(const Model& model, const std::vector<int>& prompt, const GenerateOptions& opt,
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
    auto is_stop = [&](int t) { return std::find(opt.stop_ids.begin(), opt.stop_ids.end(), t) != opt.stop_ids.end(); };

    auto t0 = clock::now();
    std::vector<float> logits = model.forward_cached(prompt, cache, true).to_host();  // to_host synchronizes
    auto t1 = clock::now();
    res.prefill_ms = ms(t0, t1);

    // The graph only makes sense if there is at least one decode step (the last token needs no forward pass).
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
        if (i + 1 == opt.max_new_tokens) break;  // last token: no further forward pass needed
        auto s0 = clock::now();
        if (graph)
            graph->step(tok, logits);
        else
            logits = model.forward_cached({tok}, cache, true).to_host();
        res.decode_ms += ms(s0, clock::now());
    }
    return res;
}

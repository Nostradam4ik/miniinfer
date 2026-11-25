// Tests of the CUDA graph for the decode step on the real model. The reference is the regular path
// (forward_cached) and PyTorch. Without model/ and reference/ the tests are skipped (see README).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include "generate.h"
#include "graph.cuh"
#include "ops_cpu.h"
#include "test_framework.h"
#include "weights.h"

#ifndef MINIINFER_ROOT
#define MINIINFER_ROOT "."
#endif

namespace {

bool files_exist() {
    static const bool ok = std::ifstream(std::string(MINIINFER_ROOT) + "/model/config.txt").good() &&
                           std::ifstream(std::string(MINIINFER_ROOT) + "/reference/generation.manifest.txt").good();
    if (!ok) std::printf("       (skipped: model/ or reference/ is missing; run tools/export_reference.py)\n");
    return ok;
}

const Model& model_for(DType dt) {
    static Model* models[3] = {nullptr, nullptr, nullptr};
    Model*& m = models[static_cast<int>(dt)];
    if (!m) m = new Model(Model::load(std::string(MINIINFER_ROOT) + "/model", dt));
    return *m;
}

TensorFile& gen_file() {
    static TensorFile* f = new TensorFile(std::string(MINIINFER_ROOT) + "/reference/generation");
    return *f;
}

double rel_diff(const std::vector<float>& got, const std::vector<float>& want) {
    double scale = 1.0;
    for (float v : want) scale = std::fmax(scale, std::fabs(static_cast<double>(v)));
    return max_abs_diff(got, want) / scale;
}

bool bitwise_equal(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// Tokens for the decode steps: deterministic, within the vocabulary.
std::vector<int> tokens(int n, int vocab, unsigned seed) {
    std::vector<int> t(n);
    unsigned s = seed;
    for (int& x : t) {
        s = s * 1664525u + 1013904223u;
        x = static_cast<int>((s >> 8) % static_cast<unsigned>(vocab));
    }
    return t;
}

const DType kDtypes[] = {DType::F32, DType::F16, DType::BF16};

}  // namespace

// Main check: the logits of every step through the graph match the regular cached path, and the
// caches (K, V) after the steps are bitwise identical.
TEST(graph_steps_match_normal_decode_for_all_dtypes) {
    if (!files_exist()) return;
    for (DType dt : kDtypes) {
        const Model& m = model_for(dt);
        const std::vector<int> prompt = gen_file().ints("gen2.prompt");
        const std::vector<int> toks = tokens(24, m.cfg.vocab_size, 7);

        KVCache normal(m.cfg, 64), graphed(m.cfg, 64);
        (void)m.forward_cached(prompt, normal, true);
        (void)m.forward_cached(prompt, graphed, true);
        DecodeGraph g(m, graphed);

        double worst = 0;
        std::vector<float> got;
        for (int tok : toks) {
            std::vector<float> want = m.forward_cached({tok}, normal, true).to_host();
            g.step(tok, got);
            CHECK(got.size() == want.size());
            worst = std::fmax(worst, rel_diff(got, want));
        }
        std::printf("       %s: %zu steps, largest relative logit difference graph/no graph: %.2e\n", dtype_name(dt),
                    toks.size(), worst);
        CHECK(worst <= 1e-6);
        CHECK(graphed.length == normal.length);
        for (int l = 0; l < m.cfg.num_layers; ++l) {
            const int n = normal.length;
            CHECK(bitwise_equal(slice_rows(normal.k[l], 0, n).to_host(), slice_rows(graphed.k[l], 0, n).to_host()));
            CHECK(bitwise_equal(slice_rows(normal.v[l], 0, n).to_host(), slice_rows(graphed.v[l], 0, n).to_host()));
        }
    }
}

// Key check: greedy generation through the graph matches PyTorch token for token.
TEST(graph_greedy_generation_matches_pytorch_for_all_dtypes) {
    if (!files_exist()) return;
    for (DType dt : kDtypes)
        for (const char* key : {"gen1", "gen2"}) {
            std::vector<int> prompt = gen_file().ints(std::string(key) + ".prompt");
            std::vector<int> want = gen_file().ints(std::string(key) + ".new");
            GenerateOptions o;
            o.max_new_tokens = static_cast<int>(want.size());
            o.use_graph = true;
            GenerateResult r = generate(model_for(dt), prompt, o);
            if (r.tokens != want) {
                size_t i = 0;
                while (i < r.tokens.size() && i < want.size() && r.tokens[i] == want[i]) ++i;
                std::ostringstream os;
                os << dtype_name(dt) << " " << key << ": mismatch at new token " << i;
                throw TestFailure(os.str());
            }
            std::printf("       %s %s: %zu tokens match PyTorch; graph capture %.1f ms, decode %.2f ms/token\n",
                        dtype_name(dt), key, want.size(), r.graph_capture_ms,
                        r.decode_ms / std::max<size_t>(1, want.size() - 1));
        }
}

// The arena really isolates the graph buffers: after capture, other tensors of the same size live their
// own life; the graph does not corrupt them and is not corrupted by them.
TEST(graph_buffers_are_isolated_from_the_pool) {
    if (!files_exist()) return;
    const Model& m = model_for(DType::BF16);
    const std::vector<int> prompt = gen_file().ints("gen1.prompt");
    const std::vector<int> toks = tokens(10, m.cfg.vocab_size, 11);
    const int H = m.cfg.hidden_size, I = m.cfg.intermediate_size, V = m.cfg.vocab_size;
    const int qkv = (m.cfg.num_heads + 2 * m.cfg.num_kv_heads) * m.cfg.head_dim;

    KVCache normal(m.cfg, 64), graphed(m.cfg, 64);
    (void)m.forward_cached(prompt, normal, true);
    (void)m.forward_cached(prompt, graphed, true);
    DecodeGraph g(m, graphed);

    // Tensors of the same sizes as the intermediates in the graph, with known contents. If the graph shared
    // memory with them, it would overwrite them, and they would overwrite its buffers.
    std::vector<Tensor> bystanders;
    const std::vector<std::vector<int>> shapes = {{1, H}, {1, qkv}, {1, I}, {1, V}, {1, m.cfg.num_heads * m.cfg.head_dim}};
    for (size_t i = 0; i < shapes.size(); ++i) {
        size_t n = 1;
        for (int d : shapes[i]) n *= d;
        bystanders.push_back(Tensor::from_host(std::vector<float>(n, 1000.f + static_cast<float>(i)), shapes[i]));
    }

    std::vector<float> got;
    for (int tok : toks) {
        std::vector<float> want = m.forward_cached({tok}, normal, true).to_host();  // the regular path uses the pool too
        g.step(tok, got);
        CHECK(rel_diff(got, want) <= 1e-6);
    }
    for (size_t i = 0; i < bystanders.size(); ++i)
        for (float v : bystanders[i].to_host()) CHECK(v == 1000.f + static_cast<float>(i));
}

TEST(graph_validates_tokens_and_cache_capacity) {
    if (!files_exist()) return;
    const Model& m = model_for(DType::BF16);
    KVCache cache(m.cfg, 4);
    DecodeGraph g(m, cache);
    std::vector<float> out;
    CHECK_THROWS(g.step(-1, out));
    CHECK_THROWS(g.step(m.cfg.vocab_size, out));
    CHECK(cache.length == 0);  // the error does not change the state
    for (int i = 0; i < 4; ++i) g.step(i + 1, out);
    CHECK(cache.length == 4);
    CHECK_THROWS(g.step(1, out));  // the cache is full
    CHECK(cache.length == 4);
    cache.reset();                  // after reset the graph works again from position zero
    g.step(1, out);
    CHECK(cache.length == 1);
}

// The same graph reproduces the same sequence after a cache reset: no state "sticks" between runs.
TEST(graph_is_reproducible_after_cache_reset) {
    if (!files_exist()) return;
    const Model& m = model_for(DType::F16);
    KVCache cache(m.cfg, 16);
    DecodeGraph g(m, cache);
    const std::vector<int> toks = tokens(8, m.cfg.vocab_size, 3);
    auto run = [&] {
        cache.reset();
        std::vector<std::vector<float>> all;
        std::vector<float> out;
        for (int t : toks) {
            g.step(t, out);
            all.push_back(out);
        }
        return all;
    };
    auto a = run(), b = run();
    for (size_t i = 0; i < a.size(); ++i) CHECK(bitwise_equal(a[i], b[i]));
}

// A graph captured at cache length zero (no prefill) works and agrees with the regular path.
TEST(graph_works_from_empty_cache) {
    if (!files_exist()) return;
    const Model& m = model_for(DType::BF16);
    KVCache normal(m.cfg, 16), graphed(m.cfg, 16);
    DecodeGraph g(m, graphed);
    const std::vector<int> toks = tokens(10, m.cfg.vocab_size, 5);
    std::vector<float> got;
    for (int t : toks) {
        std::vector<float> want = m.forward_cached({t}, normal, true).to_host();
        g.step(t, got);
        CHECK(rel_diff(got, want) <= 1e-6);
    }
}

// Two graphs on different caches (and even different models) do not interfere.
TEST(graph_two_graphs_interleaved_do_not_interfere) {
    if (!files_exist()) return;
    const Model& a = model_for(DType::BF16);
    const Model& b = model_for(DType::F32);
    const std::vector<int> prompt = gen_file().ints("gen1.prompt");
    const std::vector<int> toks = tokens(8, a.cfg.vocab_size, 9);

    auto solo = [&](const Model& m) {
        KVCache c(m.cfg, 32);
        (void)m.forward_cached(prompt, c, true);
        DecodeGraph g(m, c);
        std::vector<std::vector<float>> r;
        std::vector<float> out;
        for (int t : toks) {
            g.step(t, out);
            r.push_back(out);
        }
        return r;
    };
    auto ra = solo(a), rb = solo(b);

    KVCache ca(a.cfg, 32), cb(b.cfg, 32);
    (void)a.forward_cached(prompt, ca, true);
    (void)b.forward_cached(prompt, cb, true);
    DecodeGraph ga(a, ca), gb(b, cb);
    std::vector<float> out;
    for (size_t i = 0; i < toks.size(); ++i) {
        ga.step(toks[i], out);
        CHECK(bitwise_equal(out, ra[i]));
        gb.step(toks[i], out);
        CHECK(bitwise_equal(out, rb[i]));
    }
}

// Destroying a graph returns all resources: the arena is freed and the pool does not leak.
TEST(graph_destruction_releases_resources) {
    if (!files_exist()) return;
    const Model& m = model_for(DType::BF16);
    KVCache cache(m.cfg, 16);
    GpuAllocator& alloc = GpuAllocator::instance();
    const size_t in_use = alloc.stats().bytes_in_use;
    for (int i = 0; i < 3; ++i) {
        DecodeGraph g(m, cache);
        std::vector<float> out;
        g.step(1, out);
        cache.reset();
        CHECK(g.arena_used() > 0 && g.arena_used() <= g.arena_capacity());
    }
    CHECK(alloc.stats().bytes_in_use == in_use);
    CHECK(!alloc.arena_active());
}

// Arena too small: capture fails with a clear error, and the stream and allocator stay usable.
TEST(graph_capture_failure_leaves_everything_usable) {
    if (!files_exist()) return;
    const Model& m = model_for(DType::BF16);
    KVCache cache(m.cfg, 16);
    bool threw = false;
    try {
        DecodeGraph g(m, cache, 4096);  // too small even for the logits
    } catch (const std::exception& e) {
        threw = std::string(e.what()).find("arena") != std::string::npos || std::string(e.what()).find("Arena") != std::string::npos;
    }
    CHECK(threw);
    CHECK(!GpuAllocator::instance().arena_active());
    CHECK(exec_stream() == nullptr);
    // both the regular path and a new graph work after the failed attempt
    (void)m.forward_cached({1, 2, 3}, cache, true);
    DecodeGraph g(m, cache);
    std::vector<float> out;
    g.step(4, out);
    for (float v : out) CHECK(std::isfinite(v));
}

TEST(graph_generate_edge_cases) {
    if (!files_exist()) return;
    const Model& m = model_for(DType::BF16);
    GenerateOptions o;
    o.use_graph = true;
    o.max_new_tokens = 1;  // one token: no graph needed
    CHECK(generate(m, {504}, o).tokens.size() == 1);
    o.max_new_tokens = 0;
    CHECK(generate(m, {504}, o).tokens.empty());
    o.max_new_tokens = 2;
    GenerateResult r = generate(m, {504, 3575}, o);
    CHECK(r.tokens.size() == 2 && r.graph_capture_ms > 0);
}

// Seeded sampling through the graph gives the same tokens as without it (same logits = same choice).
TEST(graph_sampling_is_identical_to_normal_path) {
    if (!files_exist()) return;
    const Model& m = model_for(DType::BF16);
    const std::vector<int> prompt = gen_file().ints("gen2.prompt");
    GenerateOptions o;
    o.max_new_tokens = 16;
    o.sampler.temperature = 0.9f;
    o.sampler.top_p = 0.95f;
    o.sampler.seed = 12;
    std::vector<int> plain = generate(m, prompt, o).tokens;
    o.use_graph = true;
    CHECK(generate(m, prompt, o).tokens == plain);
}

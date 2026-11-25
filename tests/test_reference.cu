// Comparison with PyTorch: uses the activations written by tools/export_reference.py.
// Every test feeds a kernel the REFERENCE input (not the output of our previous kernel),
// so errors do not accumulate and the test shows exactly which kernel diverged.
// Without model/ or reference/ the tests are skipped (see README).
#include <algorithm>
#include <cstdio>
#include "generate.h"
#include "model.cuh"
#include "ops.cuh"
#include "ops_cpu.h"
#include "test_framework.h"
#include "weights.h"

#ifndef MINIINFER_ROOT
#define MINIINFER_ROOT "."
#endif

namespace {

struct Ref {
    TensorFile weights{std::string(MINIINFER_ROOT) + "/model/weights"};
    TensorFile ref{std::string(MINIINFER_ROOT) + "/reference/reference"};
    ModelConfig cfg = ModelConfig::load(std::string(MINIINFER_ROOT) + "/model/config.txt");
    float eps = 1e-5f;
};

// nullptr if the reference files are missing.
Ref* load_ref() {
    static Ref* r = nullptr;
    static bool tried = false;
    if (!tried) {
        tried = true;
        try {
            r = new Ref();
        } catch (const std::exception& e) {
            std::printf("       (skipped: %s; run tools/export_reference.py)\n", e.what());
        }
    }
    return r;
}

void check_close(const char* what, const std::vector<float>& got, const std::vector<float>& want, double tol) {
    double d = max_abs_diff(got, want);
    if (!(d <= tol)) {
        std::ostringstream os;
        os << what << ": max_diff=" << d << " (tol " << tol << ")";
        throw TestFailure(os.str());
    }
}

}  // namespace

TEST(ref_embedding_matches_pytorch) {
    Ref* r = load_ref();
    if (!r) return;
    Tensor table = r->weights.to_gpu("model.embed_tokens.weight");
    std::vector<int> ids = r->ref.ints("input_ids");
    check_close("embed_out", embedding(table, ids).to_host(), r->ref.floats("embed_out"), 0.0);
}

TEST(ref_layer0_input_rmsnorm_matches_pytorch) {
    Ref* r = load_ref();
    if (!r) return;
    Tensor x = r->ref.to_gpu("embed_out");
    Tensor w = r->weights.to_gpu("model.layers.0.input_layernorm.weight");
    check_close("l0.input_norm", rmsnorm(x, w, r->eps).to_host(), r->ref.floats("l0.input_norm"), 1e-5);
}

// Q/K/V projections of layer 0: matmul with the weights. In PyTorch the weight is [out x in], y = x * W^T,
// so the weights are transposed with the transpose kernel.
namespace {
Tensor transposed(const Tensor& w) {
    return transpose(w);
}
}  // namespace

TEST(ref_layer0_rope_matches_pytorch) {
    Ref* r = load_ref();
    if (!r) return;
    const float theta = 100000.f;
    check_close("l0.q_rope", rope(r->ref.to_gpu("l0.q_proj"), 9, 64, theta).to_host(), r->ref.floats("l0.q_rope"),
                1e-5);
    check_close("l0.k_rope", rope(r->ref.to_gpu("l0.k_proj"), 3, 64, theta).to_host(), r->ref.floats("l0.k_rope"),
                1e-5);
}

// Attention context (before o_proj) from the reference q_rope, k_rope, v_proj.
TEST(ref_layer0_attention_matches_pytorch) {
    Ref* r = load_ref();
    if (!r) return;
    Tensor out =
        attention(r->ref.to_gpu("l0.q_rope"), r->ref.to_gpu("l0.k_rope"), r->ref.to_gpu("l0.v_proj"), 9, 3, 64);
    check_close("l0.attn_ctx", out.to_host(), r->ref.floats("l0.attn_ctx"), 1e-5);
}

// Output projection: ctx * Wo^T = o_proj (which is also the output of the whole self_attn).
TEST(ref_layer0_o_proj_matches_pytorch) {
    Ref* r = load_ref();
    if (!r) return;
    Tensor wt = transposed(r->weights.to_gpu("model.layers.0.self_attn.o_proj.weight"));
    std::vector<float> got = matmul(r->ref.to_gpu("l0.attn_ctx"), wt).to_host();
    check_close("l0.o_proj", got, r->ref.floats("l0.o_proj"), 1e-4);
    check_close("l0.attn_out", got, r->ref.floats("l0.attn_out"), 1e-4);
}

// Relative error: max|diff| / max(1, max|ref|). Activations in deep layers can be large.
static double rel_diff(const std::vector<float>& got, const std::vector<float>& want) {
    double scale = 1.0;
    for (float v : want) scale = std::fmax(scale, std::fabs(static_cast<double>(v)));
    return max_abs_diff(got, want) / scale;
}

// Residual after attention and the second normalization.
TEST(ref_layer0_post_attention_norm_matches_pytorch) {
    Ref* r = load_ref();
    if (!r) return;
    Tensor x1 = add(r->ref.to_gpu("embed_out"), r->ref.to_gpu("l0.attn_out"));
    Tensor w = r->weights.to_gpu("model.layers.0.post_attention_layernorm.weight");
    check_close("l0.post_norm", rmsnorm(x1, w, r->eps).to_host(), r->ref.floats("l0.post_norm"), 1e-5);
}

// MLP of layer 0 from the reference input.
TEST(ref_layer0_mlp_matches_pytorch) {
    Ref* r = load_ref();
    if (!r) return;
    LayerWeights w = load_layer(r->weights, 0);
    check_close("l0.mlp_out", mlp_forward(r->ref.to_gpu("l0.post_norm"), w).to_host(), r->ref.floats("l0.mlp_out"),
                1e-4);
}

// The whole layer 0 from the reference embedding: output = hidden_states.1 = l0.out.
TEST(ref_layer0_output_matches_pytorch) {
    Ref* r = load_ref();
    if (!r) return;
    LayerWeights w = load_layer(r->weights, 0);
    std::vector<float> got = decoder_layer(r->ref.to_gpu("embed_out"), w, r->cfg).to_host();
    check_close("l0.out", got, r->ref.floats("l0.out"), 1e-4);
    check_close("hidden_states.1", got, r->ref.floats("hidden_states.1"), 1e-4);
}

// All 30 layers in sequence: each layer gets the OUTPUT of our previous layer, so the test
// shows where the accumulated error starts to grow. The last hidden_state (30) in
// transformers is after the final RMSNorm.
TEST(ref_all_layers_match_pytorch) {
    Ref* r = load_ref();
    if (!r) return;
    const ModelConfig& c = r->cfg;
    Tensor h = r->ref.to_gpu("hidden_states.0");
    double worst = 0;
    for (int i = 0; i < c.num_layers; ++i) {
        LayerWeights w = load_layer(r->weights, i);
        h = decoder_layer(h, w, c);
        const std::string name = "hidden_states." + std::to_string(i + 1);
        std::vector<float> got;
        if (i + 1 < c.num_layers) {
            got = h.to_host();
        } else {
            got = rmsnorm(h, r->weights.to_gpu("model.norm.weight"), c.rms_norm_eps).to_host();
        }
        double d = rel_diff(got, r->ref.floats(name));
        worst = std::fmax(worst, d);
        if (!(d <= 1e-3)) {
            std::ostringstream os;
            os << "layer " << i << " (" << name << "): relative error " << d << " > 1e-3";
            throw TestFailure(os.str());
        }
    }
    std::printf("       largest relative error over 30 layers: %.2e\n", worst);
}

// ---- full model -------------------------------------------------------------

namespace {
// The model is loaded once for all tests (reading 650 MB from disk).
const Model& shared_model() {
    static Model* m = new Model(Model::load(std::string(MINIINFER_ROOT) + "/model"));
    return *m;
}

// Indices of the k largest values of a row, largest first.
std::vector<int> top_k(const float* row, int n, int k) {
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return row[a] > row[b]; });
    idx.resize(k);
    return idx;
}
}  // namespace

// Logits at all positions match PyTorch, and the top tokens are the same.
TEST(ref_full_model_logits_match_pytorch) {
    Ref* r = load_ref();
    if (!r) return;
    const Model& m = shared_model();
    const std::vector<int> ids = r->ref.ints("input_ids");
    const int T = static_cast<int>(ids.size()), V = m.cfg.vocab_size;

    Tensor logits = m.forward(ids);
    CHECK(logits.dim(0) == T && logits.dim(1) == V);
    std::vector<float> got = logits.to_host(), want = r->ref.floats("logits");

    const double d = rel_diff(got, want);
    std::printf("       relative logit error: %.2e\n", d);
    if (!(d <= 1e-3)) {
        std::ostringstream os;
        os << "logits: relative error " << d << " > 1e-3";
        throw TestFailure(os.str());
    }
    for (int t = 0; t < T; ++t) {
        std::vector<int> a = top_k(&got[static_cast<size_t>(t) * V], V, 5);
        std::vector<int> b = top_k(&want[static_cast<size_t>(t) * V], V, 5);
        if (a != b) {
            std::ostringstream os;
            os << "position " << t << ": top-5 tokens differ (our top1=" << a[0] << ", PyTorch top1=" << b[0] << ")";
            throw TestFailure(os.str());
        }
    }
    // What the model would "say" next: ' the' (id 260) for "The capital of France is".
    const int next = top_k(&got[static_cast<size_t>(T - 1) * V], V, 1)[0];
    std::printf("       next token: %d\n", next);
    CHECK(next == 260);
}

// Causality of the whole model: the logits of a prefix do not depend on the tokens that follow it.
TEST(ref_full_model_prefix_logits_are_consistent) {
    Ref* r = load_ref();
    if (!r) return;
    const Model& m = shared_model();
    const std::vector<int> ids = r->ref.ints("input_ids");
    const int V = m.cfg.vocab_size, P = 3;
    std::vector<float> full = m.forward(ids).to_host();
    std::vector<float> prefix = m.forward(std::vector<int>(ids.begin(), ids.begin() + P)).to_host();
    std::vector<float> full_head(full.begin(), full.begin() + static_cast<size_t>(P) * V);
    CHECK(rel_diff(prefix, full_head) <= 1e-4);
}

// KV cache on the full model: prefill of 3 tokens + the rest one by one give the same logits
// as a full pass without the cache.
TEST(ref_full_model_cached_incremental_matches_uncached) {
    Ref* r = load_ref();
    if (!r) return;
    const Model& m = shared_model();
    const std::vector<int> ids = r->ref.ints("input_ids");
    const int T = static_cast<int>(ids.size()), V = m.cfg.vocab_size, P = 3;
    std::vector<float> full = m.forward(ids).to_host();

    KVCache cache(m.cfg, 64);
    std::vector<float> got = m.forward_cached(std::vector<int>(ids.begin(), ids.begin() + P), cache).to_host();
    for (int t = P; t < T; ++t) {
        std::vector<float> row = m.forward_cached({ids[t]}, cache).to_host();
        got.insert(got.end(), row.begin(), row.end());
    }
    CHECK(cache.length == T);
    CHECK(got.size() == full.size());
    const double d = rel_diff(got, full);
    std::printf("       relative difference cache/no cache: %.2e\n", d);
    CHECK(d <= 1e-4);
    (void)V;
}

TEST(model_last_only_returns_one_row_matching_full) {
    Ref* r = load_ref();
    if (!r) return;
    const Model& m = shared_model();
    const std::vector<int> ids = r->ref.ints("input_ids");
    const int T = static_cast<int>(ids.size()), V = m.cfg.vocab_size;
    KVCache cache(m.cfg, 64);
    Tensor last = m.forward_cached(ids, cache, true);
    CHECK(last.dim(0) == 1 && last.dim(1) == V);
    std::vector<float> full = m.forward(ids).to_host();
    std::vector<float> want(full.begin() + static_cast<size_t>(T - 1) * V, full.end());
    CHECK(rel_diff(last.to_host(), want) <= 1e-4);
}

TEST(model_cache_full_throws_and_reset_works) {
    Ref* r = load_ref();
    if (!r) return;
    const Model& m = shared_model();
    KVCache cache(m.cfg, 4);
    m.forward_cached({1, 2, 3}, cache);
    CHECK_THROWS(m.forward_cached({4, 5}, cache));  // 3 + 2 > 4
    CHECK(cache.length == 3);                       // a failure does not corrupt the state
    m.forward_cached({4}, cache);                   // exactly up to the edge
    CHECK_THROWS(m.forward_cached({5}, cache));
    cache.reset();
    m.forward_cached({1, 2, 3, 4}, cache);  // available again after reset
}

// ---- generation -------------------------------------------------------------

namespace {
GenerateResult greedy(const Model& m, const std::vector<int>& prompt, int n) {
    GenerateOptions o;
    o.max_new_tokens = n;  // temperature = 0 -> greedy
    return generate(m, prompt, o);
}
}  // namespace

// Key check of generation: greedy output matches model.generate() in PyTorch
// token for token, on two prompts (5 and 15 tokens).
TEST(ref_greedy_generation_matches_pytorch) {
    Ref* r = load_ref();
    if (!r) return;
    TensorFile gen(std::string(MINIINFER_ROOT) + "/reference/generation");
    const Model& m = shared_model();
    for (const char* key : {"gen1", "gen2"}) {
        std::vector<int> prompt = gen.ints(std::string(key) + ".prompt");
        std::vector<int> want = gen.ints(std::string(key) + ".new");
        GenerateResult res = greedy(m, prompt, static_cast<int>(want.size()));
        if (res.tokens != want) {
            size_t i = 0;
            while (i < res.tokens.size() && i < want.size() && res.tokens[i] == want[i]) ++i;
            std::ostringstream os;
            os << key << ": mismatch at new token " << i << " (ours " << (i < res.tokens.size() ? res.tokens[i] : -1)
               << ", PyTorch " << (i < want.size() ? want[i] : -1) << ")";
            throw TestFailure(os.str());
        }
        std::printf("       %s: %zu tokens match; prefill %.1f ms, decode %.1f ms/token\n", key, want.size(),
                    res.prefill_ms, res.decode_ms / std::max<size_t>(1, want.size() - 1));
    }
}

TEST(generate_stops_at_stop_token_and_streams_in_order) {
    Ref* r = load_ref();
    if (!r) return;
    TensorFile gen(std::string(MINIINFER_ROOT) + "/reference/generation");
    const Model& m = shared_model();
    std::vector<int> prompt = gen.ints("gen2.prompt"), want = gen.ints("gen2.new");

    GenerateOptions o;
    o.max_new_tokens = 20;
    o.stop_ids = {want[4]};  // stop at the fifth new token
    std::vector<int> streamed;
    GenerateResult res = generate(m, prompt, o, [&](int t) { streamed.push_back(t); });
    CHECK(res.hit_stop);
    CHECK(res.tokens == std::vector<int>(want.begin(), want.begin() + 4));  // the stop token is not appended
    CHECK(streamed == res.tokens);
}

TEST(generate_edge_cases) {
    Ref* r = load_ref();
    if (!r) return;
    const Model& m = shared_model();
    CHECK(greedy(m, {504}, 0).tokens.empty());      // 0 new tokens
    CHECK(greedy(m, {504}, 1).tokens.size() == 1);  // exactly one
    CHECK_THROWS(greedy(m, {}, 5));                 // empty prompt
    CHECK_THROWS(greedy(m, {504}, -1));
    CHECK_THROWS(greedy(m, std::vector<int>(10, 504), kMaxAttentionKeys));  // does not fit
    CHECK_THROWS(greedy(m, {m.cfg.vocab_size}, 3));                         // id outside the vocabulary
}

// Seeded sampling is reproducible, and top_k=1 gives the same as greedy.
TEST(generate_sampling_is_reproducible_and_top_k_1_is_greedy) {
    Ref* r = load_ref();
    if (!r) return;
    TensorFile gen(std::string(MINIINFER_ROOT) + "/reference/generation");
    const Model& m = shared_model();
    std::vector<int> prompt = gen.ints("gen2.prompt");

    GenerateOptions o;
    o.max_new_tokens = 12;
    o.sampler.temperature = 0.9f;
    o.sampler.top_p = 0.95f;
    o.sampler.seed = 7;
    std::vector<int> a = generate(m, prompt, o).tokens, b = generate(m, prompt, o).tokens;
    CHECK(a == b);
    o.sampler.seed = 8;
    CHECK(generate(m, prompt, o).tokens != a);

    GenerateOptions k1;
    k1.max_new_tokens = 12;
    k1.sampler.temperature = 1.3f;
    k1.sampler.top_k = 1;
    CHECK(generate(m, prompt, k1).tokens == greedy(m, prompt, 12).tokens);
}

// ---- model with fp16 / bf16 weights ----------------------------------------

namespace {
const Model& lowp_model(DType dt) {
    static Model* models[2] = {nullptr, nullptr};
    Model*& m = models[dt == DType::F16 ? 0 : 1];
    if (!m) m = new Model(Model::load(std::string(MINIINFER_ROOT) + "/model", dt));
    return *m;
}
}  // namespace

// The weights are half the size, and the checkpoint is bf16, so bf16 loses nothing and fp16 almost nothing.
TEST(ref_lowp_conversion_stats_and_memory) {
    Ref* r = load_ref();
    if (!r) return;
    const Model& f32 = shared_model();
    for (DType dt : {DType::F16, DType::BF16}) {
        const Model& m = lowp_model(dt);
        CHECK(m.weight_bytes() * 2 == f32.weight_bytes());  // exactly half
        const LowpStats& s = m.lowp_stats;
        std::printf("       %s: %zu elements, %zu overflow, %zu flushed to zero, max relative error %.2e\n",
                    dtype_name(dt), s.elements, s.overflow, s.flushed, s.max_rel_error);
        CHECK(s.overflow == 0);
        if (dt == DType::BF16)
            CHECK(s.max_rel_error == 0.0 && s.flushed == 0);  // the checkpoint is already bf16
        else
            CHECK(s.flushed < 1000);  // fp16: a few tiny weights become zero
    }
}

// Logits at all positions against PyTorch (fp32 reference): the difference comes only from summation order,
// because the weights themselves (bf16) are reproduced exactly.
TEST(ref_lowp_model_logits_match_pytorch) {
    Ref* r = load_ref();
    if (!r) return;
    const std::vector<int> ids = r->ref.ints("input_ids");
    const std::vector<float> want = r->ref.floats("logits");
    const int T = static_cast<int>(ids.size()), V = lowp_model(DType::F16).cfg.vocab_size;
    for (DType dt : {DType::F16, DType::BF16}) {
        KVCache cache(lowp_model(dt).cfg, 64);
        std::vector<float> got = lowp_model(dt).forward_cached(ids, cache).to_host();
        const double d = rel_diff(got, want);
        std::printf("       %s: relative logit error against PyTorch: %.2e\n", dtype_name(dt), d);
        CHECK(d <= 1e-3);
        for (int t = 0; t < T; ++t)
            CHECK(top_k(&got[static_cast<size_t>(t) * V], V, 5) == top_k(&want[static_cast<size_t>(t) * V], V, 5));
    }
}

// Key check: greedy generation with 16-bit weights matches PyTorch token for token.
TEST(ref_lowp_greedy_generation_matches_pytorch) {
    Ref* r = load_ref();
    if (!r) return;
    TensorFile gen(std::string(MINIINFER_ROOT) + "/reference/generation");
    for (DType dt : {DType::F16, DType::BF16})
        for (const char* key : {"gen1", "gen2"}) {
            std::vector<int> prompt = gen.ints(std::string(key) + ".prompt"),
                             want = gen.ints(std::string(key) + ".new");
            GenerateResult res = greedy(lowp_model(dt), prompt, static_cast<int>(want.size()));
            if (res.tokens != want) {
                size_t i = 0;
                while (i < res.tokens.size() && i < want.size() && res.tokens[i] == want[i]) ++i;
                std::ostringstream os;
                os << dtype_name(dt) << " " << key << ": mismatch at new token " << i;
                throw TestFailure(os.str());
            }
            std::printf("       %s %s: %zu tokens match, decode %.2f ms/token\n", dtype_name(dt), key, want.size(),
                        res.decode_ms / std::max<size_t>(1, want.size() - 1));
        }
}

TEST(ref_lowp_model_rejects_uncached_reference_path) {
    Ref* r = load_ref();
    if (!r) return;
    CHECK_THROWS(lowp_model(DType::BF16).forward({1, 2, 3}));  // the uncached reference path is fp32 only
}

TEST(model_rejects_bad_input) {
    Ref* r = load_ref();
    if (!r) return;
    const Model& m = shared_model();
    CHECK_THROWS(m.forward({}));
    CHECK_THROWS(m.forward({m.cfg.vocab_size}));
    CHECK_THROWS(m.forward({-1}));
}

TEST(ref_layer0_qkv_projections_match_pytorch) {
    Ref* r = load_ref();
    if (!r) return;
    Tensor x = r->ref.to_gpu("l0.input_norm");
    struct P {
        const char* w;
        const char* out;
    };
    const P projs[] = {{"model.layers.0.self_attn.q_proj.weight", "l0.q_proj"},
                       {"model.layers.0.self_attn.k_proj.weight", "l0.k_proj"},
                       {"model.layers.0.self_attn.v_proj.weight", "l0.v_proj"}};
    for (const P& p : projs) {
        Tensor wt = transposed(r->weights.to_gpu(p.w));
        check_close(p.out, matmul(x, wt).to_host(), r->ref.floats(p.out), 1e-4);
    }
}

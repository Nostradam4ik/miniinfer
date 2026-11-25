// Properties of a decoder layer that need no reference weights: on small random weights.
#include <random>
#include "model.cuh"
#include "ops_cpu.h"
#include "test_framework.h"

namespace {

ModelConfig small_config() {
    ModelConfig c;
    c.vocab_size = 50;
    c.hidden_size = 16;
    c.intermediate_size = 32;
    c.num_layers = 1;
    c.num_heads = 4;
    c.num_kv_heads = 2;
    c.head_dim = 4;
    c.rms_norm_eps = 1e-5f;
    c.rope_theta = 10000.f;
    return c;
}

Tensor rnd_tensor(std::vector<int> shape, unsigned seed, float scale = 0.3f) {
    size_t n = 1;
    for (int d : shape) n *= d;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<float> v(n);
    for (float& x : v) x = dist(gen);
    return Tensor::from_host(v, shape);
}

LayerWeights random_weights(const ModelConfig& c, unsigned seed) {
    const int H = c.hidden_size, Q = c.num_heads * c.head_dim, KV = c.num_kv_heads * c.head_dim,
              I = c.intermediate_size;
    LayerWeights w;
    w.in_norm = rnd_tensor({H}, seed + 1, 1.f);
    w.post_norm = rnd_tensor({H}, seed + 2, 1.f);
    w.wq = rnd_tensor({H, Q}, seed + 3);
    w.wk = rnd_tensor({H, KV}, seed + 4);
    w.wv = rnd_tensor({H, KV}, seed + 5);
    w.wo = rnd_tensor({Q, H}, seed + 6);
    w.w_gate = rnd_tensor({H, I}, seed + 7);
    w.w_up = rnd_tensor({H, I}, seed + 8);
    w.w_down = rnd_tensor({I, H}, seed + 9);
    w.build_fused();
    return w;
}

}  // namespace

// If o_proj and down_proj are zero, both branches give 0 and the layer is the identity (residual).
TEST(layer_with_zero_output_projections_is_identity) {
    ModelConfig c = small_config();
    LayerWeights w = random_weights(c, 1);
    w.wo = Tensor({c.num_heads * c.head_dim, c.hidden_size});  // zeros
    w.w_down = Tensor({c.intermediate_size, c.hidden_size});
    std::vector<float> x = rnd_tensor({6, c.hidden_size}, 99, 2.f).to_host();
    std::vector<float> out = decoder_layer(Tensor::from_host(x, {6, c.hidden_size}), w, c).to_host();
    CHECK_NEAR(max_abs_diff(out, x), 0.0, 0.0);
}

// MLP branch only: if wo = 0, then out = x + mlp(rmsnorm(x)).
TEST(layer_mlp_branch_is_residual_plus_mlp) {
    ModelConfig c = small_config();
    LayerWeights w = random_weights(c, 2);
    w.wo = Tensor({c.num_heads * c.head_dim, c.hidden_size});
    Tensor x = rnd_tensor({5, c.hidden_size}, 7, 1.f);
    Tensor expect =
        add(x, mlp_forward(rmsnorm(x, w.post_norm, c.rms_norm_eps), w));
    CHECK_NEAR(max_abs_diff(decoder_layer(x, w, c).to_host(), expect.to_host()), 0.0, 0.0);
}

// Causality of the whole layer: changing tokens AFTER t does not change the output at positions <= t.
TEST(layer_is_causal) {
    ModelConfig c = small_config();
    LayerWeights w = random_weights(c, 3);
    const int T = 8, cut = 5, H = c.hidden_size;
    std::vector<float> x = rnd_tensor({T, H}, 11, 1.f).to_host();
    std::vector<float> a = decoder_layer(Tensor::from_host(x, {T, H}), w, c).to_host();
    for (int t = cut; t < T; ++t)
        for (int d = 0; d < H; ++d) x[t * H + d] += 5.f;
    std::vector<float> b = decoder_layer(Tensor::from_host(x, {T, H}), w, c).to_host();
    for (int t = 0; t < cut; ++t)
        for (int d = 0; d < H; ++d) CHECK(a[t * H + d] == b[t * H + d]);
    // while the positions after cut really did change
    CHECK(max_abs_diff(std::vector<float>(a.begin() + cut * H, a.end()),
                       std::vector<float>(b.begin() + cut * H, b.end())) > 1e-3);
}

// The layer only sees the position through RoPE: permuting the tokens must change the result.
TEST(layer_depends_on_token_order) {
    ModelConfig c = small_config();
    LayerWeights w = random_weights(c, 4);
    const int H = c.hidden_size;
    std::vector<float> x = rnd_tensor({4, H}, 5, 1.f).to_host(), y = x;
    for (int d = 0; d < H; ++d) std::swap(y[0 * H + d], y[1 * H + d]);
    std::vector<float> a = decoder_layer(Tensor::from_host(x, {4, H}), w, c).to_host();
    std::vector<float> b = decoder_layer(Tensor::from_host(y, {4, H}), w, c).to_host();
    // the last token sees the same tokens in a different order, so the output must differ
    CHECK(max_abs_diff(std::vector<float>(a.end() - H, a.end()),
                       std::vector<float>(b.end() - H, b.end())) > 1e-4);
}

TEST(layer_output_shape_and_finite) {
    ModelConfig c = small_config();
    LayerWeights w = random_weights(c, 5);
    Tensor out = decoder_layer(rnd_tensor({7, c.hidden_size}, 3, 1.f), w, c);
    CHECK(out.dim(0) == 7 && out.dim(1) == c.hidden_size);
    for (float v : out.to_host()) CHECK(std::isfinite(v));
}

// ---- KV cache ---------------------------------------------------------------

// A cached layer fed one token at a time gives the same rows as the layer on the whole sequence.
TEST(cached_layer_token_by_token_matches_full) {
    ModelConfig c = small_config();
    LayerWeights w = random_weights(c, 21);
    const int T = 9, H = c.hidden_size;
    std::vector<float> x = rnd_tensor({T, H}, 5, 1.f).to_host();
    std::vector<float> full = decoder_layer(Tensor::from_host(x, {T, H}), w, c).to_host();

    KVCache cache(c, 16);
    for (int t = 0; t < T; ++t) {
        std::vector<float> row(x.begin() + t * H, x.begin() + (t + 1) * H);
        std::vector<float> out =
            decoder_layer_cached(Tensor::from_host(row, {1, H}), w, c, cache, 0, t).to_host();
        std::vector<float> want(full.begin() + t * H, full.begin() + (t + 1) * H);
        if (max_abs_diff(out, want) > 1e-5) {
            std::ostringstream os;
            os << "token " << t << ": difference " << max_abs_diff(out, want);
            throw TestFailure(os.str());
        }
    }
}

// Chunks of different sizes (prefill 3, then 2, then 4) behave the same as the whole sequence.
TEST(cached_layer_chunked_matches_full) {
    ModelConfig c = small_config();
    LayerWeights w = random_weights(c, 22);
    const int T = 9, H = c.hidden_size;
    std::vector<float> x = rnd_tensor({T, H}, 6, 1.f).to_host();
    std::vector<float> full = decoder_layer(Tensor::from_host(x, {T, H}), w, c).to_host();

    KVCache cache(c, 32);
    const int chunks[] = {3, 2, 4};
    int pos = 0;
    for (int n : chunks) {
        std::vector<float> part(x.begin() + pos * H, x.begin() + (pos + n) * H);
        std::vector<float> out =
            decoder_layer_cached(Tensor::from_host(part, {n, H}), w, c, cache, 0, pos).to_host();
        std::vector<float> want(full.begin() + pos * H, full.begin() + (pos + n) * H);
        CHECK(max_abs_diff(out, want) <= 1e-5);
        pos += n;
    }
}

// Old cache entries visible to a query are not corrupted by new tokens.
TEST(cache_keeps_earlier_entries_intact) {
    ModelConfig c = small_config();
    LayerWeights w = random_weights(c, 23);
    const int H = c.hidden_size, kv_dim = c.num_kv_heads * c.head_dim;
    KVCache cache(c, 8);
    decoder_layer_cached(rnd_tensor({3, H}, 1, 1.f), w, c, cache, 0, 0);
    std::vector<float> before = slice_rows(cache.k[0], 0, 3).to_host();
    decoder_layer_cached(rnd_tensor({2, H}, 2, 1.f), w, c, cache, 0, 3);
    std::vector<float> after = slice_rows(cache.k[0], 0, 3).to_host();
    CHECK(before == after);
    CHECK(before.size() == 3u * kv_dim);
}

TEST(kvcache_rejects_invalid_sizes) {
    ModelConfig c = small_config();
    CHECK_THROWS(KVCache(c, 0));
    CHECK_THROWS(KVCache(c, kMaxAttentionKeys + 1));
}

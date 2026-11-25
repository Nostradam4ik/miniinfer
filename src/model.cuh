#pragma once
// Llama-like model: config, weights of one layer and the forward pass of one decoder layer.
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include "lowp.cuh"
#include "ops.cuh"
#include "tensor.cuh"
#include "weights.h"

struct ModelConfig {
    int vocab_size = 0, hidden_size = 0, intermediate_size = 0;
    int num_layers = 0, num_heads = 0, num_kv_heads = 0, head_dim = 0;
    float rms_norm_eps = 1e-5f, rope_theta = 10000.f;
    bool tie_word_embeddings = true;

    // Reads model/config.txt (lines of "key value").
    static ModelConfig load(const std::string& path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("cannot open " + path);
        std::map<std::string, std::string> kv;
        std::string k, v;
        while (f >> k >> v) kv[k] = v;
        auto geti = [&](const char* key) {
            auto it = kv.find(key);
            if (it == kv.end()) throw std::runtime_error(std::string("config: missing ") + key);
            return std::stoi(it->second);
        };
        auto getf = [&](const char* key) {
            auto it = kv.find(key);
            if (it == kv.end()) throw std::runtime_error(std::string("config: missing ") + key);
            return std::stof(it->second);
        };
        ModelConfig c;
        c.vocab_size = geti("vocab_size");
        c.hidden_size = geti("hidden_size");
        c.intermediate_size = geti("intermediate_size");
        c.num_layers = geti("num_layers");
        c.num_heads = geti("num_heads");
        c.num_kv_heads = geti("num_kv_heads");
        c.head_dim = geti("head_dim");
        c.rms_norm_eps = getf("rms_norm_eps");
        c.rope_theta = getf("rope_theta");
        c.tie_word_embeddings = geti("tie_word_embeddings") != 0;
        return c;
    }
};

// Projection weights are stored already transposed ([in x out]), so y = x * W needs no extra steps.
struct LayerWeights {
    Tensor in_norm, post_norm;    // [hidden]
    Tensor wq, wk, wv, wo;        // [hidden x q_dim], [hidden x kv_dim] x2, [q_dim x hidden]
    Tensor w_gate, w_up, w_down;  // [hidden x inter] x2, [inter x hidden]
    // wq|wk|wv concatenated by columns: [hidden x (q_dim + 2*kv_dim)]. Needed by the KV-cache path,
    // where QKV is a single matmul. Duplicates the memory of wq/wk/wv (~2 MB per layer), because wq/wk/wv
    // stay for the uncached reference path (decoder_layer).
    Tensor wqkv;

    // Builds wqkv from wq, wk, wv. Call after filling those three.
    void build_fused() { wqkv = concat_cols({&wq, &wk, &wv}); }
};

inline LayerWeights load_layer(const TensorFile& f, int i) {
    const std::string p = "model.layers." + std::to_string(i) + ".";
    LayerWeights w;
    w.in_norm = f.to_gpu(p + "input_layernorm.weight");
    w.post_norm = f.to_gpu(p + "post_attention_layernorm.weight");
    w.wq = transpose(f.to_gpu(p + "self_attn.q_proj.weight"));
    w.wk = transpose(f.to_gpu(p + "self_attn.k_proj.weight"));
    w.wv = transpose(f.to_gpu(p + "self_attn.v_proj.weight"));
    w.wo = transpose(f.to_gpu(p + "self_attn.o_proj.weight"));
    w.w_gate = transpose(f.to_gpu(p + "mlp.gate_proj.weight"));
    w.w_up = transpose(f.to_gpu(p + "mlp.up_proj.weight"));
    w.w_down = transpose(f.to_gpu(p + "mlp.down_proj.weight"));
    w.build_fused();
    return w;
}

// Same, but the large weight matrices are fp16/bf16 (2 bytes per element). Norms stay fp32: they are
// tiny and RMSNorm precision matters. There is no uncached reference path for these weights,
// only the KV-cache path (decoder_layer_cached).
struct LayerWeightsLP {
    Tensor in_norm, post_norm;                  // [hidden], fp32
    LowpMatrix wqkv, wo, w_gate, w_up, w_down;  // same shapes as in LayerWeights
};

inline LayerWeightsLP load_layer_lowp(const TensorFile& f, int i, DType dt, LowpStats* stats = nullptr) {
    const std::string p = "model.layers." + std::to_string(i) + ".";
    LayerWeightsLP w;
    w.in_norm = f.to_gpu(p + "input_layernorm.weight");
    w.post_norm = f.to_gpu(p + "post_attention_layernorm.weight");
    {  // the temporary fp32 copies live only inside this block
        Tensor wq = transpose(f.to_gpu(p + "self_attn.q_proj.weight"));
        Tensor wk = transpose(f.to_gpu(p + "self_attn.k_proj.weight"));
        Tensor wv = transpose(f.to_gpu(p + "self_attn.v_proj.weight"));
        w.wqkv = to_lowp(concat_cols({&wq, &wk, &wv}), dt, stats);
    }
    w.wo = to_lowp(transpose(f.to_gpu(p + "self_attn.o_proj.weight")), dt, stats);
    w.w_gate = to_lowp(transpose(f.to_gpu(p + "mlp.gate_proj.weight")), dt, stats);
    w.w_up = to_lowp(transpose(f.to_gpu(p + "mlp.up_proj.weight")), dt, stats);
    w.w_down = to_lowp(transpose(f.to_gpu(p + "mlp.down_proj.weight")), dt, stats);
    return w;
}

// MLP: down( silu(gate(x)) * up(x) ). x: [T x hidden].
inline Tensor mlp_forward(const Tensor& x, const LayerWeights& w) {
    Tensor gate = matmul(x, w.w_gate);
    Tensor up = matmul(x, w.w_up);
    return matmul(silu_mul(gate, up), w.w_down);
}

// Self-attention block without the residual: ctx = attention(rope(q), rope(k), v), then o_proj.
inline Tensor attention_forward(const Tensor& x, const LayerWeights& w, const ModelConfig& c, int pos_offset = 0) {
    Tensor q = rope(matmul(x, w.wq), c.num_heads, c.head_dim, c.rope_theta, pos_offset);
    Tensor k = rope(matmul(x, w.wk), c.num_kv_heads, c.head_dim, c.rope_theta, pos_offset);
    Tensor v = matmul(x, w.wv);
    Tensor ctx = attention(q, k, v, c.num_heads, c.num_kv_heads, c.head_dim, pos_offset);
    return matmul(ctx, w.wo);
}

// One decoder layer (pre-norm, with residuals):
//   x1  = x  + attention(rmsnorm(x))
//   out = x1 + mlp(rmsnorm(x1))
// No KV cache here: k, v are computed from the whole sequence x (the cached path is below).
inline Tensor decoder_layer(const Tensor& x, const LayerWeights& w, const ModelConfig& c, int pos_offset = 0) {
    Tensor x1 = add(x, attention_forward(rmsnorm(x, w.in_norm, c.rms_norm_eps), w, c, pos_offset));
    return add(x1, mlp_forward(rmsnorm(x1, w.post_norm, c.rms_norm_eps), w));
}

// KV cache: for every layer stores the already computed K (after RoPE) and V for positions
// 0..length-1. New tokens are appended at the end, so each generation step only computes those.
struct KVCache {
    int max_seq = 0;
    int length = 0;            // how many positions are cached
    std::vector<Tensor> k, v;  // per layer: [max_seq x kv_dim]

    KVCache(const ModelConfig& c, int max_seq_) : max_seq(max_seq_) {
        if (max_seq_ <= 0 || max_seq_ > kMaxAttentionKeys)
            throw std::invalid_argument("KVCache: max_seq must be in [1, " + std::to_string(kMaxAttentionKeys) + "]");
        const int kv_dim = c.num_kv_heads * c.head_dim;
        for (int i = 0; i < c.num_layers; ++i) {
            k.emplace_back(std::vector<int>{max_seq_, kv_dim});
            v.emplace_back(std::vector<int>{max_seq_, kv_dim});
        }
    }
    void reset() { length = 0; }
};

// Decoder layer with the KV cache and fused ops (6 kernel launches per token instead of ~15):
//   1. rmsnorm + QKV in one GEMV                    (rmsnorm_matmul)
//   2. RoPE for Q and K + write of K, V to the cache (rope_qkv_to_cache)
//   3. attention over the whole cache
//   4. o_proj + residual                            (matmul_residual)
//   5. rmsnorm + gate + up + SiLU                   (rmsnorm_gate_up)
//   6. down_proj + residual                         (matmul_residual)
// For T > kGemvMaxRows (prefill) the fused ops fall back to separate ones. The result matches the
// reference decoder_layer (no cache) on the same sequence; the tests check this.
// LW is LayerWeights (fp32) or LayerWeightsLP (fp16/bf16): the weights arrive as WeightRef.
// d_pos != nullptr: the position is read from device memory (pos_offset is then ignored); needed for
// the CUDA graph (see graph.cuh).
template <class LW>
inline Tensor decoder_layer_cached(const Tensor& x, const LW& w, const ModelConfig& c, KVCache& cache, int layer,
                                   int pos_offset, const int* d_pos = nullptr) {
    if (WeightRef(w.wqkv).rows == 0)
        throw std::invalid_argument("decoder_layer_cached: wqkv is empty (call build_fused())");
    Tensor qkv = rmsnorm_matmul(x, w.in_norm, c.rms_norm_eps, w.wqkv);
    Tensor q = rope_qkv_to_cache(qkv, c.num_heads, c.num_kv_heads, c.head_dim, c.rope_theta, pos_offset, cache.k[layer],
                                 cache.v[layer], d_pos);
    Tensor ctx =
        attention(q, cache.k[layer], cache.v[layer], c.num_heads, c.num_kv_heads, c.head_dim, pos_offset, d_pos);
    Tensor x1 = matmul_residual(ctx, w.wo, x);
    Tensor act = rmsnorm_gate_up(x1, w.post_norm, c.rms_norm_eps, w.w_gate, w.w_up);
    return matmul_residual(act, w.w_down, x1);
}

// Shared implementation of a KV-cache step for fp32 and fp16/bf16 weights.
template <class LW>
inline Tensor forward_cached_impl(const ModelConfig& cfg, WeightRef embed, const std::vector<LW>& layers,
                                  const Tensor& final_norm, WeightRef lm_head, const std::vector<int>& ids,
                                  KVCache& cache, bool last_only) {
    if (ids.empty()) throw std::invalid_argument("Model::forward_cached: empty input");
    if (static_cast<long long>(cache.length) + static_cast<long long>(ids.size()) > cache.max_seq)
        throw std::invalid_argument("Model::forward_cached: KV cache is full (" + std::to_string(cache.length) + " + " +
                                    std::to_string(ids.size()) + " > " + std::to_string(cache.max_seq) + ")");
    if (static_cast<int>(cache.k.size()) != cfg.num_layers)
        throw std::invalid_argument("Model::forward_cached: cache does not match model");
    const int pos = cache.length;
    Tensor h = embedding(embed, ids);
    for (int i = 0; i < cfg.num_layers; ++i) h = decoder_layer_cached(h, layers[i], cfg, cache, i, pos);
    cache.length += static_cast<int>(ids.size());
    if (last_only) h = slice_rows(h, h.dim(0) - 1, 1);
    return rmsnorm_matmul(h, final_norm, cfg.rms_norm_eps, lm_head);  // final norm + lm_head
}

// One decode step (one token) with the token and position in DEVICE MEMORY: no host copies,
// synchronizations or host-side computations, so the whole step can be captured into a CUDA graph.
// Returns logits [1 x vocab]. The caller maintains the cache length (cache.length).
template <class LW>
inline Tensor forward_decode_dev(const ModelConfig& cfg, WeightRef embed, const std::vector<LW>& layers,
                                 const Tensor& final_norm, WeightRef lm_head, const int* d_token, const int* d_pos,
                                 KVCache& cache) {
    if (static_cast<int>(cache.k.size()) != cfg.num_layers)
        throw std::invalid_argument("forward_decode_dev: cache does not match model");
    Tensor h = embedding_dev(embed, d_token, 1);
    for (int i = 0; i < cfg.num_layers; ++i) h = decoder_layer_cached(h, layers[i], cfg, cache, i, 0, d_pos);
    return rmsnorm_matmul(h, final_norm, cfg.rms_norm_eps, lm_head);
}

// Full model: embedding -> decoder layers -> final RMSNorm -> lm_head.
// The weights live in GPU memory; loading reads model/weights.bin. The weight format is chosen at load
// time: fp32 (reference) or fp16/bf16 (half the memory and traffic per token).
struct Model {
    ModelConfig cfg;
    DType dtype = DType::F32;
    Tensor final_norm;  // [hidden], always fp32

    // dtype == F32:
    Tensor embed;    // [vocab x hidden]
    Tensor lm_head;  // [hidden x vocab] (already transposed; with tied weights = embed^T)
    std::vector<LayerWeights> layers;

    // dtype == F16 or BF16:
    LowpMatrix embed_lp, lm_head_lp;
    std::vector<LayerWeightsLP> layers_lp;
    LowpStats lowp_stats;  // precision lost in conversion (over all matrices)

    // dir contains config.txt, weights.bin and weights.manifest.txt (tools/export_reference.py).
    static Model load(const std::string& dir, DType dt = DType::F32) {
        Model m;
        m.dtype = dt;
        m.cfg = ModelConfig::load(dir + "/config.txt");
        TensorFile f(dir + "/weights");
        m.final_norm = f.to_gpu("model.norm.weight");
        if (dt == DType::F32) {
            m.embed = f.to_gpu("model.embed_tokens.weight");
            if (m.cfg.tie_word_embeddings)
                m.lm_head = transpose(m.embed);  // tied weights: lm_head = embed^T
            else
                m.lm_head = transpose(f.to_gpu("lm_head.weight"));
            m.layers.reserve(m.cfg.num_layers);
            for (int i = 0; i < m.cfg.num_layers; ++i) m.layers.push_back(load_layer(f, i));
        } else {
            {  // the fp32 copy of the embedding only exists here, to build embedding and lm_head from it
                Tensor embed32 = f.to_gpu("model.embed_tokens.weight");
                m.embed_lp = to_lowp(embed32, dt, &m.lowp_stats);
                if (m.cfg.tie_word_embeddings)
                    m.lm_head_lp = to_lowp(transpose(embed32), dt, &m.lowp_stats);
                else
                    m.lm_head_lp = to_lowp(transpose(f.to_gpu("lm_head.weight")), dt, &m.lowp_stats);
            }
            m.layers_lp.reserve(m.cfg.num_layers);
            for (int i = 0; i < m.cfg.num_layers; ++i) m.layers_lp.push_back(load_layer_lowp(f, i, dt, &m.lowp_stats));
        }
        return m;
    }

    // Bytes taken by the weights of the large matrices (without norms).
    size_t weight_bytes() const {
        size_t b = 0;
        if (dtype == DType::F32) {
            b += (embed.numel() + lm_head.numel()) * 4;
            for (const LayerWeights& w : layers)
                b += (w.wqkv.numel() + w.wo.numel() + w.w_gate.numel() + w.w_up.numel() + w.w_down.numel()) * 4;
        } else {
            b += embed_lp.bytes() + lm_head_lp.bytes();
            for (const LayerWeightsLP& w : layers_lp)
                b += w.wqkv.bytes() + w.wo.bytes() + w.w_gate.bytes() + w.w_up.bytes() + w.w_down.bytes();
        }
        return b;
    }

    // ids -> logits [T x vocab] through the uncached reference path. fp32 weights only.
    // Throws std::invalid_argument for empty input or bad ids.
    Tensor forward(const std::vector<int>& ids) const {
        if (dtype != DType::F32)
            throw std::invalid_argument("Model::forward: the reference path without cache requires f32 weights");
        if (ids.empty()) throw std::invalid_argument("Model::forward: empty input");
        Tensor h = embedding(embed, ids);
        for (const LayerWeights& w : layers) h = decoder_layer(h, w, cfg);
        return matmul(rmsnorm(h, final_norm, cfg.rms_norm_eps), lm_head);
    }

    // KV-cache step: ids are the NEW tokens (the whole prompt for prefill or one token for decoding).
    // Positions come from cache.length, which grows by ids.size() after the step.
    // last_only: compute logits only for the last token (enough for generation; saves a
    // [T x hidden] x [hidden x vocab] matmul). Returns [T x vocab] or [1 x vocab].
    Tensor forward_cached(const std::vector<int>& ids, KVCache& cache, bool last_only = false) const {
        if (dtype == DType::F32)
            return forward_cached_impl(cfg, embed, layers, final_norm, lm_head, ids, cache, last_only);
        return forward_cached_impl(cfg, embed_lp, layers_lp, final_norm, lm_head_lp, ids, cache, last_only);
    }

    // One token from d_token and position d_pos in device memory (see forward_decode_dev).
    Tensor forward_decode_dev(const int* d_token, const int* d_pos, KVCache& cache) const {
        if (dtype == DType::F32)
            return ::forward_decode_dev(cfg, embed, layers, final_norm, lm_head, d_token, d_pos, cache);
        return ::forward_decode_dev(cfg, embed_lp, layers_lp, final_norm, lm_head_lp, d_token, d_pos, cache);
    }
};

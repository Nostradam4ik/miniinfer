#pragma once
// Llama-подібна модель: конфіг, ваги шару і прохід одного шару декодера.
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

    // Читає model/config.txt (рядки "ключ значення").
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

// Ваги проєкцій тримаємо вже транспонованими ([in x out]), щоб y = x * W без додаткових кроків.
struct LayerWeights {
    Tensor in_norm, post_norm;                // [hidden]
    Tensor wq, wk, wv, wo;                    // [hidden x q_dim], [hidden x kv_dim] x2, [q_dim x hidden]
    Tensor w_gate, w_up, w_down;              // [hidden x inter] x2, [inter x hidden]
    // wq|wk|wv, склеєні по стовпцях: [hidden x (q_dim + 2*kv_dim)]. Потрібні шляху з KV-кешем,
    // де QKV рахується одним матмулом. Дублює пам'ять wq/wk/wv (~2 МБ на шар), бо wq/wk/wv лишаються
    // для еталонного шляху без кешу (decoder_layer).
    Tensor wqkv;

    // Збирає wqkv з wq, wk, wv. Викликати після заповнення цих трьох.
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

// Те саме, але великі матриці ваг у fp16/bf16 (2 байти на елемент). Норми лишаються fp32: вони
// крихітні, а точність RMSNorm важлива. Еталонного шляху без кешу для таких ваг немає,
// лише шлях з KV-кешем (decoder_layer_cached).
struct LayerWeightsLP {
    Tensor in_norm, post_norm;                      // [hidden], fp32
    LowpMatrix wqkv, wo, w_gate, w_up, w_down;      // ті самі форми, що й у LayerWeights
};

inline LayerWeightsLP load_layer_lowp(const TensorFile& f, int i, DType dt, LowpStats* stats = nullptr) {
    const std::string p = "model.layers." + std::to_string(i) + ".";
    LayerWeightsLP w;
    w.in_norm = f.to_gpu(p + "input_layernorm.weight");
    w.post_norm = f.to_gpu(p + "post_attention_layernorm.weight");
    {   // тимчасові fp32-копії живуть лише всередині цього блоку
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

// Self-attention блок без residual: ctx = attention(rope(q), rope(k), v), потім o_proj.
inline Tensor attention_forward(const Tensor& x, const LayerWeights& w, const ModelConfig& c,
                                int pos_offset = 0) {
    Tensor q = rope(matmul(x, w.wq), c.num_heads, c.head_dim, c.rope_theta, pos_offset);
    Tensor k = rope(matmul(x, w.wk), c.num_kv_heads, c.head_dim, c.rope_theta, pos_offset);
    Tensor v = matmul(x, w.wv);
    Tensor ctx = attention(q, k, v, c.num_heads, c.num_kv_heads, c.head_dim, pos_offset);
    return matmul(ctx, w.wo);
}

// Один шар декодера (pre-norm, з residual):
//   x1  = x  + attention(rmsnorm(x))
//   out = x1 + mlp(rmsnorm(x1))
// Поки без KV-кешу: k, v рахуються з усієї послідовності x (кеш буде на етапі 4).
inline Tensor decoder_layer(const Tensor& x, const LayerWeights& w, const ModelConfig& c,
                            int pos_offset = 0) {
    Tensor x1 = add(x, attention_forward(rmsnorm(x, w.in_norm, c.rms_norm_eps), w, c, pos_offset));
    return add(x1, mlp_forward(rmsnorm(x1, w.post_norm, c.rms_norm_eps), w));
}

// KV-кеш: для кожного шару зберігає вже порахований K (після RoPE) і V для позицій
// 0..length-1. Нові токени дописуються в кінець, тож кожен крок генерації рахує лише їх.
struct KVCache {
    int max_seq = 0;
    int length = 0;  // скільки позицій уже закешовано
    std::vector<Tensor> k, v;  // по шару: [max_seq x kv_dim]

    KVCache(const ModelConfig& c, int max_seq_) : max_seq(max_seq_) {
        if (max_seq_ <= 0 || max_seq_ > kMaxAttentionKeys)
            throw std::invalid_argument("KVCache: max_seq must be in [1, " +
                                        std::to_string(kMaxAttentionKeys) + "]");
        const int kv_dim = c.num_kv_heads * c.head_dim;
        for (int i = 0; i < c.num_layers; ++i) {
            k.emplace_back(std::vector<int>{max_seq_, kv_dim});
            v.emplace_back(std::vector<int>{max_seq_, kv_dim});
        }
    }
    void reset() { length = 0; }
};

// Шар декодера з KV-кешем і злитими операціями (6 запусків ядер на токен замість ~15):
//   1. rmsnorm + QKV одним GEMV                     (rmsnorm_matmul)
//   2. RoPE для Q і K + запис K, V у кеш            (rope_qkv_to_cache)
//   3. attention по всьому кешу
//   4. o_proj + residual                            (matmul_residual)
//   5. rmsnorm + gate + up + SiLU                   (rmsnorm_gate_up)
//   6. down_proj + residual                         (matmul_residual)
// Для T > kGemvMaxRows (prefill) злиті операції самі переходять на розділені. Результат збігається
// з еталонним decoder_layer (без кешу) на тій самій послідовності - це перевіряють тести.
// LW - LayerWeights (fp32) або LayerWeightsLP (fp16/bf16): ваги приходять як WeightRef.
// d_pos != nullptr: позиція береться з пам'яті пристрою (pos_offset тоді ігнорується); потрібно для
// CUDA-графа (див. graph.cuh).
template <class LW>
inline Tensor decoder_layer_cached(const Tensor& x, const LW& w, const ModelConfig& c,
                                   KVCache& cache, int layer, int pos_offset, const int* d_pos = nullptr) {
    if (WeightRef(w.wqkv).rows == 0) throw std::invalid_argument("decoder_layer_cached: wqkv is empty (call build_fused())");
    Tensor qkv = rmsnorm_matmul(x, w.in_norm, c.rms_norm_eps, w.wqkv);
    Tensor q = rope_qkv_to_cache(qkv, c.num_heads, c.num_kv_heads, c.head_dim, c.rope_theta, pos_offset,
                                 cache.k[layer], cache.v[layer], d_pos);
    Tensor ctx = attention(q, cache.k[layer], cache.v[layer], c.num_heads, c.num_kv_heads, c.head_dim, pos_offset,
                           d_pos);
    Tensor x1 = matmul_residual(ctx, w.wo, x);
    Tensor act = rmsnorm_gate_up(x1, w.post_norm, c.rms_norm_eps, w.w_gate, w.w_up);
    return matmul_residual(act, w.w_down, x1);
}

// Спільна реалізація кроку з KV-кешем для fp32- і fp16/bf16-ваг.
template <class LW>
inline Tensor forward_cached_impl(const ModelConfig& cfg, WeightRef embed, const std::vector<LW>& layers,
                                  const Tensor& final_norm, WeightRef lm_head, const std::vector<int>& ids,
                                  KVCache& cache, bool last_only) {
    if (ids.empty()) throw std::invalid_argument("Model::forward_cached: empty input");
    if (static_cast<long long>(cache.length) + static_cast<long long>(ids.size()) > cache.max_seq)
        throw std::invalid_argument("Model::forward_cached: KV cache is full (" + std::to_string(cache.length) +
                                    " + " + std::to_string(ids.size()) + " > " + std::to_string(cache.max_seq) + ")");
    if (static_cast<int>(cache.k.size()) != cfg.num_layers)
        throw std::invalid_argument("Model::forward_cached: cache does not match model");
    const int pos = cache.length;
    Tensor h = embedding(embed, ids);
    for (int i = 0; i < cfg.num_layers; ++i) h = decoder_layer_cached(h, layers[i], cfg, cache, i, pos);
    cache.length += static_cast<int>(ids.size());
    if (last_only) h = slice_rows(h, h.dim(0) - 1, 1);
    return rmsnorm_matmul(h, final_norm, cfg.rms_norm_eps, lm_head);  // фінальний norm + lm_head
}

// Один крок декодування (один токен) із токеном і позицією в ПАМ'ЯТІ ПРИСТРОЮ: жодних копіювань
// з хоста, синхронізацій чи розрахунків на хості, тому весь крок можна захопити в CUDA-граф.
// Повертає логіти [1 x vocab]. Довжину кешу (cache.length) викликач веде сам.
template <class LW>
inline Tensor forward_decode_dev(const ModelConfig& cfg, WeightRef embed, const std::vector<LW>& layers,
                                 const Tensor& final_norm, WeightRef lm_head, const int* d_token,
                                 const int* d_pos, KVCache& cache) {
    if (static_cast<int>(cache.k.size()) != cfg.num_layers)
        throw std::invalid_argument("forward_decode_dev: cache does not match model");
    Tensor h = embedding_dev(embed, d_token, 1);
    for (int i = 0; i < cfg.num_layers; ++i) h = decoder_layer_cached(h, layers[i], cfg, cache, i, 0, d_pos);
    return rmsnorm_matmul(h, final_norm, cfg.rms_norm_eps, lm_head);
}

// Повна модель: embedding -> шари декодера -> фінальний RMSNorm -> lm_head.
// Ваги лежать у GPU-пам'яті; завантаження читає model/weights.bin. Формат ваг обирається при
// завантаженні: fp32 (еталон) або fp16/bf16 (вдвічі менше пам'яті й трафіку на токен).
struct Model {
    ModelConfig cfg;
    DType dtype = DType::F32;
    Tensor final_norm;  // [hidden], завжди fp32

    // dtype == F32:
    Tensor embed;       // [vocab x hidden]
    Tensor lm_head;     // [hidden x vocab] (вже транспонована; при tie = embed^T)
    std::vector<LayerWeights> layers;

    // dtype == F16 або BF16:
    LowpMatrix embed_lp, lm_head_lp;
    std::vector<LayerWeightsLP> layers_lp;
    LowpStats lowp_stats;  // втрати точності при конвертації (по всіх матрицях)

    // dir містить config.txt, weights.bin і weights.manifest.txt (tools/export_reference.py).
    static Model load(const std::string& dir, DType dt = DType::F32) {
        Model m;
        m.dtype = dt;
        m.cfg = ModelConfig::load(dir + "/config.txt");
        TensorFile f(dir + "/weights");
        m.final_norm = f.to_gpu("model.norm.weight");
        if (dt == DType::F32) {
            m.embed = f.to_gpu("model.embed_tokens.weight");
            if (m.cfg.tie_word_embeddings)
                m.lm_head = transpose(m.embed);  // спільні ваги: lm_head = embed^T
            else
                m.lm_head = transpose(f.to_gpu("lm_head.weight"));
            m.layers.reserve(m.cfg.num_layers);
            for (int i = 0; i < m.cfg.num_layers; ++i) m.layers.push_back(load_layer(f, i));
        } else {
            {   // fp32-копія embedding існує лише тут, щоб зробити з неї embedding і lm_head
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

    // Скільки байтів займають ваги великих матриць (без норм).
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

    // ids -> логіти [T x vocab] еталонним шляхом без кешу. Лише для fp32-ваг.
    // Кидає std::invalid_argument для порожнього входу чи поганих id.
    Tensor forward(const std::vector<int>& ids) const {
        if (dtype != DType::F32)
            throw std::invalid_argument("Model::forward: the reference path without cache requires f32 weights");
        if (ids.empty()) throw std::invalid_argument("Model::forward: empty input");
        Tensor h = embedding(embed, ids);
        for (const LayerWeights& w : layers) h = decoder_layer(h, w, cfg);
        return matmul(rmsnorm(h, final_norm, cfg.rms_norm_eps), lm_head);
    }

    // Крок з KV-кешем: ids - НОВІ токени (весь промпт для prefill або один токен для декодування).
    // Позиції беруться з cache.length, який після кроку збільшується на ids.size().
    // last_only: рахувати логіти лише для останнього токена (для генерації достатньо; економить
    // матмул [T x hidden] x [hidden x vocab]). Повертає [T x vocab] або [1 x vocab].
    Tensor forward_cached(const std::vector<int>& ids, KVCache& cache, bool last_only = false) const {
        if (dtype == DType::F32)
            return forward_cached_impl(cfg, embed, layers, final_norm, lm_head, ids, cache, last_only);
        return forward_cached_impl(cfg, embed_lp, layers_lp, final_norm, lm_head_lp, ids, cache, last_only);
    }

    // Один токен з d_token і позицією d_pos у пам'яті пристрою (див. forward_decode_dev).
    Tensor forward_decode_dev(const int* d_token, const int* d_pos, KVCache& cache) const {
        if (dtype == DType::F32)
            return ::forward_decode_dev(cfg, embed, layers, final_norm, lm_head, d_token, d_pos, cache);
        return ::forward_decode_dev(cfg, embed_lp, layers_lp, final_norm, lm_head_lp, d_token, d_pos, cache);
    }
};

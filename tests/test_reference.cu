// Звірка з PyTorch: беремо активації, які записав tools/export_reference.py.
// Кожен тест подає ядру ЕТАЛОННИЙ вхід (а не вихід попереднього нашого ядра),
// щоб помилка не накопичувалась і показувала, яке саме ядро розійшлося.
// Якщо model/ чи reference/ немає, тести пропускаються (див. README).
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

// nullptr, якщо еталонних файлів немає.
Ref* load_ref() {
    static Ref* r = nullptr;
    static bool tried = false;
    if (!tried) {
        tried = true;
        try {
            r = new Ref();
        } catch (const std::exception& e) {
            std::printf("       (пропущено: %s; запустіть tools/export_reference.py)\n", e.what());
        }
    }
    return r;
}

void check_close(const char* what, const std::vector<float>& got, const std::vector<float>& want,
                 double tol) {
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
    check_close("l0.input_norm", rmsnorm(x, w, r->eps).to_host(), r->ref.floats("l0.input_norm"),
                1e-5);
}

// Проєкції Q/K/V шару 0: matmul з вагами. У PyTorch вага [out x in], y = x * W^T,
// тому ваги транспонуємо ядром transpose.
namespace {
Tensor transposed(const Tensor& w) { return transpose(w); }
}  // namespace

TEST(ref_layer0_rope_matches_pytorch) {
    Ref* r = load_ref();
    if (!r) return;
    const float theta = 100000.f;
    check_close("l0.q_rope", rope(r->ref.to_gpu("l0.q_proj"), 9, 64, theta).to_host(),
                r->ref.floats("l0.q_rope"), 1e-5);
    check_close("l0.k_rope", rope(r->ref.to_gpu("l0.k_proj"), 3, 64, theta).to_host(),
                r->ref.floats("l0.k_rope"), 1e-5);
}

// Контекст attention (перед o_proj) з еталонних q_rope, k_rope, v_proj.
TEST(ref_layer0_attention_matches_pytorch) {
    Ref* r = load_ref();
    if (!r) return;
    Tensor out = attention(r->ref.to_gpu("l0.q_rope"), r->ref.to_gpu("l0.k_rope"),
                           r->ref.to_gpu("l0.v_proj"), 9, 3, 64);
    check_close("l0.attn_ctx", out.to_host(), r->ref.floats("l0.attn_ctx"), 1e-5);
}

// Вихідна проєкція: ctx * Wo^T = o_proj (і це ж вихід усього self_attn).
TEST(ref_layer0_o_proj_matches_pytorch) {
    Ref* r = load_ref();
    if (!r) return;
    Tensor wt = transposed(r->weights.to_gpu("model.layers.0.self_attn.o_proj.weight"));
    std::vector<float> got = matmul(r->ref.to_gpu("l0.attn_ctx"), wt).to_host();
    check_close("l0.o_proj", got, r->ref.floats("l0.o_proj"), 1e-4);
    check_close("l0.attn_out", got, r->ref.floats("l0.attn_out"), 1e-4);
}

// Відносна похибка: max|diff| / max(1, max|ref|). Активації в глибоких шарах бувають великими.
static double rel_diff(const std::vector<float>& got, const std::vector<float>& want) {
    double scale = 1.0;
    for (float v : want) scale = std::fmax(scale, std::fabs(static_cast<double>(v)));
    return max_abs_diff(got, want) / scale;
}

// Residual після attention і друга нормалізація.
TEST(ref_layer0_post_attention_norm_matches_pytorch) {
    Ref* r = load_ref();
    if (!r) return;
    Tensor x1 = add(r->ref.to_gpu("embed_out"), r->ref.to_gpu("l0.attn_out"));
    Tensor w = r->weights.to_gpu("model.layers.0.post_attention_layernorm.weight");
    check_close("l0.post_norm", rmsnorm(x1, w, r->eps).to_host(), r->ref.floats("l0.post_norm"), 1e-5);
}

// MLP шару 0 з еталонного входу.
TEST(ref_layer0_mlp_matches_pytorch) {
    Ref* r = load_ref();
    if (!r) return;
    LayerWeights w = load_layer(r->weights, 0);
    check_close("l0.mlp_out", mlp_forward(r->ref.to_gpu("l0.post_norm"), w).to_host(),
                r->ref.floats("l0.mlp_out"), 1e-4);
}

// Весь шар 0 з еталонного embedding: вихід = hidden_states.1 = l0.out.
TEST(ref_layer0_output_matches_pytorch) {
    Ref* r = load_ref();
    if (!r) return;
    LayerWeights w = load_layer(r->weights, 0);
    std::vector<float> got = decoder_layer(r->ref.to_gpu("embed_out"), w, r->cfg).to_host();
    check_close("l0.out", got, r->ref.floats("l0.out"), 1e-4);
    check_close("hidden_states.1", got, r->ref.floats("hidden_states.1"), 1e-4);
}

// Усі 30 шарів послідовно: кожен шар отримує ВИХІД нашого попереднього шару, тож тест
// показує, де саме накопичена похибка починає рости. Останній hidden_state (30) у
// transformers - після фінального RMSNorm.
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
            os << "шар " << i << " (" << name << "): відносна похибка " << d << " > 1e-3";
            throw TestFailure(os.str());
        }
    }
    std::printf("       найбільша відносна похибка по 30 шарах: %.2e\n", worst);
}

// ---- повна модель ----------------------------------------------------------

namespace {
// Модель завантажується один раз на всі тести (читання 650 МБ з диска).
const Model& shared_model() {
    static Model* m = new Model(Model::load(std::string(MINIINFER_ROOT) + "/model"));
    return *m;
}

// Індекси k найбільших значень рядка, від найбільшого.
std::vector<int> top_k(const float* row, int n, int k) {
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return row[a] > row[b]; });
    idx.resize(k);
    return idx;
}
}  // namespace

// Логіти всіх позицій збігаються з PyTorch, і топ-токени однакові.
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
    std::printf("       відносна похибка логітів: %.2e\n", d);
    if (!(d <= 1e-3)) {
        std::ostringstream os;
        os << "логіти: відносна похибка " << d << " > 1e-3";
        throw TestFailure(os.str());
    }
    for (int t = 0; t < T; ++t) {
        std::vector<int> a = top_k(&got[static_cast<size_t>(t) * V], V, 5);
        std::vector<int> b = top_k(&want[static_cast<size_t>(t) * V], V, 5);
        if (a != b) {
            std::ostringstream os;
            os << "позиція " << t << ": топ-5 токенів відрізняється (наш top1=" << a[0]
               << ", PyTorch top1=" << b[0] << ")";
            throw TestFailure(os.str());
        }
    }
    // Те, що модель "сказала б" далі: ' the' (id 260) для "The capital of France is".
    const int next = top_k(&got[static_cast<size_t>(T - 1) * V], V, 1)[0];
    std::printf("       наступний токен: %d\n", next);
    CHECK(next == 260);
}

// Причинність усієї моделі: логіти префікса не залежать від токенів, що йдуть після нього.
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

// KV-кеш на повній моделі: prefill 3 токенів + решта по одному дають ті самі логіти,
// що й повний прохід без кешу.
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
    std::printf("       відносна різниця кеш/без кешу: %.2e\n", d);
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
    CHECK(cache.length == 3);                        // збій не псує стан
    m.forward_cached({4}, cache);                    // рівно до краю
    CHECK_THROWS(m.forward_cached({5}, cache));
    cache.reset();
    m.forward_cached({1, 2, 3, 4}, cache);           // після reset знову доступно
}

// ---- генерація -------------------------------------------------------------

namespace {
GenerateResult greedy(const Model& m, const std::vector<int>& prompt, int n) {
    GenerateOptions o;
    o.max_new_tokens = n;  // temperature = 0 -> greedy
    return generate(m, prompt, o);
}
}  // namespace

// Головна перевірка етапу 4: greedy-генерація збігається з model.generate() у PyTorch
// токен у токен, на двох промптах (5 і 15 токенів).
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
            os << key << ": розбіжність на новому токені " << i << " (наш "
               << (i < res.tokens.size() ? res.tokens[i] : -1) << ", PyTorch "
               << (i < want.size() ? want[i] : -1) << ")";
            throw TestFailure(os.str());
        }
        std::printf("       %s: %zu токенів збігаються; prefill %.1f мс, декодування %.1f мс/токен\n", key,
                    want.size(), res.prefill_ms, res.decode_ms / std::max<size_t>(1, want.size() - 1));
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
    o.stop_ids = {want[4]};  // зупинка на п'ятому новому токені
    std::vector<int> streamed;
    GenerateResult res = generate(m, prompt, o, [&](int t) { streamed.push_back(t); });
    CHECK(res.hit_stop);
    CHECK(res.tokens == std::vector<int>(want.begin(), want.begin() + 4));  // стоп-токен не додається
    CHECK(streamed == res.tokens);
}

TEST(generate_edge_cases) {
    Ref* r = load_ref();
    if (!r) return;
    const Model& m = shared_model();
    CHECK(greedy(m, {504}, 0).tokens.empty());           // 0 нових токенів
    CHECK(greedy(m, {504}, 1).tokens.size() == 1);       // рівно один
    CHECK_THROWS(greedy(m, {}, 5));                      // порожній промпт
    CHECK_THROWS(greedy(m, {504}, -1));
    CHECK_THROWS(greedy(m, std::vector<int>(10, 504), kMaxAttentionKeys));  // не вміщається
    CHECK_THROWS(greedy(m, {m.cfg.vocab_size}, 3));      // id поза словником
}

// Семплінг із зерном відтворюється, і top_k=1 дає те саме, що greedy.
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

// ---- модель з вагами fp16 / bf16 -------------------------------------------------

namespace {
const Model& lowp_model(DType dt) {
    static Model* models[2] = {nullptr, nullptr};
    Model*& m = models[dt == DType::F16 ? 0 : 1];
    if (!m) m = new Model(Model::load(std::string(MINIINFER_ROOT) + "/model", dt));
    return *m;
}
}  // namespace

// Ваги вдвічі менші, а чекпойнт у bf16, тож bf16 не втрачає нічого, а fp16 - майже нічого.
TEST(ref_lowp_conversion_stats_and_memory) {
    Ref* r = load_ref();
    if (!r) return;
    const Model& f32 = shared_model();
    for (DType dt : {DType::F16, DType::BF16}) {
        const Model& m = lowp_model(dt);
        CHECK(m.weight_bytes() * 2 == f32.weight_bytes());  // рівно вдвічі менше
        const LowpStats& s = m.lowp_stats;
        std::printf("       %s: %zu елементів, переповнень %zu, занулено %zu, макс. відносна похибка %.2e\n",
                    dtype_name(dt), s.elements, s.overflow, s.flushed, s.max_rel_error);
        CHECK(s.overflow == 0);
        if (dt == DType::BF16) CHECK(s.max_rel_error == 0.0 && s.flushed == 0);  // чекпойнт уже bf16
        else CHECK(s.flushed < 1000);  // fp16: кілька надмалих ваг стають нулями
    }
}

// Логіти всіх позицій проти PyTorch (fp32 еталон): розбіжність лише від порядку сумування,
// бо самі ваги (bf16) відтворені точно.
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
        std::printf("       %s: відносна похибка логітів проти PyTorch: %.2e\n", dtype_name(dt), d);
        CHECK(d <= 1e-3);
        for (int t = 0; t < T; ++t)
            CHECK(top_k(&got[static_cast<size_t>(t) * V], V, 5) == top_k(&want[static_cast<size_t>(t) * V], V, 5));
    }
}

// Головна перевірка: greedy-генерація з 16-бітними вагами збігається з PyTorch токен у токен.
TEST(ref_lowp_greedy_generation_matches_pytorch) {
    Ref* r = load_ref();
    if (!r) return;
    TensorFile gen(std::string(MINIINFER_ROOT) + "/reference/generation");
    for (DType dt : {DType::F16, DType::BF16})
        for (const char* key : {"gen1", "gen2"}) {
            std::vector<int> prompt = gen.ints(std::string(key) + ".prompt"), want = gen.ints(std::string(key) + ".new");
            GenerateResult res = greedy(lowp_model(dt), prompt, static_cast<int>(want.size()));
            if (res.tokens != want) {
                size_t i = 0;
                while (i < res.tokens.size() && i < want.size() && res.tokens[i] == want[i]) ++i;
                std::ostringstream os;
                os << dtype_name(dt) << " " << key << ": розбіжність на новому токені " << i;
                throw TestFailure(os.str());
            }
            std::printf("       %s %s: %zu токенів збігаються, декодування %.2f мс/токен\n", dtype_name(dt), key,
                        want.size(), res.decode_ms / std::max<size_t>(1, want.size() - 1));
        }
}

TEST(ref_lowp_model_rejects_uncached_reference_path) {
    Ref* r = load_ref();
    if (!r) return;
    CHECK_THROWS(lowp_model(DType::BF16).forward({1, 2, 3}));  // еталонний шлях без кешу лише для fp32
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
    struct P { const char* w; const char* out; };
    const P projs[] = {{"model.layers.0.self_attn.q_proj.weight", "l0.q_proj"},
                       {"model.layers.0.self_attn.k_proj.weight", "l0.k_proj"},
                       {"model.layers.0.self_attn.v_proj.weight", "l0.v_proj"}};
    for (const P& p : projs) {
        Tensor wt = transposed(r->weights.to_gpu(p.w));
        check_close(p.out, matmul(x, wt).to_host(), r->ref.floats(p.out), 1e-4);
    }
}

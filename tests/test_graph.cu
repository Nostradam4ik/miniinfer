// Тести CUDA-графа кроку декодування на справжній моделі. Еталон - звичайний шлях
// (forward_cached) і PyTorch. Без model/ і reference/ тести пропускаються (див. README).
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
    if (!ok) std::printf("       (пропущено: немає model/ або reference/; запустіть tools/export_reference.py)\n");
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

// Токени для кроків декодування: детерміновані, у межах словника.
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

// Основна перевірка: логіти кожного кроку через граф збігаються зі звичайним кешованим шляхом, а
// кеші (K, V) після кроків ідентичні побітово.
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
        std::printf("       %s: %zu кроків, найбільша відносна різниця логітів граф/без графа: %.2e\n", dtype_name(dt),
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

// Головна перевірка: greedy-генерація через граф збігається з PyTorch токен у токен.
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
                os << dtype_name(dt) << " " << key << ": розбіжність на новому токені " << i;
                throw TestFailure(os.str());
            }
            std::printf("       %s %s: %zu токенів збігаються з PyTorch; захоплення графа %.1f мс, декодування %.2f мс/токен\n",
                        dtype_name(dt), key, want.size(), r.graph_capture_ms,
                        r.decode_ms / std::max<size_t>(1, want.size() - 1));
        }
}

// Арена справді ізолює буфери графа: після захоплення інші тензори тієї ж ширини живуть своїм життям,
// а граф їх не псує і сам не псується ними.
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

    // Тензори тих самих розмірів, що й проміжні в графі, із відомим вмістом. Якби граф розділяв з ними
    // пам'ять, то перезаписав би їх, а вони - його буфери.
    std::vector<Tensor> bystanders;
    const std::vector<std::vector<int>> shapes = {{1, H}, {1, qkv}, {1, I}, {1, V}, {1, m.cfg.num_heads * m.cfg.head_dim}};
    for (size_t i = 0; i < shapes.size(); ++i) {
        size_t n = 1;
        for (int d : shapes[i]) n *= d;
        bystanders.push_back(Tensor::from_host(std::vector<float>(n, 1000.f + static_cast<float>(i)), shapes[i]));
    }

    std::vector<float> got;
    for (int tok : toks) {
        std::vector<float> want = m.forward_cached({tok}, normal, true).to_host();  // звичайний шлях теж вживає пул
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
    CHECK(cache.length == 0);  // помилка не змінює стан
    for (int i = 0; i < 4; ++i) g.step(i + 1, out);
    CHECK(cache.length == 4);
    CHECK_THROWS(g.step(1, out));  // кеш повний
    CHECK(cache.length == 4);
    cache.reset();                  // після reset граф працює знову з нульової позиції
    g.step(1, out);
    CHECK(cache.length == 1);
}

// Той самий граф відтворює ту саму послідовність після reset кешу: стан не "залипає" між запусками.
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

// Граф, захоплений із нульової довжини кешу (без prefill), працює й узгоджується зі звичайним шляхом.
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

// Два графа на різних кешах (і навіть різних моделях) не заважають один одному.
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

// Знищення графа повертає всі ресурси: арена звільняється, пул не "тече".
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

// Арена замала: захоплення завершується зрозумілою помилкою, а потік і алокатор лишаються робочими.
TEST(graph_capture_failure_leaves_everything_usable) {
    if (!files_exist()) return;
    const Model& m = model_for(DType::BF16);
    KVCache cache(m.cfg, 16);
    bool threw = false;
    try {
        DecodeGraph g(m, cache, 4096);  // замало навіть для логітів
    } catch (const std::exception& e) {
        threw = std::string(e.what()).find("arena") != std::string::npos || std::string(e.what()).find("Arena") != std::string::npos;
    }
    CHECK(threw);
    CHECK(!GpuAllocator::instance().arena_active());
    CHECK(exec_stream() == nullptr);
    // і нормальний шлях та новий граф працюють після невдалої спроби
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
    o.max_new_tokens = 1;  // один токен: граф не потрібен
    CHECK(generate(m, {504}, o).tokens.size() == 1);
    o.max_new_tokens = 0;
    CHECK(generate(m, {504}, o).tokens.empty());
    o.max_new_tokens = 2;
    GenerateResult r = generate(m, {504, 3575}, o);
    CHECK(r.tokens.size() == 2 && r.graph_capture_ms > 0);
}

// Семплінг із зерном через граф дає ті самі токени, що й без графа (однакові логіти = однаковий вибір).
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

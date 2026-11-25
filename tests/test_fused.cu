// Тести злитих операцій. Еталон - прості CPU-обчислення в double, а не інші GPU-ядра:
// так злиття перевіряється проти незалежного обчислення.
#include <cmath>
#include <cstring>
#include <random>
#include "lowp.cuh"
#include "ops.cuh"
#include "ops_cpu.h"
#include "test_framework.h"

namespace {
std::vector<float> rnd(size_t n, unsigned seed, float scale = 1.f) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<float> v(n);
    for (float& x : v) x = dist(gen);
    return v;
}

// Брудний блок у пулі такого розміру: зіпсує результат, якщо ядро не запише всі елементи.
void poison_pool(std::vector<int> shape) {
    size_t n = 1;
    for (int d : shape) n *= d;
    Tensor dirty = Tensor::from_host(std::vector<float>(n, std::nanf("")), shape);
}

std::string describe(const char* what, int M, int K, int N, double d) {
    std::ostringstream os;
    os << what << " M=" << M << " K=" << K << " N=" << N << " max_diff=" << d;
    return os.str();
}

struct Shape { int M, K, N; };
// M = 1..4 -> злите ядро; M = 5, 9 -> резерв (кілька операцій). Обидва мають давати той самий результат.
const Shape kShapes[] = {{1, 1, 1},     {1, 7, 5},     {1, 16, 32},   {1, 33, 33},   {1, 576, 192}, {1, 576, 576},
                         {1, 1536, 576}, {1, 576, 1536}, {2, 100, 65}, {4, 64, 64},  {5, 576, 192}, {9, 33, 70}};
}  // namespace

TEST(fused_rmsnorm_matmul_matches_cpu) {
    for (const Shape& s : kShapes) {
        std::vector<float> x = rnd(static_cast<size_t>(s.M) * s.K, 1 + s.K, 3.f), nw = rnd(s.K, 2 + s.N);
        std::vector<float> w = rnd(static_cast<size_t>(s.K) * s.N, 3 + s.M);
        std::vector<float> ref = cpu_matmul(cpu_rmsnorm(x, nw, s.M, s.K, 1e-5f), w, s.M, s.N, s.K);
        poison_pool({s.M, s.N});
        Tensor out = rmsnorm_matmul(Tensor::from_host(x, {s.M, s.K}), Tensor::from_host(nw, {s.K}), 1e-5f,
                                    Tensor::from_host(w, {s.K, s.N}));
        CHECK(out.dim(0) == s.M && out.dim(1) == s.N);
        double d = max_abs_diff(out.to_host(), ref);
        if (d > 2e-3) throw TestFailure(describe("rmsnorm_matmul", s.M, s.K, s.N, d));
    }
}

TEST(fused_matmul_residual_matches_cpu) {
    for (const Shape& s : kShapes) {
        std::vector<float> x = rnd(static_cast<size_t>(s.M) * s.K, 4 + s.K), w = rnd(static_cast<size_t>(s.K) * s.N, 5 + s.N);
        std::vector<float> res = rnd(static_cast<size_t>(s.M) * s.N, 6 + s.M, 5.f);
        std::vector<float> ref = cpu_matmul(x, w, s.M, s.N, s.K);
        for (size_t i = 0; i < ref.size(); ++i) ref[i] += res[i];
        poison_pool({s.M, s.N});
        Tensor out = matmul_residual(Tensor::from_host(x, {s.M, s.K}), Tensor::from_host(w, {s.K, s.N}),
                                     Tensor::from_host(res, {s.M, s.N}));
        double d = max_abs_diff(out.to_host(), ref);
        if (d > 2e-3) throw TestFailure(describe("matmul_residual", s.M, s.K, s.N, d));
    }
}

TEST(fused_rmsnorm_gate_up_matches_cpu) {
    for (const Shape& s : kShapes) {
        std::vector<float> x = rnd(static_cast<size_t>(s.M) * s.K, 7 + s.K, 3.f), nw = rnd(s.K, 8 + s.N);
        std::vector<float> wg = rnd(static_cast<size_t>(s.K) * s.N, 9 + s.M), wu = rnd(static_cast<size_t>(s.K) * s.N, 10 + s.M);
        std::vector<float> h = cpu_rmsnorm(x, nw, s.M, s.K, 1e-5f);
        std::vector<float> g = cpu_matmul(h, wg, s.M, s.N, s.K), u = cpu_matmul(h, wu, s.M, s.N, s.K), ref(g.size());
        for (size_t i = 0; i < g.size(); ++i) ref[i] = static_cast<float>(g[i] / (1.0 + std::exp(-static_cast<double>(g[i]))) * u[i]);
        poison_pool({s.M, s.N});
        Tensor out = rmsnorm_gate_up(Tensor::from_host(x, {s.M, s.K}), Tensor::from_host(nw, {s.K}), 1e-5f,
                                     Tensor::from_host(wg, {s.K, s.N}), Tensor::from_host(wu, {s.K, s.N}));
        double d = max_abs_diff(out.to_host(), ref);
        if (d > 2e-3) throw TestFailure(describe("rmsnorm_gate_up", s.M, s.K, s.N, d));
    }
}

// Злите ядро і резерв дають практично однаковий результат для одного й того ж рядка.
TEST(fused_path_and_fallback_agree_on_same_row) {
    const int K = 576, N = 300;
    std::vector<float> row = rnd(K, 1, 2.f), nw = rnd(K, 2), w = rnd(static_cast<size_t>(K) * N, 3);
    Tensor W = Tensor::from_host(w, {K, N}), NW = Tensor::from_host(nw, {K});
    std::vector<float> one = rmsnorm_matmul(Tensor::from_host(row, {1, K}), NW, 1e-5f, W).to_host();  // злите
    std::vector<float> many;
    for (int i = 0; i < kGemvMaxRows + 1; ++i) many.insert(many.end(), row.begin(), row.end());
    std::vector<float> big = rmsnorm_matmul(Tensor::from_host(many, {kGemvMaxRows + 1, K}), NW, 1e-5f, W).to_host();  // резерв
    std::vector<float> first(big.begin(), big.begin() + N);
    CHECK(max_abs_diff(one, first) < 1e-3);
}

TEST(fused_ops_validate_shapes) {
    Tensor x({2, 4}), w({4, 3}), nw({4}), res({2, 3});
    CHECK_THROWS(rmsnorm_matmul(x, Tensor({5}), 1e-5f, w));       // norm_w не того розміру
    CHECK_THROWS(rmsnorm_matmul(x, nw, 1e-5f, Tensor({5, 3})));   // K не збігається
    CHECK_THROWS(matmul_residual(x, w, Tensor({2, 4})));          // residual не того розміру
    CHECK_THROWS(matmul_residual(x, Tensor({5, 3}), res));
    CHECK_THROWS(rmsnorm_gate_up(x, nw, 1e-5f, w, Tensor({4, 2})));  // Wg і Wu різні
    CHECK_THROWS(rmsnorm_gate_up(x, Tensor({3}), 1e-5f, w, w));
}

// ---- rope_qkv_to_cache -------------------------------------------------------

namespace {
// Колонки [from, to) матриці [rows x width] (host).
std::vector<float> cols(const std::vector<float>& m, int rows, int width, int from, int to) {
    std::vector<float> r;
    for (int i = 0; i < rows; ++i) r.insert(r.end(), m.begin() + i * width + from, m.begin() + i * width + to);
    return r;
}
}  // namespace

TEST(rope_qkv_to_cache_matches_separate_ops) {
    struct S { int T, H, KV, hd, pos, max_seq; float theta; };
    const S shapes[] = {{1, 1, 1, 2, 0, 4, 10000.f},  {1, 9, 3, 64, 17, 40, 100000.f}, {5, 9, 3, 64, 0, 8, 100000.f},
                        {3, 4, 2, 8, 5, 10, 10000.f}, {7, 2, 1, 16, 20, 30, 500.f}};
    for (const S& s : shapes) {
        const int width = (s.H + 2 * s.KV) * s.hd, kv_dim = s.KV * s.hd;
        std::vector<float> qkv = rnd(static_cast<size_t>(s.T) * width, 11 + s.T);
        const std::vector<float> q = cols(qkv, s.T, width, 0, s.H * s.hd);
        const std::vector<float> k = cols(qkv, s.T, width, s.H * s.hd, (s.H + s.KV) * s.hd);
        const std::vector<float> v = cols(qkv, s.T, width, (s.H + s.KV) * s.hd, width);

        // кеш, повний NaN: перевіримо, що записано рівно рядки pos..pos+T-1 і нічого більше
        Tensor ck = Tensor::from_host(std::vector<float>(static_cast<size_t>(s.max_seq) * kv_dim, std::nanf("")), {s.max_seq, kv_dim});
        Tensor cv = Tensor::from_host(std::vector<float>(static_cast<size_t>(s.max_seq) * kv_dim, std::nanf("")), {s.max_seq, kv_dim});
        poison_pool({s.T, s.H * s.hd});
        Tensor q_out = rope_qkv_to_cache(Tensor::from_host(qkv, {s.T, width}), s.H, s.KV, s.hd, s.theta, s.pos, ck, cv);

        CHECK(q_out.dim(0) == s.T && q_out.dim(1) == s.H * s.hd);
        double dq = max_abs_diff(q_out.to_host(), cpu_rope(q, s.T, s.H, s.hd, s.theta, s.pos));
        if (dq > 2e-4) throw TestFailure("Q: max_diff=" + std::to_string(dq));

        std::vector<float> hk = ck.to_host(), hv = cv.to_host();
        const std::vector<float> want_k = cpu_rope(k, s.T, s.KV, s.hd, s.theta, s.pos);
        for (int r = 0; r < s.max_seq; ++r)
            for (int d = 0; d < kv_dim; ++d) {
                const float gk = hk[static_cast<size_t>(r) * kv_dim + d], gv = hv[static_cast<size_t>(r) * kv_dim + d];
                if (r >= s.pos && r < s.pos + s.T) {
                    const int t = r - s.pos;
                    if (std::fabs(gk - want_k[static_cast<size_t>(t) * kv_dim + d]) > 2e-4)
                        throw TestFailure("K cache row " + std::to_string(r) + " differs");
                    if (gv != v[static_cast<size_t>(t) * kv_dim + d])
                        throw TestFailure("V cache row " + std::to_string(r) + " differs");
                } else if (!std::isnan(gk) || !std::isnan(gv)) {
                    throw TestFailure("запис за межами рядків pos..pos+T-1: рядок " + std::to_string(r));
                }
            }
    }
}

TEST(rope_qkv_to_cache_validates_args) {
    const int H = 2, KV = 1, hd = 4, width = (H + 2 * KV) * hd;
    Tensor qkv({3, width}), ck({8, KV * hd}), cv({8, KV * hd});
    CHECK_THROWS(rope_qkv_to_cache(Tensor({3, width + 1}), H, KV, hd, 1e4f, 0, ck, cv));  // не та ширина
    CHECK_THROWS(rope_qkv_to_cache(qkv, H, KV, 3, 1e4f, 0, ck, cv));                      // непарний head_dim
    CHECK_THROWS(rope_qkv_to_cache(qkv, H, KV, hd, 1e4f, -1, ck, cv));                    // від'ємна позиція
    CHECK_THROWS(rope_qkv_to_cache(qkv, H, KV, hd, 1e4f, 6, ck, cv));                     // 6 + 3 > 8
    Tensor small_cv({8, KV * hd + 1});
    CHECK_THROWS(rope_qkv_to_cache(qkv, H, KV, hd, 1e4f, 0, ck, small_cv));
    rope_qkv_to_cache(qkv, H, KV, hd, 1e4f, 5, ck, cv);  // рівно до краю: дозволено
}

// ---- concat_cols -------------------------------------------------------------

TEST(concat_cols_joins_columns) {
    Tensor a = Tensor::from_host({1, 2, 3, 4}, {2, 2}), b = Tensor::from_host({5, 6}, {2, 1}),
           c = Tensor::from_host({7, 8, 9, 10, 11, 12}, {2, 3});
    Tensor out = concat_cols({&a, &b, &c});
    CHECK(out.dim(0) == 2 && out.dim(1) == 6);
    const float want[] = {1, 2, 5, 7, 8, 9, 3, 4, 6, 10, 11, 12};
    std::vector<float> h = out.to_host();
    for (int i = 0; i < 12; ++i) CHECK(h[i] == want[i]);
}

TEST(concat_cols_validates) {
    Tensor a({2, 2}), b({3, 2});
    CHECK_THROWS(concat_cols({&a, &b}));  // різна кількість рядків
    CHECK_THROWS(concat_cols({}));
    Tensor c({2, 2, 2});
    CHECK_THROWS(concat_cols({&c}));
}

// ---- варіанти з позицією / id у пам'яті пристрою (для CUDA-графа) -------------------------

namespace {
// Одне int на пристрої.
struct DevInts {
    int* p = nullptr;
    explicit DevInts(const std::vector<int>& v) {
        CUDA_CHECK(cudaMalloc(&p, v.size() * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
    ~DevInts() { cudaFree(p); }
    DevInts(const DevInts&) = delete;
    DevInts& operator=(const DevInts&) = delete;
};
}  // namespace

TEST(attention_with_device_position_is_bitwise_equal_to_host_position) {
    struct S { int T, S, H, KV, hd, pos; };
    const S shapes[] = {{1, 10, 9, 3, 64, 6}, {1, 64, 4, 2, 8, 63}, {1, 7, 2, 1, 16, 0}, {3, 20, 6, 2, 16, 12}};
    for (const S& s : shapes) {
        Tensor q = Tensor::from_host(rnd(static_cast<size_t>(s.T) * s.H * s.hd, 1, 2.f), {s.T, s.H * s.hd});
        Tensor k = Tensor::from_host(rnd(static_cast<size_t>(s.S) * s.KV * s.hd, 2, 2.f), {s.S, s.KV * s.hd});
        Tensor v = Tensor::from_host(rnd(static_cast<size_t>(s.S) * s.KV * s.hd, 3), {s.S, s.KV * s.hd});
        DevInts pos({s.pos});
        std::vector<float> host = attention(q, k, v, s.H, s.KV, s.hd, s.pos).to_host();
        std::vector<float> dev = attention(q, k, v, s.H, s.KV, s.hd, 0 /*ігнорується*/, pos.p).to_host();
        CHECK(host == dev);
    }
}

// Позиція з пристрою поза межами кешу: NaN у виході замість читання/запису за межами shared memory.
TEST(attention_with_out_of_range_device_position_returns_nan) {
    const int S = 10, H = 2, KV = 1, hd = 8;
    Tensor q = Tensor::from_host(rnd(H * hd, 1), {1, H * hd});
    Tensor k = Tensor::from_host(rnd(S * KV * hd, 2), {S, KV * hd}), v = Tensor::from_host(rnd(S * KV * hd, 3), {S, KV * hd});
    for (int bad : {S, S + 5, -1, -100}) {
        DevInts pos({bad});
        for (float x : attention(q, k, v, H, KV, hd, 0, pos.p).to_host()) CHECK(std::isnan(x));
    }
    DevInts ok({S - 1});  // остання допустима позиція
    for (float x : attention(q, k, v, H, KV, hd, 0, ok.p).to_host()) CHECK(std::isfinite(x));
}

TEST(rope_qkv_with_device_position_is_bitwise_equal_to_host_position) {
    struct S { int T, H, KV, hd, pos, max_seq; };
    const S shapes[] = {{1, 9, 3, 64, 17, 40}, {1, 4, 2, 8, 0, 4}, {3, 2, 1, 16, 5, 10}};
    for (const S& s : shapes) {
        const int width = (s.H + 2 * s.KV) * s.hd, kv_dim = s.KV * s.hd;
        std::vector<float> qkv = rnd(static_cast<size_t>(s.T) * width, 7);
        auto nan_cache = [&] {
            return Tensor::from_host(std::vector<float>(static_cast<size_t>(s.max_seq) * kv_dim, std::nanf("")), {s.max_seq, kv_dim});
        };
        Tensor ck1 = nan_cache(), cv1 = nan_cache(), ck2 = nan_cache(), cv2 = nan_cache();
        DevInts pos({s.pos});
        std::vector<float> q1 = rope_qkv_to_cache(Tensor::from_host(qkv, {s.T, width}), s.H, s.KV, s.hd, 1e5f, s.pos, ck1, cv1).to_host();
        std::vector<float> q2 = rope_qkv_to_cache(Tensor::from_host(qkv, {s.T, width}), s.H, s.KV, s.hd, 1e5f, 0, ck2, cv2, pos.p).to_host();
        CHECK(q1 == q2);
        // побітове порівняння кешів (NaN != NaN, тож порівнюємо як байти)
        std::vector<float> a = ck1.to_host(), b = ck2.to_host(), c = cv1.to_host(), d = cv2.to_host();
        CHECK(std::memcmp(a.data(), b.data(), a.size() * 4) == 0);
        CHECK(std::memcmp(c.data(), d.data(), c.size() * 4) == 0);
    }
}

// Рядок кешу поза межами не пишеться (ядро його пропускає), решта кешу не чіпається.
TEST(rope_qkv_with_out_of_range_device_position_leaves_cache_untouched) {
    const int H = 2, KV = 1, hd = 8, max_seq = 6, width = (H + 2 * KV) * hd, kv_dim = KV * hd;
    Tensor qkv = Tensor::from_host(rnd(width, 1), {1, width});
    for (int bad : {max_seq, max_seq + 3, -1}) {
        Tensor ck = Tensor::from_host(std::vector<float>(max_seq * kv_dim, 7.f), {max_seq, kv_dim});
        Tensor cv = Tensor::from_host(std::vector<float>(max_seq * kv_dim, 7.f), {max_seq, kv_dim});
        DevInts pos({bad});
        (void)rope_qkv_to_cache(qkv, H, KV, hd, 1e4f, 0, ck, cv, pos.p);
        for (float x : ck.to_host()) CHECK(x == 7.f);
        for (float x : cv.to_host()) CHECK(x == 7.f);
    }
}

TEST(embedding_dev_matches_host_ids_and_flags_invalid_ids) {
    const int V = 30, D = 70;
    std::vector<float> table = rnd(static_cast<size_t>(V) * D, 5, 4.f);
    Tensor t32 = Tensor::from_host(table, {V, D});
    LowpMatrix tf16 = to_lowp(t32, DType::F16), tbf = to_lowp(t32, DType::BF16);
    const std::vector<int> ids = {29, 0, 7};
    DevInts d_ids(ids);
    CHECK(embedding(t32, ids).to_host() == embedding_dev(t32, d_ids.p, 3).to_host());
    CHECK(embedding(tf16, ids).to_host() == embedding_dev(tf16, d_ids.p, 3).to_host());
    CHECK(embedding(tbf, ids).to_host() == embedding_dev(tbf, d_ids.p, 3).to_host());

    DevInts bad({5, -1, V, 6});  // рядки 1 і 2 недопустимі
    std::vector<float> out = embedding_dev(t32, bad.p, 4).to_host();
    for (int d = 0; d < D; ++d) {
        CHECK(out[0 * D + d] == table[5 * D + d]);
        CHECK(std::isnan(out[1 * D + d]) && std::isnan(out[2 * D + d]));
        CHECK(out[3 * D + d] == table[6 * D + d]);
    }
    CHECK_THROWS(embedding_dev(t32, nullptr, 2));
    CHECK(embedding_dev(t32, nullptr, 0).numel() == 0);
}

// Запис у кеш завжди в межах буфера, навіть при хибній позиції з пристрою: кеш обрамлений зі
// сторін "сторожовими" рядками, і після виклику вони мають лишитись недоторканими. Без обрамлення
// запис за межі потрапив би в запас блоку пулу й лишився б непомітним.
TEST(rope_qkv_device_position_never_writes_outside_the_cache) {
    const int H = 2, KV = 1, hd = 8, cache_rows = 6, pad = 16, width = (H + 2 * KV) * hd, kv_dim = KV * hd;
    const int total_rows = cache_rows + 2 * pad;
    Tensor qkv = Tensor::from_host(rnd(width, 1), {1, width});
    Tensor q_out = Tensor::from_host(std::vector<float>(H * hd, 0.f), {1, H * hd});
    // кілька допустимих позицій і багато хибних, але такі, що неправильний запис лишився б у обрамленні
    for (int pos : {0, 3, 5, cache_rows, cache_rows + 1, cache_rows + pad - 1, -1, -pad + 1}) {
        Tensor ck = Tensor::from_host(std::vector<float>(static_cast<size_t>(total_rows) * kv_dim, 7.f), {total_rows, kv_dim});
        Tensor cv = Tensor::from_host(std::vector<float>(static_cast<size_t>(total_rows) * kv_dim, 7.f), {total_rows, kv_dim});
        DevInts d_pos({pos});
        launch_rope_qkv_cache(qkv.data(), q_out.data(), ck.data() + static_cast<size_t>(pad) * kv_dim,
                              cv.data() + static_cast<size_t>(pad) * kv_dim, 1, H, KV, hd, 1e4f, 0, d_pos.p, cache_rows);
        CUDA_CHECK(cudaDeviceSynchronize());
        const std::vector<float> hk = ck.to_host(), hv = cv.to_host();
        const bool valid = pos >= 0 && pos < cache_rows;
        for (int r = 0; r < total_rows; ++r)
            for (int d = 0; d < kv_dim; ++d) {
                const bool in_cache = r >= pad && r < pad + cache_rows;
                const bool written_row = valid && r == pad + pos;
                const float k = hk[static_cast<size_t>(r) * kv_dim + d], v = hv[static_cast<size_t>(r) * kv_dim + d];
                if (!written_row && (k != 7.f || v != 7.f)) {
                    std::ostringstream os;
                    os << "pos=" << pos << ": змінено рядок " << (r - pad) << (in_cache ? " (всередині кешу)" : " (ПОЗА кешем)");
                    throw TestFailure(os.str());
                }
            }
        if (valid) {  // а допустима позиція справді записала свій рядок
            bool changed = false;
            for (int d = 0; d < kv_dim; ++d) changed |= hv[static_cast<size_t>(pad + pos) * kv_dim + d] != 7.f;
            CHECK(changed);
        }
    }
}

// Велика позиція: shared memory має бути виділена під увесь кеш, а не лише під хостову позицію
// (яка при d_pos дорівнює нулю). Невеликі розміри ховали б помилку в запасі вирівнювання.
TEST(attention_with_device_position_is_correct_for_large_caches) {
    const int S = 3000, H = 2, KV = 1, hd = 8;
    Tensor q = Tensor::from_host(rnd(H * hd, 1, 2.f), {1, H * hd});
    Tensor k = Tensor::from_host(rnd(static_cast<size_t>(S) * KV * hd, 2, 2.f), {S, KV * hd});
    Tensor v = Tensor::from_host(rnd(static_cast<size_t>(S) * KV * hd, 3), {S, KV * hd});
    for (int pos : {S - 1, 1500, 2047, 2048}) {
        DevInts d_pos({pos});
        std::vector<float> host = attention(q, k, v, H, KV, hd, pos).to_host();
        std::vector<float> dev = attention(q, k, v, H, KV, hd, 0, d_pos.p).to_host();
        CHECK(host == dev);
        CHECK_NEAR(max_abs_diff(dev, cpu_attention(q.to_host(), k.to_host(), v.to_host(), 1, S, H, KV, hd, pos)), 0.0, 1e-5);
    }
}

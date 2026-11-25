// Тести ваг у fp16/bf16: конвертація, GEMV/matmul/злиті операції з такими вагами, embedding.
// Еталон - CPU на ТИХ САМИХ округлених вагах (round_to_dtype), тож перевіряється коректність
// ядер, а не похибка округлення (її окремо показує LowpStats і тести проти PyTorch).
#include <cmath>
#include <random>
#include "gemv.h"
#include "lowp.cuh"
#include "ops.cuh"
#include "ops_cpu.h"
#include "test_framework.h"

namespace {

const DType kLowp[] = {DType::F16, DType::BF16};

std::vector<float> rnd(size_t n, unsigned seed, float scale = 1.f) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<float> v(n);
    for (float& x : v) x = dist(gen);
    return v;
}

std::vector<float> rounded(std::vector<float> w, DType dt) {
    for (float& x : w) x = round_to_dtype(x, dt);
    return w;
}

LowpMatrix lowp_of(const std::vector<float>& w, int rows, int cols, DType dt, LowpStats* st = nullptr) {
    return to_lowp(Tensor::from_host(w, {rows, cols}), dt, st);
}

// Брудний блок у пулі потрібного розміру (NaN): виявляє ядра, що не записують увесь вихід.
void poison_pool(std::vector<int> shape) {
    size_t n = 1;
    for (int d : shape) n *= d;
    Tensor dirty = Tensor::from_host(std::vector<float>(n, std::nanf("")), shape);
}

// Повертає налаштування векторизації GEMV після тесту, навіть якщо він кинув виняток.
struct VecGuard {
    ~VecGuard() { gemv_set_vec_override(0); }
};

std::string fail_msg(const char* what, DType dt, int M, int K, int N, double d) {
    std::ostringstream os;
    os << what << " " << dtype_name(dt) << " M=" << M << " K=" << K << " N=" << N << " max_diff=" << d;
    return os.str();
}

struct Shape { int M, K, N; };
// M = 1..4 -> GEMV, M = 5, 9 -> резерв через reg-ядро. N: кратне 4, кратне 2, непарне.
const Shape kShapes[] = {{1, 1, 1},      {1, 7, 5},      {1, 16, 32},   {1, 33, 33},    {1, 576, 192}, {1, 576, 960},
                         {1, 1536, 576}, {2, 100, 66},   {4, 64, 64},   {3, 50, 7},     {1, 64, 4100}, {5, 576, 192},
                         {9, 33, 70},    {70, 40, 130}};

}  // namespace

// ---- конвертація -----------------------------------------------------------

TEST(lowp_conversion_matches_host_rounding) {
    // Звичайні, межові й особливі значення: нулі, найбільше fp16, субнормальні, дуже малі, великі.
    std::vector<float> v = rnd(5000, 1, 3.f);
    const float edge[] = {0.f, -0.f, 1.f, -1.f, 65504.f, -65504.f, 6.1e-5f, 5.96e-8f, 1e-9f, 1e-30f, 1e30f,
                          3.14159265f, 1.0f + 1.0f / 2048.f, 1.0f + 1.0f / 256.f, 0.1f, 123456.f};
    v.insert(v.end(), std::begin(edge), std::end(edge));
    const int n = static_cast<int>(v.size());
    for (DType dt : kLowp) {
        std::vector<float> got = from_lowp(lowp_of(v, 1, n, dt)).to_host();
        for (int i = 0; i < n; ++i) {
            const float want = round_to_dtype(v[i], dt);
            // inf == inf у IEEE, тож порівняння працює й для переповнення
            if (!(got[i] == want)) {
                std::ostringstream os;
                os << dtype_name(dt) << ": x=" << v[i] << " got " << got[i] << " want " << want;
                throw TestFailure(os.str());
            }
        }
    }
}

TEST(lowp_known_roundings) {
    Tensor x = Tensor::from_host({3.14159265f, 65504.f, 70000.f, 1.0f + 1.0f / 2048.f}, {1, 4});
    std::vector<float> h = from_lowp(to_lowp(x, DType::F16)).to_host();
    CHECK(h[0] == 3.140625f);        // 11 значущих біт
    CHECK(h[1] == 65504.f);          // найбільше скінченне fp16, точно
    CHECK(std::isinf(h[2]));         // поза діапазоном
    CHECK(h[3] == 1.0f);             // рівно посередині між 1 і 1+2^-10: округлення до парного
    std::vector<float> b = from_lowp(to_lowp(x, DType::BF16)).to_host();
    CHECK(b[0] == 3.140625f);        // 8 значущих біт
    CHECK(b[2] == 70144.f);          // 70000 у bf16 скінченне (діапазон як у fp32), крок 512 -> 137*512
    CHECK(b[3] == 1.0f);
}

TEST(lowp_stats_count_overflow_flush_and_error) {
    Tensor x = Tensor::from_host({70000.f, -70000.f, 1e-9f, 0.5f, 0.f, 1.0f + 1.0f / 2048.f}, {1, 6});
    LowpStats s16, sbf;
    to_lowp(x, DType::F16, &s16);
    to_lowp(x, DType::BF16, &sbf);
    CHECK(s16.elements == 6);
    CHECK(s16.overflow == 2);   // обидва 70000 -> inf
    CHECK(s16.flushed == 1);    // 1e-9 -> 0 (менше найменшого субнормального fp16)
    CHECK(s16.max_rel_error >= 1.0 - 1e-6);  // зануління = 100% похибки
    CHECK(sbf.overflow == 0);   // bf16 має діапазон fp32
    CHECK(sbf.flushed == 0);
    CHECK(sbf.max_rel_error < 0.01);  // 7 біт мантиси: до ~0.4% (2^-8)
}

TEST(lowp_stats_are_zero_for_exactly_representable_values) {
    // значення вже округлені до bf16, тож перетворення без втрат
    std::vector<float> v = rounded(rnd(2000, 4), DType::BF16);
    LowpStats s;
    lowp_of(v, 1, 2000, DType::BF16, &s);
    CHECK(s.max_rel_error == 0.0 && s.overflow == 0 && s.flushed == 0);
}

TEST(lowp_stats_accumulate_across_calls) {
    LowpStats s;
    lowp_of(rnd(100, 1), 1, 100, DType::F16, &s);
    lowp_of(rnd(50, 2), 1, 50, DType::F16, &s);
    CHECK(s.elements == 150);
}

TEST(lowp_matrix_basics) {
    LowpMatrix m({3, 5}, DType::BF16);
    CHECK(m.rows() == 3 && m.cols() == 5 && m.numel() == 15 && m.bytes() == 30);
    CHECK(m.dtype() == DType::BF16);
    WeightRef r = m;
    CHECK(r.rows == 3 && r.cols == 5 && r.dtype == DType::BF16 && r.data == m.data());
    LowpMatrix moved = std::move(m);
    CHECK(moved.data() != nullptr && m.data() == nullptr);  // NOLINT: після переміщення порожній
    CHECK_THROWS(LowpMatrix({2, 2}, DType::F32));
    CHECK_THROWS(LowpMatrix({2, 2, 2}, DType::F16));
    CHECK_THROWS(to_lowp(Tensor({4}), DType::F16));         // не 2D
    CHECK_THROWS(to_lowp(Tensor({2, 2}), DType::F32));
    CHECK(to_lowp(Tensor({0, 3}), DType::F16).numel() == 0);  // порожня матриця коректна
}

// ---- GEMV і злиті операції з 16-бітними вагами ----------------------------------

TEST(lowp_matmul_matches_cpu_on_rounded_weights) {
    for (DType dt : kLowp)
        for (const Shape& s : kShapes) {
            std::vector<float> a = rnd(static_cast<size_t>(s.M) * s.K, 11 + s.K), w = rnd(static_cast<size_t>(s.K) * s.N, 12 + s.N);
            poison_pool({s.M, s.N});
            Tensor out = matmul(Tensor::from_host(a, {s.M, s.K}), lowp_of(w, s.K, s.N, dt));  // Auto: GEMV або Reg
            const double d = max_abs_diff(out.to_host(), cpu_matmul(a, rounded(w, dt), s.M, s.N, s.K));
            if (d > 2e-3) throw TestFailure(fail_msg("matmul", dt, s.M, s.K, s.N, d));
        }
}

TEST(lowp_rmsnorm_matmul_matches_cpu) {
    for (DType dt : kLowp)
        for (const Shape& s : kShapes) {
            std::vector<float> x = rnd(static_cast<size_t>(s.M) * s.K, 21 + s.K, 3.f), nw = rnd(s.K, 22 + s.N);
            std::vector<float> w = rnd(static_cast<size_t>(s.K) * s.N, 23 + s.M);
            poison_pool({s.M, s.N});
            Tensor out = rmsnorm_matmul(Tensor::from_host(x, {s.M, s.K}), Tensor::from_host(nw, {s.K}), 1e-5f,
                                        lowp_of(w, s.K, s.N, dt));
            const double d = max_abs_diff(out.to_host(),
                                          cpu_matmul(cpu_rmsnorm(x, nw, s.M, s.K, 1e-5f), rounded(w, dt), s.M, s.N, s.K));
            if (d > 2e-3) throw TestFailure(fail_msg("rmsnorm_matmul", dt, s.M, s.K, s.N, d));
        }
}

TEST(lowp_matmul_residual_matches_cpu) {
    for (DType dt : kLowp)
        for (const Shape& s : kShapes) {
            std::vector<float> x = rnd(static_cast<size_t>(s.M) * s.K, 31 + s.K), w = rnd(static_cast<size_t>(s.K) * s.N, 32 + s.N);
            std::vector<float> res = rnd(static_cast<size_t>(s.M) * s.N, 33 + s.M, 5.f);
            std::vector<float> ref = cpu_matmul(x, rounded(w, dt), s.M, s.N, s.K);
            for (size_t i = 0; i < ref.size(); ++i) ref[i] += res[i];
            poison_pool({s.M, s.N});
            Tensor out = matmul_residual(Tensor::from_host(x, {s.M, s.K}), lowp_of(w, s.K, s.N, dt),
                                         Tensor::from_host(res, {s.M, s.N}));
            const double d = max_abs_diff(out.to_host(), ref);
            if (d > 2e-3) throw TestFailure(fail_msg("matmul_residual", dt, s.M, s.K, s.N, d));
        }
}

TEST(lowp_rmsnorm_gate_up_matches_cpu) {
    for (DType dt : kLowp)
        for (const Shape& s : kShapes) {
            std::vector<float> x = rnd(static_cast<size_t>(s.M) * s.K, 41 + s.K, 3.f), nw = rnd(s.K, 42 + s.N);
            std::vector<float> wg = rnd(static_cast<size_t>(s.K) * s.N, 43 + s.M), wu = rnd(static_cast<size_t>(s.K) * s.N, 44 + s.M);
            const std::vector<float> h = cpu_rmsnorm(x, nw, s.M, s.K, 1e-5f);
            std::vector<float> g = cpu_matmul(h, rounded(wg, dt), s.M, s.N, s.K), u = cpu_matmul(h, rounded(wu, dt), s.M, s.N, s.K), ref(g.size());
            for (size_t i = 0; i < g.size(); ++i)
                ref[i] = static_cast<float>(g[i] / (1.0 + std::exp(-static_cast<double>(g[i]))) * u[i]);
            poison_pool({s.M, s.N});
            Tensor out = rmsnorm_gate_up(Tensor::from_host(x, {s.M, s.K}), Tensor::from_host(nw, {s.K}), 1e-5f,
                                         lowp_of(wg, s.K, s.N, dt), lowp_of(wu, s.K, s.N, dt));
            const double d = max_abs_diff(out.to_host(), ref);
            if (d > 2e-3) throw TestFailure(fail_msg("rmsnorm_gate_up", dt, s.M, s.K, s.N, d));
        }
}

// Ширина векторизації не змінює результат (навіть побітово): кожен стовпець накопичує ті самі k
// у тому самому порядку. Це перевіряє, що векторизовані завантаження читають правильні елементи.
TEST(gemv_vector_widths_give_bitwise_identical_results) {
    VecGuard guard;
    const Shape shapes[] = {{1, 576, 960}, {2, 100, 68}, {1, 33, 4100}, {3, 64, 66}, {1, 50, 7}};
    for (DType dt : {DType::F32, DType::F16, DType::BF16})
        for (const Shape& s : shapes) {
            std::vector<float> x = rnd(static_cast<size_t>(s.M) * s.K, 51, 2.f), nw = rnd(s.K, 52);
            std::vector<float> w = rnd(static_cast<size_t>(s.K) * s.N, 53), res = rnd(static_cast<size_t>(s.M) * s.N, 54);
            Tensor X = Tensor::from_host(x, {s.M, s.K}), NW = Tensor::from_host(nw, {s.K}), R = Tensor::from_host(res, {s.M, s.N});
            Tensor W32 = Tensor::from_host(w, {s.K, s.N});
            LowpMatrix Wl;
            if (dt != DType::F32) Wl = to_lowp(W32, dt);
            WeightRef W = dt == DType::F32 ? WeightRef(W32) : WeightRef(Wl);

            std::vector<float> base_a, base_b, base_c;
            for (int vec : {1, 2, 4}) {
                gemv_set_vec_override(vec);
                std::vector<float> a = rmsnorm_matmul(X, NW, 1e-5f, W).to_host();
                std::vector<float> b = matmul_residual(X, W, R).to_host();
                std::vector<float> c = rmsnorm_gate_up(X, NW, 1e-5f, W, W).to_host();
                if (vec == 1) { base_a = a; base_b = b; base_c = c; continue; }
                if (a != base_a || b != base_b || c != base_c) {
                    std::ostringstream os;
                    os << dtype_name(dt) << " M=" << s.M << " K=" << s.K << " N=" << s.N << ": VEC=" << vec << " дає інший результат, ніж VEC=1";
                    throw TestFailure(os.str());
                }
            }
        }
}

TEST(gemv_vec_override_is_validated) {
    VecGuard guard;
    CHECK_THROWS(gemv_set_vec_override(3));
    CHECK_THROWS(gemv_set_vec_override(8));
    gemv_set_vec_override(2);
    CHECK(gemv_choose_vec(66, DType::F16) == 2);
    CHECK(gemv_choose_vec(7, DType::F16) == 1);   // непарне N: безпечний відкат до скалярного шляху
    gemv_set_vec_override(4);
    CHECK(gemv_choose_vec(66, DType::F16) == 2);  // 66 % 4 != 0 -> 2
    CHECK(gemv_choose_vec(64, DType::F16) == 4);
}

// ---- embedding -------------------------------------------------------------

TEST(lowp_embedding_returns_rounded_rows) {
    const int V = 30, D = 70;
    std::vector<float> table = rnd(static_cast<size_t>(V) * D, 61, 4.f);
    for (DType dt : kLowp) {
        LowpMatrix t = lowp_of(table, V, D, dt);
        poison_pool({3, D});
        std::vector<float> out = embedding(t, {29, 0, 7}).to_host();
        const int ids[] = {29, 0, 7};
        for (int i = 0; i < 3; ++i)
            for (int d = 0; d < D; ++d)
                CHECK(out[i * D + d] == round_to_dtype(table[ids[i] * D + d], dt));
        CHECK_THROWS(embedding(t, {V}));
    }
}

// ---- перевірки аргументів ------------------------------------------------------

TEST(lowp_ops_validate_dtypes_and_kernels) {
    Tensor x({1, 8}), nw({8});
    LowpMatrix f16({8, 4}, DType::F16), bf({8, 4}, DType::BF16);
    CHECK_THROWS(rmsnorm_gate_up(x, nw, 1e-5f, f16, bf));                 // різні формати Wg і Wu
    CHECK_THROWS(matmul(x, f16, MatmulKind::Naive));                     // навчальні ядра лише fp32
    CHECK_THROWS(matmul(x, f16, MatmulKind::Tiled));
    CHECK_THROWS(rmsnorm_matmul(Tensor({1, 7}), Tensor({7}), 1e-5f, f16));  // K не збігається з вагами
    (void)matmul(x, f16, MatmulKind::Reg);                               // Reg і Gemv працюють
    (void)matmul(x, f16, MatmulKind::Gemv);
}

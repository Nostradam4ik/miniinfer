#include <cstring>
#include <random>
#include "ops.cuh"
#include "ops_cpu.h"
#include "test_framework.h"

namespace {

const MatmulKind kKinds[] = {MatmulKind::Naive, MatmulKind::Tiled, MatmulKind::Reg, MatmulKind::Gemv,
                             MatmulKind::Auto};
const char* kind_name(MatmulKind k) {
    switch (k) {
        case MatmulKind::Naive: return "naive";
        case MatmulKind::Tiled: return "tiled";
        case MatmulKind::Reg: return "reg";
        case MatmulKind::Gemv: return "gemv";
        default: return "auto";
    }
}

std::vector<float> random_vec(size_t n, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> v(n);
    for (float& x : v) x = dist(gen);
    return v;
}

}  // namespace

// Ручний приклад: [[1,2,3],[4,5,6]] * [[7,8],[9,10],[11,12]] = [[58,64],[139,154]]
TEST(matmul_known_values_all_kinds) {
    Tensor A = Tensor::from_host({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor B = Tensor::from_host({7, 8, 9, 10, 11, 12}, {3, 2});
    for (MatmulKind k : kKinds) {
        std::vector<float> c = matmul(A, B, k).to_host();
        CHECK_NEAR(c[0], 58, 1e-5);
        CHECK_NEAR(c[1], 64, 1e-5);
        CHECK_NEAR(c[2], 139, 1e-5);
        CHECK_NEAR(c[3], 154, 1e-5);
    }
}

// A * I = A: перевірка без CPU-еталона.
TEST(matmul_identity) {
    const int n = 33;
    std::vector<float> a = random_vec(n * n, 1), eye(n * n, 0.f);
    for (int i = 0; i < n; ++i) eye[i * n + i] = 1.f;
    Tensor A = Tensor::from_host(a, {n, n}), I = Tensor::from_host(eye, {n, n});
    for (MatmulKind k : kKinds)
        CHECK_NEAR(max_abs_diff(matmul(A, I, k).to_host(), a), 0.0, 1e-6);
}

// Різні форми, у т.ч. не кратні розміру плитки (16 і 64): саме тут ховаються
// помилки в перевірках меж.
TEST(matmul_matches_cpu_for_many_shapes) {
    struct Shape { int M, K, N; };
    const Shape shapes[] = {{1, 1, 1},    {1, 7, 5},     {5, 1, 7},   {16, 16, 16},
                            {17, 33, 65}, {64, 64, 64},  {65, 64, 63}, {100, 200, 50},
                            {257, 129, 255}, {3, 300, 2}};
    for (const Shape& s : shapes) {
        std::vector<float> a = random_vec(static_cast<size_t>(s.M) * s.K, 10 + s.M);
        std::vector<float> b = random_vec(static_cast<size_t>(s.K) * s.N, 20 + s.N);
        std::vector<float> ref = cpu_matmul(a, b, s.M, s.N, s.K);
        Tensor A = Tensor::from_host(a, {s.M, s.K}), B = Tensor::from_host(b, {s.K, s.N});
        for (MatmulKind k : kKinds) {
            Tensor C = matmul(A, B, k);
            CHECK(C.dim(0) == s.M && C.dim(1) == s.N);
            double diff = max_abs_diff(C.to_host(), ref);
            if (diff > 1e-3) {
                std::ostringstream os;
                os << kind_name(k) << " M=" << s.M << " K=" << s.K << " N=" << s.N
                   << " max_diff=" << diff;
                throw TestFailure(os.str());
            }
        }
    }
}

TEST(matmul_all_kinds_agree) {
    const int M = 130, K = 70, N = 90;
    Tensor A = Tensor::from_host(random_vec(M * K, 5), {M, K});
    Tensor B = Tensor::from_host(random_vec(K * N, 6), {K, N});
    std::vector<float> naive = matmul(A, B, MatmulKind::Naive).to_host();
    CHECK_NEAR(max_abs_diff(matmul(A, B, MatmulKind::Tiled).to_host(), naive), 0.0, 1e-4);
    CHECK_NEAR(max_abs_diff(matmul(A, B, MatmulKind::Reg).to_host(), naive), 0.0, 1e-4);
}

// Буфер з "отрутою" (NaN) за межами даних: будь-яке читання за межами A чи B
// дає NaN у результаті, а будь-який запис за межі C змінює отруту.
// Без цього тесту такі баги ховаються, бо пам'ять за буфером зазвичай нульова.
TEST(matmul_does_not_touch_memory_out_of_bounds) {
    const size_t kPad = 20000;  // з запасом більше, ніж може зачепити ядро
    struct Shape { int M, K, N; };
    const Shape shapes[] = {{1, 1, 1}, {17, 33, 65}, {65, 20, 63}, {5, 100, 7}, {70, 3, 70}};
    for (const Shape& s : shapes) {
        const size_t na = static_cast<size_t>(s.M) * s.K, nb = static_cast<size_t>(s.K) * s.N,
                     nc = static_cast<size_t>(s.M) * s.N;
        std::vector<float> a = random_vec(na, 31), b = random_vec(nb, 32);
        std::vector<float> ref = cpu_matmul(a, b, s.M, s.N, s.K);

        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, (na + kPad) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, (nb + kPad) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC, (nc + kPad) * sizeof(float)));
        CUDA_CHECK(cudaMemset(dA, 0xFF, (na + kPad) * sizeof(float)));  // 0xFFFFFFFF = NaN
        CUDA_CHECK(cudaMemset(dB, 0xFF, (nb + kPad) * sizeof(float)));
        CUDA_CHECK(cudaMemset(dC, 0xFF, (nc + kPad) * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA, a.data(), na * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, b.data(), nb * sizeof(float), cudaMemcpyHostToDevice));

        for (MatmulKind k : kKinds) {
            CUDA_CHECK(cudaMemset(dC, 0xFF, (nc + kPad) * sizeof(float)));
            launch_matmul(k, dA, dB, dC, s.M, s.N, s.K);
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<float> c(nc + kPad);
            CUDA_CHECK(cudaMemcpy(c.data(), dC, c.size() * sizeof(float), cudaMemcpyDeviceToHost));
            std::vector<float> head(c.begin(), c.begin() + nc);
            bool ok = max_abs_diff(head, ref) < 1e-3;  // NaN дає false
            for (size_t i = nc; i < c.size() && ok; ++i) {
                unsigned bits;
                std::memcpy(&bits, &c[i], 4);
                if (bits != 0xFFFFFFFFu) ok = false;  // ядро писало за межі C
            }
            if (!ok) {
                cudaFree(dA); cudaFree(dB); cudaFree(dC);
                std::ostringstream os;
                os << kind_name(k) << " M=" << s.M << " K=" << s.K << " N=" << s.N
                   << ": читання/запис за межами буфера";
                throw TestFailure(os.str());
            }
        }
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }
}

TEST(matmul_shape_mismatch_throws) {
    Tensor A({2, 3}), B({4, 2});
    CHECK_THROWS(matmul(A, B));
}

TEST(matmul_rejects_non_2d) {
    Tensor A({2, 3, 4}), B({4, 2});
    CHECK_THROWS(matmul(A, B));
}

// Нульове K: порожня сума = нулі, а не сміття в пам'яті.
TEST(matmul_zero_inner_dimension_gives_zeros) {
    Tensor A({3, 0}), B({0, 4});
    for (MatmulKind k : kKinds)
        for (float v : matmul(A, B, k).to_host()) CHECK(v == 0.f);
}

// ---- GEMV (малі M) ---------------------------------------------------------

// Форми декодування: M = 1..5, N і K різні, у т.ч. не кратні 32 (ширина блоку) і 16 (розбиття K).
TEST(gemv_matches_cpu_for_decode_shapes) {
    struct S { int M, K, N; };
    const S shapes[] = {{1, 1, 1},     {1, 5, 1},     {1, 16, 32},   {1, 17, 33},   {1, 576, 192},
                        {1, 576, 576}, {1, 1536, 576}, {1, 576, 1536}, {1, 100, 1000}, {2, 33, 65},
                        {3, 576, 49},  {4, 7, 100},    {5, 64, 64},   {1, 3, 4097}};
    for (const S& s : shapes) {
        std::vector<float> a = random_vec(static_cast<size_t>(s.M) * s.K, 300 + s.K);
        std::vector<float> b = random_vec(static_cast<size_t>(s.K) * s.N, 400 + s.N);
        std::vector<float> ref = cpu_matmul(a, b, s.M, s.N, s.K);
        Tensor C = matmul(Tensor::from_host(a, {s.M, s.K}), Tensor::from_host(b, {s.K, s.N}), MatmulKind::Gemv);
        double d = max_abs_diff(C.to_host(), ref);
        if (d > 1e-3) {
            std::ostringstream os;
            os << "M=" << s.M << " K=" << s.K << " N=" << s.N << " max_diff=" << d;
            throw TestFailure(os.str());
        }
    }
}

// Рядки не змішуються: кожен рядок M дає той самий результат, що й окремий запуск з M=1.
TEST(gemv_rows_are_independent) {
    const int M = 4, K = 70, N = 90;
    std::vector<float> a = random_vec(M * K, 1), b = random_vec(K * N, 2);
    Tensor B = Tensor::from_host(b, {K, N});
    std::vector<float> all = matmul(Tensor::from_host(a, {M, K}), B, MatmulKind::Gemv).to_host();
    for (int m = 0; m < M; ++m) {
        std::vector<float> row(a.begin() + m * K, a.begin() + (m + 1) * K);
        std::vector<float> one = matmul(Tensor::from_host(row, {1, K}), B, MatmulKind::Gemv).to_host();
        CHECK(std::equal(one.begin(), one.end(), all.begin() + m * N));
    }
}

// Auto: малі M ідуть у Gemv, великі в Reg; результат у обох випадках правильний.
TEST(matmul_auto_is_correct_on_both_sides_of_threshold) {
    const int K = 64, N = 70;
    for (int M : {1, kGemvMaxRows, kGemvMaxRows + 1, 100}) {
        std::vector<float> a = random_vec(static_cast<size_t>(M) * K, 5), b = random_vec(K * N, 6);
        Tensor C = matmul(Tensor::from_host(a, {M, K}), Tensor::from_host(b, {K, N}));
        CHECK(max_abs_diff(C.to_host(), cpu_matmul(a, b, M, N, K)) < 1e-3);
    }
}

// Tensor core matmul (MatmulKind::Wmma): fp32 A, bf16 B. The kernel rounds A to bf16, so the CPU reference
// rounds both inputs the same way and multiplies in double precision. This separates kernel bugs from the
// cost of the format.
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <sstream>
#include "lowp.cuh"
#include "ops.cuh"
#include "test_framework.h"
#include "wmma.h"

namespace {

std::vector<float> uniform_values(size_t n, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> v(n);
    for (float& x : v) x = dist(gen);
    return v;
}

std::vector<float> small_integers(size_t n, int period) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = static_cast<float>(static_cast<int>(i % period) - period / 2);
    return v;
}

struct Reference {
    std::vector<double> value;  // round(A) * round(B)
    std::vector<double> bound;  // |round(A)| * |round(B)|, the scale of the fp32 accumulation error
};

Reference rounded_reference(const std::vector<float>& a, const std::vector<float>& b, int M, int N, int K) {
    Reference r{std::vector<double>(static_cast<size_t>(M) * N), std::vector<double>(static_cast<size_t>(M) * N)};
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k) {
            const double x = round_to_dtype(a[static_cast<size_t>(i) * K + k], DType::BF16);
            for (int j = 0; j < N; ++j) {
                const double y = round_to_dtype(b[static_cast<size_t>(k) * N + j], DType::BF16);
                r.value[static_cast<size_t>(i) * N + j] += x * y;
                r.bound[static_cast<size_t>(i) * N + j] += std::fabs(x * y);
            }
        }
    return r;
}

// tol = 0 asks for exact equality. NaN fails in both cases.
void check_against(const float* got, const Reference& ref, double tol, const char* what, int M, int K, int N) {
    for (size_t i = 0; i < ref.value.size(); ++i)
        if (!(std::fabs(got[i] - ref.value[i]) <= tol * ref.bound[i])) {
            std::ostringstream os;
            os << what << " M=" << M << " K=" << K << " N=" << N << " element " << i << ": got " << got[i] << ", want "
               << ref.value[i];
            throw TestFailure(os.str());
        }
}

LowpMatrix bf16_matrix(const std::vector<float>& b, int K, int N) {
    return to_lowp(Tensor::from_host(b, {K, N}), DType::BF16);
}

}  // namespace

TEST(wmma_matches_rounded_cpu_reference) {
    struct Shape {
        int M, K, N;
    };
    const Shape shapes[] = {{1, 32, 64},     {5, 576, 960},   {31, 64, 128}, {32, 576, 576}, {63, 32, 192},
                            {64, 576, 1536}, {65, 1536, 576}, {130, 96, 64}, {512, 576, 576}};
    for (const Shape& s : shapes) {
        const std::vector<float> a = uniform_values(static_cast<size_t>(s.M) * s.K, 700 + s.M);
        const std::vector<float> b = uniform_values(static_cast<size_t>(s.K) * s.N, 800 + s.N);
        const LowpMatrix B = bf16_matrix(b, s.K, s.N);
        const Tensor C = matmul(Tensor::from_host(a, {s.M, s.K}), B, MatmulKind::Wmma);
        CHECK(C.dim(0) == s.M && C.dim(1) == s.N);
        check_against(C.to_host().data(), rounded_reference(a, b, s.M, s.N, s.K), 1e-5, "random", s.M, s.K, s.N);
    }
}

// Small integers are exact in bf16 and in the fp32 accumulator, so the result must be equal.
TEST(wmma_small_integers_are_exact) {
    const int K = 64, N = 128;
    for (int M : {1, 5, 63, 65, 130}) {
        const std::vector<float> a = small_integers(static_cast<size_t>(M) * K, 7);
        const std::vector<float> b = small_integers(static_cast<size_t>(K) * N, 5);
        const LowpMatrix B = bf16_matrix(b, K, N);
        const Tensor C = matmul(Tensor::from_host(a, {M, K}), B, MatmulKind::Wmma);
        check_against(C.to_host().data(), rounded_reference(a, b, M, N, K), 0.0, "integers", M, K, N);
    }
}

// NaN beyond the data: a read outside A or B puts NaN into the result, a write outside C changes the poison.
TEST(wmma_does_not_touch_memory_out_of_bounds) {
    const size_t kPad = 20000;
    const int K = 64, N = 128;
    for (int M : {5, 65, 70}) {
        const size_t na = static_cast<size_t>(M) * K, nb = static_cast<size_t>(K) * N, nc = static_cast<size_t>(M) * N;
        const std::vector<float> a = uniform_values(na, 31), b = uniform_values(nb, 32);
        std::vector<__nv_bfloat16> bh(nb);
        for (size_t i = 0; i < nb; ++i) bh[i] = __float2bfloat16_rn(b[i]);

        float *dA, *dC;
        __nv_bfloat16* dB;
        CUDA_CHECK(cudaMalloc(&dA, (na + kPad) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, (nb + kPad) * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&dC, (nc + kPad) * sizeof(float)));
        CUDA_CHECK(cudaMemset(dA, 0xFF, (na + kPad) * sizeof(float)));
        CUDA_CHECK(cudaMemset(dB, 0xFF, (nb + kPad) * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemset(dC, 0xFF, (nc + kPad) * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA, a.data(), na * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, bh.data(), nb * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

        launch_matmul_wmma(dA, dB, DType::BF16, dC, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> c(nc + kPad);
        CUDA_CHECK(cudaMemcpy(c.data(), dC, c.size() * sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);

        check_against(c.data(), rounded_reference(a, b, M, N, K), 1e-5, "poisoned", M, K, N);
        for (size_t i = nc; i < c.size(); ++i) {
            unsigned bits;
            std::memcpy(&bits, &c[i], 4);
            CHECK(bits == 0xFFFFFFFFu);  // the kernel wrote outside C
        }
    }
}

// The output comes from the pool uninitialized: the kernel has to write every element.
TEST(wmma_overwrites_dirty_pool_block) {
    const int M = 65, K = 64, N = 128;
    const std::vector<float> a = uniform_values(static_cast<size_t>(M) * K, 1), b = uniform_values(K * N, 2);
    const Tensor A = Tensor::from_host(a, {M, K});
    const LowpMatrix B = bf16_matrix(b, K, N);
    {
        const Tensor dirty = Tensor::from_host(
            std::vector<float>(static_cast<size_t>(M) * N, std::numeric_limits<float>::quiet_NaN()), {M, N});
    }
    const Tensor C = matmul(A, B, MatmulKind::Wmma);
    for (float v : C.to_host()) CHECK(!std::isnan(v));
}

TEST(wmma_zero_inner_dimension_gives_zeros) {
    const Tensor A({3, 0});
    const LowpMatrix B({0, 64}, DType::BF16);
    for (float v : matmul(A, B, MatmulKind::Wmma).to_host()) CHECK(v == 0.f);
}

TEST(wmma_rejects_unsupported_input) {
    CHECK_THROWS(matmul(Tensor({2, 48}), bf16_matrix(std::vector<float>(48 * 64, 0.f), 48, 64), MatmulKind::Wmma));
    CHECK_THROWS(matmul(Tensor({2, 64}), bf16_matrix(std::vector<float>(64 * 96, 0.f), 64, 96), MatmulKind::Wmma));
    const Tensor A({2, 64}), B32({64, 128});
    CHECK_THROWS(matmul(A, B32, MatmulKind::Wmma));
    const LowpMatrix B16 = to_lowp(B32, DType::F16);
    CHECK_THROWS(matmul(A, B16, MatmulKind::Wmma));

    Tensor a_buf({3, 64}), c_buf({2, 128});
    const LowpMatrix b_buf = bf16_matrix(std::vector<float>(65 * 128, 0.f), 65, 128);
    const auto* bp = static_cast<const __nv_bfloat16*>(b_buf.data());
    CHECK_THROWS(launch_matmul_wmma(a_buf.data() + 1, bp, DType::BF16, c_buf.data(), 2, 128, 64));
    CHECK_THROWS(launch_matmul_wmma(a_buf.data(), bp + 1, DType::BF16, c_buf.data(), 2, 128, 64));
    // The raw launcher with fp32 weights cannot use tensor cores.
    CHECK_THROWS(launch_matmul(MatmulKind::Wmma, a_buf.data(), B32.data(), c_buf.data(), 2, 128, 64));
}

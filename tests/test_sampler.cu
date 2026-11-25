// Sampler tests: CPU only, no GPU. The statistical checks use tight tolerances on a large enough
// sample with a fixed seed, so they are not flaky.
#include <map>
#include "sampler.h"
#include "test_framework.h"

namespace {
SamplerConfig cfg(float temp, int top_k = 0, float top_p = 1.f, unsigned seed = 1) {
    SamplerConfig c;
    c.temperature = temp;
    c.top_k = top_k;
    c.top_p = top_p;
    c.seed = seed;
    return c;
}

std::map<int, int> histogram(Sampler& s, const std::vector<float>& logits, int n) {
    std::map<int, int> h;
    for (int i = 0; i < n; ++i) ++h[s.sample(logits)];
    return h;
}
}  // namespace

TEST(sampler_greedy_picks_argmax) {
    Sampler s(cfg(0.f));
    CHECK(s.sample({0.1f, 3.f, -1.f, 2.9f}) == 1);
    CHECK(s.sample({5.f}) == 0);
}

TEST(sampler_greedy_tie_picks_lowest_index) {
    Sampler s(cfg(0.f));
    CHECK(s.sample({1.f, 7.f, 7.f, 7.f}) == 1);
}

TEST(sampler_greedy_ignores_top_k_and_top_p) {
    Sampler s(cfg(0.f, 2, 0.1f));
    CHECK(s.sample({0.f, 1.f, 2.f, 1.5f}) == 2);
}

TEST(sampler_nan_logits_throw) {
    Sampler g(cfg(0.f)), t(cfg(1.f));
    CHECK_THROWS(g.sample({1.f, std::nanf(""), 2.f}));
    CHECK_THROWS(t.sample({1.f, std::nanf(""), 2.f}));
}

TEST(sampler_empty_logits_throw) { CHECK_THROWS(Sampler(cfg(1.f)).sample({})); }

TEST(sampler_invalid_config_throws) {
    CHECK_THROWS(Sampler(cfg(-1.f)));
    CHECK_THROWS(Sampler(cfg(std::nanf(""))));
    CHECK_THROWS(Sampler(cfg(1.f, -1)));
    CHECK_THROWS(Sampler(cfg(1.f, 0, 0.f)));
    CHECK_THROWS(Sampler(cfg(1.f, 0, 1.5f)));
}

TEST(sampler_top_k_1_equals_argmax) {
    Sampler s(cfg(1.5f, 1));
    for (int i = 0; i < 200; ++i) CHECK(s.sample({0.f, 1.f, 4.f, 3.9f}) == 2);
}

TEST(sampler_top_k_restricts_candidates) {
    Sampler s(cfg(1.f, 3));
    auto h = histogram(s, {5.f, 4.f, 3.f, 2.f, 1.f, 0.f}, 3000);
    for (auto& [tok, n] : h) CHECK(tok <= 2);
    CHECK(h.size() == 3);  // all three candidates really occur
}

// Frequencies must match softmax: logits {0,1,2}, T=1.
TEST(sampler_frequencies_match_softmax) {
    const std::vector<float> logits = {0.f, 1.f, 2.f};
    double z = 1 + std::exp(1.0) + std::exp(2.0);
    const double want[] = {1 / z, std::exp(1.0) / z, std::exp(2.0) / z};
    const int N = 40000;
    Sampler s(cfg(1.f));
    auto h = histogram(s, logits, N);
    for (int t = 0; t < 3; ++t) CHECK_NEAR(static_cast<double>(h[t]) / N, want[t], 0.01);
}

TEST(sampler_temperature_sharpens_and_flattens) {
    const std::vector<float> logits = {0.f, 1.f, 2.f};
    const int N = 10000;
    Sampler cold(cfg(0.1f)), hot(cfg(10.f));
    double p_cold = static_cast<double>(histogram(cold, logits, N)[2]) / N;
    double p_hot = static_cast<double>(histogram(hot, logits, N)[2]) / N;
    CHECK(p_cold > 0.99);              // almost always the best one
    CHECK(p_hot > 0.3 && p_hot < 0.45);  // almost uniform (1/3 + a slight skew)
}

// Distribution {0.5, 0.3, 0.15, 0.05}, top_p = 0.7: the sum 0.5 < 0.7, with the second 0.8 >= 0.7,
// so only tokens 0 and 1 remain, in the ratio 0.5 : 0.3.
TEST(sampler_top_p_keeps_smallest_nucleus) {
    const std::vector<float> logits = {std::log(0.5f), std::log(0.3f), std::log(0.15f), std::log(0.05f)};
    const int N = 20000;
    Sampler s(cfg(1.f, 0, 0.7f));
    auto h = histogram(s, logits, N);
    CHECK(h.count(2) == 0 && h.count(3) == 0);
    CHECK_NEAR(static_cast<double>(h[0]) / N, 0.5 / 0.8, 0.015);
}

TEST(sampler_top_p_always_keeps_at_least_one_token) {
    Sampler s(cfg(1.f, 0, 0.0001f));
    for (int i = 0; i < 100; ++i) CHECK(s.sample({1.f, 5.f, 2.f}) == 1);
}

TEST(sampler_never_picks_minus_infinity) {
    const float ninf = -std::numeric_limits<float>::infinity();
    Sampler s(cfg(1.f));
    auto h = histogram(s, {ninf, 0.f, ninf, 0.f}, 2000);
    CHECK(h.count(0) == 0 && h.count(2) == 0);
    Sampler g(cfg(0.f));
    CHECK(g.sample({ninf, ninf, 0.f}) == 2);
    CHECK_THROWS(Sampler(cfg(1.f)).sample({ninf, ninf}));
}

TEST(sampler_is_deterministic_for_same_seed) {
    std::vector<float> logits(50);
    for (int i = 0; i < 50; ++i) logits[i] = std::sin(i * 0.7f);
    Sampler a(cfg(1.f, 0, 1.f, 42)), b(cfg(1.f, 0, 1.f, 42)), c(cfg(1.f, 0, 1.f, 43));
    std::vector<int> sa, sb, sc;
    for (int i = 0; i < 100; ++i) { sa.push_back(a.sample(logits)); sb.push_back(b.sample(logits)); sc.push_back(c.sample(logits)); }
    CHECK(sa == sb);
    CHECK(sa != sc);
}

// exp(1000) overflows, so without subtracting the maximum we would get NaN. The distribution must be
// the same as for logits {0, 1, -1}: softmax = {0.245, 0.665, 0.090}.
TEST(sampler_large_logits_give_correct_distribution) {
    const int N = 40000;
    Sampler s(cfg(1.f));
    auto h = histogram(s, {1000.f, 1001.f, 999.f}, N);
    const double z = 1 + std::exp(1.0) + std::exp(-1.0);
    CHECK_NEAR(static_cast<double>(h[0]) / N, 1 / z, 0.01);
    CHECK_NEAR(static_cast<double>(h[1]) / N, std::exp(1.0) / z, 0.01);
    CHECK_NEAR(static_cast<double>(h[2]) / N, std::exp(-1.0) / z, 0.01);
}

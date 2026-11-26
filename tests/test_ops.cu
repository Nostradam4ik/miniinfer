#include <algorithm>
#include <cstring>
#include <limits>
#include <random>
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
}  // namespace

// ---- embedding -------------------------------------------------------------

TEST(embedding_picks_rows) {
    // a 4x3 table: row i = {i, i+0.5, i+0.25}
    std::vector<float> t;
    for (int i = 0; i < 4; ++i) t.insert(t.end(), {float(i), i + 0.5f, i + 0.25f});
    Tensor table = Tensor::from_host(t, {4, 3});
    std::vector<float> out = embedding(table, {3, 0, 3, 1}).to_host();
    const float want[] = {3, 3.5f, 3.25f, 0, 0.5f, 0.25f, 3, 3.5f, 3.25f, 1, 1.5f, 1.25f};
    CHECK(out.size() == 12);
    for (int i = 0; i < 12; ++i) CHECK(out[i] == want[i]);
}

TEST(embedding_out_of_range_throws) {
    Tensor table({4, 3});
    CHECK_THROWS(embedding(table, {0, 4}));
    CHECK_THROWS(embedding(table, {-1}));
}

TEST(embedding_empty_ids) {
    Tensor table({4, 3});
    Tensor out = embedding(table, {});
    CHECK(out.dim(0) == 0 && out.dim(1) == 3);
}

// ---- write_rows / slice_rows -----------------------------------------------

TEST(write_rows_copies_into_offset_and_leaves_rest) {
    Tensor dst = Tensor::from_host(std::vector<float>(5 * 3, 9.f), {5, 3});
    Tensor src = Tensor::from_host({1, 2, 3, 4, 5, 6}, {2, 3});
    write_rows(dst, src, 2);
    const float want[] = {9, 9, 9, 9, 9, 9, 1, 2, 3, 4, 5, 6, 9, 9, 9};
    std::vector<float> h = dst.to_host();
    for (int i = 0; i < 15; ++i) CHECK(h[i] == want[i]);
}

TEST(write_rows_rejects_overflow_and_bad_shapes) {
    Tensor dst({4, 3}), src({2, 3});
    CHECK_THROWS(write_rows(dst, src, 3));  // 3 + 2 > 4
    CHECK_THROWS(write_rows(dst, src, -1));
    CHECK_THROWS(write_rows(dst, Tensor({2, 4}), 0));  // different number of columns
    write_rows(dst, src, 2);                           // exactly up to the edge: allowed
}

TEST(slice_rows_extracts_range) {
    Tensor x = Tensor::from_host({1, 2, 3, 4, 5, 6, 7, 8}, {4, 2});
    std::vector<float> h = slice_rows(x, 1, 2).to_host();
    const float want[] = {3, 4, 5, 6};
    CHECK(h.size() == 4);
    for (int i = 0; i < 4; ++i) CHECK(h[i] == want[i]);
    CHECK(slice_rows(x, 4, 0).numel() == 0);  // empty slice at the end
    CHECK_THROWS(slice_rows(x, 3, 2));
    CHECK_THROWS(slice_rows(x, -1, 1));
}

// ---- transpose / add / silu_mul -------------------------------------------

TEST(transpose_known_values) {
    Tensor t = transpose(Tensor::from_host({1, 2, 3, 4, 5, 6}, {2, 3}));
    CHECK(t.dim(0) == 3 && t.dim(1) == 2);
    const float want[] = {1, 4, 2, 5, 3, 6};
    std::vector<float> h = t.to_host();
    for (int i = 0; i < 6; ++i) CHECK(h[i] == want[i]);
}

// Shapes around the tile size 32 and non-multiples of it: bounds bugs hide here.
TEST(transpose_matches_cpu_and_roundtrips) {
    struct S {
        int R, C;
    };
    const S shapes[] = {{1, 1}, {1, 40}, {40, 1}, {31, 33}, {32, 32}, {33, 31}, {64, 65}, {100, 7}, {576, 1536}};
    for (const S& s : shapes) {
        std::vector<float> x = rnd(static_cast<size_t>(s.R) * s.C, 100 + s.R);
        Tensor t = transpose(Tensor::from_host(x, {s.R, s.C}));
        CHECK(t.dim(0) == s.C && t.dim(1) == s.R);
        std::vector<float> h = t.to_host();
        for (int i = 0; i < s.R; ++i)
            for (int j = 0; j < s.C; ++j)
                if (h[static_cast<size_t>(j) * s.R + i] != x[static_cast<size_t>(i) * s.C + j]) {
                    std::ostringstream os;
                    os << "R=" << s.R << " C=" << s.C << " mismatch at (" << i << "," << j << ")";
                    throw TestFailure(os.str());
                }
        CHECK(transpose(t).to_host() == x);  // twice = original
    }
}

TEST(transpose_rejects_non_2d) {
    CHECK_THROWS(transpose(Tensor({2, 3, 4})));
}

TEST(add_sums_elementwise) {
    std::vector<float> a = rnd(1000, 1), b = rnd(1000, 2);
    std::vector<float> c = add(Tensor::from_host(a, {10, 100}), Tensor::from_host(b, {10, 100})).to_host();
    for (size_t i = 0; i < a.size(); ++i) CHECK(c[i] == a[i] + b[i]);
}

TEST(add_shape_mismatch_throws) {
    CHECK_THROWS(add(Tensor({2, 3}), Tensor({3, 2})));
}

TEST(silu_mul_known_values) {
    Tensor g = Tensor::from_host({0.f, 1.f, -1.f, 2.f}, {4});
    Tensor u = Tensor::from_host({5.f, 1.f, 1.f, 0.5f}, {4});
    std::vector<float> o = silu_mul(g, u).to_host();
    CHECK_NEAR(o[0], 0.0, 1e-7);                           // silu(0) = 0
    CHECK_NEAR(o[1], 1.0 / (1.0 + std::exp(-1.0)), 1e-6);  // 0.7310586
    CHECK_NEAR(o[2], -1.0 / (1.0 + std::exp(1.0)), 1e-6);  // -0.2689414
    CHECK_NEAR(o[3], 2.0 / (1.0 + std::exp(-2.0)) * 0.5, 1e-6);
}

// Extreme inputs must not produce NaN: exp(-x) at x = -1000 overflows to inf.
TEST(silu_mul_is_stable_for_extreme_inputs) {
    Tensor g = Tensor::from_host({-1000.f, -100.f, 100.f, 1000.f}, {4});
    Tensor u = Tensor::from_host({1.f, 1.f, 2.f, 1.f}, {4});
    std::vector<float> o = silu_mul(g, u).to_host();
    for (float v : o) CHECK(std::isfinite(v));
    CHECK_NEAR(o[0], 0.0, 1e-6);
    CHECK_NEAR(o[2], 200.0, 1e-3);
    CHECK_NEAR(o[3], 1000.0, 1e-3);
}

TEST(silu_mul_shape_mismatch_throws) {
    CHECK_THROWS(silu_mul(Tensor({4}), Tensor({5})));
}

// ---- rope ------------------------------------------------------------------

TEST(rope_position_zero_is_identity) {
    const int T = 1, H = 3, hd = 8;
    std::vector<float> x = rnd(T * H * hd, 1);
    CHECK_NEAR(max_abs_diff(rope(Tensor::from_host(x, {T, H * hd}), H, hd, 10000.f).to_host(), x), 0.0, 1e-7);
}

// hd = 2, any theta: angle = pos * theta^0 = pos, (1,0) -> (cos pos, sin pos).
TEST(rope_known_rotation) {
    Tensor x = Tensor::from_host({1.f, 0.f, 1.f, 0.f}, {2, 2});  // two tokens, one head
    std::vector<float> y = rope(x, 1, 2, 10000.f).to_host();
    CHECK_NEAR(y[0], 1.0, 1e-6);  // position 0
    CHECK_NEAR(y[1], 0.0, 1e-6);
    CHECK_NEAR(y[2], std::cos(1.0), 1e-6);  // position 1
    CHECK_NEAR(y[3], std::sin(1.0), 1e-6);
}

TEST(rope_matches_cpu_for_many_shapes) {
    struct S {
        int T, H, hd;
        float theta;
        int off;
    };
    const S shapes[] = {{1, 1, 2, 10000.f, 0},  {5, 9, 64, 100000.f, 0}, {7, 3, 64, 100000.f, 11},
                        {16, 2, 8, 10000.f, 3}, {33, 4, 16, 500.f, 100}, {3, 1, 128, 1e6f, 1000}};
    for (const S& s : shapes) {
        std::vector<float> x = rnd(static_cast<size_t>(s.T) * s.H * s.hd, 60 + s.T);
        std::vector<float> ref = cpu_rope(x, s.T, s.H, s.hd, s.theta, s.off);
        Tensor y = rope(Tensor::from_host(x, {s.T, s.H * s.hd}), s.H, s.hd, s.theta, s.off);
        double d = max_abs_diff(y.to_host(), ref);
        if (d > 2e-4) {  // float sinf/cosf at large angles (pos ~1000) give an error of ~1e-4
            std::ostringstream os;
            os << "T=" << s.T << " H=" << s.H << " hd=" << s.hd << " off=" << s.off << " diff=" << d;
            throw TestFailure(os.str());
        }
    }
}

// The rotation preserves the length of each pair: this catches a sign error (a*c + b*s).
TEST(rope_preserves_pair_norm) {
    const int T = 6, H = 2, hd = 16;
    std::vector<float> x = rnd(T * H * hd, 3);
    std::vector<float> y = rope(Tensor::from_host(x, {T, H * hd}), H, hd, 10000.f, 5).to_host();
    for (int th = 0; th < T * H; ++th)
        for (int i = 0; i < hd / 2; ++i) {
            double a = x[th * hd + i], b = x[th * hd + i + hd / 2];
            double c = y[th * hd + i], d = y[th * hd + i + hd / 2];
            CHECK_NEAR(a * a + b * b, c * c + d * d, 1e-5);
        }
}

// KV cache: rope of a tail with pos_offset = rope of the full sequence cut down to the tail.
TEST(rope_offset_equals_slice_of_full) {
    const int T = 10, H = 2, hd = 8, off = 6;
    std::vector<float> x = rnd(T * H * hd, 4);
    std::vector<float> full = rope(Tensor::from_host(x, {T, H * hd}), H, hd, 10000.f).to_host();
    std::vector<float> tail_in(x.begin() + off * H * hd, x.end());
    std::vector<float> tail = rope(Tensor::from_host(tail_in, {T - off, H * hd}), H, hd, 10000.f, off).to_host();
    std::vector<float> want(full.begin() + off * H * hd, full.end());
    CHECK_NEAR(max_abs_diff(tail, want), 0.0, 1e-6);
}

TEST(rope_invalid_args_throw) {
    Tensor x({2, 12});
    CHECK_THROWS(rope(x, 3, 3, 10000.f));  // odd head_dim
    CHECK_THROWS(rope(x, 2, 4, 10000.f));  // 2*4 != 12
    CHECK_THROWS(rope(x, 3, 4, 10000.f, -1));
}

// ---- attention -------------------------------------------------------------

TEST(attention_single_token_returns_v) {
    // A single token sees only itself: softmax over one element = 1, result = v.
    const int H = 2, KV = 2, hd = 4;
    Tensor q = Tensor::from_host(rnd(H * hd, 1), {1, H * hd});
    Tensor k = Tensor::from_host(rnd(KV * hd, 2), {1, KV * hd});
    std::vector<float> v = rnd(KV * hd, 3);
    std::vector<float> out = attention(q, k, Tensor::from_host(v, {1, KV * hd}), H, KV, hd).to_host();
    CHECK_NEAR(max_abs_diff(out, v), 0.0, 1e-6);
}

// Q = 0: all scores are 0, softmax is uniform, the result is the mean of V over the visible keys.
TEST(attention_uniform_scores_average_values) {
    const int T = 4, H = 1, KV = 1, hd = 2;
    Tensor q({T, hd});  // zeros
    Tensor k = Tensor::from_host(rnd(T * hd, 1), {T, hd});
    std::vector<float> v = rnd(T * hd, 2);
    std::vector<float> out = attention(q, k, Tensor::from_host(v, {T, hd}), H, KV, hd).to_host();
    for (int t = 0; t < T; ++t)
        for (int d = 0; d < hd; ++d) {
            double mean = 0;
            for (int j = 0; j <= t; ++j) mean += v[j * hd + d];
            CHECK_NEAR(out[t * hd + d], mean / (t + 1), 1e-6);
        }
}

TEST(attention_matches_cpu_for_many_shapes) {
    struct S {
        int T, S, H, KV, hd, off;
    };
    const S shapes[] = {{1, 1, 1, 1, 4, 0},    {5, 5, 9, 3, 64, 0},   {7, 7, 4, 2, 8, 0},    {1, 20, 9, 3, 64, 19},
                        {3, 40, 6, 2, 16, 37}, {33, 33, 2, 1, 32, 0}, {70, 70, 4, 4, 16, 0}, {2, 300, 3, 1, 8, 298}};
    for (const S& s : shapes) {
        std::vector<float> q = rnd(static_cast<size_t>(s.T) * s.H * s.hd, 70 + s.T, 2.f);
        std::vector<float> k = rnd(static_cast<size_t>(s.S) * s.KV * s.hd, 80 + s.S, 2.f);
        std::vector<float> v = rnd(static_cast<size_t>(s.S) * s.KV * s.hd, 90 + s.S);
        std::vector<float> ref = cpu_attention(q, k, v, s.T, s.S, s.H, s.KV, s.hd, s.off);
        Tensor out = attention(Tensor::from_host(q, {s.T, s.H * s.hd}), Tensor::from_host(k, {s.S, s.KV * s.hd}),
                               Tensor::from_host(v, {s.S, s.KV * s.hd}), s.H, s.KV, s.hd, s.off);
        double d = max_abs_diff(out.to_host(), ref);
        if (d > 1e-5) {
            std::ostringstream os;
            os << "T=" << s.T << " S=" << s.S << " H=" << s.H << " KV=" << s.KV << " hd=" << s.hd << " off=" << s.off
               << " diff=" << d;
            throw TestFailure(os.str());
        }
    }
}

// The kernel loads head widths that are multiples of 4 as float4 and uses a scalar loop for the rest, so
// both kinds are needed. Several shapes have more keys than a block has threads (each thread then handles
// more than one key).
TEST(attention_matches_cpu_for_any_head_width_and_many_keys) {
    struct S {
        int T, S, H, KV, hd, off;
    };
    const S shapes[] = {{2, 2, 2, 1, 2, 0},      {5, 5, 2, 2, 6, 0},      {9, 200, 3, 1, 10, 191},
                        {3, 300, 9, 3, 64, 297}, {140, 140, 2, 1, 12, 0}, {2, 260, 2, 1, 66, 258}};
    for (const S& s : shapes) {
        // Small q and k keep the scores moderate, so that fp32 softmax stays close to the double reference.
        std::vector<float> q = rnd(static_cast<size_t>(s.T) * s.H * s.hd, 170 + s.T, 0.5f);
        std::vector<float> k = rnd(static_cast<size_t>(s.S) * s.KV * s.hd, 180 + s.S, 0.5f);
        std::vector<float> v = rnd(static_cast<size_t>(s.S) * s.KV * s.hd, 190 + s.S);
        std::vector<float> ref = cpu_attention(q, k, v, s.T, s.S, s.H, s.KV, s.hd, s.off);
        Tensor out = attention(Tensor::from_host(q, {s.T, s.H * s.hd}), Tensor::from_host(k, {s.S, s.KV * s.hd}),
                               Tensor::from_host(v, {s.S, s.KV * s.hd}), s.H, s.KV, s.hd, s.off);
        double d = max_abs_diff(out.to_host(), ref);
        if (d > 1e-5) {
            std::ostringstream os;
            os << "T=" << s.T << " S=" << s.S << " H=" << s.H << " KV=" << s.KV << " hd=" << s.hd << " off=" << s.off
               << " diff=" << d;
            throw TestFailure(os.str());
        }
    }
}

// Causality: changing keys/values AFTER position t cannot change the output at positions <= t.
TEST(attention_is_causal) {
    const int T = 8, H = 2, KV = 1, hd = 8, cut = 4;
    std::vector<float> q = rnd(T * H * hd, 1), k = rnd(T * KV * hd, 2), v = rnd(T * KV * hd, 3);
    Tensor Q = Tensor::from_host(q, {T, H * hd});
    std::vector<float> a =
        attention(Q, Tensor::from_host(k, {T, KV * hd}), Tensor::from_host(v, {T, KV * hd}), H, KV, hd).to_host();
    for (int j = cut; j < T; ++j)
        for (int d = 0; d < KV * hd; ++d) {
            k[j * KV * hd + d] += 100.f;
            v[j * KV * hd + d] -= 50.f;
        }
    std::vector<float> b =
        attention(Q, Tensor::from_host(k, {T, KV * hd}), Tensor::from_host(v, {T, KV * hd}), H, KV, hd).to_host();
    for (int t = 0; t < cut; ++t)
        for (int d = 0; d < H * hd; ++d) CHECK(a[t * H * hd + d] == b[t * H * hd + d]);
}

// GQA: heads of one group with the same Q must give the same result,
// and heads of different groups (different K/V) must differ.
TEST(attention_gqa_heads_share_kv_groups) {
    const int T = 5, H = 4, KV = 2, hd = 8;
    std::vector<float> head = rnd(T * hd, 9), q(T * H * hd);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < H; ++h)
            for (int d = 0; d < hd; ++d) q[(t * H + h) * hd + d] = head[t * hd + d];  // same Q
    std::vector<float> out =
        attention(Tensor::from_host(q, {T, H * hd}), Tensor::from_host(rnd(T * KV * hd, 1), {T, KV * hd}),
                  Tensor::from_host(rnd(T * KV * hd, 2), {T, KV * hd}), H, KV, hd)
            .to_host();
    const int t = T - 1;
    for (int d = 0; d < hd; ++d) {
        CHECK(out[(t * H + 0) * hd + d] == out[(t * H + 1) * hd + d]);  // group 0
        CHECK(out[(t * H + 2) * hd + d] == out[(t * H + 3) * hd + d]);  // group 1
    }
    CHECK(max_abs_diff(std::vector<float>(out.begin() + (t * H + 0) * hd, out.begin() + (t * H + 1) * hd),
                       std::vector<float>(out.begin() + (t * H + 2) * hd, out.begin() + (t * H + 3) * hd)) > 1e-3);
}

// KV cache: attention of a tail with pos_offset = the matching rows of attention over the whole sequence.
TEST(attention_offset_equals_slice_of_full) {
    const int T = 9, H = 3, KV = 1, hd = 8, off = 5;
    std::vector<float> q = rnd(T * H * hd, 1), k = rnd(T * KV * hd, 2), v = rnd(T * KV * hd, 3);
    Tensor K = Tensor::from_host(k, {T, KV * hd}), V = Tensor::from_host(v, {T, KV * hd});
    std::vector<float> full = attention(Tensor::from_host(q, {T, H * hd}), K, V, H, KV, hd).to_host();
    std::vector<float> q_tail(q.begin() + off * H * hd, q.end());
    std::vector<float> tail = attention(Tensor::from_host(q_tail, {T - off, H * hd}), K, V, H, KV, hd, off).to_host();
    CHECK_NEAR(max_abs_diff(tail, std::vector<float>(full.begin() + off * H * hd, full.end())), 0.0, 1e-6);
}

// Large Q.K values: without subtracting the maximum, exp would overflow. The result must be finite.
TEST(attention_is_numerically_stable_for_large_scores) {
    const int T = 3, hd = 4;
    std::vector<float> big(T * hd, 100.f);
    Tensor Q = Tensor::from_host(big, {T, hd}), K = Tensor::from_host(big, {T, hd});
    Tensor V = Tensor::from_host(rnd(T * hd, 5), {T, hd});
    for (float x : attention(Q, K, V, 1, 1, hd).to_host()) CHECK(std::isfinite(x));
}

TEST(attention_invalid_args_throw) {
    Tensor q({2, 8}), k({2, 4}), v({2, 4});
    CHECK_THROWS(attention(q, k, v, 3, 2, 4));     // 3 is not a multiple of 2
    CHECK_THROWS(attention(q, k, v, 2, 1, 5));     // shapes do not match
    CHECK_THROWS(attention(q, k, v, 2, 1, 4, 1));  // S < pos_offset + T
    Tensor big_k({kMaxAttentionKeys + 1, 4}), big_q({1, 8});
    CHECK_THROWS(attention(big_q, big_k, big_k, 2, 1, 4, kMaxAttentionKeys));  // too many keys
}

// Prefill uses one block per (token, KV head) when the scores of the whole group fit into shared memory.
TEST(attention_shared_kv_selection) {
    CHECK(attention_uses_shared_kv(5, 5, 9, 3, 64, false));
    CHECK(attention_uses_shared_kv(512, 4085, 9, 3, 64, false));
    CHECK(!attention_uses_shared_kv(4, 4, 9, 3, 64, false));       // decode sizes
    CHECK(!attention_uses_shared_kv(512, 4086, 9, 3, 64, false));  // the scores do not fit
    CHECK(!attention_uses_shared_kv(8, 8, 9, 3, 64, true));        // position on the device
    CHECK(!attention_uses_shared_kv(8, 8, 4, 2, 64, false));       // 2 query heads per KV head
    CHECK(!attention_uses_shared_kv(8, 8, 9, 3, 6, false));        // width not a multiple of 4
}

// One call with many query rows against calls on at most kGemvMaxRows rows each, which use the per-head
// kernel: the bits must be equal. Cache rows after the last visible key are NaN and must not reach the result.
TEST(attention_shared_kv_is_bitwise_equal_to_per_head) {
    struct S {
        int T, pos;
    };
    const S shapes[] = {{5, 0}, {8, 0}, {31, 7}, {129, 0}, {200, 300}, {5, 4080}, {5, 4081}};
    const int H = 9, KV = 3, hd = 64;
    for (const S& s : shapes) {
        const int keys = s.pos + s.T + 3;
        std::vector<float> q = rnd(static_cast<size_t>(s.T) * H * hd, 270 + s.T, 0.5f);
        std::vector<float> k = rnd(static_cast<size_t>(keys) * KV * hd, 280 + s.pos, 0.5f);
        std::vector<float> v = rnd(static_cast<size_t>(keys) * KV * hd, 290 + s.pos);
        for (size_t i = static_cast<size_t>(s.pos + s.T) * KV * hd; i < k.size(); ++i)
            k[i] = v[i] = std::numeric_limits<float>::quiet_NaN();
        const Tensor K = Tensor::from_host(k, {keys, KV * hd}), V = Tensor::from_host(v, {keys, KV * hd});
        const std::vector<float> whole =
            attention(Tensor::from_host(q, {s.T, H * hd}), K, V, H, KV, hd, s.pos).to_host();
        std::vector<float> pieces;
        for (int row = 0; row < s.T; row += kGemvMaxRows) {
            const int rows = std::min(kGemvMaxRows, s.T - row);
            const std::vector<float> part(q.begin() + static_cast<size_t>(row) * H * hd,
                                          q.begin() + static_cast<size_t>(row + rows) * H * hd);
            const std::vector<float> out =
                attention(Tensor::from_host(part, {rows, H * hd}), K, V, H, KV, hd, s.pos + row).to_host();
            pieces.insert(pieces.end(), out.begin(), out.end());
        }
        for (float x : whole) CHECK(std::isfinite(x));
        if (whole.size() != pieces.size() ||
            std::memcmp(whole.data(), pieces.data(), whole.size() * sizeof(float)) != 0) {
            std::ostringstream os;
            os << "T=" << s.T << " pos=" << s.pos << ": not bitwise equal";
            throw TestFailure(os.str());
        }
    }
}

TEST(attention_shared_kv_matches_cpu) {
    struct S {
        int T, pos;
    };
    const S shapes[] = {{5, 0}, {33, 12}, {130, 70}};
    const int H = 9, KV = 3, hd = 64;
    for (const S& s : shapes) {
        const int keys = s.pos + s.T;
        std::vector<float> q = rnd(static_cast<size_t>(s.T) * H * hd, 370 + s.T, 0.5f);
        std::vector<float> k = rnd(static_cast<size_t>(keys) * KV * hd, 380 + s.T, 0.5f);
        std::vector<float> v = rnd(static_cast<size_t>(keys) * KV * hd, 390 + s.T);
        const std::vector<float> ref = cpu_attention(q, k, v, s.T, keys, H, KV, hd, s.pos);
        const Tensor out = attention(Tensor::from_host(q, {s.T, H * hd}), Tensor::from_host(k, {keys, KV * hd}),
                                     Tensor::from_host(v, {keys, KV * hd}), H, KV, hd, s.pos);
        CHECK_NEAR(max_abs_diff(out.to_host(), ref), 0.0, 1e-5);
    }
}

// ---- rmsnorm ---------------------------------------------------------------

TEST(rmsnorm_known_value) {
    // x = {3, 4}: mean(x^2) = 12.5, rstd = 1/sqrt(12.5), w = {1, 2}
    Tensor x = Tensor::from_host({3.f, 4.f}, {1, 2});
    Tensor w = Tensor::from_host({1.f, 2.f}, {2});
    std::vector<float> y = rmsnorm(x, w, 0.f).to_host();
    const double rstd = 1.0 / std::sqrt(12.5);
    CHECK_NEAR(y[0], 3 * rstd, 1e-6);
    CHECK_NEAR(y[1], 4 * rstd * 2, 1e-6);
}

// Various D: below the block size (256), not a multiple of the warp (32), above a block, the real 576.
TEST(rmsnorm_matches_cpu_for_many_shapes) {
    struct S {
        int T, D;
    };
    const S shapes[] = {{1, 1}, {1, 7}, {3, 31}, {2, 33}, {4, 256}, {5, 257}, {5, 576}, {2, 1000}};
    for (const S& s : shapes) {
        std::vector<float> x = rnd(static_cast<size_t>(s.T) * s.D, 40 + s.D, 3.f);
        std::vector<float> w = rnd(s.D, 50 + s.D);
        std::vector<float> ref = cpu_rmsnorm(x, w, s.T, s.D, 1e-5f);
        Tensor y = rmsnorm(Tensor::from_host(x, {s.T, s.D}), Tensor::from_host(w, {s.D}), 1e-5f);
        double d = max_abs_diff(y.to_host(), ref);
        if (d > 1e-5) {
            std::ostringstream os;
            os << "T=" << s.T << " D=" << s.D << " max_diff=" << d;
            throw TestFailure(os.str());
        }
    }
}

TEST(rmsnorm_shape_errors_throw) {
    CHECK_THROWS(rmsnorm(Tensor({2, 4}), Tensor({5}), 1e-5f));
    CHECK_THROWS(rmsnorm(Tensor({2, 4}), Tensor({2, 4}), 1e-5f));
}

// Rows are processed independently: changing one row does not affect the others.
TEST(rmsnorm_rows_are_independent) {
    const int T = 4, D = 300;
    std::vector<float> x = rnd(T * D, 7, 2.f), w = rnd(D, 8);
    Tensor W = Tensor::from_host(w, {D});
    std::vector<float> a = rmsnorm(Tensor::from_host(x, {T, D}), W, 1e-5f).to_host();
    for (int d = 0; d < D; ++d) x[2 * D + d] *= 100.f;  // change only row 2
    std::vector<float> b = rmsnorm(Tensor::from_host(x, {T, D}), W, 1e-5f).to_host();
    for (int t : {0, 1, 3})
        for (int d = 0; d < D; ++d) CHECK(a[t * D + d] == b[t * D + d]);
}

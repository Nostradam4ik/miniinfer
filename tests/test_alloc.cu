#include <cmath>
#include <random>
#include "gpu_alloc.cuh"
#include "model.cuh"
#include "ops.cuh"
#include "ops_cpu.h"
#include "test_framework.h"

// ---- size rounding ----------------------------------------------------------

TEST(alloc_bucket_size_known_values) {
    const size_t cases[][2] = {{1, 512},     {512, 512},   {513, 640},   {640, 640},   {641, 768},
                               {1000, 1024}, {1024, 1024}, {1025, 1280}, {4097, 5120}, {1u << 20, 1u << 20}};
    for (auto& c : cases) CHECK(GpuAllocator::bucket_size(c[0]) == c[1]);
}

TEST(alloc_bucket_size_is_bounded_waste) {
    std::mt19937 gen(5);
    std::uniform_int_distribution<size_t> dist(1, 200u << 20);
    for (int i = 0; i < 20000; ++i) {
        size_t b = dist(gen), k = GpuAllocator::bucket_size(b);
        CHECK(k >= b);
        if (b > 512) CHECK(static_cast<double>(k) < 1.25 * static_cast<double>(b) + 1);  // waste < 25%
        CHECK(GpuAllocator::bucket_size(k) == k);                                        // a step rounds to itself
    }
}

// ---- pool -------------------------------------------------------------------

TEST(alloc_reuses_freed_block_of_same_bucket) {
    GpuAllocator& a = GpuAllocator::instance();
    if (!a.pooling()) return;
    void* p = a.allocate(1000);
    a.release(p, 1000);
    const auto before = a.stats();
    void* q = a.allocate(900);  // same step (1024)
    CHECK(q == p);
    CHECK(a.stats().pool_hits == before.pool_hits + 1);
    CHECK(a.stats().cuda_mallocs == before.cuda_mallocs);
    a.release(q, 900);
}

TEST(alloc_different_buckets_do_not_mix) {
    GpuAllocator& a = GpuAllocator::instance();
    if (!a.pooling()) return;
    void* small = a.allocate(1000);  // step 1024
    a.release(small, 1000);
    void* big = a.allocate(5000);  // another step: must not get the small block
    CHECK(big != small);
    a.release(big, 5000);
}

TEST(alloc_zero_bytes_returns_null) {
    GpuAllocator& a = GpuAllocator::instance();
    CHECK(a.allocate(0) == nullptr);
    a.release(nullptr, 0);  // does not crash
}

TEST(alloc_stats_track_bytes_in_use) {
    GpuAllocator& a = GpuAllocator::instance();
    const size_t base = a.stats().bytes_in_use;
    {
        Tensor t({1000, 100});
        CHECK(a.stats().bytes_in_use >= base + 1000 * 100 * sizeof(float));
    }
    CHECK(a.stats().bytes_in_use == base);  // the tensor returned its memory
}

TEST(alloc_cannot_switch_mode_while_tensors_alive) {
    Tensor t({4, 4});
    CHECK_THROWS(GpuAllocator::instance().set_pooling(false));
    CHECK_THROWS(GpuAllocator::instance().set_pooling(true));
}

// A tensor from the pool after a block is returned: the zeroing constructor really clears "dirty" memory.
TEST(tensor_zero_constructor_clears_recycled_memory) {
    {
        Tensor dirty = Tensor::from_host(std::vector<float>(4096, 7.f), {64, 64});
    }
    Tensor z({64, 64});  // same step: the block was just filled with sevens
    for (float v : z.to_host()) CHECK(v == 0.f);
}

// ---- safety of uninitialized ------------------------------------------------
// Kernels that return an uninitialized tensor must write EVERY element. We check this by handing
// them blocks full of NaN: any element left unwritten spoils the result.

namespace {
// Fills the pool with NaN blocks of the size of a tensor with this shape.
void poison_pool(const std::vector<int>& shape) {
    size_t n = 1;
    for (int d : shape) n *= d;
    Tensor dirty = Tensor::from_host(std::vector<float>(n, std::nanf("")), shape);
}

std::vector<float> rnd(size_t n, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> v(n);
    for (float& x : v) x = dist(gen);
    return v;
}
}  // namespace

TEST(ops_do_not_read_uninitialized_output_memory) {
    if (!GpuAllocator::instance().pooling()) return;
    // matmul, all three kernels, including sizes that are not multiples of the tile
    const int M = 70, K = 33, N = 65;
    std::vector<float> a = rnd(M * K, 1), b = rnd(K * N, 2);
    Tensor A = Tensor::from_host(a, {M, K}), B = Tensor::from_host(b, {K, N});
    for (MatmulKind k : {MatmulKind::Naive, MatmulKind::Tiled, MatmulKind::Reg}) {
        poison_pool({M, N});
        CHECK(max_abs_diff(matmul(A, B, k).to_host(), cpu_matmul(a, b, M, N, K)) < 1e-4);
    }
    // K = 0 -> zeros even if the block is dirty
    poison_pool({3, 4});
    for (float v : matmul(Tensor({3, 0}), Tensor({0, 4})).to_host()) CHECK(v == 0.f);

    // rmsnorm
    const int T = 5, D = 300;
    std::vector<float> x = rnd(T * D, 3), w = rnd(D, 4);
    poison_pool({T, D});
    CHECK(max_abs_diff(rmsnorm(Tensor::from_host(x, {T, D}), Tensor::from_host(w, {D}), 1e-5f).to_host(),
                       cpu_rmsnorm(x, w, T, D, 1e-5f)) < 1e-5);

    // rope
    const int H = 3, hd = 8;
    std::vector<float> r = rnd(T * H * hd, 5);
    poison_pool({T, H * hd});
    CHECK(max_abs_diff(rope(Tensor::from_host(r, {T, H * hd}), H, hd, 10000.f, 2).to_host(),
                       cpu_rope(r, T, H, hd, 10000.f, 2)) < 1e-5);

    // attention
    const int QH = 4, KV = 2, S = 9;
    std::vector<float> q = rnd(T * QH * hd, 6), k = rnd(S * KV * hd, 7), v = rnd(S * KV * hd, 8);
    poison_pool({T, QH * hd});
    CHECK(max_abs_diff(attention(Tensor::from_host(q, {T, QH * hd}), Tensor::from_host(k, {S, KV * hd}),
                                 Tensor::from_host(v, {S, KV * hd}), QH, KV, hd, S - T)
                           .to_host(),
                       cpu_attention(q, k, v, T, S, QH, KV, hd, S - T)) < 1e-5);

    // transpose
    std::vector<float> t = rnd(37 * 41, 9);
    poison_pool({41, 37});
    std::vector<float> tt = transpose(Tensor::from_host(t, {37, 41})).to_host();
    for (int i = 0; i < 37; ++i)
        for (int j = 0; j < 41; ++j) CHECK(tt[j * 37 + i] == t[i * 41 + j]);

    // add, silu_mul
    std::vector<float> e1 = rnd(1000, 10), e2 = rnd(1000, 11);
    poison_pool({10, 100});
    std::vector<float> sum = add(Tensor::from_host(e1, {10, 100}), Tensor::from_host(e2, {10, 100})).to_host();
    for (int i = 0; i < 1000; ++i) CHECK(sum[i] == e1[i] + e2[i]);
    poison_pool({10, 100});
    for (float x2 : silu_mul(Tensor::from_host(e1, {10, 100}), Tensor::from_host(e2, {10, 100})).to_host())
        CHECK(std::isfinite(x2));

    // embedding and slice_rows
    Tensor table = Tensor::from_host(rnd(20 * 6, 12), {20, 6});
    poison_pool({3, 6});
    std::vector<float> emb = embedding(table, {19, 0, 7}).to_host(), tab = table.to_host();
    for (int d = 0; d < 6; ++d) CHECK(emb[d] == tab[19 * 6 + d]);
    poison_pool({2, 6});
    std::vector<float> sl = slice_rows(table, 5, 2).to_host();
    for (int i = 0; i < 12; ++i) CHECK(sl[i] == tab[5 * 6 + i]);
}

// ---- arena -----------------------------------------------------------------------

TEST(arena_serves_allocations_and_ignores_releases) {
    GpuAllocator& a = GpuAllocator::instance();
    GpuArena arena(1 << 20);
    const auto before = a.stats();
    void* p1 = nullptr;
    void* p2 = nullptr;
    {
        ArenaScope scope(arena);
        CHECK(a.arena_active());
        p1 = a.allocate(1000);
        p2 = a.allocate(5000);
        CHECK(arena.contains(p1) && arena.contains(p2) && p1 != p2);
        CHECK(arena.used() == 1024 + 5120);  // 256-byte alignment
        a.release(p1, 1000);                 // does not return to the pool
    }
    CHECK(!a.arena_active());
    CHECK(a.stats().arena_allocs == before.arena_allocs + 2);
    CHECK(a.stats().cuda_mallocs == before.cuda_mallocs);  // no cudaMalloc from the arena
    CHECK(a.stats().bytes_cached == before.bytes_cached);  // and nothing went to the pool
    a.release(p2, 5000);                                   // also ignored after the scope
    CHECK(a.stats().bytes_cached == before.bytes_cached);
    CHECK(a.stats().bytes_in_use == before.bytes_in_use);
}

TEST(arena_tensors_do_not_enter_the_pool_when_destroyed) {
    GpuAllocator& a = GpuAllocator::instance();
    if (!a.pooling()) return;
    GpuArena arena(1 << 20);
    const size_t cached_before = a.stats().bytes_cached;
    {
        ArenaScope scope(arena);
        Tensor t = Tensor::uninitialized({64, 64});  // leaves the scope only together with the tensor
        CHECK(arena.contains(t.data()));
    }  // t is destroyed after (or with) the scope: the arena address must not go into the pool
    CHECK(a.stats().bytes_cached == cached_before);
    Tensor fresh = Tensor::uninitialized({64, 64});
    CHECK(!arena.contains(fresh.data()));  // new pool tensors do not get arena memory
}

TEST(arena_exhaustion_throws_and_allocator_stays_usable) {
    GpuAllocator& a = GpuAllocator::instance();
    GpuArena arena(4096);
    {
        ArenaScope scope(arena);
        (void)a.allocate(4000);
        CHECK_THROWS(a.allocate(1000));
    }
    CHECK(!a.arena_active());
    Tensor t({8, 8});  // regular allocations work after leaving the scope
    CHECK(t.numel() == 64);
}

TEST(arena_rejects_nesting_and_zero_capacity) {
    GpuArena a1(4096), a2(4096);
    ArenaScope scope(a1);
    CHECK_THROWS(ArenaScope(a2));
    CHECK_THROWS(GpuArena(0));
}

TEST(arena_destruction_ends_active_scope) {
    GpuAllocator& a = GpuAllocator::instance();
    {
        GpuArena arena(4096);
        a.begin_arena(&arena);
        CHECK(a.arena_active());
    }  // the arena was destroyed while active: the allocator must not keep a dangling pointer
    CHECK(!a.arena_active());
    Tensor t({4, 4});
    CHECK(t.numel() == 16);
}

// ---- execution stream --------------------------------------------------------

TEST(scoped_stream_sets_and_restores_even_on_exception) {
    CHECK(exec_stream() == nullptr);  // the default stream by default
    cudaStream_t s = nullptr;
    CUDA_CHECK(cudaStreamCreate(&s));
    {
        ScopedStream outer(s);
        CHECK(exec_stream() == s);
        try {
            ScopedStream inner(nullptr);
            CHECK(exec_stream() == nullptr);
            throw std::runtime_error("boom");
        } catch (const std::runtime_error&) {
        }
        CHECK(exec_stream() == s);  // the inner scope restored the outer one
    }
    CHECK(exec_stream() == nullptr);
    CUDA_CHECK(cudaStreamDestroy(s));
}

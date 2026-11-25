// bench_layer_ops.exe - the cost of each fused layer operation at real model shapes (M = 1).
//
//   bench_layer_ops.exe [--dtype f32|f16|bf16]
//
// Each operation runs 300 times in a row (including taking the output from the pool, as in the model) and
// is timed with a CUDA event. The result is compared with the cost of an empty kernel (the launch floor):
// if an operation costs about the same, it is bound by the launch, not by its work.
#include <chrono>
#include <cstdio>
#include <string>
#include "gemv.h"
#include "lowp.cuh"
#include "ops.cuh"

__global__ void empty_kernel() {}

template <typename F>
static double time_us(F&& f, int iters = 300) {
    f();
    CUDA_CHECK(cudaDeviceSynchronize());
    GpuTimer t;
    t.start();
    for (int i = 0; i < iters; ++i) f();
    return t.stop() * 1000.0 / iters;
}

// Same, but `reps` calls of the operation are captured into a CUDA graph and launched with one command:
// this is the real GPU execution time without the host cost of launching every kernel.
template <typename F>
static double time_graph_us(F&& f, int reps = 100) {
    f();  // warm-up (kernel loading)
    CUDA_CHECK(cudaDeviceSynchronize());
    GpuArena arena(192u << 20);  // every call produces its own output, so leave plenty of room
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    {
        ScopedStream ss(s);
        ArenaScope as(arena);
        CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        for (int i = 0; i < reps; ++i) f();
        CUDA_CHECK(cudaStreamEndCapture(s, &graph));
    }
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
    CUDA_CHECK(cudaGraphLaunch(exec, s));  // warm-up
    CUDA_CHECK(cudaStreamSynchronize(s));
    const int launches = 20;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < launches; ++i) CUDA_CHECK(cudaGraphLaunch(exec, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() /
                      (static_cast<double>(launches) * reps);
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(s);
    return us;
}

int main(int argc, char** argv) {
    DType dt = DType::F32;
    bool use_graph = false;  // --graph: time of each operation inside a CUDA graph (without launch cost)
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--graph") use_graph = true;
        if (a == "--dtype" && i + 1 < argc) {
            std::string t = argv[i + 1];
            dt = t == "f16" ? DType::F16 : t == "bf16" ? DType::BF16 : DType::F32;
        }
    }
    auto meas = [&](auto&& f, int iters = 300) { return use_graph ? time_graph_us(f) : time_us(f, iters); };

    const int H = 576, Q = 576, KV = 192, I = 1536, V = 49152, NH = 9, NKV = 3, HD = 64, S = 20;
    auto W = [&](int k, int n) -> std::pair<Tensor, LowpMatrix> {
        Tensor w({k, n});
        LowpMatrix lp;
        if (dt != DType::F32) lp = to_lowp(w, dt);
        return {std::move(w), std::move(lp)};
    };
    auto ref = [&](const std::pair<Tensor, LowpMatrix>& p) -> WeightRef {
        return dt == DType::F32 ? WeightRef(p.first) : WeightRef(p.second);
    };

    auto wqkv = W(H, Q + 2 * KV), wo = W(Q, H), wg = W(H, I), wu = W(H, I), wd = W(I, H), wlm = W(H, V);
    Tensor x({1, H}), nw({H}), ctxt({1, Q}), act({1, I}), qkv({1, Q + 2 * KV});
    Tensor ck({64, KV}), cv({64, KV}), q({1, Q}), kk({S, KV}), vv({S, KV});

    const double launch = time_us([&] { empty_kernel<<<1, 32>>>(); }, 2000);

    struct Row { const char* name; double us; int per_token; };
    std::vector<Row> rows;
    rows.push_back({"rmsnorm + QKV gemv     (576->960)", meas([&] { Tensor o = rmsnorm_matmul(x, nw, 1e-5f, ref(wqkv)); }), 30});
    rows.push_back({"rope + K/V cache write", meas([&] { Tensor o = rope_qkv_to_cache(qkv, NH, NKV, HD, 1e5f, 19, ck, cv); }), 30});
    rows.push_back({"attention (S=20, 9 heads)", meas([&] { Tensor o = attention(q, kk, vv, NH, NKV, HD, 19); }), 30});
    rows.push_back({"o_proj + residual       (576->576)", meas([&] { Tensor o = matmul_residual(ctxt, ref(wo), x); }), 30});
    rows.push_back({"rmsnorm + gate/up/silu  (576->1536)", meas([&] { Tensor o = rmsnorm_gate_up(x, nw, 1e-5f, ref(wg), ref(wu)); }), 30});
    rows.push_back({"down + residual         (1536->576)", meas([&] { Tensor o = matmul_residual(act, ref(wd), x); }), 30});
    rows.push_back({"final norm + lm_head    (576->49152)", meas([&] { Tensor o = rmsnorm_matmul(x, nw, 1e-5f, ref(wlm)); }), 1});
    std::vector<int> ids = {504};
    Tensor table({V, H});
    LowpMatrix table_lp;
    if (dt != DType::F32) table_lp = to_lowp(table, dt);
    WeightRef table_ref = dt == DType::F32 ? WeightRef(table) : WeightRef(table_lp);
    int* d_id = nullptr;
    CUDA_CHECK(cudaMalloc(&d_id, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_id, ids.data(), sizeof(int), cudaMemcpyHostToDevice));
    if (use_graph)  // variant with the ids on the device: it can be captured into a graph
        rows.push_back({"embedding (1 token, device id)", meas([&] { Tensor o = embedding_dev(table_ref, d_id, 1); }), 1});
    else
        rows.push_back({"embedding (1 token, incl. sync)", time_us([&] { Tensor o = embedding(table_ref, ids); }, 100), 1});

    std::printf("weights: %s | timing: %s | empty kernel launch: %.2f us\n\n", dtype_name(dt),
                use_graph ? "inside a CUDA graph (GPU time)" : "back-to-back host launches", launch);
    std::printf("%-40s %10s %10s %12s\n", "operation", "us / call", "calls/tok", "ms / token");
    double total = 0;
    for (const Row& r : rows) {
        const double ms = r.us * r.per_token / 1000.0;
        total += ms;
        std::printf("%-40s %10.1f %10d %12.2f\n", r.name, r.us, r.per_token, ms);
    }
    std::printf("\nsum of operations per token: %.2f ms  (isolated back-to-back calls)\n", total);
    return 0;
}

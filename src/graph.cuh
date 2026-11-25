#pragma once
// CUDA graph for the decode step.
//
// Why: one token is ~180 short kernels, and the cost of launching each (~7 us) is comparable to its
// work. The graph records the whole sequence once and then launches with one command: the driver
// gets a ready list of kernels and runs them on the GPU without the host in between.
//
// How it works:
//   1. The token and position live in device memory (d_state_); kernels read them from there, so the
//      same graph serves any step. Before each launch the host copies 8 bytes there.
//   2. The step is captured from the same operations as the regular path (forward_decode_dev), so the
//      result has to match it rather than being a separate implementation.
//   3. All intermediate tensors of the step come from an arena owned by the graph, so their addresses
//      do not go back to the pool and get overwritten by something else between launches.
//
// Limitation: the graph is tied to specific Model and KVCache objects (their addresses are baked in),
// and both must outlive the graph. Only suitable for single-token steps.
#include <cstring>
#include <stdexcept>
#include <vector>
#include "model.cuh"

class DecodeGraph {
public:
    // arena_bytes - memory reserved for the intermediate tensors of a step (megabytes for this model).
    // Capture happens in the constructor; the cache is not changed (pos stays cache.length).
    DecodeGraph(const Model& model, KVCache& cache, size_t arena_bytes = 16u << 20)
        : model_(model), cache_(cache), vocab_(model.cfg.vocab_size), arena_(arena_bytes) {
        CUDA_CHECK(cudaStreamCreate(&stream_));  // blocking stream: ordered with the default one
        CUDA_CHECK(cudaMalloc(&d_state_, 2 * sizeof(int)));
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_state_), 2 * sizeof(int), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_logits_), static_cast<size_t>(vocab_) * sizeof(float),
                                 cudaHostAllocDefault));
        try {
            capture();
        } catch (...) {
            release();
            throw;
        }
    }

    ~DecodeGraph() { release(); }
    DecodeGraph(const DecodeGraph&) = delete;
    DecodeGraph& operator=(const DecodeGraph&) = delete;

    // One step: processes `token` at position cache.length, appends K, V to the cache and increments
    // cache.length. Writes the logits of the next token (vocab_size numbers) into `logits`.
    void step(int token, std::vector<float>& logits) {
        if (token < 0 || token >= vocab_)
            throw std::invalid_argument("DecodeGraph::step: token id " + std::to_string(token) + " out of range");
        if (cache_.length >= cache_.max_seq)
            throw std::invalid_argument("DecodeGraph::step: KV cache is full (" + std::to_string(cache_.length) +
                                        " of " + std::to_string(cache_.max_seq) + ")");
        h_state_[0] = token;
        h_state_[1] = cache_.length;
        CUDA_CHECK(cudaMemcpyAsync(d_state_, h_state_, 2 * sizeof(int), cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaGraphLaunch(exec_, stream_));
        CUDA_CHECK(cudaMemcpyAsync(h_logits_, logits_.data(), static_cast<size_t>(vocab_) * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        ++cache_.length;
        logits.assign(h_logits_, h_logits_ + vocab_);
    }

    // Diagnostics: runs the graph n times in a row without copies and returns the mean launch time (ms).
    // Token and position stay those of the last step(); the cache and cache.length do not change
    // (the graph overwrites the same cache row every time).
    double time_replays(int n) {
        GpuTimer timer;
        timer.start();  // event in the default stream; the graph in the blocking stream is ordered with it
        for (int i = 0; i < n; ++i) CUDA_CHECK(cudaGraphLaunch(exec_, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        return timer.stop() / n;
    }

    int vocab_size() const { return vocab_; }
    size_t arena_used() const { return arena_.used(); }
    size_t arena_capacity() const { return arena_.capacity(); }

private:
    void capture() {
        // Warm-up: one regular run of the same step, so all kernels are loaded before capture
        // (lazy module loading may fail during capture). It writes the cache row at
        // position cache.length, but the real step will overwrite it.
        h_state_[0] = 0;
        h_state_[1] = cache_.length;
        CUDA_CHECK(cudaMemcpy(d_state_, h_state_, 2 * sizeof(int), cudaMemcpyHostToDevice));
        if (cache_.length < cache_.max_seq) {
            ScopedStream warm(stream_);
            Tensor tmp = model_.forward_decode_dev(d_state_, d_state_ + 1, cache_);
            CUDA_CHECK(cudaStreamSynchronize(stream_));
        }

        cudaGraph_t graph = nullptr;
        {
            ScopedStream use_stream(stream_);
            ArenaScope use_arena(arena_);
            CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
            try {
                logits_ = model_.forward_decode_dev(d_state_, d_state_ + 1, cache_);
            } catch (...) {
                cudaGraph_t discarded = nullptr;
                cudaStreamEndCapture(stream_, &discarded);  // bring the stream back to a working state
                if (discarded) cudaGraphDestroy(discarded);
                throw;
            }
            CUDA_CHECK(cudaStreamEndCapture(stream_, &graph));
        }
        graph_ = graph;
        CUDA_CHECK(cudaGraphInstantiate(&exec_, graph_, 0));
    }

    void release() noexcept {
        if (stream_) cudaStreamSynchronize(stream_);
        if (exec_) cudaGraphExecDestroy(exec_);
        if (graph_) cudaGraphDestroy(graph_);
        exec_ = nullptr;
        graph_ = nullptr;
        logits_ = Tensor();  // arena tensor: releasing it does nothing, but do it before destroying the arena
        if (h_logits_) cudaFreeHost(h_logits_);
        if (h_state_) cudaFreeHost(h_state_);
        if (d_state_) cudaFree(d_state_);
        if (stream_) cudaStreamDestroy(stream_);
        h_logits_ = nullptr;
        h_state_ = nullptr;
        d_state_ = nullptr;
        stream_ = nullptr;
    }

    const Model& model_;
    KVCache& cache_;
    int vocab_;
    GpuArena arena_;      // declared BEFORE logits_: destroyed after it
    Tensor logits_;       // [1 x vocab] in the arena: the graph writes here
    cudaStream_t stream_ = nullptr;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t exec_ = nullptr;
    int* d_state_ = nullptr;  // {token, position} on the device
    int* h_state_ = nullptr;  // the same on the host (pinned memory)
    float* h_logits_ = nullptr;
};

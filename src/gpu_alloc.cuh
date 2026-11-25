#pragma once
// Caching GPU memory allocator.
//
// Why it exists: cudaMalloc is expensive, and cudaFree also synchronizes the whole device. The model
// creates hundreds of temporary tensors per token, so without a pool the time goes into allocating
// memory instead of computing. Here a freed block stays in a list and is handed to the next request
// of the same size.
//
// Sizes are rounded up to quarter-octave steps (512, 640, 768, 896, 1024, 1280, ...), so memory
// waste never exceeds 25%. One host thread, one CUDA stream (the default one): reusing a block is
// safe because all operations in a stream are ordered.
#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "cuda_utils.cuh"

// Arena: one large GPU memory block handed out by bumping a pointer (no reuse). CUDA graphs need it:
// every intermediate tensor of a captured step has to live as long as the graph, and its address
// must not go back to the pool, otherwise someone else would overwrite buffers the graph still
// references.
// The graph owns the arena; tensors allocated from it must not outlive the arena.
class GpuArena {
public:
    explicit GpuArena(size_t bytes);  // allocates the memory (outside capture!) and registers with the allocator
    ~GpuArena();
    GpuArena(const GpuArena&) = delete;
    GpuArena& operator=(const GpuArena&) = delete;

    size_t capacity() const { return capacity_; }
    size_t used() const { return offset_; }
    bool contains(const void* p) const {
        const char* c = static_cast<const char*>(p);
        return c >= base_ && c < base_ + capacity_;
    }

    void* bump(size_t bytes) {
        const size_t aligned = (bytes + 255) & ~static_cast<size_t>(255);
        if (offset_ + aligned > capacity_)
            throw std::runtime_error("GpuArena exhausted: need " + std::to_string(aligned) + " bytes, " +
                                     std::to_string(capacity_ - offset_) + " left of " + std::to_string(capacity_));
        void* p = base_ + offset_;
        offset_ += aligned;
        return p;
    }

private:
    char* base_ = nullptr;
    size_t capacity_ = 0;
    size_t offset_ = 0;
};

class GpuAllocator {
public:
    struct Stats {
        size_t requests = 0;      // how many times memory was requested
        size_t pool_hits = 0;     // how many were served from the pool without cudaMalloc
        size_t cuda_mallocs = 0;  // real cudaMalloc calls
        size_t cuda_frees = 0;    // real cudaFree calls
        size_t arena_allocs = 0;  // how many blocks were served from an arena
        size_t bytes_in_use = 0;  // currently held by tensors
        size_t bytes_cached = 0;  // sitting in the pool
    };

    // Deliberately "leaked" at process exit: the destructor of a static object could run after the
    // CUDA context has already been destroyed.
    static GpuAllocator& instance() {
        static GpuAllocator* a = new GpuAllocator();
        return *a;
    }

    void* allocate(size_t bytes) {
        std::lock_guard<std::mutex> lock(mu_);
        ++stats_.requests;
        if (bytes == 0) return nullptr;
        if (active_arena_) {  // graph capture: memory comes from the arena, no cudaMalloc, never returns to the pool
            ++stats_.arena_allocs;
            return active_arena_->bump(bytes);
        }
        const size_t size = pooling_ ? bucket_size(bytes) : bytes;
        if (pooling_) {
            auto it = free_.find(size);
            if (it != free_.end() && !it->second.empty()) {
                void* p = it->second.back();
                it->second.pop_back();
                ++stats_.pool_hits;
                stats_.bytes_cached -= size;
                stats_.bytes_in_use += size;
                return p;
            }
        }
        void* p = nullptr;
        cudaError_t err = cudaMalloc(&p, size);
        if (err != cudaSuccess && pooling_ && stats_.bytes_cached > 0) {
            cudaGetLastError();  // clear the error; release the cache and try again
            trim_locked();
            err = cudaMalloc(&p, size);
        }
        CUDA_CHECK(err);
        ++stats_.cuda_mallocs;
        stats_.bytes_in_use += size;
        return p;
    }

    // bytes is the same size that was passed to allocate.
    void release(void* p, size_t bytes) noexcept {
        if (!p) return;
        std::lock_guard<std::mutex> lock(mu_);
        if (is_arena_pointer(p)) return;  // arena memory is only released by the arena itself
        const size_t size = pooling_ ? bucket_size(bytes) : bytes;
        stats_.bytes_in_use -= size;
        if (pooling_) {
            free_[size].push_back(p);
            stats_.bytes_cached += size;
        } else {
            cudaFree(p);
            ++stats_.cuda_frees;
        }
    }

    // Enable/disable the pool (disabled = cudaMalloc/cudaFree for every tensor, for comparison).
    // Only switch while no tensors are alive: the block size depends on the mode.
    void set_pooling(bool on) {
        std::lock_guard<std::mutex> lock(mu_);
        if (stats_.bytes_in_use != 0) throw std::runtime_error("GpuAllocator::set_pooling: tensors are still alive");
        trim_locked();
        pooling_ = on;
    }
    bool pooling() const { return pooling_; }

    // All following allocate() calls are served from the arena (until end_arena). Graph capture only.
    void begin_arena(GpuArena* a) {
        std::lock_guard<std::mutex> lock(mu_);
        if (active_arena_) throw std::runtime_error("GpuAllocator::begin_arena: an arena is already active");
        active_arena_ = a;
    }
    void end_arena() {
        std::lock_guard<std::mutex> lock(mu_);
        active_arena_ = nullptr;
    }
    bool arena_active() const {
        std::lock_guard<std::mutex> lock(mu_);
        return active_arena_ != nullptr;
    }

    // Returns the cache to the system.
    void trim() {
        std::lock_guard<std::mutex> lock(mu_);
        trim_locked();
    }

    Stats stats() const {
        std::lock_guard<std::mutex> lock(mu_);
        return stats_;
    }
    // Resets the call counters (but not the current amount of memory).
    void reset_counters() {
        std::lock_guard<std::mutex> lock(mu_);
        stats_.requests = stats_.pool_hits = stats_.cuda_mallocs = stats_.cuda_frees = 0;
    }

    // Block size for a request: the nearest quarter-octave step (at least 512 bytes).
    static size_t bucket_size(size_t bytes) {
        if (bytes <= 512) return 512;
        size_t p = 512;
        while (p * 2 < bytes) p *= 2;  // p < bytes <= 2p
        const size_t step = p / 4;
        return ((bytes + step - 1) / step) * step;
    }

private:
    friend class GpuArena;

    void register_arena(const GpuArena* a) {
        std::lock_guard<std::mutex> lock(mu_);
        arenas_.push_back(a);
    }
    void unregister_arena(const GpuArena* a) {
        std::lock_guard<std::mutex> lock(mu_);
        for (size_t i = 0; i < arenas_.size(); ++i)
            if (arenas_[i] == a) {
                arenas_.erase(arenas_.begin() + static_cast<std::ptrdiff_t>(i));
                break;
            }
        if (active_arena_ == a) active_arena_ = nullptr;
    }
    bool is_arena_pointer(const void* p) const {
        for (const GpuArena* a : arenas_)
            if (a->contains(p)) return true;
        return false;
    }

    void trim_locked() {
        for (auto& kv : free_) {
            for (void* p : kv.second) {
                cudaFree(p);
                ++stats_.cuda_frees;
            }
            stats_.bytes_cached -= kv.first * kv.second.size();
            kv.second.clear();
        }
    }

    mutable std::mutex mu_;
    std::unordered_map<size_t, std::vector<void*>> free_;
    Stats stats_;
    bool pooling_ = true;
    GpuArena* active_arena_ = nullptr;
    std::vector<const GpuArena*> arenas_;
};

inline GpuArena::GpuArena(size_t bytes) : capacity_(bytes) {
    if (bytes == 0) throw std::invalid_argument("GpuArena: capacity must be positive");
    void* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, bytes));
    base_ = static_cast<char*>(p);
    GpuAllocator::instance().register_arena(this);
}

inline GpuArena::~GpuArena() {
    GpuAllocator::instance().unregister_arena(this);
    cudaFree(base_);
}

// While this object is alive, all memory allocations go to the arena.
class ArenaScope {
public:
    explicit ArenaScope(GpuArena& a) { GpuAllocator::instance().begin_arena(&a); }
    ~ArenaScope() { GpuAllocator::instance().end_arena(); }
    ArenaScope(const ArenaScope&) = delete;
    ArenaScope& operator=(const ArenaScope&) = delete;
};

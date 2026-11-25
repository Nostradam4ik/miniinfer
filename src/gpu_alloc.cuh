#pragma once
// Кешуючий алокатор пам'яті GPU.
//
// Чому він потрібен: cudaMalloc дорогий, а cudaFree ще й синхронізує весь пристрій. Модель
// створює сотні тимчасових тензорів на кожен токен, тож без пулу час іде на виділення пам'яті,
// а не на обчислення. Тут звільнений блок лишається в списку і віддається наступному запиту
// такого ж розміру.
//
// Розміри округлюються вгору до сходинок 1/4 октави (512, 640, 768, 896, 1024, 1280, ...),
// тому втрати пам'яті не перевищують 25 %. Один потік виконання, один CUDA stream
// (стандартний): повторне використання блоку безпечне, бо всі операції в потоці впорядковані.
#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "cuda_utils.cuh"

// Арена: один великий блок пам'яті GPU, з якого видається пам'ять "бампом" (без повторного
// використання). Потрібна для CUDA-графів: усі проміжні тензори кроку, захопленого в граф, мають
// жити стільки, скільки живе граф, і їхні адреси не можуть потрапити назад у пул, інакше хтось
// інший перезапише буфери, на які граф ще посилається. Арена належить графові; тензори, видані
// з неї, не повинні пережити арену.
class GpuArena {
public:
    explicit GpuArena(size_t bytes);  // виділяє пам'ять (поза захопленням!) і реєструється в алокаторі
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
        size_t requests = 0;      // скільки разів просили пам'ять
        size_t pool_hits = 0;     // скільки віддано з пулу без cudaMalloc
        size_t cuda_mallocs = 0;  // реальні виклики cudaMalloc
        size_t cuda_frees = 0;    // реальні виклики cudaFree
        size_t arena_allocs = 0;  // скільки блоків видано з арени
        size_t bytes_in_use = 0;  // зараз віддано тензорам
        size_t bytes_cached = 0;  // лежить у пулі
    };

    // Намірено "протікає" при завершенні процесу: деструктор статичного об'єкта міг би
    // спрацювати вже після знищення контексту CUDA.
    static GpuAllocator& instance() {
        static GpuAllocator* a = new GpuAllocator();
        return *a;
    }

    void* allocate(size_t bytes) {
        std::lock_guard<std::mutex> lock(mu_);
        ++stats_.requests;
        if (bytes == 0) return nullptr;
        if (active_arena_) {  // захоплення графа: пам'ять із арени, без cudaMalloc і без повернення в пул
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
            cudaGetLastError();  // скидаємо помилку; віддаємо кеш і пробуємо ще раз
            trim_locked();
            err = cudaMalloc(&p, size);
        }
        CUDA_CHECK(err);
        ++stats_.cuda_mallocs;
        stats_.bytes_in_use += size;
        return p;
    }

    // bytes - той самий розмір, що передавали в allocate.
    void release(void* p, size_t bytes) noexcept {
        if (!p) return;
        std::lock_guard<std::mutex> lock(mu_);
        if (is_arena_pointer(p)) return;  // пам'ять арени звільняє лише сама арена
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

    // Увімкнути/вимкнути пул (вимкнений = cudaMalloc/cudaFree на кожен тензор, для порівняння).
    // Перемикати можна лише коли немає живих тензорів: розмір блоку залежить від режиму.
    void set_pooling(bool on) {
        std::lock_guard<std::mutex> lock(mu_);
        if (stats_.bytes_in_use != 0)
            throw std::runtime_error("GpuAllocator::set_pooling: tensors are still alive");
        trim_locked();
        pooling_ = on;
    }
    bool pooling() const { return pooling_; }

    // Усі наступні allocate() видаються з арени (до end_arena). Лише для захоплення графа.
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

    // Повертає кеш системі.
    void trim() {
        std::lock_guard<std::mutex> lock(mu_);
        trim_locked();
    }

    Stats stats() const {
        std::lock_guard<std::mutex> lock(mu_);
        return stats_;
    }
    // Обнуляє лічильники викликів (але не поточний обсяг пам'яті).
    void reset_counters() {
        std::lock_guard<std::mutex> lock(mu_);
        stats_.requests = stats_.pool_hits = stats_.cuda_mallocs = stats_.cuda_frees = 0;
    }

    // Розмір блоку для запиту: найближча сходинка 1/4 октави (не менше 512 байтів).
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

// Під час життя цього об'єкта всі виділення пам'яті йдуть в арену.
class ArenaScope {
public:
    explicit ArenaScope(GpuArena& a) { GpuAllocator::instance().begin_arena(&a); }
    ~ArenaScope() { GpuAllocator::instance().end_arena(); }
    ArenaScope(const ArenaScope&) = delete;
    ArenaScope& operator=(const ArenaScope&) = delete;
};

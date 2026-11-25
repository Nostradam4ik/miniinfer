#pragma once
// CUDA-граф для кроку декодування.
//
// Навіщо: один токен це ~180 коротких ядер, і вартість запуску кожного (~7 мкс) порівнянна з його
// роботою. Граф записує всю послідовність один раз, а потім запускається однією командою: драйвер
// отримує готовий список ядер і запускає їх на GPU без участі хоста між ними.
//
// Як це працює:
//   1. Токен і позиція лежать у пам'яті пристрою (d_state_); ядра читають їх звідти, тож один і той
//      самий граф підходить для будь-якого кроку. Перед кожним запуском хост копіює туди 8 байтів.
//   2. Крок захоплюється тими самими операціями, що й звичайний шлях (forward_decode_dev), тож
//      результат має збігатися з ним, а не бути окремою реалізацією.
//   3. Усі проміжні тензори кроку видаються з арени, що належить графові, щоб їхні адреси не
//      потрапили назад у пул і не були перезаписані чимось іншим між запусками.
//
// Обмеження: граф прив'язаний до конкретних об'єктів Model і KVCache (їхні адреси вшиті в нього),
// обидва мають жити довше за граф. Підходить лише для кроків з одним токеном.
#include <cstring>
#include <stdexcept>
#include <vector>
#include "model.cuh"

class DecodeGraph {
public:
    // arena_bytes - скільки пам'яті відвести під проміжні тензори кроку (мегабайти для цієї моделі).
    // Захоплення відбувається в конструкторі; кеш при цьому не змінюється (pos лишається cache.length).
    DecodeGraph(const Model& model, KVCache& cache, size_t arena_bytes = 16u << 20)
        : model_(model), cache_(cache), vocab_(model.cfg.vocab_size), arena_(arena_bytes) {
        CUDA_CHECK(cudaStreamCreate(&stream_));  // блокуючий потік: впорядкований зі стандартним
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

    // Один крок: обробляє `token` на позиції cache.length, дописує K, V у кеш і збільшує cache.length.
    // Записує логіти наступного токена (vocab_size чисел) у `logits`.
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

    // Діагностика: запускає граф n разів підряд без копіювань і повертає середній час запуску (мс).
    // Токен і позиція лишаються тими, що були в останньому step(); кеш і cache.length не змінюються
    // (граф щоразу перезаписує той самий рядок кешу).
    double time_replays(int n) {
        GpuTimer timer;
        timer.start();  // подія в стандартному потоці; граф у блокуючому потоці впорядкований із ним
        for (int i = 0; i < n; ++i) CUDA_CHECK(cudaGraphLaunch(exec_, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        return timer.stop() / n;
    }

    int vocab_size() const { return vocab_; }
    size_t arena_used() const { return arena_.used(); }
    size_t arena_capacity() const { return arena_.capacity(); }

private:
    void capture() {
        // Прогрів: один звичайний прогін того самого кроку, щоб усі ядра були завантажені до захоплення
        // (ліниве завантаження модулів під час захоплення може не спрацювати). Він пише рядок кешу на
        // позиції cache.length, але справжній крок перепише його.
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
                cudaStreamEndCapture(stream_, &discarded);  // повертаємо потік у робочий стан
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
        logits_ = Tensor();  // тензор з арени: звільнення нічого не робить, але зробимо це до знищення арени
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
    GpuArena arena_;      // оголошена ПЕРЕД logits_: знищується після нього
    Tensor logits_;       // [1 x vocab] в арені: сюди пише граф
    cudaStream_t stream_ = nullptr;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t exec_ = nullptr;
    int* d_state_ = nullptr;  // {токен, позиція} на пристрої
    int* h_state_ = nullptr;  // те саме на хості (закріплена пам'ять)
    float* h_logits_ = nullptr;
};

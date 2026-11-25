#pragma once
// Операції на GPU. Усі матриці row-major, float.
#include "tensor.cuh"

// Naive/Tiled/Reg - для великих M (prefill, бенчмарки). Gemv - для малих M (декодування по
// одному токену): ядро, обмежене пропускною здатністю пам'яті, а не латентністю плиток.
// Auto обирає Gemv при M <= kGemvMaxRows і Reg інакше.
enum class MatmulKind { Naive, Tiled, Reg, Gemv, Auto };
constexpr int kGemvMaxRows = 4;

// Низькорівневі запуски на сирих вказівниках (для бенчмарків).
// C[M x N] = A[M x K] * B[K x N]
void launch_matmul(MatmulKind kind, const float* A, const float* B, float* C, int M,
                   int N, int K);

// Високорівневий інтерфейс. Кидає std::invalid_argument, якщо форми не збігаються.
// B - матриця ваг у fp32 (Tensor), fp16 або bf16 (LowpMatrix); перетворення на WeightRef неявне.
// Naive і Tiled приймають лише fp32. Накопичення завжди в fp32.
Tensor matmul(const Tensor& A, WeightRef B, MatmulKind kind = MatmulKind::Auto);

// out[t] = table[ids[t]]. table: [V x D], результат: [T x D].
// Кидає std::invalid_argument, якщо якийсь id поза [0, V).
Tensor embedding(WeightRef table, const std::vector<int>& ids);

// Те саме, але id лежать у пам'яті пристрою (T штук, d_ids). Без копіювання з хоста й синхронізації,
// тож придатне для CUDA-графа. Недопустимий id дає рядок із NaN (хост його не бачить).
Tensor embedding_dev(WeightRef table, const int* d_ids, int T);

// RoPE: x [T x (num_heads*head_dim)]. Рядок t має позицію pos_offset + t
// (pos_offset знадобиться з KV-кешем). head_dim має бути парним.
Tensor rope(const Tensor& x, int num_heads, int head_dim, float theta, int pos_offset = 0);

// Причинний attention з GQA (кілька голів Q ділять одну голову K/V).
// q: [T x (H*hd)], k, v: [S x (KV*hd)], результат [T x (H*hd)].
// Запит t (позиція pos_offset + t) бачить ключі 0..pos_offset+t, тож S >= pos_offset + T.
// Обмеження: S <= kMaxAttentionKeys (оцінки зберігаються в shared memory).
constexpr int kMaxAttentionKeys = 12288;
// d_pos != nullptr: позиція першого запиту береться з пам'яті пристрою (одне int), а pos_offset
// ігнорується. Це потрібно для CUDA-графа. Хост тоді не перевіряє позицію; ядро заповнює вихід NaN,
// якщо позиція виходить за межі кешу k, v.
Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v, int num_heads,
                 int num_kv_heads, int head_dim, int pos_offset = 0, const int* d_pos = nullptr);

// ---- Злиті операції (менше запусків ядер на токен) -------------------------------------------
// Для малих M (<= kGemvMaxRows, тобто декодування) кожна виконується одним запуском.
// Для більших M діє безпечний резерв: ті самі результати через окремі операції.

// rmsnorm(x, norm_w) * W. x: [M x K], norm_w: [K], W: [K x N].
Tensor rmsnorm_matmul(const Tensor& x, const Tensor& norm_w, float eps, WeightRef W);

// x * W + residual. x: [M x K], W: [K x N], residual: [M x N].
Tensor matmul_residual(const Tensor& x, WeightRef W, const Tensor& residual);

// silu(h * Wg) * (h * Wu), де h = rmsnorm(x, norm_w). x: [M x K], Wg, Wu: [K x N].
Tensor rmsnorm_gate_up(const Tensor& x, const Tensor& norm_w, float eps, WeightRef Wg, WeightRef Wu);

// Результат матмула x * Wqkv ділиться на Q, K, V за формою [T x (H + 2*KV)*hd]. RoPE застосовується
// до Q і K, повернутий тензор - обернений Q [T x H*hd]; обернений K і V записуються в кеш
// (cache_k, cache_v: [max_seq x KV*hd]) у рядки pos_offset .. pos_offset + T - 1.
// d_pos != nullptr: позиція (pos_offset) береться з пам'яті пристрою, див. attention(). Рядки кешу поза
// межами при цьому не пишуться (ядро їх пропускає), а attention тоді повертає NaN.
Tensor rope_qkv_to_cache(const Tensor& qkv, int num_heads, int num_kv_heads, int head_dim, float theta,
                         int pos_offset, Tensor& cache_k, Tensor& cache_v, const int* d_pos = nullptr);

// Сирий запуск ядра RoPE + запис у кеш на готових буферах (для тестів меж і бенчмарків). Без перевірок
// аргументів: qkv [T x (H+2*KV)*hd], q_out [T x H*hd], cache_k/cache_v [cache_rows x KV*hd].
void launch_rope_qkv_cache(const float* qkv, float* q_out, float* cache_k, float* cache_v, int T, int num_heads,
                           int num_kv_heads, int head_dim, float theta, int pos_offset, const int* d_pos,
                           int cache_rows);

// Склеює 2D-тензори з однаковою кількістю рядків вздовж стовпців: [R x C1], [R x C2] -> [R x (C1 + C2)].
Tensor concat_cols(const std::vector<const Tensor*>& parts);

// Копіює всі рядки src у dst, починаючи з рядка row_offset (для KV-кешу).
// Обидва тензори 2D з однаковою кількістю стовпців. Кидає std::invalid_argument при виході за межі.
void write_rows(Tensor& dst, const Tensor& src, int row_offset);

// Копія рядків [start, start + count) тензора x.
Tensor slice_rows(const Tensor& x, int start, int count);

// Транспонування 2D-тензора [R x C] -> [C x R].
Tensor transpose(const Tensor& x);

// Поелементна сума a + b (residual-зв'язки). Форми мають збігатися.
Tensor add(const Tensor& a, const Tensor& b);

// MLP-активація Llama: silu(gate) * up, поелементно. silu(x) = x / (1 + exp(-x)).
Tensor silu_mul(const Tensor& gate, const Tensor& up);

// RMSNorm по рядках: y = x / sqrt(mean(x^2) + eps) * w. x: [T x D], w: [D].
Tensor rmsnorm(const Tensor& x, const Tensor& w, float eps);

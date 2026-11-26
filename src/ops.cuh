#pragma once
// GPU operations. All matrices are row-major float.
#include "tensor.cuh"

// Naive/Tiled/Reg are for large M (prefill, benchmarks). Gemv is for small M (single-token
// decoding): a kernel bound by memory bandwidth rather than by tile latency.
// Wmma runs on tensor cores: bf16 weights only, with limits on the shapes (see wmma.h), and it rounds A to
// bf16, so its results differ from the other kernels. Auto picks Gemv when M <= kGemvMaxRows and Reg
// otherwise; it never picks Wmma.
enum class MatmulKind { Naive, Tiled, Reg, Gemv, Wmma, Auto };
constexpr int kGemvMaxRows = 4;

// Low-level launches on raw pointers (for benchmarks).
// C[M x N] = A[M x K] * B[K x N]
void launch_matmul(MatmulKind kind, const float* A, const float* B, float* C, int M, int N, int K);

// High-level interface. Throws std::invalid_argument if the shapes do not match.
// B is a weight matrix in fp32 (Tensor), fp16 or bf16 (LowpMatrix); conversion to WeightRef is implicit.
// Naive and Tiled accept only fp32. Accumulation is always in fp32.
Tensor matmul(const Tensor& A, WeightRef B, MatmulKind kind = MatmulKind::Auto);

// out[t] = table[ids[t]]. table: [V x D], result: [T x D].
// Throws std::invalid_argument if any id is outside [0, V).
Tensor embedding(WeightRef table, const std::vector<int>& ids);

// Same, but the ids are in device memory (T of them, d_ids). No host copy and no synchronization,
// so it works inside a CUDA graph. An invalid id yields a row of NaN (the host does not see the id).
Tensor embedding_dev(WeightRef table, const int* d_ids, int T);

// RoPE: x [T x (num_heads*head_dim)]. Row t has position pos_offset + t
// (pos_offset is used by the KV cache). head_dim must be even.
Tensor rope(const Tensor& x, int num_heads, int head_dim, float theta, int pos_offset = 0);

// Causal attention with GQA (several Q heads share one K/V head).
// q: [T x (H*hd)], k, v: [S x (KV*hd)], result [T x (H*hd)].
// Query t (position pos_offset + t) sees keys 0..pos_offset+t, so S >= pos_offset + T.
// Limit: S <= kMaxAttentionKeys (the scores are kept in shared memory).
constexpr int kMaxAttentionKeys = 12288;
// d_pos != nullptr: the position of the first query is read from device memory (one int) and pos_offset
// is ignored. Needed for the CUDA graph. The host then cannot validate the position; the kernel fills
// the output with NaN if the position falls outside the k, v cache.
Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v, int num_heads, int num_kv_heads, int head_dim,
                 int pos_offset = 0, const int* d_pos = nullptr);

// Whether attention() runs the kernel with one block per (query row, KV head), which loads K and V once for
// all query heads that share the KV head. key_capacity = pos_offset + T. Both kernels give the same bits.
bool attention_uses_shared_kv(int T, int key_capacity, int num_heads, int num_kv_heads, int head_dim, bool device_pos);

// ---- Fused operations (fewer kernel launches per token) --------------------------------------
// For small M (<= kGemvMaxRows, i.e. decoding) each one is a single launch.
// For larger M there is a safe fallback: the same results through separate operations. Their matmuls use
// the kernel given by `large`.

// rmsnorm(x, norm_w) * W. x: [M x K], norm_w: [K], W: [K x N].
Tensor rmsnorm_matmul(const Tensor& x, const Tensor& norm_w, float eps, WeightRef W,
                      MatmulKind large = MatmulKind::Reg);

// x * W + residual. x: [M x K], W: [K x N], residual: [M x N].
Tensor matmul_residual(const Tensor& x, WeightRef W, const Tensor& residual, MatmulKind large = MatmulKind::Reg);

// silu(h * Wg) * (h * Wu), where h = rmsnorm(x, norm_w). x: [M x K], Wg, Wu: [K x N].
Tensor rmsnorm_gate_up(const Tensor& x, const Tensor& norm_w, float eps, WeightRef Wg, WeightRef Wu,
                       MatmulKind large = MatmulKind::Reg);

// The result of the matmul x * Wqkv is split into Q, K, V with shape [T x (H + 2*KV)*hd]. RoPE is applied
// to Q and K; the returned tensor is the rotated Q [T x H*hd]; the rotated K and V are written into the
// cache (cache_k, cache_v: [max_seq x KV*hd]) at rows pos_offset .. pos_offset + T - 1.
// d_pos != nullptr: the position (pos_offset) is read from device memory, see attention(). Cache rows out
// of range are then not written (the kernel skips them), and attention returns NaN.
Tensor rope_qkv_to_cache(const Tensor& qkv, int num_heads, int num_kv_heads, int head_dim, float theta, int pos_offset,
                         Tensor& cache_k, Tensor& cache_v, const int* d_pos = nullptr);

// Raw launch of the RoPE + cache write kernel on ready buffers (for bounds tests and benchmarks). No
// argument checks: qkv [T x (H+2*KV)*hd], q_out [T x H*hd], cache_k/cache_v [cache_rows x KV*hd].
void launch_rope_qkv_cache(const float* qkv, float* q_out, float* cache_k, float* cache_v, int T, int num_heads,
                           int num_kv_heads, int head_dim, float theta, int pos_offset, const int* d_pos,
                           int cache_rows);

// Concatenates 2D tensors with the same number of rows along columns: [R x C1], [R x C2] -> [R x (C1 + C2)].
Tensor concat_cols(const std::vector<const Tensor*>& parts);

// Copies all rows of src into dst starting at row row_offset (for the KV cache).
// Both tensors are 2D with the same number of columns. Throws std::invalid_argument if out of bounds.
void write_rows(Tensor& dst, const Tensor& src, int row_offset);

// A copy of rows [start, start + count) of tensor x.
Tensor slice_rows(const Tensor& x, int start, int count);

// Transpose of a 2D tensor [R x C] -> [C x R].
Tensor transpose(const Tensor& x);

// Elementwise sum a + b (residual connections). The shapes must match.
Tensor add(const Tensor& a, const Tensor& b);

// Llama MLP activation: silu(gate) * up, elementwise. silu(x) = x / (1 + exp(-x)).
Tensor silu_mul(const Tensor& gate, const Tensor& up);

// Row-wise RMSNorm: y = x / sqrt(mean(x^2) + eps) * w. x: [T x D], w: [D].
Tensor rmsnorm(const Tensor& x, const Tensor& w, float eps);

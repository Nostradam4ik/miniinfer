#pragma once
// CPU-еталони: повільні, прості, максимально точні. Проти них перевіряємо ядра GPU.
#include <cmath>
#include <vector>

// C[M x N] = A[M x K] * B[K x N], row-major. Накопичення в double для точності.
inline std::vector<float> cpu_matmul(const std::vector<float>& A,
                                     const std::vector<float>& B, int M, int N, int K) {
    std::vector<float> C(static_cast<size_t>(M) * N);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            double s = 0.0;
            for (int k = 0; k < K; ++k)
                s += static_cast<double>(A[static_cast<size_t>(i) * K + k]) *
                     B[static_cast<size_t>(k) * N + j];
            C[static_cast<size_t>(i) * N + j] = static_cast<float>(s);
        }
    return C;
}

// RMSNorm: y = x / sqrt(mean(x^2) + eps) * w, по кожному рядку x[T x D].
inline std::vector<float> cpu_rmsnorm(const std::vector<float>& x,
                                      const std::vector<float>& w, int T, int D,
                                      float eps) {
    std::vector<float> y(static_cast<size_t>(T) * D);
    for (int t = 0; t < T; ++t) {
        double ss = 0.0;
        for (int d = 0; d < D; ++d) {
            double v = x[static_cast<size_t>(t) * D + d];
            ss += v * v;
        }
        double rstd = 1.0 / std::sqrt(ss / D + eps);
        for (int d = 0; d < D; ++d)
            y[static_cast<size_t>(t) * D + d] =
                static_cast<float>(x[static_cast<size_t>(t) * D + d] * rstd * w[d]);
    }
    return y;
}

// RoPE (стиль Llama "rotate_half"): у кожній голові елемент i обертається разом з
// i + hd/2 на кут pos * theta^(-2i/hd). x: [T x (heads*hd)], позиція рядка t = pos_offset + t.
inline std::vector<float> cpu_rope(const std::vector<float>& x, int T, int heads, int hd,
                                   float theta, int pos_offset) {
    std::vector<float> y(x.size());
    const int half = hd / 2;
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < heads; ++h) {
            const size_t base = (static_cast<size_t>(t) * heads + h) * hd;
            for (int i = 0; i < half; ++i) {
                double ang = (pos_offset + t) * std::pow(static_cast<double>(theta), -2.0 * i / hd);
                double c = std::cos(ang), s = std::sin(ang);
                double a = x[base + i], b = x[base + i + half];
                y[base + i] = static_cast<float>(a * c - b * s);
                y[base + i + half] = static_cast<float>(b * c + a * s);
            }
        }
    return y;
}

// Причинний attention з GQA. q: [T x H*hd], k, v: [S x KV*hd].
// Запит t має абсолютну позицію pos_offset + t і бачить ключі 0..pos_offset+t.
inline std::vector<float> cpu_attention(const std::vector<float>& q, const std::vector<float>& k,
                                        const std::vector<float>& v, int T, int S, int H, int KV,
                                        int hd, int pos_offset) {
    std::vector<float> out(static_cast<size_t>(T) * H * hd);
    const double scale = 1.0 / std::sqrt(static_cast<double>(hd));
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < H; ++h) {
            const int kvh = h / (H / KV);
            const int nvalid = pos_offset + t + 1;
            std::vector<double> p(nvalid);
            double mx = -INFINITY;
            for (int j = 0; j < nvalid; ++j) {
                double dot = 0;
                for (int d = 0; d < hd; ++d)
                    dot += static_cast<double>(q[(static_cast<size_t>(t) * H + h) * hd + d]) *
                           k[(static_cast<size_t>(j) * KV + kvh) * hd + d];
                p[j] = dot * scale;
                mx = std::fmax(mx, p[j]);
            }
            double sum = 0;
            for (double& e : p) { e = std::exp(e - mx); sum += e; }
            for (int d = 0; d < hd; ++d) {
                double acc = 0;
                for (int j = 0; j < nvalid; ++j)
                    acc += p[j] / sum * v[(static_cast<size_t>(j) * KV + kvh) * hd + d];
                out[(static_cast<size_t>(t) * H + h) * hd + d] = static_cast<float>(acc);
            }
        }
    (void)S;
    return out;
}

// Максимальна абсолютна різниця. NaN/inf у будь-якому масиві => INFINITY
// (std::fmax ігнорує NaN, тому порівнювати треба явно).
inline double max_abs_diff(const std::vector<float>& x, const std::vector<float>& y) {
    if (x.size() != y.size()) return INFINITY;
    double m = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        double d = std::fabs(static_cast<double>(x[i]) - y[i]);
        if (std::isnan(d) || std::isinf(d)) return INFINITY;
        if (d > m) m = d;
    }
    return m;
}

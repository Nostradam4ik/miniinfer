#pragma once
// Вибір наступного токена з логітів (CPU). Працює на копії одного рядка логітів: ~50 тис.
// чисел, це мізерно порівняно з прогоном моделі.
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

struct SamplerConfig {
    float temperature = 0.f;  // 0 = greedy (завжди найімовірніший токен)
    int top_k = 0;            // 0 = вимкнено; інакше лише k найімовірніших
    float top_p = 1.f;        // ядро: найменша множина з сумарною ймовірністю >= top_p
    unsigned seed = 0;
};

class Sampler {
public:
    explicit Sampler(SamplerConfig cfg) : cfg_(cfg), rng_(cfg.seed) {
        if (!(cfg.temperature >= 0.f) || !std::isfinite(cfg.temperature))
            throw std::invalid_argument("Sampler: temperature must be finite and >= 0");
        if (cfg.top_k < 0) throw std::invalid_argument("Sampler: top_k must be >= 0");
        if (!(cfg.top_p > 0.f && cfg.top_p <= 1.f))
            throw std::invalid_argument("Sampler: top_p must be in (0, 1]");
    }

    int sample(const std::vector<float>& logits) {
        if (logits.empty()) throw std::invalid_argument("Sampler: empty logits");
        for (float v : logits)
            if (std::isnan(v)) throw std::runtime_error("Sampler: logits contain NaN");

        if (cfg_.temperature == 0.f) return argmax(logits);

        // Кандидати, від найбільшого логіта (стабільно: при рівності раніший індекс).
        std::vector<int> idx(logits.size());
        std::iota(idx.begin(), idx.end(), 0);
        auto better = [&](int a, int b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); };
        size_t keep = idx.size();
        if (cfg_.top_k > 0) keep = std::min(keep, static_cast<size_t>(cfg_.top_k));
        if (keep < idx.size()) {
            std::partial_sort(idx.begin(), idx.begin() + keep, idx.end(), better);
            idx.resize(keep);
        } else {
            std::sort(idx.begin(), idx.end(), better);
        }

        // Softmax з температурою; віднімаємо максимум для стійкості. -inf дає ймовірність 0.
        const double mx = logits[idx[0]];
        if (std::isinf(mx) && mx < 0) throw std::runtime_error("Sampler: all logits are -inf");
        std::vector<double> p(idx.size());
        double sum = 0;
        for (size_t i = 0; i < idx.size(); ++i) {
            p[i] = std::exp((static_cast<double>(logits[idx[i]]) - mx) / cfg_.temperature);
            sum += p[i];
        }
        for (double& v : p) v /= sum;

        // top-p: залишаємо найменший префікс із сумою >= top_p (мінімум один токен).
        if (cfg_.top_p < 1.f) {
            double cum = 0;
            size_t n = 0;
            while (n < p.size()) {
                cum += p[n++];
                if (cum >= cfg_.top_p) break;
            }
            p.resize(n);
            idx.resize(n);
            double s = 0;
            for (double v : p) s += v;
            for (double& v : p) v /= s;
        }

        std::uniform_real_distribution<double> uni(0.0, 1.0);
        double u = uni(rng_), acc = 0;
        for (size_t i = 0; i < p.size(); ++i) {
            acc += p[i];
            if (u < acc) return idx[i];
        }
        return idx.back();  // похибка округлення: u майже 1
    }

    static int argmax(const std::vector<float>& logits) {
        int best = 0;
        for (size_t i = 1; i < logits.size(); ++i)
            if (logits[i] > logits[best]) best = static_cast<int>(i);  // при рівності лишається менший індекс
        return best;
    }

private:
    SamplerConfig cfg_;
    std::mt19937 rng_;
};

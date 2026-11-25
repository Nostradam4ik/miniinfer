// generate.exe - command-line generation. Takes token ids (tokenization is in tools/chat.py).
//
//   generate.exe --ids "504 3575 282 4649 314" --max 30 --temp 0.8 --top-p 0.95 --seed 1
//
// stdout: one "tok <id>" line per new token (flushed, for streaming).
// stderr: diagnostics (time, speed) and errors. Exit code 0 on success.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include "generate.h"

static void usage() {
    std::fprintf(stderr,
                 "Usage: generate --ids \"<id id ...>\" [options]\n"
                 "  --model DIR     model directory (default: model)\n"
                 "  --dtype T       weight format: f32, f16 or bf16 (default: f32)\n"
                 "  --graph         run decode steps through a CUDA graph (faster)\n"
                 "  --max N         new tokens to generate (default: 64)\n"
                 "  --temp T        temperature; 0 = greedy (default: 0)\n"
                 "  --top-k K       keep K most likely tokens; 0 = off (default: 0)\n"
                 "  --top-p P       nucleus sampling, (0, 1] (default: 1)\n"
                 "  --seed S        random seed (default: 0)\n"
                 "  --stop \"<ids>\"  stop on any of these token ids\n");
}

static std::vector<int> parse_ids(const std::string& s) {
    std::vector<int> v;
    std::istringstream is(s);
    std::string tok;
    while (is >> tok) {
        char* end = nullptr;
        long x = std::strtol(tok.c_str(), &end, 10);
        if (*end != '\0') throw std::invalid_argument("not an integer: '" + tok + "'");
        v.push_back(static_cast<int>(x));
    }
    return v;
}

int main(int argc, char** argv) {
    try {
        std::string model_dir = "model";
        DType dtype = DType::F32;
        std::vector<int> ids;
        GenerateOptions opt;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
                return argv[++i];
            };
            if (a == "--help" || a == "-h") {
                usage();
                return 0;
            } else if (a == "--model")
                model_dir = next();
            else if (a == "--dtype") {
                const std::string t = next();
                if (t == "f32")
                    dtype = DType::F32;
                else if (t == "f16")
                    dtype = DType::F16;
                else if (t == "bf16")
                    dtype = DType::BF16;
                else
                    throw std::invalid_argument("--dtype must be f32, f16 or bf16");
            } else if (a == "--graph")
                opt.use_graph = true;
            else if (a == "--ids")
                ids = parse_ids(next());
            else if (a == "--max")
                opt.max_new_tokens = std::stoi(next());
            else if (a == "--temp")
                opt.sampler.temperature = std::stof(next());
            else if (a == "--top-k")
                opt.sampler.top_k = std::stoi(next());
            else if (a == "--top-p")
                opt.sampler.top_p = std::stof(next());
            else if (a == "--seed")
                opt.sampler.seed = static_cast<unsigned>(std::stoul(next()));
            else if (a == "--stop")
                opt.stop_ids = parse_ids(next());
            else
                throw std::invalid_argument("unknown option: " + a);
        }
        if (ids.empty()) {
            usage();
            throw std::invalid_argument("--ids is required");
        }

        Model model = Model::load(model_dir, dtype);
        GenerateResult res = generate(model, ids, opt, [](int t) {
            std::printf("tok %d\n", t);
            std::fflush(stdout);
        });

        const size_t n = res.tokens.size();
        std::fprintf(stderr, "[prefill %zu tokens: %.1f ms | decode %zu tokens: %.1f ms = %.1f tok/s%s%s]\n",
                     ids.size(), res.prefill_ms, n > 0 ? n - 1 : 0, res.decode_ms,
                     res.decode_ms > 0 ? 1000.0 * (n - 1) / res.decode_ms : 0.0,
                     res.graph_capture_ms > 0
                         ? (" | graph capture " + std::to_string(res.graph_capture_ms) + " ms").c_str()
                         : "",
                     res.hit_stop ? " | stopped on stop token" : "");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}

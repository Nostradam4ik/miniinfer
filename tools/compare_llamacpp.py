"""Speed comparison of MiniInfer and llama.cpp on the same model and GPU.

Example:
  .venv\\Scripts\\python.exe tools\\compare_llamacpp.py --llama-bench C:\\path\\llama-bench.exe ^
      --gguf C:\\path\\SmolLM2-135M-BF16.gguf --gguf C:\\path\\SmolLM2-135M-Q8_0.gguf

The configurations are run in turns over several rounds (the order is rotated every round), so that
warm-up and clock changes of a laptop GPU affect all of them equally. For both engines the MEAN time
per token is used (llama-bench reports the mean); the summary is the median over the rounds.
If llama.cpp cannot find the CUDA DLLs, add CUDA\\bin to PATH.
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT)
    if p.returncode != 0:
        raise SystemExit(f"error ({p.returncode}): {' '.join(cmd)}\n{p.stderr[-2000:]}")
    return p.stdout


def prompt_ids(n):
    return " ".join(str(1000 + i * 37 % 20000) for i in range(n))


def mini_decode(exe, steps, graph):
    cmd = [exe, "--dtype", "bf16", "--steps", str(steps)] + (["--graph"] if graph else [])
    return 1000.0 / float(re.search(r"decode per token: .*?mean ([\d.]+) ms", run(cmd)).group(1))


def mini_prefill(exe, n):
    out = run([exe, "--dtype", "bf16", "--steps", "1", "--ids", prompt_ids(n)])
    return n * 1000.0 / float(re.search(r"prefill \(\d+ tokens\): ([\d.]+) ms", out).group(1))


def llama(exe, gguf, n_prompt, n_gen):
    out = run([exe, "-m", gguf, "-ngl", "99", "-p", str(n_prompt), "-n", str(n_gen), "-r", "1", "-o", "json"])
    rows = json.loads(out)
    if len(rows) != 1:
        raise SystemExit(f"expected one llama-bench result, got {len(rows)}")
    return float(rows[0]["avg_ts"])


def gpu_temp():
    return run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"]).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-bench", required=True)
    ap.add_argument("--gguf", action="append", required=True, help="may be given several times")
    ap.add_argument("--mini", default=os.path.join(ROOT, "build", "bench_decode.exe"))
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--prefill", type=int, nargs="+", default=[128, 512])
    ap.add_argument("--pause", type=float, default=3.0, help="pause between runs, s")
    args = ap.parse_args()

    # key: (test, configuration) -> function returning tok/s
    jobs = {
        ("decode", "MiniInfer bf16 + CUDA graph"): lambda: mini_decode(args.mini, args.steps, True),
        ("decode", "MiniInfer bf16 eager"): lambda: mini_decode(args.mini, args.steps, False),
    }
    for g in args.gguf:
        name = "llama.cpp " + os.path.splitext(os.path.basename(g))[0].split("-")[-1]
        jobs[("decode", name)] = lambda g=g: llama(args.llama_bench, g, 0, args.steps)
    for n in args.prefill:
        jobs[(f"prefill {n}", "MiniInfer bf16")] = lambda n=n: mini_prefill(args.mini, n)
        for g in args.gguf:
            name = "llama.cpp " + os.path.splitext(os.path.basename(g))[0].split("-")[-1]
            jobs[(f"prefill {n}", name)] = lambda g=g, n=n: llama(args.llama_bench, g, n, 0)

    keys = list(jobs)
    results = {k: [] for k in keys}
    for r in range(args.rounds):
        print(f"round {r + 1}/{args.rounds}, GPU {gpu_temp()} C", flush=True)
        shift = r * 3 % len(keys)
        for k in keys[shift:] + keys[:shift]:
            time.sleep(args.pause)
            results[k].append(jobs[k]())
            print(f"  {k[0]:<12} {k[1]:<30} {results[k][-1]:9.1f} tok/s", flush=True)

    print(f"\nMedian over {args.rounds} rounds, tok/s (decode: {args.steps} tokens):")
    print("| test | engine | tok/s | min | max |\n|---|---|---|---|---|")
    for k in sorted(keys, key=lambda k: k[0]):
        v = results[k]
        print(f"| {k[0]} | {k[1]} | {statistics.median(v):.0f} | {min(v):.0f} | {max(v):.0f} |")


if __name__ == "__main__":
    main()

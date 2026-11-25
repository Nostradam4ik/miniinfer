"""Text interface to generate.exe: tokenizes the prompt and streams the answer.

Example:
  .venv\\Scripts\\python.exe tools\\chat.py "Once upon a time" --max 60 --temp 0.8 --top-p 0.95

The tokenizer is loaded from model/ (saved there by tools/export_reference.py).
This is a base model (not instruct): it continues text, it does not answer questions.
"""
import argparse
import os
import subprocess
import sys

from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", help="beginning of the text")
    ap.add_argument("--max", type=int, default=64, help="number of new tokens")
    ap.add_argument("--temp", type=float, default=0.0, help="temperature; 0 = greedy")
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=["f32", "f16", "bf16"], default="bf16",
                    help="weight format (bf16: half the memory, no precision loss)")
    ap.add_argument("--no-graph", action="store_true", help="do not use a CUDA graph for decoding")
    ap.add_argument("--exe", default=os.path.join(ROOT, "build", "generate.exe"))
    ap.add_argument("--model", default=os.path.join(ROOT, "model"))
    args = ap.parse_args()

    if not os.path.exists(args.exe):
        sys.exit(f"{args.exe} not found: run build.bat first")
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(args.prompt).input_ids
    if not ids:
        sys.exit("empty prompt")

    cmd = [
        args.exe, "--model", args.model,
        "--ids", " ".join(map(str, ids)),
        "--max", str(args.max), "--temp", str(args.temp),
        "--top-k", str(args.top_k), "--top-p", str(args.top_p), "--seed", str(args.seed),
        "--dtype", args.dtype,
    ]
    if not args.no_graph:
        cmd.append("--graph")
    if tok.eos_token_id is not None:
        cmd += ["--stop", str(tok.eos_token_id)]

    sys.stdout.reconfigure(encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    print(args.prompt, end="", flush=True)

    # Incremental detokenization: a token can be part of a UTF-8 character; decode then returns
    # "�" and we wait for the next token instead of printing garbage.
    generated, printed = [], ""
    for line in proc.stdout:
        if not line.startswith("tok "):
            continue
        generated.append(int(line.split()[1]))
        text = tok.decode(generated)
        if text.endswith("�"):
            continue
        print(text[len(printed):], end="", flush=True)
        printed = text
    print()

    err = proc.stderr.read()
    code = proc.wait()
    if err.strip():
        print(err.strip(), file=sys.stderr)
    sys.exit(code)


if __name__ == "__main__":
    main()

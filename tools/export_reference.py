"""Export for the C++ engine: model weights + PyTorch reference activations.

Usage:  .venv\\Scripts\\python.exe tools\\export_reference.py

Creates the model/ directory (fp32 weights + manifest.txt + config.json) and reference/
(input tokens and intermediate results for checking the kernels).
Manifest format (one line per tensor):  name dtype ndim d0 d1 ... byte_offset
All tensors are raw little-endian float32 (int32 for tokens).
"""

import json
import os
import sys

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
PROMPT = "The capital of France is"
# Natural English written for this project, on several topics. Used to measure how often a faster, less exact
# mode predicts the same next token as PyTorch.
LONG_TEXT = (
    "The city library will close its main reading room for six weeks this spring while the roof is "
    "repaired. Visitors can still borrow and return books at a desk near the side entrance, and the "
    "quiet study area will move to the second floor. Opening hours stay the same on weekdays.\n\n"
    "The harbor wakes up long before the town does. At four in the morning the first boats start their "
    "engines, and the smell of diesel mixes with salt and wet rope. Maria has sold coffee from a small "
    "cart near the pier for eleven years. She says she can tell how the night went by the way the "
    "fishermen walk toward her. When the catch was good, they talk loudly and ask for sugar. When it was "
    "poor, they pay in silence and look at the water.\n\n"
    "A graphics processor is built around a simple bargain. Each of its threads is slow compared with a "
    "core of a central processor, but there are thousands of them, and they share the cost of fetching "
    "instructions. The bargain pays off only when neighboring threads do similar work on neighboring "
    "data. A program that ignores this rule still gives the right answer, yet it may spend most of its "
    "time waiting for memory.\n\n"
    "To make the bread, mix five hundred grams of flour with ten grams of salt and a small spoon of dry "
    "yeast. Add water that feels slightly warm on your wrist, and stir until no dry patches remain. Cover "
    "the bowl and leave it on the counter overnight. In the morning, fold the dough over itself a few "
    "times, let it rest for an hour, and bake it in a very hot pot with the lid on.\n\n"
    "In 1815 a volcano in what is now Indonesia erupted so violently that the following year had no real "
    "summer in parts of Europe and North America. Snow fell in June, crops failed, and the price of grain "
    "rose sharply. Few people at the time connected the cold weather with a mountain on the other side of "
    "the world. The link was understood only much later, when scientists studied how ash and gas travel "
    "through the upper atmosphere.\n\n"
    '"Are you sure this is the right road?" asked Tom, holding the map upside down. His sister did not '
    "answer at once. The path had narrowed to a line of flattened grass, and somewhere below them a "
    'stream was making far more noise than a stream should. "I am sure it was the right road an hour '
    'ago," she said at last, and they both started to laugh.'
)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "model")
REF_DIR = os.path.join(ROOT, "reference")


class BinWriter:
    """Writes tensors into one .bin file and keeps a manifest."""

    def __init__(self, directory, name):
        os.makedirs(directory, exist_ok=True)
        self.bin = open(os.path.join(directory, name + ".bin"), "wb")
        self.man = open(os.path.join(directory, name + ".manifest.txt"), "w", encoding="utf-8")
        self.offset = 0

    def add(self, name, array):
        arr = np.ascontiguousarray(array)
        if arr.dtype == np.float64 or arr.dtype == np.float16:
            arr = arr.astype(np.float32)
        if arr.dtype not in (np.float32, np.int32):
            raise ValueError(f"{name}: unsupported dtype {arr.dtype}")
        data = arr.tobytes()
        dims = " ".join(str(d) for d in arr.shape)
        self.man.write(f"{name} {arr.dtype.name} {arr.ndim} {dims} {self.offset}\n")
        self.bin.write(data)
        self.offset += len(data)

    def close(self):
        self.bin.close()
        self.man.close()


def write_long_text_reference(tok, model):
    """reference/long.*: PyTorch's top-1 token at every position of LONG_TEXT, full logits at some of them."""
    ids = tok(LONG_TEXT, return_tensors="pt").input_ids
    count = ids.shape[1]
    if not 450 <= count <= 600:
        raise SystemExit(f"LONG_TEXT must be 450..600 tokens, got {count}")
    with torch.no_grad():
        logits = model(ids).logits[0]
    positions = sorted(set(range(0, count, 32)) | {count - 1})
    w = BinWriter(REF_DIR, "long")
    w.add("long.ids", ids[0].numpy().astype(np.int32))
    w.add("long.top1", logits.argmax(dim=-1).numpy().astype(np.int32))
    w.add("long.positions", np.array(positions, dtype=np.int32))
    w.add("long.logits", logits[positions].numpy())
    w.close()
    print(f"long text reference written: {count} tokens, logits at {len(positions)} positions")


def main():
    print("Loading the model...", MODEL_ID)
    path = snapshot_download(MODEL_ID)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
    model.eval()
    cfg = model.config
    print(cfg)

    # --- config needed by the engine ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    tok.save_pretrained(MODEL_DIR)  # tokenizer for tools/chat.py
    engine_cfg = {
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "num_layers": cfg.num_hidden_layers,
        "num_heads": cfg.num_attention_heads,
        "num_kv_heads": cfg.num_key_value_heads,
        "head_dim": cfg.hidden_size // cfg.num_attention_heads,
        "rms_norm_eps": cfg.rms_norm_eps,
        "rope_theta": getattr(cfg, "rope_theta", None) or cfg.rope_parameters["rope_theta"],
        "tie_word_embeddings": bool(cfg.tie_word_embeddings),
    }
    with open(os.path.join(MODEL_DIR, "config.json"), "w") as f:
        json.dump(engine_cfg, f, indent=2)
    # For C++ without a JSON parser: one "key value" per line.
    with open(os.path.join(MODEL_DIR, "config.txt"), "w") as f:
        for k, v in engine_cfg.items():
            f.write(f"{k} {int(v) if isinstance(v, bool) else v}\n")
    print("config:", engine_cfg)

    # --- weights ---
    w = BinWriter(MODEL_DIR, "weights")
    for name, p in model.state_dict().items():
        w.add(name, p.detach().cpu().numpy())
    w.close()
    print("weights written, bytes:", w.offset)

    # --- reference activations ---
    ids = tok(PROMPT, return_tensors="pt").input_ids
    print("prompt ids:", ids.tolist())

    captured = {}

    def hook(name):
        def fn(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            captured[name] = t.detach().cpu().numpy()[0]  # drop the batch dimension

        return fn

    layer0 = model.model.layers[0]
    model.model.embed_tokens.register_forward_hook(hook("embed_out"))
    layer0.input_layernorm.register_forward_hook(hook("l0.input_norm"))
    layer0.self_attn.q_proj.register_forward_hook(hook("l0.q_proj"))
    layer0.self_attn.k_proj.register_forward_hook(hook("l0.k_proj"))
    layer0.self_attn.v_proj.register_forward_hook(hook("l0.v_proj"))
    layer0.self_attn.o_proj.register_forward_hook(hook("l0.o_proj"))
    layer0.self_attn.register_forward_hook(hook("l0.attn_out"))
    layer0.post_attention_layernorm.register_forward_hook(hook("l0.post_norm"))
    layer0.mlp.register_forward_hook(hook("l0.mlp_out"))
    layer0.register_forward_hook(hook("l0.out"))
    model.model.norm.register_forward_hook(hook("final_norm"))

    # Input of o_proj = result of attention before the output projection (context over all heads).
    def pre_hook(name):
        def fn(_mod, args):
            captured[name] = args[0].detach().cpu().numpy()[0]

        return fn

    layer0.self_attn.o_proj.register_forward_pre_hook(pre_hook("l0.attn_ctx"))

    with torch.no_grad():
        out = model(ids, output_hidden_states=True)

    # RoPE of layer 0 computed with the transformers functions (independent of our code).
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    T = ids.shape[1]
    H, KV, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads
    q = torch.from_numpy(captured["l0.q_proj"]).view(1, T, H, hd).transpose(1, 2)
    k = torch.from_numpy(captured["l0.k_proj"]).view(1, T, KV, hd).transpose(1, 2)
    pos = torch.arange(T)[None]
    cos, sin = model.model.rotary_emb(q, pos)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    captured["l0.q_rope"] = q_rot.transpose(1, 2).reshape(T, H * hd).numpy()
    captured["l0.k_rope"] = k_rot.transpose(1, 2).reshape(T, KV * hd).numpy()

    r = BinWriter(REF_DIR, "reference")
    r.add("input_ids", ids[0].numpy().astype(np.int32))
    for k_, v_ in captured.items():
        r.add(k_, v_)
    # Output of every layer (hidden_states[0] = embeddings, then 30 layers; the last is after the final norm)
    for i, hs in enumerate(out.hidden_states):
        r.add(f"hidden_states.{i}", hs[0].numpy())
    r.add("logits", out.logits[0].numpy())
    r.close()

    write_long_text_reference(tok, model)

    # Reference greedy generation (to check the KV cache and the generation loop in C++).
    gen_prompts = {
        "gen1": PROMPT,
        "gen2": "Once upon a time, in a small village near the sea, there lived",
    }
    g = BinWriter(REF_DIR, "generation")
    for key, text in gen_prompts.items():
        pids = tok(text, return_tensors="pt").input_ids
        with torch.no_grad():
            full = model.generate(pids, max_new_tokens=20, do_sample=False, pad_token_id=tok.eos_token_id)
        new = full[0, pids.shape[1] :]
        g.add(f"{key}.prompt", pids[0].numpy().astype(np.int32))
        g.add(f"{key}.new", new.numpy().astype(np.int32))
        print(key, repr(text), "->", repr(tok.decode(new)))
    g.close()

    # What the model predicts next
    nxt = int(out.logits[0, -1].argmax())
    print("next token:", nxt, repr(tok.decode([nxt])))
    print("reference written, bytes:", r.offset)
    print(f"DONE: {PROMPT!r} -> {tok.decode([nxt])!r}")


if __name__ == "__main__":
    sys.exit(main())

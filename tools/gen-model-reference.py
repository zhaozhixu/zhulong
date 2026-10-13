#!/usr/bin/env python
"""SmolLM2-135M layer-parity reference.

A bf16 forward faithful to HF Llama semantics (f32 rms-norm internals,
f32 softmax, rope tables cast to bf16 at use), written directly in torch
over the safetensors weights. Emits tests/data/model/manifest.rktd plus
one raw bf16 blob per checkpoint: the embedding output, every block
output, and the logits.
"""

import pathlib

import torch
from safetensors.torch import load_file

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "tests/data/model"
OUT.mkdir(parents=True, exist_ok=True)

W = load_file(ROOT / "weights/SmolLM2-135M/model.safetensors")

LAYERS, HIDDEN, HEADS, KV, HD = 30, 576, 9, 3, 64
THETA, EPS = 100000.0, 1e-5
HALF = HD // 2
IDS = [1, 542, 3067, 12, 8999, 42, 7, 2025]
L = len(IDS)


def rms(x, w):
    x32 = x.float()
    var = x32.pow(2).mean(-1, keepdim=True)
    return w * (x32 * torch.rsqrt(var + EPS)).to(torch.bfloat16)


def rot_half(t):
    return torch.cat((-t[..., HALF:], t[..., :HALF]), -1)


inv = 1.0 / (THETA ** (torch.arange(0, HD, 2).float() / HD))
freqs = torch.outer(torch.arange(L).float(), inv)
emb = torch.cat((freqs, freqs), -1)
cos = emb.cos().to(torch.bfloat16)
sin = emb.sin().to(torch.bfloat16)
mask = torch.triu(torch.full((L, L), torch.finfo(torch.bfloat16).min), 1).to(torch.bfloat16)

records = []


def save(name, t):
    fn = f"{name}.bin"
    (OUT / fn).write_bytes(t.contiguous().view(torch.uint16).numpy().tobytes())
    records.append((name, list(t.shape), fn))


x = W["model.embed_tokens.weight"][torch.tensor(IDS)].unsqueeze(0)
save("h0", x)

for i in range(LAYERS):
    p = f"model.layers.{i}."

    def lin(t, n):
        return t @ W[p + n + ".weight"].T

    h = rms(x, W[p + "input_layernorm.weight"])
    q = lin(h, "self_attn.q_proj").view(1, L, HEADS, HD).transpose(1, 2)
    k = lin(h, "self_attn.k_proj").view(1, L, KV, HD).transpose(1, 2)
    v = lin(h, "self_attn.v_proj").view(1, L, KV, HD).transpose(1, 2)
    q = q * cos + rot_half(q) * sin
    k = k * cos + rot_half(k) * sin
    rep = HEADS // KV
    k = k[:, :, None].expand(1, KV, rep, L, HD).reshape(1, HEADS, L, HD)
    v = v[:, :, None].expand(1, KV, rep, L, HD).reshape(1, HEADS, L, HD)
    s = q @ k.transpose(-2, -1) * (HD**-0.5) + mask
    probs = torch.softmax(s.float(), -1).to(torch.bfloat16)
    o = (probs @ v).transpose(1, 2).reshape(1, L, HIDDEN)
    x = x + lin(o, "self_attn.o_proj")
    h2 = rms(x, W[p + "post_attention_layernorm.weight"])
    x = x + lin(torch.nn.functional.silu(lin(h2, "mlp.gate_proj")) * lin(h2, "mlp.up_proj"),
                "mlp.down_proj")
    save(f"h{i + 1}", x)

xf = rms(x, W["model.norm.weight"])
save("logits", xf @ W["model.embed_tokens.weight"].T)

with open(OUT / "manifest.rktd", "w") as f:
    f.write("((ids {})\n (checkpoints\n".format(" ".join(map(str, IDS))))
    for name, shape, fn in records:
        f.write("  ({} ({}) \"{}\")\n".format(name, " ".join(map(str, shape)), fn))
    f.write("))\n")

print(f"model reference: {len(records)} checkpoints, L={L}")

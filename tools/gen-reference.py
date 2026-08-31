#!/usr/bin/env python3
"""Generate operator parity reference data.

Each case runs the pinned torch op on seeded inputs and records the input
and output tensors as raw little-endian .bin blobs plus one s-expression
manifest (cases.rktd) that the Racket test runner reads. Deterministic
(fixed seed) and regenerable at will; output lives under tests/data/ and
is not committed.
"""

from pathlib import Path

import torch
import torch.nn.functional as F

OUT = Path(__file__).resolve().parent.parent / "tests" / "data" / "ops"
OUT.mkdir(parents=True, exist_ok=True)

DT = {"f32": torch.float32, "bf16": torch.bfloat16, "i64": torch.int64}
DT_NAME = {v: k for k, v in DT.items()}


def to_bytes(t: torch.Tensor) -> bytes:
    t = t.contiguous()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.uint16)
    return t.numpy().tobytes()


def sx(v) -> str:
    if isinstance(v, bool):
        return "#t" if v else "#f"
    if isinstance(v, (list, tuple)):
        return "(" + " ".join(sx(x) for x in v) + ")"
    return str(v)


cases = []


def emit(name, dtype, inputs, attrs, out, tag=None):
    tag = tag or f"{name}-{dtype}-{len(cases)}"
    parts = []
    for i, (kind, v) in enumerate(inputs):
        if kind == "s":
            parts.append(f"(scalar {v})")
        else:
            fn = f"{tag}-in{i}.bin"
            (OUT / fn).write_bytes(to_bytes(v))
            parts.append(f'(tensor "{fn}" {sx(list(v.shape))} {DT_NAME[v.dtype]})')
    ofn = f"{tag}-out.bin"
    (OUT / ofn).write_bytes(to_bytes(out))
    attrs_s = "(" + " ".join(f"({k} . {sx(v)})" for k, v in attrs.items()) + ")"
    cases.append(
        f"((name . {name}) (attrs . {attrs_s})"
        f' (inputs . ({" ".join(parts)}))'
        f' (output . ("{ofn}" {sx(list(out.shape))} {DT_NAME[out.dtype]})))'
    )


torch.manual_seed(0)

for dts in ["f32", "bf16"]:
    dt = DT[dts]
    a = torch.randn(2, 3).to(dt)
    b = torch.randn(2, 3).to(dt)
    c = torch.randn(1, 3).to(dt)
    pos = (torch.rand(2, 3) + 0.5).to(dt)

    emit("add", dts, [("t", a), ("t", b)], {}, torch.add(a, b))
    emit("add", dts, [("t", a), ("t", c)], {}, torch.add(a, c), tag=f"add-bc-{dts}")
    emit("add", dts, [("t", a), ("s", 1.5)], {}, torch.add(a, 1.5), tag=f"add-s-{dts}")
    emit("sub", dts, [("t", a), ("t", b)], {}, torch.sub(a, b))
    emit("mul", dts, [("t", a), ("t", b)], {}, torch.mul(a, b))
    emit("div", dts, [("t", a), ("t", pos)], {}, torch.div(a, pos))
    emit("neg", dts, [("t", a)], {}, torch.neg(a))
    emit("abs", dts, [("t", a)], {}, torch.abs(a))
    emit("sqr", dts, [("t", a)], {}, torch.square(a))
    emit("rsqrt", dts, [("t", pos)], {}, torch.rsqrt(pos))
    emit("silu", dts, [("t", a)], {}, F.silu(a))

    x = torch.randn(2, 3, 4).to(dt)
    emit("mean", dts, [("t", x)], {"axis": -1, "keep-dim": True},
         torch.mean(x, -1, keepdim=True))
    emit("mean", dts, [("t", x)], {"axis": 0, "keep-dim": False},
         torch.mean(x, 0), tag=f"mean0-{dts}")
    emit("amax", dts, [("t", x)], {"axis": -1, "keep-dim": True},
         torch.amax(x, -1, keepdim=True))
    emit("argmax", dts, [("t", x)], {"axis": -1, "keep-dim": False},
         torch.argmax(x, -1))

    s2 = torch.randn(2, 5).to(dt)
    emit("softmax", dts, [("t", s2)], {"axis": -1}, F.softmax(s2, dim=-1))

    m1 = torch.randn(3, 4).to(dt)
    m2 = torch.randn(4, 5).to(dt)
    emit("matmul", dts, [("t", m1), ("t", m2)], {}, torch.matmul(m1, m2))
    m3 = torch.randn(2, 3, 4).to(dt)
    emit("matmul", dts, [("t", m3), ("t", m2)], {}, torch.matmul(m3, m2),
         tag=f"matmul-n2-{dts}")
    q = torch.randn(2, 3, 4, 5).to(dt)
    k = torch.randn(2, 3, 5, 6).to(dt)
    emit("matmul", dts, [("t", q), ("t", k)], {}, torch.matmul(q, k),
         tag=f"matmul-bmm-{dts}")

    w = torch.randn(10, 4).to(dt)
    idx = torch.randint(0, 10, (2, 3))
    emit("embedding", dts, [("t", w), ("t", idx)], {}, F.embedding(idx, w))

    emit("reshape", dts, [("t", x)], {"shape": [4, 6]}, x.reshape(4, 6))
    emit("transpose", dts, [("t", x)], {"dim0": 0, "dim1": 2},
         x.transpose(0, 2).contiguous())
    emit("narrow", dts, [("t", x)], {"axis": 1, "start": 1, "len": 2},
         x.narrow(1, 1, 2).contiguous())
    emit("cat", dts, [("t", a), ("t", b)], {"axis": 0}, torch.cat([a, b], 0))
    emit("cat", dts, [("t", a), ("t", b)], {"axis": -1}, torch.cat([a, b], -1),
         tag=f"cat1-{dts}")
    emit("broadcast-to", dts, [("t", c)], {"shape": [4, 2, 3]},
         torch.broadcast_to(c, (4, 2, 3)).contiguous())
    emit("unsqueeze", dts, [("t", a)], {"axis": 1}, a.unsqueeze(1))
    other = "bf16" if dts == "f32" else "f32"
    emit("cast", dts, [("t", a)], {"dtype": other}, a.to(DT[other]))

(OUT / "cases.rktd").write_text("(\n" + "\n".join(cases) + "\n)\n")
print(f"wrote {len(cases)} cases to {OUT}")

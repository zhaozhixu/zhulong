# Zhulong: Status

Status as of 2026-08-31. The architecture and its rationale live in
[design.md](design.md); the module map and build order live in
[impl-plan.md](impl-plan.md); coding rules are in STYLE.md at the repo
root. This document only tracks where the build stands.

## Done (build steps 1–5)

**Foundation.** Typed core (shape algebra; tensor structs with real and
symbolic variants), the `zhulong` #lang with `define-layer` /
`define-model`, the operator registry with `operator-apply` dispatch
over `current-backend`, STYLE.md, raco fmt formatting, linked
dual-package layout (`src/` framework, `models/` model library).

**FFI bridge.** Library discovery against the project venv's torch,
Stable C ABI bindings (version, device/dtype constants, ctensor
lifecycle: create / metadata / copy / item / delete / reinterpret),
lazy symbol resolution so modules load without libtorch.

**Eager operator layer.** 22 operators end-to-end from three kernel
sources: Stable-ABI kernels, libtorch's generated per-backend shims
(`aoti_torch_cpu_*`, `aoti_torch_aten_*`), and a 9-kernel custom C++
shim built at setup. Views are Racket-side stride math; matmul folds
every supported rank combination into one mm/bmm call via a single
shared plan analysis. Registration is declarative: `define-operator`
(semantic registry) mirrored by `define-operator-impl` (eager table),
with the calling convention held by construction. Verified by a 56-case
byte-exact parity suite (f32 and bf16) generated from the same pinned
torch the runtime dlopens.

**Model forward.** safetensors parser (pure typed IO), a
bytes-to-tensor loading path, and `models/llama.rkt` — a Llama-family
forward in branch-free dataflow form, faithful to HF semantics (f32
rms-norm internals, f32 softmax, host-built rope/mask tables cast at
use, GQA expansion). Verified against a torch reference over
SmolLM2-135M: 32 checkpoints (embedding, all 30 blocks, logits)
**bit-identical in bf16** — measured worst relative difference 0.0.

The full suite is 378 passing tests.

## Pinned toolchain

Racket 9.2 (CS). Python 3.14 in a uv-managed venv whose single torch
2.13.0 installation serves both reference generation and the dylibs the
runtime dlopens — the two sides cannot drift. Development model:
SmolLM2-135M (LlamaForCausalLM, 30 layers, hidden 576, GQA 9/3, bf16,
tied embeddings); SmolLM2-1.7B is the later 1B-class acceptance target.
Details and measured facts in VERSIONS.

## Next (build steps 6–9, detail in impl-plan.md §3)

6. **End-to-end eager** — KV cache arena, `kv-append` / `kv-read`
   state operators, composer's single-request generation loop, greedy
   decoding of a pre-tokenized prompt against the reference.
7. **Graph and tracer** — node table with S-expression codec, the
   tracer backend, the interpreting executor, eager-versus-graph
   differential testing.
8. **Islands** — boundary weaving, delimited control, trace reports.
9. **Wrap-up** — tokenizer (BPE), CLI, probe hooks, native packaging:
   text in, text out on a clean machine, zero Python.

Then phase two and beyond: AOTI compilation service, dual-mode
execution, CUDA graph capture on the decode path, and the optimization
plane proper (eqsat rewriting, planning, scheduling).

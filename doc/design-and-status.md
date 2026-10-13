# Zhulong: Design and Status

Status as of 2026-08-31. Zhulong (烛龙) is a Racket-hosted large-model
inference system: Racket owns model definition, symbolic optimization,
and orchestration; heavy compute is delegated to libtorch kernels and
AOTInductor artifacts. The project is a research vehicle first and a
natively distributable application base second — its yardstick is
iteration speed on optimization strategies and interactive capability,
not raw throughput.

## Architecture

The system is organized as three planes over one S-expression graph IR.

**Definition plane** — what the model is. A shallowly embedded DSL:
operators are ordinary Racket functions dispatching on the current
tensor backend. Under the eager backend a call executes immediately via
FFI and returns a real tensor; under the tracer backend (planned) the
same call records a graph node and returns a symbolic tensor. One body
of model code, two readings. Control flow is stage-polymorphic:
branches and loops on symbolic tensor values lift into graph control
nodes; what cannot lift falls back to eager islands rather than
whole-graph breaks. The host/graph boundary is explicit and small —
`item` (0-d tensor to host scalar) and `fetch` (rank-deep host list
mirror) are the only crossings.

**Optimization plane** — how to compute it best. A pass pipeline over
the IR: equality-saturation rewriting (rules are data: serializable,
hot-loadable `defrule` forms), parallel planning (device assignment and
sharding as annotations, communication materialized as explicit IR
nodes), and communication scheduling. Cost models are fed by a
measurement loop around the pipeline.

**Execution plane** — how to put it on hardware. Bottom-up: an FFI
bridge over PyTorch's Stable C ABI, dual-mode execution (interpreted
glue around compiled subgraphs), a build-time Python sidecar that
compiles subgraphs through AOTInductor into C-ABI artifacts (no Python
at runtime), a multi-stream executor, and lazy stream combinators for
orchestration. Decode steps are shape-stable by design, enabling
whole-step CUDA graph capture later.

### Design principles

- **Brain, not muscle.** No hand-written CUDA kernels (Triton-generated
  excepted). High-performance external kernels integrate behind thin C
  shims as implementation choices, never as new operator identities.
- **One operator vocabulary, ATen-aligned.** An operator's identity is
  one symbol threading the whole system: the semantic registry, each
  backend's implementation table, the reference generator, the parity
  manifests, and (once the graph lands) node op fields are all keyed by
  the same name.
- **Semantic registry plus facet tables.** The registry holds only
  backend-neutral semantic facts — the shape rule and the dtype rule —
  and stays FFI-free. Knowledge tied to a backend or to hardware (eager
  implementations, FX emission, capability manifests, measured costs)
  lives in separate tables keyed by the operator name and never flows
  back into the registry.
- **Device-neutral operator identity.** There is no cpu-matmul versus
  cuda-matmul. Device is a property of data (eager operators follow
  their inputs) and of placement (a planning-pass annotation on the
  graph). Placement is an optimizer decision variable; baking it into
  identity would freeze it before optimization begins.
- **Uniform calling convention.** Operands are positional; attributes
  are keywords whose key is the parameter name. The registration macro
  (`define-operator`) and the implementation macro
  (`define-operator-impl`) are mirrored declaration forms deriving keys
  from parameter names, so the two sides agree by construction.
- **Weak scalar dtype.** Racket reals enter rules with shape `'()` and
  dtype deferred, mirroring ATen's Tensor/Scalar split; the eager
  backend converts scalars into the dtype the tensor operands agreed
  on, so scalars never win type promotion.
- **bf16 native.** Weights load zero-copy-ready as bf16; parity against
  references is a tolerance-based methodology, not a system dtype
  constraint. Quantization is scoped as a weight format plus operator
  vocabulary extension.
- **Pure semantics, in-place implementation.** The KV cache is threaded
  linearly through the IR; a linearity pass licenses destructive
  updates while the optimizer sees a pure function. State operations
  are rewrite barriers.
- **Typed by default.** Modules are Typed Racket except where the
  machinery demands otherwise (macros, delimited control, FFI bindings,
  model DSL code, tests). Generated contracts are dropped on the one
  hot edge (untyped model code into the operator API) via
  typed/racket/unsafe, with tests standing guard; tensor handles are
  opaque `CTensor` values.

## Current status

Built bottom-up; every layer is parity-tested against torch before the
next goes on top.

**Foundation (done).** Typed core (shape algebra, tensor structs with
real/symbolic variants), the `zhulong` #lang with `define-layer` /
`define-model`, registry and dispatch (`operator-apply` over
`current-backend`), STYLE.md coding rules, raco fmt formatting, linked
dual-package layout (`src/` framework, `models/` application library).

**FFI bridge (done).** Library discovery against the project venv's
torch, Stable C ABI bindings (version, device/dtype constants, ctensor
lifecycle: create / metadata / copy / item / delete / reinterpret),
lazy symbol resolution so modules load without libtorch.

**Eager operator layer (done).** 22 operators end-to-end, assembled
from three kernel sources: Stable-ABI kernels, libtorch's generated
per-backend shims (`aoti_torch_cpu_*`, `aoti_torch_aten_*`), and a
9-kernel custom C++ shim built at setup. Views (reshape, transpose,
narrow, broadcast-to, unsqueeze) are Racket-side stride math over
`_reinterpret_tensor`; matmul folds every supported rank combination
into one mm/bmm call via a single shared plan analysis. Verified by a
56-case byte-exact parity suite (f32 and bf16) generated from the same
pinned torch that the runtime dlopens.

**Model forward (done).** A safetensors parser (pure typed IO), a
bytes-to-tensor loading path, and `models/llama.rkt` — a Llama-family
forward in branch-free dataflow form, faithful to HF semantics (f32
rms-norm internals, f32 softmax, host-built rope/mask tables cast to
model dtype at use, GQA head expansion). Verified against a torch
reference over SmolLM2-135M: 32 checkpoints (embedding, all 30 blocks,
logits) **bit-identical in bf16** — the measured maximum relative
difference is 0.0, with the tolerance headroom unused.

The full suite is 378 passing tests.

**Pinned toolchain.** Racket 9.2 (CS); Python 3.14 in a uv-managed venv
whose single torch 2.13.0 installation serves both reference generation
and the dylibs the runtime dlopens — the two sides cannot drift.
Development model: SmolLM2-135M (LlamaForCausalLM, 30 layers, hidden
576, GQA 9/3, bf16, tied embeddings); SmolLM2-1.7B is the later
1B-class acceptance target.

## Next

1. **Decode loop** — KV cache arena (contiguous max-len layout behind
   an opaque interface, block-table indirection reserved),
   `kv-append` / `kv-read` as state operators, and a greedy
   single-request generation loop over pre-tokenized input.
2. **Graph path** — the id-indexed node table with S-expression codec,
   the tracer backend (delimited control, auto-lifting, eager
   islands), the interpreting executor, and eager-versus-graph
   differential testing.
3. **Productization of phase 1** — tokenizer, CLI, in-vivo probes, and
   native packaging via raco distribute.
4. **Optimization plane** — rewrite engine, AOTI compilation service,
   dual-mode execution, CUDA graph capture on the decode path.

Detailed design rationale lives in the working documents under
`debug/` (untracked); coding rules are in `STYLE.md` at the repo root.

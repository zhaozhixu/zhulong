# Zhulong Implementation Plan: Phase One

Companion to the design document (doc/design.md), covering its §7 phase
one. The design document answers *what and why*; this one answers *how*:
in what style the code is written, how it is divided into modules, and
in what order it is delivered. Where the two conflict, the design
document wins; when implementation reveals the need for an architecture
change, the design document changes first, then the code.

## 0. Go decision

Work can start. The design decisions that would block coding (shape
policy, wrapper load-bearing order, numeric precision, the boundary
between eqsat and state) are closed in the design document; the open
questions in its §9 block nothing in phase one. The three things that
still had to be pinned before coding are pinned by this document: code
style (§1), module boundaries (§2), build order (§3). The only external
prerequisite was selecting and locking a libtorch version — done at FFI
bring-up (build step 4), where the Stable C ABI's actually usable
surface was verified.

## 1. Code style

The canonical rules live in STYLE.md at the repo root (English,
maintained alongside the code, doubling as context material for
LLM-assisted coding); this section keeps no copy. Highlights:
functional and data-oriented, operator-apply as the single dispatch
point, a macro whitelist, typed/racket by default (lang, tracer, FFI
bindings, model code, and tests stay dynamic), heterogeneous payloads
honestly typed Any, generated contracts dropped on the measured hot
edge via typed/racket/unsafe, tests in tests/ by default, comments in
English, raco fmt formatting throughout (DSL forms registered in
.fmt.rkt), ffi/unsafe quarantined in runtime/ffi/.

## 2. Module layout

### 2.1 Organizing principles

Source files are laid flat by module level: a directory appears only
where clearly parallel submodules exist (operators/, optimizer/rules/,
runtime/ffi/); everything else is a single file. Directories follow the
design document's three planes, plus utils for shared facilities. The
framework (the `zhulong` collection) and the model library (the
`models` collection) are separate packages; zhulong contains no model
knowledge.

Two code locations differ from the design document's plane narrative —
code organization, not architecture changes. safetensors parsing lives
in utils/: pure IO machinery, conceptually still a definition-plane
component (design §2.1). The generation end of the compilation service
lives in optimizer/compiler.rkt: the design document (§5.2) lists the
compilation service under the execution plane, while this plan treats
"IR→FX generation, sidecar invocation, content-hash artifact caching"
as the tail of the optimization pipeline; "dlopen the artifact,
dual-mode switching" stays in runtime/executor. The mechanism matches
the design document, and the hand-off object between the two sides is
an artifact file, not a code dependency.

The operator-library / layer-library boundary: operators/ corresponds
one-to-one with the registry and is a **superset** of the ATen-aligned
core plus our own extensions — the state operators (kv-append/kv-read)
are extensions today; if phase three promotes fused macro-operators
(e.g. fused-attention) to rewrite targets, they enter the same way, as
registry entries whose eager reference implementation may be the
decomposed composition, with only the compiled path mapping to a fused
kernel. The single admission criterion: a registry entry (shape/dtype
rules) plus an eager implementation-table entry. mlp, rms-norm, RoPE,
and GQA attention are composite layers written in the DSL that trace
into multiple nodes; they belong to models/layers.rkt — the
application-level library shared across models (the design document's
§3.1 "library components").

### 2.2 The hub: the operator registry

The whole system has one operator vocabulary, keyed by the operator
name (one symbol) across every plane. operator.rkt defines the registry
structure, operator-apply single-point dispatch, and current-backend;
operators/ files, one per family, use define-operator so that a single
declaration registers the entry and derives the user function. A
registry entry holds only backend-neutral semantic facts — the operator
name, the shape rule, the dtype rule — so the registry is FFI-free and
testable without libtorch. Knowledge tied to a specific backend or
hardware never flows back into the registry; each plane keeps its own
facet table keyed by operator name: the eager implementation table
(define-operator-impl in runtime/ffi/ops.rkt, landed), FX emission
templates (optimizer/compiler.rkt, phase two), backend capability
manifests and measured cost data (phase three on, lowering and the
tuning loop). The registry and the implementation tables are pinned
together by one calling convention: same name symbol; operands
positional in the same order; attributes as keywords with key =
parameter name — both macros derive keys from parameter names, and
their mirrored header shapes make agreement hold by construction.
Consumers: eager.rkt (through eager-run into the implementation table),
tracer.rkt (shape/dtype rules for symbolic inference),
tools/gen-wrapper.rkt (exported manifest), optimizer/compiler.rkt
(IR→FX translation, phase two), and the rule validator (reusing shape
rules when instantiating pattern variables, phase three). The design
document's two promises — "manifest-driven wrapper" and "rule
validation reuses shape rules" — land on the registry and its same-name
facet tables.

Eager-side C bindings do not enter the registry; they are assembled by
hand per operator inside runtime/ffi, from three sources: kernels the
Stable ABI exposes directly, libtorch's own generated per-backend shims
(aoti_torch_cpu_* / aoti_torch_aten_*), and a custom thin shim (zl_*,
compiled at build time). At tens of operators, hand binding is
tractable and more reliable; revisit generating bindings from ATen
declarations when the operator surface reaches hundreds.

Dispatch: every user-visible operator is a thin function derived from
its declaration by define-operator, whose body is the single line
`(operator-apply 'name (list operand ...) attrs)`; operator-apply reads
current-backend and forwards. This is the only dispatch point in the
project.

### 2.3 Directory tree

```
src/                          ; framework (collection `zhulong`, set by info.rkt)
  main.rkt                    ; language & library entry: reader submodule + public API re-exports

  ;; -- definition plane --
  lang.rkt                    ; DSL syntax: define-layer/define-model/with-eager, boundary weaving (step 8)
  tensor.rkt                  ; tensor types & public API (real + symbolic);
                              ;   ctensor lifecycle machinery comes from runtime/ffi
  shape.rkt                   ; shape algebra (typed): broadcasting, reduction, pure algorithms
  operator.rkt                ; operator type, registry, operator-apply, current-backend, define-operator
  operators/                  ; define-operator declarations by family -- ATen-aligned core + extensions
    elementwise.rkt           ;   add/mul/rsqrt/mean/...
    matmul.rkt
    softmax.rkt
    index.rkt                 ;   narrow/cat/embedding/...
    state.rkt                 ;   kv-append/kv-read (fused macro-operators may join in phase three)
  graph.rkt                   ; node table, traversal, shape/dtype inference, S-expression codec, linearity check
  eager.rkt                   ; eager backend: tensor unwrap/rewrap, eager-run into the ffi impl table;
                              ;   bytes->tensor materialization
  tracer.rkt                  ; tracer backend: graph building, delimited control, auto-eager islands,
                              ;   trace reports and with-eager echo

  ;; -- optimization plane (phase one has only the rules/ data enumeration) --
  optimizer/
    rule.rkt                  ; eqsat rewrite engine (phase three)
    rules/                    ; rule-library data files
    parallel.rkt              ; parallel planning pass (phase four)
    scheduler.rkt             ; communication scheduling pass (phase four)
    compiler.rkt              ; IR->FX generation, sidecar client, content-hash artifact cache (phase two)

  ;; -- execution plane --
  runtime/
    ffi/                      ; unsafe quarantine
      raw.rkt                 ;   dlopen & raw Stable C ABI symbols (visible only inside ffi/)
      wrapper.rkt             ;   wrapper shim library bindings (Racket face of the generated shim)
      ops.rkt                 ;   eager implementation table (define-operator-impl, keyed by operator
                              ;   name) + view stride math
    arena.rkt                 ; KV cache: opaque interface, contiguous layout, block-table reserved,
                              ;   use-after-consume assertions in debug mode
    executor.rkt              ; phase one: graph interpretation (topological order through eager);
                              ;   phase two on: dual-mode & artifact loading, multi-stream, CUDA graph capture
    composer.rkt              ; phase one: single-request generation loop + sampling (greedy/temperature/
                              ;   top-k/top-p); phase two on: lazy stream combinators, continuous batching
    vivo.rkt                  ; phase one: probe hooks; phase three: writable hot reload

  ;; -- shared facilities --
  utils/
    safetensor.rkt            ; pure safetensors parsing: header/metadata/offsets, zero dependencies;
                              ;   tensor materialization glue lives in eager.rkt (bytes->tensor)
    tokenizer.rkt             ; tokenizer.json parsing & BPE (before build step 9)
  cli.rkt                     ; inference program entry (module+ main, raco exe target)

models/                       ; model library collection (#lang zhulong, application level)
  layers.rkt                  ; layers shared across models: rms-norm, RoPE, GQA attention, mlp
  llama.rkt                   ; structure, config, weight-name mapping (about a hundred lines;
                              ;   becomes a directory if the family grows)

tests/                        ; parity and integration tests (plus one <module>-test.rkt per module)
  operators.rkt               ;   per-operator parity (data-driven over the generated manifest)
  model-test.rkt              ;   per-layer activations and end-to-end generation parity
  diff.rkt                    ;   differential harness: eager vs graph interpretation (build step 7)
  data/                       ;   generated reference data (gitignored; tools/gen-*.py output)
tools/                        ; development-only tools, not shipped
  gen-wrapper.rkt             ;   registry manifest -> C++ shim source and build
  gen-reference.py            ;   per-operator reference data from the pinned torch (fixed seed)
  gen-model-reference.py      ;   model-level reference forward (checkpoints + manifest)
  sidecar/                    ;   Python compilation service (phase two)
```

Package structure: src/ and models/ are two linked single-collection
packages (src's collection name is `zhulong`, set by info.rkt and
pinned with --name at install; requires always use zhulong/... paths,
independent of the directory name); the raco exe target is src/cli.rkt.
Type policy: the src core defaults to typed/racket; lang.rkt,
tracer.rkt, runtime/ffi/, and models/ stay dynamic (STYLE rule 8).

### 2.4 Dependency rules

The planes form a DAG, not strict layers. Four invariants:

1. **unsafe quarantine**: ffi/unsafe appears only in runtime/ffi/;
   every other module (including tensor.rkt) uses ctensors through its
   safe exports.
2. **Offline-testable set**: operator.rkt, operators/, graph.rkt
   (including the linearity check), and utils/ do not depend on FFI and
   are fully testable without libtorch.
3. **One-way planes**: the definition plane does not require the
   optimization plane; the optimization plane consumes only the
   definition plane (graph.rkt, operator.rkt) and never requires the
   execution plane — the compiler's product (.so artifacts) is handed
   over through the filesystem and dlopen'd by runtime/executor, with
   no code dependency between the two sides. The execution plane may
   depend on the definition plane.
4. **Model library outermost**: models/ (layers.rkt and the model
   definitions) uses only zhulong's public API.

Within the definition plane: tensor.rkt → runtime/ffi; eager.rkt →
operator.rkt + tensor.rkt; tracer.rkt → operator.rkt + graph.rkt (no
FFI); lang.rkt and main.rkt at the surface.

### 2.5 Named but not built

The optimization plane's four engine files and the higher duties of
composer/vivo are named today and filled on schedule: compiler.rkt in
phase two, rule.rkt in phase three, parallel.rkt and scheduler.rkt in
phase four; composer.rkt carries only the single-request loop and
sampling in phase one, vivo.rkt only probe hooks. Phase one writes no
speculative interfaces for the future — the one exception the design
document explicitly requires: the arena's block-table indirection.

## 3. Build order

No milestones, no schedule. Written bottom-up: each module or feature
gets its module+ test and (when it crosses modules) a tests/
integration test immediately, all green before the next layer. The
order is the dependency order:

1. **Skeleton** — info.rkt for both collections, Makefile
   (setup/compile/test), .gitignore. *Done.*
2. **Definition-plane type layer** (offline, no libtorch) —
   tensor.rkt, operator.rkt, the first operators/ entries. *Done.*
3. **Language surface** — lang.rkt (define-layer/define-model v0),
   main.rkt (reader submodule), the first layer in models/layers.rkt.
   *Done.*
4. **FFI and eager** — VERSIONS locks libtorch; runtime/ffi/raw.rkt
   (dlopen and Stable C ABI, producing the usable-surface inventory);
   tensor materialization; tools/gen-wrapper.rkt and
   runtime/ffi/wrapper.rkt for the gaps; eager.rkt;
   tools/gen-reference.py and tests/operators.rkt per-operator parity
   (one round each of f32 and bf16). *Done — see §5.*
5. **Weights and model** — utils/safetensor.rkt; models/llama.rkt;
   per-layer parity against a torch reference. *Done — see §6.*
6. **End-to-end eager** — runtime/arena.rkt (contiguous layout, debug
   assertions); kv-append/kv-read in operators/state.rkt;
   composer.rkt (single-request loop + sampling); greedy decoding of a
   pre-tokenized prompt, parity against the reference.
7. **Graph and tracer** — graph.rkt (node table, shape inference,
   linearity check, codec); tracer.rkt; runtime/executor.rkt;
   tests/diff.rkt (eager and graph execution bit-identical).
8. **Islands** — boundary weaving in lang.rkt, tracer delimited
   control, trace reports and with-eager echo.
9. **Wrap-up** — utils/tokenizer.rkt (BPE); cli.rkt; vivo.rkt probe
   hooks; raco exe/distribute packaging — text in, text out on a clean
   machine, zero Python.

Rule-library enumeration (optimizer/rules/, co-reviewed with the
registry's operator granularity) proceeds in parallel with coding.
Completion is judged by parity and differential tests, matching the
design document's phase-one deliverables.

## 4. Engineering facilities

- **Versions pinned.** The VERSIONS file records the Racket version and
  the exact libtorch version; phase two extends it to the design
  document's §5.2 triple. Upgrading a version is an explicit commit.
- **Device policy.** Development and unit tests default to CPU (macOS);
  references are generated on CPU; bf16 parity runs on CPU (slow but
  correct; acceptance uses short sequences). CUDA bindings
  (stream/event/graph) join runtime/ffi/ in phase two, activated on a
  Linux+CUDA machine. Phase-one acceptance contains no GPU items.
- **Reference data** — recorded outputs of the reference
  implementation: the same inputs are fed to the pinned torch, and its
  outputs become the standard our implementation must reproduce within
  tolerance. Never committed — tools/gen-reference.py and
  tools/gen-model-reference.py regenerate on demand with fixed seeds
  into the gitignored tests/data/; model weights download on demand
  (make weights) and are likewise untracked. A fresh machine or CI runs
  generation first (requires Python and the pinned torch, development
  only).
- **CI.** Starts as local scripting (make): full compile → unit tests →
  reference parity → leak smoke. The differential tests join after
  build step 7; the scripts migrate unchanged to remote CI later.
- **Boundary-weaving cost.** Under eager, with-boundary is one
  parameter read and one branch — negligible against FFI tensor
  operators; if profiling shows visible overhead, lang/ gets an
  expansion-time specialization for the eager path. Deferred until
  there is data.

## 5. Step 4 record: target model and operator inventory

**Target model**: HuggingFaceTB/SmolLM2-135M — pure LlamaForCausalLM
(hidden 576, 30 layers, GQA 9/3, intermediate 1536, vocab 49152, tied
embeddings, rope_theta 1e5, eps 1e-5, bf16, no attention bias),
Apache-2.0 and ungated; at 135M, CPU per-layer parity runs in seconds.
The phase-one "1B-class" acceptance later swaps in SmolLM2-1.7B — the
same family and operator set, weights only. Llama-3.2 is gated and not
an automation target.

**Three simplifying decisions** (all from the binding-time analysis):
linear weights pre-transposed at load so runtime needs only matmul (no
linear operator, no runtime transpose); RoPE cos/sin tables and the
causal mask are host-precomputed constants (baked at construction), so
no arange/cos/sin/triu operators; rms-norm stays decomposed (fusion is
phase-three eqsat work — a live example of IR-granularity/rule
co-design), softmax stays a single native operator (numeric stability,
and a rewrite target).

**Registry inventory (24 operators)**:
- elementwise (10): add, sub, mul, div, neg, abs, sqr, rsqrt, silu, cast
- reductions (3): mean (keep-dim), amax, argmax
- matmul (1): matmul (internally mm/bmm plus views)
- softmax (1)
- view/layout (6): reshape, transpose, narrow, cat, broadcast-to, unsqueeze
- embedding (1)
- state (2): kv-append, kv-read
- deferred (sampling extensions): topk, multinomial

**The usable surface, in three parts** (measured against torch 2.13.0:
libtorch_cpu exports 465 aoti_torch_* symbols):
1. **Covered directly by the Stable ABI**: lifecycle
   (create_tensor_from_blob_v2, empty_strided, delete, copy_, clone),
   metadata (sizes/strides/dtype/data_ptr/storage_offset), mm/bmm/addmm,
   SDPA flash v2, _reinterpret_tensor and as_strided, the full-dtype
   item_* and scalar_to_tensor_* families, dtype/device constant
   functions.
2. **Wrapper-free on the Racket side** (own stride math plus existing
   ABI combinations): the whole view family
   (reshape/transpose/narrow/broadcast-to/unsqueeze = shape.rkt stride
   algebra + _reinterpret_tensor), cat and kv-append (empty_strided +
   narrow views + copy_), cast (empty_strided at the target dtype +
   copy_, which converts dtypes), item/fetch (the item_* family),
   matmul's batched forms (views folded to 2D/3D, then mm/bmm).
3. **Needs the C++ wrapper**: the elementwise families, mean/amax,
   softmax, embedding/index_select, argmax (later topk/multinomial).

**Step 4 complete**: 22 operators × f32/bf16, 56 cases, byte-exact
parity all green (tests/operators.rkt, data-driven). Every deviation
from the plan was favorable: the custom wrapper shrank to **9 kernels**
(sub/div/neg/rsqrt/silu/mean/softmax/embedding/argmax) — torch 2.13's
generated per-backend shims (aoti_torch_cpu_*, 161 of them) directly
provide add/mul/abs/cat/mm/bmm, and aoti_torch_aten_amax is likewise
ready-made; the view family / cat / cast / matmul batch folding landed
on the Racket side as planned; kv-append/kv-read depend on the arena
and arrive with build step 6, verified by the differential harness.
Pitfalls on record: clone preserves layout for strided views
(preserve_format), so contiguization must materialize explicitly via
empty_strided + copy_; when the typed operator API returns tensors to
untyped model code, any-wrap contracts cannot protect opaque ctensors —
the hot-edge valve STYLE rule 8 anticipated (unsafe-provide) was pulled
for **correctness**, not performance, with ctensors typed as the opaque
CTensor. Two future freebies found in the generated shim:
_fused_rms_norm and _scaled_dot_product_flash_attention_for_cpu already
have C entry points — phase-three fusion targets at zero wrapper cost.
Also landed: VERSIONS (python 3.14 + torch 2.13.0 + SmolLM2-135M),
tools/setup-python.sh (uv creates the root .venv with
torch/numpy/safetensors — Python reference generation and the dlopened
libtorch dylibs share one installation, eliminating skew; raw.rkt
auto-detects .venv when ZHULONG_LIBTORCH is unset), runtime/ffi/raw.rkt
(dlopen, abi_version, dtype/device constants, symbol resolver),
tests/ffi-test.rkt (smoke, skips when libtorch is absent).

## 6. Step 5 record: weights and model forward

**Complete.** utils/safetensor.rkt (typed, pure IO: little-endian
header length, JSON directory, ranged reads; byte counts validated
against shape × dtype at open); the materialization glue landed in
eager.rkt as bytes->tensor (not tensor.rkt as first sketched — this
keeps tensor.rkt free of even indirect FFI paths), over an owning
bytes->ctensor in runtime/ffi/ops.rkt. models/llama.rkt implements the
Llama-family forward in branch-free dataflow form, faithful to HF
semantics: f32 rms-norm internals, f32 softmax, host-built rope/mask
tables (racket/flonum's flsingle mimics the reference's f32 arithmetic
chain) cast to the model dtype at use, GQA expansion via
unsqueeze/broadcast-to/reshape, rotate-half via narrow/neg/cat.

Two deviations from the plan, both deliberate. Linear weights are
**not** pre-transposed at load: transpose is a zero-copy stride view
and matmul's fold contiguizes once per call, so a `linear` layer
(matmul over a transposed view) costs one materialization — revisit
pre-transposition if decode profiling says so. And there is no
tolerances.rkt: the parity turned out **bit-exact** — 32 checkpoints
(embedding, all 30 blocks, logits) over SmolLM2-135M against
tools/gen-model-reference.py, worst relative difference 0.0 — so a
single inline 2% relative guard in tests/model-test.rkt stands as
unused headroom; a tolerance table earns its keep only when a backend
with different kernel selection arrives.

One real bug found and fixed on the way: the eager backend used to
materialize Real operands as f64 scalar tensors, which win ATen type
promotion (bf16 × 0.125 → f64); eager-run now converts scalars into the
dtype the tensor operands agreed on — the weak-scalar semantics the
registry's dtype rules had promised all along. Exercised by every
attention layer (scaling) and every rms-norm (eps).

Also landed: make weights (downloads SmolLM2-135M, gitignored under
weights/) and make model-reference; the full suite stands at 378 tests.

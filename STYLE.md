# Zhulong Style Guide

Canonical coding rules for this repository. This file is the single source of
truth — the implementation plan (debug/zhulong-impl-plan-zh.md §1) defers to
it — and doubles as context material for LLM-assisted coding: a conservative,
predictable subset of Racket is the countermeasure to the niche-language
corpus disadvantage.

## Paradigm

1. **Functional-first, data-oriented.** Data is immutable structs
   (`#:transparent`) and S-expressions; behavior is pure functions. Mutable
   state is allowed only in four named places: ctensors and the arena
   (runtime/), registry and definition tables at module-load time
   (operator.rkt, lang.rkt), trace context (tracer instances), and caches.
   New mutable state requires review.
2. **No racket/class, no units.** The project's only polymorphism is
   operator-apply dispatching on current-backend. A backend is a closed struct
   with function slots: eager is a singleton; the tracer creates a fresh
   instance per trace, carrying that trace's state.
3. **Macros do syntax and binding only, never computation.** Whitelist:
   define-layer, define-model, with-eager, with-tensor, defrule,
   define-operator (operator.rkt), define-operator-impl
   (runtime/ffi/ops.rkt), plus lang.rkt internals. Every macro expands to
   calls into plain, independently testable functions. syntax-parse
   appears only in modules that define whitelisted macros.
4. **Delimited control stays inside tracer.rkt.** prompt/abort/continuations
   appear in no other module and no public interface.
5. **Parameter whitelist:** current-backend; current-device (default
   placement for newly created tensors only — operators follow their
   inputs' device, and multi-device placement is a phase-4 plan annotation
   on the graph, never ambient state). Parameters are for true dynamic
   scoping; configuration is passed explicitly. Ownership: the cell
   belongs to the module defining the mechanism it configures
   (operator.rkt for dispatch, tensor.rkt for creation); only extent
   owners — mode entry points (trace, with-eager, the executor's island
   runner) and session setup — may parameterize it; everything in between
   neither reads nor writes it. Parameterize is required rather than a
   settable global because bindings must unwind correctly across the
   tracer's aborts and stay per-thread under concurrent requests.
6. **Strict evaluation throughout the core.** Laziness appears only in the
   phase-2 orchestration stream combinators.

## Conventions

7. Naming: kebab-case; predicates end in `?`; mutators end in `!`;
   conversions read `a->b`. Multi-word names lead with the noun and put
   the verb after it (operator-register!, ctensor-delete!,
   shape-broadcast), so names sharing a subject sort and complete
   together; exceptions are established Racket idioms (make-*,
   call-with-*, current-*, define-*) and `a->b` conversions. Private by
   default; export explicitly. Source stays ASCII: write `lambda`, never
   the `λ` glyph (editors may render it as the glyph; the file must not
   contain it).
8. **Typed by default; untyped where the machinery demands it.** Modules
   are `#lang typed/racket` except: lang.rkt (macros), tracer.rkt
   (delimited control), FFI binding modules (ffi/unsafe forms), models/
   and all user DSL code (dynamic by design — the shallow-embedding
   contract), and test files. Type honestly where precision pays
   (Shape/Dtype algebra, ShapeRule/DtypeRule signatures) and use `Any`
   for genuinely heterogeneous payloads (attr values, backend results),
   asserting at the seam into precise code. Measured facts: typed→typed
   calls are contract-free (~zero overhead); a typed↔untyped crossing
   costs up to ~1µs — noise against GPU-scale operator costs, and
   structurally irrelevant once hot paths compile away (AOTI artifacts,
   CUDA-graph replay). The payoff is maintainability: machine-checked
   interfaces double as a spec for LLM-assisted coding, catching
   plausible-but-wrong generated code at compile time — which offsets
   Typed Racket's thinner corpus. The untyped exceptions above exist for
   expressiveness, not performance. As hygiene, still keep crossings off
   per-op/per-token paths: install backends from typed code so values do
   not round-trip through untyped land; for the one unavoidable hot edge
   (untyped model code → the operator API), measure under eager
   execution and, if it shows, drop the generated contracts on that edge
   with typed/racket/unsafe (unsafe-provide) — tests then guard it. This
   valve is in effect: generated any-wrap contracts cannot protect an
   opaque ctensor at all, so the operator API surface (operator-apply, the
   operator functions, the eager backend) ships via unsafe-provide for
   correctness, with ctensors typed as the opaque CTensor. contract-out
   remains the boundary form for the public surfaces of untyped modules.
9. Destructure IR and S-expressions with racket/match; no car/cdr chains.
   Internal bindings use define and match-define, not let/let* — except
   for deliberate shadowing, a deliberately narrowed expression-local
   scope, or emphasizing a group of independent parallel bindings.
   Loops that need result annotations become annotated internal helpers
   (see rule 17).
10. Errors are structured exceptions carrying srcloc; FFI error codes are
    translated into exceptions at the safe layer and never leak upward.
11. **Tests live in tests/ by default**: one `<module>-test.rkt` per module,
    wrapped in `(module+ test ...)`. In-source `module+ test` is reserved
    for white-box tests that must reach unexported internals. Every module
    starts with one English header comment stating its responsibility and
    allowed dependency direction. Comments are written in English; no
    narrative inline comments. Comments never cite STYLE.md or the design
    documents by rule or section number — references rot; state the
    constraint itself in place. Comments describe what the code is, never
    the plan around it (no "until X lands", "phase-N placeholder", "will
    be added later") — schedule talk lives in the plan document.
12. One file, one responsibility.
13. Third-party runtime dependencies default to zero; additions require
    review. Currently: base, typed-racket-lib (main distribution),
    rackunit. Dev-only tools (fmt) are not runtime dependencies.

## Project rules

14. **Two views of the graph.** In memory the graph is an id-indexed node
    table (structs); serialized, displayed, and diffed it is S-expressions.
    Only graph.rkt provides the codec between the two; no other module
    assembles the textual form by hand.
15. **Require paths.** Relative within a collection (the flat layout keeps
    `..` to at most one level); collection paths across collections
    (models, tests → zhulong). Moves are compiler-checked: require is
    statically resolved, and raco setup reports every affected site.
16. **unsafe isolation.** Only runtime/ffi/ may require ffi/unsafe;
    raw.rkt is visible only inside runtime/ffi/. operator.rkt and
    graph.rkt stay FFI-free so graph construction, shape inference, and
    the linearity check are testable without libtorch.

## Documentation

18. Documentation lives in the source, for now: the module header comment
    (responsibility, dependency direction) plus the typed signatures are
    the per-function documentation. A Scribble manual is deliberately
    deferred — prose maintained in two places drifts, and the API is
    still moving. Revisit when the public API stabilizes, preferably via
    auto-generation from typed modules and their comments (potentially an
    upstream contribution: scribble/srcdoc-style extraction for Typed
    Racket); do not hand-maintain a parallel manual before then.

## Formatting

17. All .rkt files are formatted with raco fmt (default width); `make fmt`
    formats the tree, and formatting is never hand-tuned. Project forms
    are registered in `.fmt.rkt` at the repo root (define-layer,
    define-model, and struct format define-like); extend that file when
    new define-like or with-like forms are added. Typed Racket
    annotations use the standalone form — `(: name type)` before the
    definition, including for internal defines — never the infix
    `define`/`let`/`for` colon, which fmt renders poorly; a named let
    that needs a result annotation becomes an annotated internal helper
    instead.

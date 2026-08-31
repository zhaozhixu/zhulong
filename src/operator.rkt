#lang typed/racket
;; Operator registry — the single source of truth for operators — plus the
;; operator-apply dispatch point. operators/*.rkt fill the table at module load;
;; it is read-only afterwards.
;; Rule signatures are enforced types (ShapeRule/DtypeRule). Scalar
;; arguments stay Racket reals all the way down, mirroring ATen's
;; Tensor/Scalar overload split — they never materialize on device. Their
;; shape is concretely '(); their dtype is weak (PyTorch wrapped numbers /
;; JAX weak types): it defers to the tensor operands rather than
;; committing early, marked #f in Maybe-Dtype and resolved only by
;; dtype-resolve. Backend results stay Any on purpose: what a backend
;; returns is its own business (real tensors, symbolic tensors, test
;; doubles).
;; Operand versus attribute: an operand (a Value) participates in the
;; tensor dataflow — an edge in the graph — and may be a tensor or a
;; weak-dtyped scalar; an attribute configures the operator — node
;; metadata — and can never be a tensor (axis, keep-dim, a target shape,
;; a dtype). The test: could it ever be a tensor, or vary with runtime
;; data? Then it is an operand. This mirrors ATen's Tensor/Scalar
;; arguments versus attributes.

(require (for-syntax racket/base
                     syntax/parse)
         typed/racket/unsafe
         "tensor.rkt")

(provide define-operator
         Attrs
         Value
         Maybe-Dtype
         ShapeRule
         DtypeRule
         (struct-out operator)
         (struct-out backend)
         operator-register!
         operator-ref
         operator-names
         current-backend
         attr-ref
         value-shape
         value-dtype
         dtype-resolve
         same-dtype-rule)

;; operator-apply returns tensors to untyped model code on every call — the hot
;; edge where generated contracts are dropped (tests stand guard).
(unsafe-provide operator-apply)

(define-type Attrs (Listof (Pairof Symbol Any)))
(define-type Value (U Tensor Real))
;; #f marks the weak dtype of a scalar argument.
(define-type Maybe-Dtype (U Dtype #f))
(define-type ShapeRule (-> (Listof Shape) Attrs Shape))
(define-type DtypeRule (-> (Listof Maybe-Dtype) Attrs Dtype))

(struct operator
  ([name : Symbol] [shape-rule : ShapeRule] [dtype-rule : DtypeRule])
  #:transparent)

(struct backend
  ([name : Symbol] [run-op : (-> operator (Listof Value) Attrs Any)])
  #:transparent)

(: table (Mutable-HashTable Symbol operator))
(define table (make-hasheq))

(: operator-register! (-> operator Void))
(define (operator-register! op)
  (define name (operator-name op))
  (when (hash-has-key? table name)
    (raise-arguments-error 'operator-register! "duplicate operator" "name" name))
  (hash-set! table name op))

(: operator-ref (-> Symbol operator))
(define (operator-ref name)
  (hash-ref table
            name
            (lambda () (raise-arguments-error 'operator-ref "unknown operator" "name" name))))

(: operator-names (-> (Listof Symbol)))
(define (operator-names)
  (sort (hash-keys table) symbol<?))

;; The dispatch target of operator-apply. A parameter precisely because the same
;; model code runs under different backends in different dynamic extents
;; (eager call vs. trace); exactly one backend is current within an extent.
;; eager is a singleton; the tracer installs a fresh instance per trace that
;; carries that trace's state — there is deliberately no current-trace.
(: current-backend (Parameterof (Option backend)))
(define current-backend (make-parameter #f))

(: operator-apply (->* (Symbol (Listof Value)) (Attrs) Any))
(define (operator-apply name args [attrs '()])
  (define op (operator-ref name))
  (define b (current-backend))
  (unless b
    (raise-arguments-error 'operator-apply "no current backend" "operator" name))
  ((backend-run-op b) op args attrs))

(: missing Symbol)
(define missing (gensym 'missing))

(: attr-ref (->* (Attrs Symbol) (Any) Any))
(define (attr-ref attrs key [default missing])
  (define p (assq key attrs))
  (cond
    [p (cdr p)]
    [(eq? default missing)
     (raise-arguments-error 'attr-ref "missing attribute" "key" key "attrs" attrs)]
    [else default]))

(: value-shape (-> Value Shape))
(define (value-shape v)
  (if (real? v)
      '()
      (tensor-shape v)))

(: value-dtype (-> Value Maybe-Dtype))
(define (value-dtype v)
  (if (real? v)
      #f
      (tensor-dtype v)))

;; The one place weak-dtype deferral lives: strong dtypes must agree (no
;; promotion lattice yet — revisit with mixed precision), weak scalars
;; defer, and an all-scalar application is a user error: plain Racket
;; arithmetic does that job.
(: dtype-resolve (-> Symbol (Listof Maybe-Dtype) Dtype))
(define (dtype-resolve who dtypes)
  (: strong (Listof Dtype))
  (define strong
    (for/list ([d (in-list dtypes)]
               #:when d)
      d))
  (match strong
    ['()
     (raise-arguments-error who
                            "all-scalar application; use plain Racket arithmetic"
                            "dtypes"
                            dtypes)]
    [(cons d rest)
     (for ([d2 (in-list rest)])
       (unless (eq? d d2)
         (raise-arguments-error who "dtype mismatch" "dtypes" dtypes)))
     d]))

;; The common dtype rule: output dtype is the agreed dtype of the inputs.
(: same-dtype-rule (-> Symbol DtypeRule))
(define ((same-dtype-rule who) dtypes attrs)
  (dtype-resolve who dtypes))

(begin-for-syntax
  (define-syntax-class attr-spec
    #:datum-literals (:)
    (pattern [id:id : ty:expr]
      #:with decl #'[id : ty]
      #:with key #''id
      #:with ref #'id)
    (pattern [id:id : ty:expr default:expr]
      #:with decl #'[id : ty default]
      #:with key #''id
      #:with ref #'id)))

;; One declaration per operator: registers the semantic entry and defines
;; the user-facing keyword function over the uniform protocol. Attribute
;; keys are the parameter names; #:name overrides the registry symbol when
;; the function name differs from it (absolute vs. abs).
(define-syntax (define-operator stx)
  (syntax-parse stx
    [(_ (name:id operand:id ... (~seq kw:keyword a:attr-spec) ...)
        (~alt (~once (~seq #:shape shape-rule:expr))
              (~once (~seq #:dtype dtype-rule:expr))
              (~optional (~seq #:name op-name:id))) ...)
     #'(begin
         (operator-register! (operator (~? 'op-name 'name) shape-rule dtype-rule))
         (define (name [operand : Value] ... (~@ kw a.decl) ...)
           :
           Any
           (operator-apply (~? 'op-name 'name) (list operand ...) (list (cons a.key a.ref) ...))))]))

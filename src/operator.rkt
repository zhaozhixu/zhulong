#lang typed/racket
;; Operator registry — the single source of truth for operators — plus the
;; apply-op dispatch point (the only polymorphism site in the project).
;; operators/*.rkt fill the table at module load; it is read-only afterwards.
;; Rule signatures are enforced types (ShapeRule/DtypeRule). Scalar
;; arguments stay Racket reals all the way down, mirroring ATen's
;; Tensor/Scalar overload split — they never materialize on device. Their
;; shape is concretely '(); their dtype is weak (PyTorch wrapped numbers /
;; JAX weak types): it defers to the tensor operands rather than
;; committing early, marked #f in Maybe-Dtype and resolved only by
;; resolve-dtype. Backend results stay Any on purpose: what a backend
;; returns is its own business (real tensors, symbolic tensors, test
;; doubles).

(require "tensor.rkt")

(provide Attrs
         Value
         Maybe-Dtype
         ShapeRule
         DtypeRule
         (struct-out operator)
         (struct-out backend)
         register-operator!
         operator-ref
         operator-names
         current-backend
         apply-op
         attr-ref
         value-shape
         value-dtype
         resolve-dtype)

(define-type Attrs (Listof (Pairof Symbol Any)))
(define-type Value (U Tensor Real))
;; #f marks the weak dtype of a scalar argument.
(define-type Maybe-Dtype (U Dtype #f))
(define-type ShapeRule (-> (Listof Shape) Attrs Shape))
(define-type DtypeRule (-> (Listof Maybe-Dtype) Attrs Dtype))

;; c-template feeds wrapper generation (#f until the FFI lands); fx-template
;; is a phase-2 placeholder.
(struct operator
  ([name : Symbol] [shape-rule : ShapeRule]
                   [dtype-rule : DtypeRule]
                   [c-template : Any]
                   [fx-template : Any])
  #:transparent)

(struct backend
  ([name : Symbol] [run-op : (-> operator (Listof Value) Attrs Any)])
  #:transparent)

(: table (Mutable-HashTable Symbol operator))
(define table (make-hasheq))

(: register-operator! (-> operator Void))
(define (register-operator! op)
  (define name (operator-name op))
  (when (hash-has-key? table name)
    (raise-arguments-error 'register-operator! "duplicate operator" "name" name))
  (hash-set! table name op))

(: operator-ref (-> Symbol operator))
(define (operator-ref name)
  (hash-ref table name (λ () (raise-arguments-error 'operator-ref "unknown operator" "name" name))))

(: operator-names (-> (Listof Symbol)))
(define (operator-names)
  (sort (hash-keys table) symbol<?))

;; The dispatch target of apply-op. A parameter precisely because the same
;; model code runs under different backends in different dynamic extents
;; (eager call vs. trace); exactly one backend is current within an extent.
;; eager is a singleton; the tracer installs a fresh instance per trace that
;; carries that trace's state — there is deliberately no current-trace.
(: current-backend (Parameterof (Option backend)))
(define current-backend (make-parameter #f))

(: apply-op (->* (Symbol (Listof Value)) (Attrs) Any))
(define (apply-op name args [attrs '()])
  (define op (operator-ref name))
  (define b (current-backend))
  (unless b
    (raise-arguments-error 'apply-op "no current backend" "operator" name))
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
(: resolve-dtype (-> Symbol (Listof Maybe-Dtype) Dtype))
(define (resolve-dtype who dtypes)
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

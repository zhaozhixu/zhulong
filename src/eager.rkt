#lang typed/racket
;; The eager backend: unwraps real-tensor arguments to ctensors, runs
;; the operator through runtime/ffi/ops, and wraps the resulting ctensor
;; back into a real-tensor, reading shape and dtype from the ctensor
;; itself as ground truth.

(require typed/racket/unsafe
         "tensor.rkt"
         "operator.rkt")

;; unsafe: ctensors are opaque cpointers, which the generated
;; any-wrap/c contracts cannot protect — and this is the per-operator hot
;; edge anyway. Correctness is guarded by the FFI smoke tests and the
;; operator parity suite.
(unsafe-require/typed "runtime/ffi/ops.rkt"
                      [eager-available? (-> Boolean)]
                      [eager-run (-> Symbol (Listof Any) (Listof (Pairof Symbol Any)) CTensor)]
                      [ctensor-shape (-> CTensor (Listof Integer))]
                      [ctensor-dtype-sym (-> CTensor Symbol)])

(provide eager-available?)

;; The backend value and the thunk result both carry tensors across the
;; typed/untyped line — same hot edge, same treatment.
(unsafe-provide eager-backend call-with-eager)

(: unwrap (-> Value Any))
(define (unwrap v)
  (cond
    [(real-tensor? v) (real-tensor-handle v)]
    [(sym-tensor? v) (error 'eager "symbolic tensor under the eager backend")]
    [else v]))

(: run-op (-> operator (Listof Value) Attrs Any))
(define (run-op op args attrs)
  (define h (eager-run (operator-name op) (map unwrap args) attrs))
  (real-tensor (assert (ctensor-shape h) shape?) (assert (ctensor-dtype-sym h) dtype?) h 'cpu))

(define eager-backend (backend 'eager run-op))

(: call-with-eager (All (A) (-> (-> A) A)))
(define (call-with-eager thunk)
  (parameterize ([current-backend eager-backend])
    (thunk)))

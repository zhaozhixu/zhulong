#lang typed/racket
;; Registry entries and user-facing functions: elementwise family plus the
;; reductions mean/amax/argmax. Dtypes resolve through dtype-resolve:
;; strong dtypes must agree, weak scalars defer. attr values arrive as Any
;; and are asserted at the seam into the precisely typed shape algebra.

(require typed/racket/unsafe
         "../shape.rkt"
         "../tensor.rkt"
         "../operator.rkt")

;; Operator functions return tensors to untyped model code — the hot edge
;; where generated contracts are dropped (tests stand guard).
(unsafe-provide add sub mul div neg absolute sqr rsqrt silu to-dtype mean amax argmax)

(: broadcast-shape-rule (-> Symbol ShapeRule))
(define ((broadcast-shape-rule who) shapes attrs)
  (match shapes
    [(list a b)
     (or (shape-broadcast a b)
         (raise-arguments-error who "shapes not broadcastable" "left" a "right" b))]
    [_ (raise-arguments-error who "expects 2 inputs" "got" (length shapes))]))

(: unary-shape-rule ShapeRule)
(define (unary-shape-rule shapes attrs)
  (match shapes
    [(list s) s]
    [_ (raise-arguments-error 'unary-shape-rule "expects 1 input" "got" (length shapes))]))

(: reduce-shape-rule ShapeRule)
(define (reduce-shape-rule shapes attrs)
  (match shapes
    [(list s)
     (shape-reduce s
                   (assert (attr-ref attrs 'axis) exact-integer?)
                   (and (attr-ref attrs 'keep-dim #f) #t))]
    [_ (raise-arguments-error 'reduce-shape-rule "expects 1 input" "got" (length shapes))]))

(: cast-dtype-rule DtypeRule)
(define (cast-dtype-rule dtypes attrs)
  (assert (attr-ref attrs 'dtype) dtype?))

(: i64-dtype-rule DtypeRule)
(define (i64-dtype-rule dtypes attrs)
  'i64)

(define-operator (add a b)
  #:shape (broadcast-shape-rule 'add)
  #:dtype (same-dtype-rule 'add))

(define-operator (sub a b)
  #:shape (broadcast-shape-rule 'sub)
  #:dtype (same-dtype-rule 'sub))

(define-operator (mul a b)
  #:shape (broadcast-shape-rule 'mul)
  #:dtype (same-dtype-rule 'mul))

(define-operator (div a b)
  #:shape (broadcast-shape-rule 'div)
  #:dtype (same-dtype-rule 'div))

(define-operator (neg x)
  #:shape unary-shape-rule
  #:dtype (same-dtype-rule 'neg))

(define-operator (absolute x)
  #:name abs
  #:shape unary-shape-rule
  #:dtype (same-dtype-rule 'abs))

(define-operator (sqr x)
  #:shape unary-shape-rule
  #:dtype (same-dtype-rule 'sqr))

(define-operator (rsqrt x)
  #:shape unary-shape-rule
  #:dtype (same-dtype-rule 'rsqrt))

(define-operator (silu x)
  #:shape unary-shape-rule
  #:dtype (same-dtype-rule 'silu))

(define-operator (mean x #:axis [axis : Integer] #:keep-dim? [keep-dim : Any #f])
  #:shape reduce-shape-rule
  #:dtype (same-dtype-rule 'mean))

(define-operator (amax x #:axis [axis : Integer] #:keep-dim? [keep-dim : Any #f])
  #:shape reduce-shape-rule
  #:dtype (same-dtype-rule 'amax))

(define-operator (argmax x #:axis [axis : Integer] #:keep-dim? [keep-dim : Any #f])
  #:shape reduce-shape-rule
  #:dtype i64-dtype-rule)

(define-operator (to-dtype x #:dtype [dtype : Dtype])
  #:name cast
  #:shape unary-shape-rule
  #:dtype cast-dtype-rule)

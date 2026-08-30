#lang typed/racket
;; Registry entries and user-facing functions: elementwise family plus mean.
;; Dtypes resolve through resolve-dtype: strong dtypes must agree, weak
;; scalars defer. attr values arrive as Any and are asserted at the seam
;; into the precisely typed shape algebra.

(require "../shape.rkt"
         "../tensor.rkt"
         "../operator.rkt")

(provide add
         mul
         sqr
         rsqrt
         mean)

(: broadcast-shape-rule (-> Symbol ShapeRule))
(define ((broadcast-shape-rule who) shapes attrs)
  (match shapes
    [(list a b)
     (or (broadcast-shapes a b)
         (raise-arguments-error who "shapes not broadcastable" "left" a "right" b))]
    [_ (raise-arguments-error who "expects 2 inputs" "got" (length shapes))]))

(: same-dtype-rule (-> Symbol DtypeRule))
(define ((same-dtype-rule who) dtypes attrs)
  (resolve-dtype who dtypes))

(: unary-shape-rule ShapeRule)
(define (unary-shape-rule shapes attrs)
  (match shapes
    [(list s) s]
    [_ (raise-arguments-error 'unary-shape-rule "expects 1 input" "got" (length shapes))]))

(: mean-shape-rule ShapeRule)
(define (mean-shape-rule shapes attrs)
  (match shapes
    [(list s)
     (reduce-shape s
                   (assert (attr-ref attrs 'axis) exact-integer?)
                   (and (attr-ref attrs 'keep-dim #f) #t))]
    [_ (raise-arguments-error 'mean "expects 1 input" "got" (length shapes))]))

(register-operator! (operator 'add (broadcast-shape-rule 'add) (same-dtype-rule 'add) #f #f))
(register-operator! (operator 'mul (broadcast-shape-rule 'mul) (same-dtype-rule 'mul) #f #f))
(register-operator! (operator 'sqr unary-shape-rule (same-dtype-rule 'sqr) #f #f))
(register-operator! (operator 'rsqrt unary-shape-rule (same-dtype-rule 'rsqrt) #f #f))
(register-operator! (operator 'mean mean-shape-rule (same-dtype-rule 'mean) #f #f))

(: add (-> Value Value Any))
(define (add a b)
  (apply-op 'add (list a b)))

(: mul (-> Value Value Any))
(define (mul a b)
  (apply-op 'mul (list a b)))

(: sqr (-> Value Any))
(define (sqr x)
  (apply-op 'sqr (list x)))

(: rsqrt (-> Value Any))
(define (rsqrt x)
  (apply-op 'rsqrt (list x)))

(: mean (-> Value #:axis Integer [#:keep-dim? Any] Any))
(define (mean x #:axis axis #:keep-dim? [keep-dim #f])
  (apply-op 'mean (list x) (list (cons 'axis axis) (cons 'keep-dim keep-dim))))

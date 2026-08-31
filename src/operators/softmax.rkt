#lang typed/racket
;; softmax registry entry: kept as a single native operator for numeric
;; stability, and as a rewrite target.

(require typed/racket/unsafe
         "../shape.rkt"
         "../tensor.rkt"
         "../operator.rkt")

(unsafe-provide softmax)

(: softmax-shape-rule ShapeRule)
(define (softmax-shape-rule shapes attrs)
  (match shapes
    [(list s)
     (define rank (length s))
     (define ax (assert (attr-ref attrs 'axis) exact-integer?))
     (define nd
       (if (negative? ax)
           (+ rank ax)
           ax))
     (unless (and (>= nd 0) (< nd rank))
       (raise-arguments-error 'softmax "axis out of range" "axis" ax "shape" s))
     s]
    [_ (raise-arguments-error 'softmax "expects 1 input" "got" (length shapes))]))

(define-operator (softmax x #:axis [axis : Integer])
  #:shape softmax-shape-rule
  #:dtype (same-dtype-rule 'softmax))

#lang typed/racket

(provide Shape
         shape?
         shape-numel
         shape-broadcast
         shape-reduce)

(define-type Shape (Listof Nonnegative-Integer))

(define-predicate shape? Shape)

(: shape-numel (-> Shape Nonnegative-Integer))
(define (shape-numel s)
  (apply * s))

(: merge-dim (-> Nonnegative-Integer Nonnegative-Integer (U Nonnegative-Integer #f)))
(define (merge-dim da db)
  (cond
    [(= da db) da]
    [(= da 1) db]
    [(= db 1) da]
    [else #f]))

;; numpy broadcasting: right-aligned, size-1 dims stretch; #f if incompatible.
(: shape-broadcast (-> Shape Shape (U Shape #f)))
(define (shape-broadcast a b)
  (: go (-> Shape Shape Shape (U Shape #f)))
  (define (go ra rb acc)
    (match* (ra rb)
      [('() '()) acc]
      [('() (cons d rest)) (go '() rest (cons d acc))]
      [((cons d rest) '()) (go rest '() (cons d acc))]
      [((cons da rest-a) (cons db rest-b))
       (define d (merge-dim da db))
       (and d (go rest-a rest-b (cons d acc)))]))
  (go (reverse a) (reverse b) '()))

;; Result shape of a reduction: normalizes a possibly-negative axis, then
;; drops that axis (keep? = #f) or pins it to 1 (keep? = #t).
(: shape-reduce (-> Shape Integer Boolean Shape))
(define (shape-reduce s axis keep?)
  (define rank (length s))
  (define ax
    (if (negative? axis)
        (+ rank axis)
        axis))
  (unless (and (>= ax 0) (< ax rank))
    (raise-arguments-error 'shape-reduce "axis out of range" "axis" axis "shape" s))
  (: dims Shape)
  (define dims
    (for/list ([d (in-list s)]
               [i (in-naturals)]
               #:unless (and (= i ax) (not keep?)))
      (if (= i ax) 1 d)))
  dims)

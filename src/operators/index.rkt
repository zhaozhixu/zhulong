#lang typed/racket
;; View, layout, and indexing entries: embedding, reshape, transpose,
;; narrow, cat, broadcast-to, unsqueeze. Views are pure stride math at
;; execution time; here live their shape rules and user functions.

(require typed/racket/unsafe
         racket/list
         "../shape.rkt"
         "../tensor.rkt"
         "../operator.rkt")

(unsafe-provide embedding reshape transpose narrow cat broadcast-to unsqueeze)

(: aint (-> Symbol Attrs Symbol Integer))
(define (aint who attrs key)
  (assert (attr-ref attrs key) exact-integer?))

(: ashape (-> Attrs Shape))
(define (ashape attrs)
  (assert (attr-ref attrs 'shape) shape?))

(: norm-axis (-> Symbol Integer Integer Integer))
(define (norm-axis who rank ax)
  (define nd
    (if (negative? ax)
        (+ rank ax)
        ax))
  (unless (and (>= nd 0) (< nd rank))
    (raise-arguments-error who "axis out of range" "axis" ax "rank" rank))
  nd)

(: one-input (-> Symbol (Listof Shape) Shape))
(define (one-input who shapes)
  (match shapes
    [(list s) s]
    [_ (raise-arguments-error who "expects 1 input" "got" (length shapes))]))

;; --- embedding ---

(: embedding-shape-rule ShapeRule)
(define (embedding-shape-rule shapes attrs)
  (match shapes
    [(list ws is)
     (unless (= 2 (length ws))
       (raise-arguments-error 'embedding "weight must be rank 2" "weight" ws))
     (append is (list (second ws)))]
    [_ (raise-arguments-error 'embedding "expects 2 inputs" "got" (length shapes))]))

(: embedding-dtype-rule DtypeRule)
(define (embedding-dtype-rule dtypes attrs)
  (match dtypes
    [(list wd id)
     (unless (eq? id 'i64)
       (raise-arguments-error 'embedding "indices must be i64" "indices-dtype" id))
     (assert wd dtype?)]
    [_ (raise-arguments-error 'embedding "expects 2 inputs" "got" (length dtypes))]))

(define-operator (embedding weight indices)
  #:shape embedding-shape-rule
  #:dtype embedding-dtype-rule)

;; --- reshape ---

(: reshape-shape-rule ShapeRule)
(define (reshape-shape-rule shapes attrs)
  (define s (one-input 'reshape shapes))
  (define target (ashape attrs))
  (unless (= (shape-numel s) (shape-numel target))
    (raise-arguments-error 'reshape "element counts differ" "shape" s "target" target))
  target)

(define-operator (reshape x #:shape [shape : Shape])
  #:shape reshape-shape-rule
  #:dtype (same-dtype-rule 'reshape))

;; --- transpose ---

(: transpose-shape-rule ShapeRule)
(define (transpose-shape-rule shapes attrs)
  (define s (one-input 'transpose shapes))
  (define rank (length s))
  (define d0 (norm-axis 'transpose rank (aint 'transpose attrs 'dim0)))
  (define d1 (norm-axis 'transpose rank (aint 'transpose attrs 'dim1)))
  (list-set (list-set s d0 (list-ref s d1)) d1 (list-ref s d0)))

(define-operator (transpose x #:dim0 [dim0 : Integer] #:dim1 [dim1 : Integer])
  #:shape transpose-shape-rule
  #:dtype (same-dtype-rule 'transpose))

;; --- narrow ---

(: narrow-shape-rule ShapeRule)
(define (narrow-shape-rule shapes attrs)
  (define s (one-input 'narrow shapes))
  (define rank (length s))
  (define d (norm-axis 'narrow rank (aint 'narrow attrs 'axis)))
  (define start (aint 'narrow attrs 'start))
  (define len (aint 'narrow attrs 'len))
  (unless (and (>= start 0) (>= len 0) (<= (+ start len) (list-ref s d)))
    (raise-arguments-error 'narrow
                           "slice out of range"
                           "start"
                           start
                           "len"
                           len
                           "size"
                           (list-ref s d)))
  (list-set s d (assert len exact-nonnegative-integer?)))

(define-operator (narrow x #:axis [axis : Integer] #:start [start : Integer] #:len [len : Integer])
  #:shape narrow-shape-rule
  #:dtype (same-dtype-rule 'narrow))

;; --- cat ---

(: cat-shape-rule ShapeRule)
(define (cat-shape-rule shapes attrs)
  (when (null? shapes)
    (raise-arguments-error 'cat "expects at least 1 input"))
  (define s0 (first shapes))
  (define rank (length s0))
  (define d (norm-axis 'cat rank (aint 'cat attrs 'axis)))
  (for ([s (in-list (rest shapes))])
    (unless (= (length s) rank)
      (raise-arguments-error 'cat "ranks disagree" "first" s0 "other" s))
    (for ([a (in-list s0)]
          [b (in-list s)]
          [i (in-naturals)])
      (unless (or (= i d) (= a b))
        (raise-arguments-error 'cat "shapes disagree outside the cat axis" "first" s0 "other" s))))
  (define total
    (for/fold ([acc
                :
                Integer
                0])
              ([s (in-list shapes)])
      (+ acc (list-ref s d))))
  (list-set s0 d (assert total exact-nonnegative-integer?)))

;; cat is variadic over its operands — outside the fixed-arity header
;; define-operator generates — so its entry and function stay hand-written.
(operator-register! (operator 'cat cat-shape-rule (same-dtype-rule 'cat)))

(: cat (->* (#:axis Integer) #:rest Value Any))
(define (cat #:axis axis . ts)
  (operator-apply 'cat ts (list (cons 'axis axis))))

;; --- broadcast-to ---

(: broadcast-to-shape-rule ShapeRule)
(define (broadcast-to-shape-rule shapes attrs)
  (define s (one-input 'broadcast-to shapes))
  (define target (ashape attrs))
  (define pad (- (length target) (length s)))
  (unless (>= pad 0)
    (raise-arguments-error 'broadcast-to "target rank too small" "shape" s "target" target))
  (for ([a (in-list (append (make-list pad 1) s))]
        [t (in-list target)])
    (unless (or (= a t) (= a 1))
      (raise-arguments-error 'broadcast-to "shape not broadcastable" "shape" s "target" target)))
  target)

(define-operator (broadcast-to x #:shape [shape : Shape])
  #:shape broadcast-to-shape-rule
  #:dtype (same-dtype-rule 'broadcast-to))

;; --- unsqueeze ---

(: unsqueeze-shape-rule ShapeRule)
(define (unsqueeze-shape-rule shapes attrs)
  (define s (one-input 'unsqueeze shapes))
  (define d (norm-axis 'unsqueeze (add1 (length s)) (aint 'unsqueeze attrs 'axis)))
  (append (take s d) (list 1) (drop s d)))

(define-operator (unsqueeze x #:axis [axis : Integer])
  #:shape unsqueeze-shape-rule
  #:dtype (same-dtype-rule 'unsqueeze))

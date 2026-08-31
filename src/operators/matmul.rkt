#lang typed/racket
;; matmul registry entry. matmul-plan is the single rank analysis shared
;; by the shape rule and the eager implementation: every supported case
;; (2Dx2D, NDx2D, equal-batch NDxND) folds into one mm or bmm call, and
;; the plan reports the folded operand shapes, the folded output shape,
;; and the final output shape.

(require typed/racket/unsafe
         racket/list
         "../shape.rkt"
         "../tensor.rkt"
         "../operator.rkt")

(provide matmul-plan)

(unsafe-provide matmul)

(: matmul-plan (-> Shape Shape (Values (U 'mm 'bmm) Shape Shape Shape Shape)))
(define (matmul-plan sa sb)
  (define ra (length sa))
  (define rb (length sb))
  (define (inner-check [k1 : Nonnegative-Integer] [k2 : Nonnegative-Integer])
    (unless (= k1 k2)
      (raise-arguments-error 'matmul "inner dimensions disagree" "a" sa "b" sb)))
  (cond
    [(and (= ra 2) (= rb 2))
     (inner-check (second sa) (first sb))
     (define out (list (first sa) (second sb)))
     (values 'mm sa sb out out)]
    [(and (>= ra 3) (= rb 2))
     (inner-check (last sa) (first sb))
     (define lead (drop-right sa 1))
     (define m (apply * lead))
     (values 'mm (list m (last sa)) sb (list m (second sb)) (append lead (list (second sb))))]
    [(and (= ra rb) (>= ra 3))
     (unless (equal? (drop-right sa 2) (drop-right sb 2))
       (raise-arguments-error 'matmul "batch dimensions disagree" "a" sa "b" sb))
     (inner-check (last sa) (list-ref sb (- rb 2)))
     (define batch (drop-right sa 2))
     (define bsz (apply * batch))
     (define m (list-ref sa (- ra 2)))
     (define n (last sb))
     (values 'bmm
             (list bsz m (last sa))
             (list bsz (last sa) n)
             (list bsz m n)
             (append batch (list m n)))]
    [else (raise-arguments-error 'matmul "unsupported rank combination" "a" sa "b" sb)]))

(: matmul-shape-rule ShapeRule)
(define (matmul-shape-rule shapes attrs)
  (match shapes
    [(list sa sb)
     (define-values (kind af bf of out) (matmul-plan sa sb))
     out]
    [_ (raise-arguments-error 'matmul "expects 2 inputs" "got" (length shapes))]))

(define-operator (matmul a b)
  #:shape matmul-shape-rule
  #:dtype (same-dtype-rule 'matmul))

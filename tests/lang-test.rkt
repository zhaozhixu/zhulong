#lang racket/base
(module+ test
  (require rackunit
           zhulong/lang)

  (define-layer (double x) (* x 2))
  (define-model (quad x) (double (double x)))
  (define-layer (scale x #:by [k 2]) (* x k))

  (check-equal? (double 3) 6)
  (check-equal? (quad 3) 12)
  (check-equal? (scale 3) 6)
  (check-equal? (scale 3 #:by 10) 30)

  (check-equal? (def-info-kind (definition-info 'double)) 'layer)
  (check-equal? (def-info-kind (definition-info 'quad)) 'model)
  (check-false (definition-info 'nope))
  (check-not-false (memq 'scale (definition-names)))

  (define loc (def-info-srcloc (definition-info 'double)))
  (check-true (srcloc? loc))
  (check-true (exact-positive-integer? (srcloc-line loc))))

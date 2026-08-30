#lang racket/base
(module+ test
  (require rackunit
           zhulong/tensor)

  (check-equal? (dtype-size 'bf16) 2)
  (check-false (dtype? 'f64))

  (check-true (shape? '()))
  (check-true (shape? '(2 3)))
  (check-false (shape? '(2 -1)))
  (check-equal? (shape-numel '()) 1)
  (check-equal? (shape-numel '(2 3 4)) 24)
  (check-equal? (shape-numel '(0 3)) 0)

  (check-equal? (broadcast-shapes '(2 3) '(1 3)) '(2 3))
  (check-equal? (broadcast-shapes '(3) '(2 3)) '(2 3))
  (check-equal? (broadcast-shapes '(2 1 4) '(5 4)) '(2 5 4))
  (check-equal? (broadcast-shapes '() '(2 3)) '(2 3))
  (check-equal? (broadcast-shapes '(0 3) '(1 3)) '(0 3))
  (check-false (broadcast-shapes '(2 3) '(4 3)))

  (define t (sym-tensor '(2 3) 'f32 0))
  (check-true (tensor? t))
  (check-equal? (tensor-shape t) '(2 3))
  (check-equal? (tensor-dtype t) 'f32)
  (check-equal? (tensor-rank t) 2)
  (check-equal? (tensor-numel t) 6)

  (check-exn exn:fail? (λ () (current-device 'tpu))))

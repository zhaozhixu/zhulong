#lang racket/base
(module+ test
  (require rackunit
           zhulong/operator
           zhulong/operators/elementwise
           zhulong/tensor)

  (define (shape-of name shapes [attrs '()])
    ((operator-shape-rule (operator-ref name)) shapes attrs))
  (define (dtype-of name dtypes)
    ((operator-dtype-rule (operator-ref name)) dtypes '()))

  (check-equal? (shape-of 'add '((2 3) (1 3))) '(2 3))
  (check-equal? (shape-of 'add '((2 3) ())) '(2 3))
  (check-exn #rx"not broadcastable" (lambda () (shape-of 'add '((2 3) (4 3)))))
  (check-equal? (shape-of 'sqr '((2 3))) '(2 3))
  (check-equal? (shape-of 'mean '((2 3 4)) '((axis . -1) (keep-dim . #t))) '(2 3 1))
  (check-equal? (shape-of 'mean '((2 3 4)) '((axis . -1))) '(2 3))
  (check-equal? (shape-of 'mean '((2 3 4)) '((axis . 0) (keep-dim . #f))) '(3 4))
  (check-equal? (shape-of 'mean '((3)) '((axis . 0))) '())
  (check-exn #rx"axis out of range" (lambda () (shape-of 'mean '((2 3)) '((axis . 5)))))

  (check-equal? (dtype-of 'add '(bf16 bf16)) 'bf16)
  (check-equal? (dtype-of 'add '(bf16 #f)) 'bf16)
  (check-exn #rx"dtype mismatch" (lambda () (dtype-of 'add '(bf16 f32))))
  (check-exn #rx"all-scalar" (lambda () (dtype-of 'add '(#f #f))))

  (define echo
    (backend 'echo (lambda (op args attrs) (list (operator-name op) (map value-shape args) attrs))))
  (parameterize ([current-backend echo])
    (define x (sym-tensor '(2 3) 'f32 0))
    (check-equal? (add x 1.0) '(add ((2 3) ()) ()))
    (check-equal? (mean x #:axis -1 #:keep-dim? #t) '(mean ((2 3)) ((axis . -1) (keep-dim . #t))))))

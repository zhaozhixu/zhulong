#lang racket/base
(module+ test
  (require rackunit
           racket/list
           zhulong/operator
           zhulong/tensor)

  (define toy
    (operator 'toy (lambda (shapes attrs) (first shapes)) (lambda (dtypes attrs) (first dtypes))))
  (operator-register! toy)
  (check-exn #rx"duplicate operator" (lambda () (operator-register! toy)))
  (check-exn #rx"unknown operator" (lambda () (operator-ref 'nope)))
  (check-not-false (memq 'toy (operator-names)))

  (check-exn #rx"no current backend"
             (lambda () (operator-apply 'toy (list (sym-tensor '(1) 'f32 0)))))

  (define echo (backend 'echo (lambda (op args attrs) (list (operator-name op) (length args) attrs))))
  (parameterize ([current-backend echo])
    (check-equal? (operator-apply 'toy (list 1 2) '((k . 3))) '(toy 2 ((k . 3)))))

  (check-equal? (attr-ref '((axis . -1)) 'axis) -1)
  (check-equal? (attr-ref '() 'axis 'none) 'none)
  (check-exn #rx"missing attribute" (lambda () (attr-ref '() 'axis)))

  (check-equal? (value-shape 1.5) '())
  (check-equal? (value-shape (sym-tensor '(2 3) 'f32 0)) '(2 3))
  (check-false (value-dtype 1.5))
  (check-equal? (value-dtype (sym-tensor '(2 3) 'bf16 0)) 'bf16))

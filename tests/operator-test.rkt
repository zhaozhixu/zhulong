#lang racket/base
(module+ test
  (require rackunit
           racket/list
           zhulong/operator
           zhulong/tensor)

  (define toy
    (operator 'toy (λ (shapes attrs) (first shapes)) (λ (dtypes attrs) (first dtypes)) #f #f))
  (register-operator! toy)
  (check-exn #rx"duplicate operator" (λ () (register-operator! toy)))
  (check-exn #rx"unknown operator" (λ () (operator-ref 'nope)))
  (check-not-false (memq 'toy (operator-names)))

  (check-exn #rx"no current backend" (λ () (apply-op 'toy (list (sym-tensor '(1) 'f32 0)))))

  (define echo (backend 'echo (λ (op args attrs) (list (operator-name op) (length args) attrs))))
  (parameterize ([current-backend echo])
    (check-equal? (apply-op 'toy (list 1 2) '((k . 3))) '(toy 2 ((k . 3)))))

  (check-equal? (attr-ref '((axis . -1)) 'axis) -1)
  (check-equal? (attr-ref '() 'axis 'none) 'none)
  (check-exn #rx"missing attribute" (λ () (attr-ref '() 'axis)))

  (check-equal? (value-shape 1.5) '())
  (check-equal? (value-shape (sym-tensor '(2 3) 'f32 0)) '(2 3))
  (check-false (value-dtype 1.5))
  (check-equal? (value-dtype (sym-tensor '(2 3) 'bf16 0)) 'bf16))

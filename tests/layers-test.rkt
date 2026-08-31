#lang racket/base
(module+ test
  (require rackunit
           zhulong/lang
           zhulong/operator
           zhulong/tensor
           models/layers)

  (check-true (procedure? rms-norm))
  (check-equal? (def-info-kind (definition-info 'rms-norm)) 'layer)
  (check-exn #rx"no current backend" (lambda () (rms-norm (sym-tensor '(2 3) 'f32 0))))

  ;; Shape-propagating spy: a minimal symbolic backend that checks the call
  ;; sequence and the keep-dim broadcast through rms-norm.
  (define seen '())
  (define spy
    (backend 'spy
             (lambda (op args attrs)
               (define shape ((operator-shape-rule op) (map value-shape args) attrs))
               (define dtype ((operator-dtype-rule op) (map value-dtype args) attrs))
               (set! seen (cons (operator-name op) seen))
               (sym-tensor shape dtype (length seen)))))
  (parameterize ([current-backend spy])
    (define y (rms-norm (sym-tensor '(2 4) 'bf16 0)))
    (check-equal? (tensor-shape y) '(2 4))
    (check-equal? (tensor-dtype y) 'bf16)
    (check-equal? (reverse seen) '(sqr mean add rsqrt mul))))

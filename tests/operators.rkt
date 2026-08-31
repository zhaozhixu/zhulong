#lang racket/base
(module+ test
  (require racket/file
           racket/list
           racket/match
           racket/runtime-path
           rackunit
           (only-in zhulong)
           zhulong/eager
           zhulong/operator
           zhulong/tensor
           (prefix-in ops: zhulong/runtime/ffi/ops)
           (prefix-in raw: zhulong/runtime/ffi/raw))

  (define-runtime-path data-dir "data/ops")

  (define (dtype-size* dt)
    (case dt
      [(f32) 4]
      [(bf16) 2]
      [(i64) 8]))

  (define (decode dt bs)
    (define w (dtype-size* dt))
    (for/list ([i (in-range (quotient (bytes-length bs) w))])
      (case dt
        [(f32) (floating-point-bytes->real bs #f (* i 4) (* (add1 i) 4))]
        [(i64) (integer-bytes->integer bs #t #f (* i 8) (* (add1 i) 8))]
        [(bf16)
         (floating-point-bytes->real (bytes 0 0 (bytes-ref bs (* i 2)) (bytes-ref bs (add1 (* i 2))))
                                     #f)])))

  (define (load-tensor fn shape dt)
    (define bs (file->bytes (build-path data-dir fn)))
    (real-tensor shape dt (raw:bytes->ctensor dt shape bs) 'cpu))

  (cond
    [(and (ops:eager-available?) (file-exists? (build-path data-dir "cases.rktd")))
     (define cases (with-input-from-file (build-path data-dir "cases.rktd") read))
     (for ([c (in-list cases)])
       (define (field k)
         (cdr (assq k c)))
       (define name (field 'name))
       (define args
         (for/list ([in (in-list (field 'inputs))])
           (match in
             [(list 'scalar v) v]
             [(list 'tensor fn shape dt) (load-tensor fn shape dt)])))
       (match-define (list ofn oshape odt) (field 'output))
       (define expected (file->bytes (build-path data-dir ofn)))
       (define r (call-with-eager (lambda () (operator-apply name args (field 'attrs)))))
       (check-pred real-tensor? r (format "~a: result kind" name))
       (check-equal? (tensor-shape r) oshape (format "~a: shape" name))
       (check-equal? (tensor-dtype r) odt (format "~a: dtype" name))
       (define hc (ops:ctensor-contiguous (real-tensor-handle r)))
       (define got (raw:ctensor->bytes hc (bytes-length expected)))
       (if (equal? got expected)
           (check-true #t)
           (let ([maxdiff (for/fold ([m 0.0])
                                    ([x (in-list (decode odt got))]
                                     [y (in-list (decode odt expected))])
                            (max m (abs (- x y))))])
             (check-true #f (format "~a (~a): bytes differ, max abs diff ~a" name ofn maxdiff)))))
     (printf "operator parity: ~a cases compared\n" (length cases))]
    [else
     (displayln
      "parity prerequisites missing; run tools/gen-wrapper.rkt and tools/gen-reference.py")]))

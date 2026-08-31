#lang racket/base
(module+ test
  (require rackunit
           zhulong/runtime/ffi/raw)

  (cond
    [(libtorch-available?)
     (check-true (positive? (abi-version)))
     (check-regexp-match #rx"^2\\.13\\." (abi-version-string))
     (check-equal? (device-code 'cpu) 0)
     (check-equal? (dtype-code 'bf16) 15)
     (check-equal? (dtype-code 'f32) 6)
     (check-equal? (dtype-code 'i64) 4)
     (check-exn #rx"unknown key" (lambda () (dtype-code 'f64)))

     (define t (make-ctensor 'f32 '(2 3)))
     (check-equal? (ctensor-sizes t) '(2 3))
     (check-equal? (ctensor-strides t) '(3 1))
     (check-equal? (ctensor-dtype-code t) (dtype-code 'f32))
     (check-equal? (ctensor-device-code t) (device-code 'cpu))
     (check-equal? (ctensor-storage-offset t) 0)
     (ctensor-delete! t)

     (define bs
       (apply bytes-append
              (for/list ([x '(1.0 2.0 3.0 4.0)])
                (real->floating-point-bytes x 4 #f))))
     (define t2 (bytes->ctensor 'f32 '(2 2) bs))
     (check-equal? (ctensor->bytes t2 16) bs)
     (ctensor-delete! t2)

     ;; cast via copy_ (f32 -> bf16 -> f32); 1.5 is exact in bf16
     (define t15 (bytes->ctensor 'f32 '(1) (real->floating-point-bytes 1.5 4 #f)))
     (define tb (make-ctensor 'bf16 '(1)))
     (ctensor-copy! tb t15)
     (define t15* (make-ctensor 'f32 '(1)))
     (ctensor-copy! t15* tb)
     (check-equal? (ctensor-item-f32 t15*) 1.5)
     (for-each ctensor-delete! (list t15 tb t15*))

     (define ti (bytes->ctensor 'i64 '(1) (integer->integer-bytes 42 8 #t)))
     (check-equal? (ctensor-item-i64 ti) 42)
     (ctensor-delete! ti)

     ;; leak smoke: explicit create/delete cycle
     (for ([i (in-range 100000)])
       (ctensor-delete! (make-ctensor 'f32 '(8))))]
    [else (displayln "libtorch not present (run tools/setup-python.sh); skipping FFI smoke")]))

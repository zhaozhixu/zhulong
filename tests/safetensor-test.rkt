#lang racket/base
(module+ test
  (require rackunit
           racket/file
           zhulong/utils/safetensor)

  (define header
    (string->bytes/utf-8
     (string-append "{\"a\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]},"
                    "\"b\":{\"dtype\":\"I64\",\"shape\":[3],\"data_offsets\":[16,40]},"
                    "\"__metadata__\":{\"format\":\"pt\"}}")))
  (define path (make-temporary-file "st-~a.safetensors"))
  (call-with-output-file path
                         #:exists 'truncate
                         (lambda (out)
                           (write-bytes (integer->integer-bytes (bytes-length header) 8 #f #f) out)
                           (write-bytes header out)
                           (write-bytes (make-bytes 40 7) out)))

  (define st (safetensor-open path))
  (check-equal? (safetensor-names st) '(a b))
  (check-true (safetensor-member? st 'a))
  (check-false (safetensor-member? st 'nope))
  (check-equal? (safetensor-shape st 'a) '(2 2))
  (check-equal? (safetensor-dtype st 'a) 'f32)
  (check-equal? (safetensor-shape st 'b) '(3))
  (check-equal? (safetensor-dtype st 'b) 'i64)
  (check-equal? (safetensor-bytes st 'a) (make-bytes 16 7))
  (check-equal? (bytes-length (safetensor-bytes st 'b)) 24)
  (check-exn #rx"unknown tensor" (lambda () (safetensor-bytes st 'nope)))
  (delete-file path)

  (define bad-header
    (string->bytes/utf-8 "{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,4]}}"))
  (define bad-path (make-temporary-file "st-~a.safetensors"))
  (call-with-output-file bad-path
                         #:exists 'truncate
                         (lambda (out)
                           (write-bytes (integer->integer-bytes (bytes-length bad-header) 8 #f #f)
                                        out)
                           (write-bytes bad-header out)
                           (write-bytes (make-bytes 4 0) out)))
  (check-exn #rx"byte range does not match shape" (lambda () (safetensor-open bad-path)))
  (delete-file bad-path))

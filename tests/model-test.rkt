#lang racket/base
;; Layer parity: the Racket SmolLM2 forward against the torch reference,
;; checkpoint by checkpoint under a relative tolerance (bf16 forwards are
;; not bit-stable across implementations; see the design notes).
(module+ test
  (require rackunit
           racket/file
           racket/match
           racket/runtime-path
           zhulong
           (prefix-in ops: zhulong/runtime/ffi/ops)
           (prefix-in raw: zhulong/runtime/ffi/raw)
           models/llama)

  (define-runtime-path model-dir "data/model")
  (define-runtime-path weights-path "../weights/SmolLM2-135M/model.safetensors")
  (define manifest-path (build-path model-dir "manifest.rktd"))

  (define (bf16->f32-list bs)
    (for/list ([i (in-range 0 (bytes-length bs) 2)])
      (define u (integer-bytes->integer bs #f #f i (+ i 2)))
      (floating-point-bytes->real (integer->integer-bytes (arithmetic-shift u 16) 4 #f #f) #f)))

  (define (tensor->f32-list t)
    (define hc (ops:ctensor-contiguous (real-tensor-handle t)))
    (bf16->f32-list (raw:ctensor->bytes hc (* 2 (tensor-numel t)))))

  (cond
    [(not (and (file-exists? weights-path) (file-exists? manifest-path) (ops:eager-available?)))
     (displayln "model parity: weights or reference missing; run `make weights model-reference`")]
    [else
     (define manifest (with-input-from-file manifest-path read))
     (define ids (cdr (assq 'ids manifest)))
     (define checkpoints (cdr (assq 'checkpoints manifest)))
     (define m (llama-load weights-path smollm2-135m #:max-len (length ids)))
     (define-values (hs logits) (call-with-eager (lambda () (llama-forward m ids))))
     (define got (append hs (list logits)))
     (check-equal? (length got) (length checkpoints))
     (define worst 0.0)
     (for ([cp (in-list checkpoints)]
           [t (in-list got)])
       (match-define (list name shape file) cp)
       (check-equal? (tensor-shape t) shape (format "~a shape" name))
       (define expect (bf16->f32-list (file->bytes (build-path model-dir file))))
       (define actual (tensor->f32-list t))
       (define scale
         (for/fold ([s 1.0]) ([v (in-list expect)])
           (max s (abs v))))
       (define d
         (for/fold ([d 0.0])
                   ([a (in-list actual)]
                    [b (in-list expect)])
           (max d (abs (- a b)))))
       (set! worst (max worst (/ d scale)))
       (check-true (<= d (* 0.02 scale)) (format "~a: max|diff|=~a scale=~a" name d scale)))
     (printf "model parity: ~a checkpoints, worst relative diff ~a\n" (length checkpoints) worst)]))

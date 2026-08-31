#lang racket/base
;; Bindings for the custom C++ shim (build/libzl_shim.dylib): the few
;; kernels neither the stable ABI nor the generated per-backend shims
;; expose. Regenerate the dylib with tools/gen-wrapper.rkt.

(require ffi/unsafe
         racket/runtime-path
         "raw.rkt")

(provide shim-available?
         zl-sub
         zl-div
         zl-neg
         zl-rsqrt
         zl-silu
         zl-mean
         zl-softmax
         zl-embedding
         zl-argmax)

(define-runtime-path this-dir ".")

(define shim-path (simplify-path (build-path this-dir "../../.." "build/libzl_shim.dylib")))

(define shim (and (file-exists? shim-path) (ffi-lib shim-path)))

(define (shim-available?)
  (and shim #t))

(define (shim-fn name type)
  (unless shim
    (error 'shim-fn "shim not built at ~a; run: racket tools/gen-wrapper.rkt" shim-path))
  (get-ffi-obj name shim type))

(define (lazy-shim name type)
  (define f #f)
  (lambda args
    (unless f
      (set! f (shim-fn name type)))
    (apply f args)))

(define (unary-type who)
  (_fun _pointer
        (out : (_ptr o _pointer))
        ->
        (code : _int32)
        ->
        (begin
          (check-code who code)
          out)))

(define (binary-type who)
  (_fun _pointer
        _pointer
        (out : (_ptr o _pointer))
        ->
        (code : _int32)
        ->
        (begin
          (check-code who code)
          out)))

(define zl-sub (lazy-shim "zl_sub" (binary-type 'sub)))
(define zl-div (lazy-shim "zl_div" (binary-type 'div)))
(define zl-neg (lazy-shim "zl_neg" (unary-type 'neg)))
(define zl-rsqrt (lazy-shim "zl_rsqrt" (unary-type 'rsqrt)))
(define zl-silu (lazy-shim "zl_silu" (unary-type 'silu)))
(define zl-embedding (lazy-shim "zl_embedding" (binary-type 'embedding)))

(define zl-mean
  (lazy-shim "zl_mean"
             (_fun _pointer
                   (_list i _int64)
                   _int64
                   _int32
                   (out : (_ptr o _pointer))
                   ->
                   (code : _int32)
                   ->
                   (begin
                     (check-code 'mean code)
                     out))))

(define zl-softmax
  (lazy-shim "zl_softmax"
             (_fun _pointer
                   _int64
                   (out : (_ptr o _pointer))
                   ->
                   (code : _int32)
                   ->
                   (begin
                     (check-code 'softmax code)
                     out))))

(define zl-argmax
  (lazy-shim "zl_argmax"
             (_fun _pointer
                   _int64
                   _int32
                   (out : (_ptr o _pointer))
                   ->
                   (code : _int32)
                   ->
                   (begin
                     (check-code 'argmax code)
                     out))))

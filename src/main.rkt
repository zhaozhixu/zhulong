#lang racket/base
;; Language and library entry: `(require zhulong)` and `#lang zhulong` both
;; resolve here. Re-exports racket/base plus the public DSL API; the reader
;; submodule makes #lang zhulong work.

(require racket/base
         "lang.rkt"
         "tensor.rkt"
         "operator.rkt"
         "operators/elementwise.rkt"
         "operators/matmul.rkt"
         "operators/softmax.rkt"
         "operators/index.rkt"
         "eager.rkt")

(provide (all-from-out racket/base)
         (all-from-out "lang.rkt")
         (all-from-out "tensor.rkt")
         (all-from-out "operator.rkt")
         (all-from-out "operators/elementwise.rkt")
         (all-from-out "operators/matmul.rkt")
         (all-from-out "operators/softmax.rkt")
         (all-from-out "operators/index.rkt")
         (all-from-out "eager.rkt"))

(module reader syntax/module-reader
  zhulong)

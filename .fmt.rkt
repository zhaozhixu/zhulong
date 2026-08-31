#lang racket/base
;; raco fmt configuration: teach the formatter this project's define-like
;; DSL forms so their bodies indent like define (picked up automatically
;; when raco fmt runs from the repo root).

(require fmt)

(provide the-formatter-map)

(define (the-formatter-map s)
  (case s
    [("define-layer" "define-model" "define-operator" "define-operator-impl")
     (format-define-like)]
    [("struct") (format-define-like)]
    [else #f]))

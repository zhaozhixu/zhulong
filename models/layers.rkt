#lang zhulong
;; Shared layer library (application level). Layers are written in
;; branch-free dataflow form. v0: rms-norm, exercising the full
;; #lang zhulong chain.

(provide rms-norm)

(define-layer (rms-norm x #:eps [eps 1e-6])
  (mul x (rsqrt (add (mean (sqr x) #:axis -1 #:keep-dim? #t) eps))))
